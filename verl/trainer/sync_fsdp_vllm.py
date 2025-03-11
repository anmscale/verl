import os
import torch
import torch.distributed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig
)
from vllm import LLM, SamplingParams
from typing import Dict
import warnings
import time

import functools
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name
from torch.distributed.device_mesh import init_device_mesh

from verl.utils import hf_tokenizer
from verl.utils.model import print_model_size, update_model_config

# Suppress warnings
warnings.filterwarnings("ignore")

# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
def get_fsdp_wrap_policy(module, config=None):
    if config is None:
        config = {}

    if config.get('disable', False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
                                                    default_transformer_cls_names_to_wrap)
    min_num_params = config.get('min_num_params', 0)
    auto_wrap_policy = None
    if min_num_params > 0:
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
    return auto_wrap_policy

def init_fn(x: torch.nn.Module):
    if not torch.distributed.get_rank() == 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x

def get_init_weight_context_manager(use_meta_tensor=True):
    from accelerate import init_empty_weights
    cpu_init_weights = lambda: torch.device('cpu')
    if use_meta_tensor:
        init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context

def init_distributed():
    """Initialize distributed training environment"""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank

def build_fsdp_model(model_path: str, local_rank: int):
    """Build FSDP-wrapped DeepSeek model"""
    print(f"[Rank {local_rank}] Loading model from {model_path}")
    
    local_path = model_path
    world_size = torch.distributed.get_world_size()
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
    # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
    tokenizer = hf_tokenizer(local_path, trust_remote_code=False)

    torch_dtype = torch.bfloat16 


    # override model kwargs
    actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=False)

    override_config_kwargs = {
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
    }
    
    update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)


    # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
    init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                            torch_dtype=torch_dtype,
                                                            config=actor_model_config,
                                                            attn_implementation='flash_attention_2',
                                                            trust_remote_code=False)
        # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
        actor_module.to(torch_dtype)


    torch.distributed.barrier()

    mixed_precision = None

    auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config={'min_num_params': 0})

    print(f'wrap_policy: {auto_wrap_policy}')

    if auto_wrap_policy is None:
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD

    actor_module_fsdp = FSDP(
        actor_module,
        param_init_fn=init_fn,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=sharding_strategy,  # zero3
        mixed_precision=mixed_precision,
        sync_module_states=True,
        device_mesh=device_mesh)


    print(f"[Rank {local_rank}] Model loaded successfully")
    return actor_module_fsdp, tokenizer, actor_model_config

        

def build_vllm_engine(actor_module_fsdp, tokenizer, model_config, local_rank: int):
    """Build vLLM inference engine"""
    print(f"[Rank {local_rank}] Initializing vLLM engine...")
    
    # from verl/verl/trainer/config/ppo_trainer.yaml
    vllm_config = {
        "tensor_parallel_size": torch.distributed.get_world_size(),
        "dtype": "bfloat16",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.5,
        "max_model_len": 1024,
        "skip_tokenizer_init": False,
        "load_format": "dummy_dtensor"
    }
    
    # Import from verl's third_party instead of vllm directly
    from verl.third_party.vllm import LLM
    
    inference_engine = LLM(
        actor_module_fsdp,
        tokenizer=tokenizer,
        model_hf_config=model_config,
        tensor_parallel_size=vllm_config["tensor_parallel_size"],
        dtype=vllm_config["dtype"],
        enforce_eager=vllm_config["enforce_eager"],
        gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
        max_model_len=vllm_config["max_model_len"],
        skip_tokenizer_init=vllm_config["skip_tokenizer_init"],
        load_format=vllm_config["load_format"]
    )
    
    # Offload model weights to reduce peak memory usage
    inference_engine.offload_model_weights()
    
    print(f"[Rank {local_rank}] vLLM engine initialized")
    return inference_engine

def sync_weights(fsdp_model, vllm_engine, local_rank: int):
    """Synchronize weights from FSDP model to vLLM engine"""
    print(f"[Rank {local_rank}] Starting weight synchronization...")
    
    torch.cuda.synchronize()
    sync_start = time.perf_counter()
    
    with FSDP.state_dict_type(fsdp_model,
                             state_dict_type=StateDictType.SHARDED_STATE_DICT,
                             state_dict_config=ShardedStateDictConfig()):
        actor_weights = fsdp_model.state_dict()
    
    vllm_engine.sync_model_weights(actor_weights, load_format='dtensor')
    
    torch.cuda.synchronize()
    sync_time = time.perf_counter() - sync_start
    
    print(f"[Rank {local_rank}] weight synchronization completed in {sync_time:.2f} seconds")


def main():
    local_rank = init_distributed()
    world_size = torch.distributed.get_world_size()
    print(f"[Rank {local_rank}] Initialized distributed environment (World size: {world_size})")
    
    model_path = "deepseek-ai/deepseek-llm-7b-chat"
    
    try:
        # Build FSDP model
        model_fsdp, tokenizer, model_config = build_fsdp_model(model_path, local_rank)
        
        # Build vLLM engine
        vllm_engine = build_vllm_engine(model_fsdp, tokenizer, model_config, local_rank)
        
        # weight sync and then test generation
        sync_weights(model_fsdp, vllm_engine, local_rank)
        
        prompt = "Human: Write a short story about a robot learning to paint.\n\nAssistant:"
        sampling_params = SamplingParams(
            n=1,
            temperature=0.7,
            max_tokens=100,
            stop=["Human:", "\n\nHuman:"]
        )
        
        outputs = vllm_engine.generate(
            prompts=[prompt],
            sampling_params=sampling_params
        )
        if local_rank == 0:
            token_ids = outputs[0].tolist()  
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)

            print(f"Generated text: {generated_text}")
        
            
    except Exception as e:
        print(f"[Rank {local_rank}] Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
