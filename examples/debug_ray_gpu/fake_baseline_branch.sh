set -x

export EXP_NAME='fake_baseline_branch'
export WANDB_API_KEY='34b8f32abb7ba71277361c99f84d9bea484b5d3b'
export BATCH_MULTIPLIER=1
python3 -m verl.trainer.main_ppo \
    data.train_files=/mnt/cluster_storage/data/gsm8k/train.parquet \
    data.val_files=/mnt/cluster_storage/data/gsm8k/test.parquet \
    data.train_batch_size=$((1024 * BATCH_MULTIPLIER)) \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.fake_data=True \
    data.fake_data_emb_size=1024 \
    actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((256 * BATCH_MULTIPLIER)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$((16 * BATCH_MULTIPLIER)) \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$((32 * BATCH_MULTIPLIER)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$((32 * BATCH_MULTIPLIER)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=deepseek-ai/deepseek-llm-7b-chat \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=$((32 * BATCH_MULTIPLIER)) \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_ray_gpu_obj' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.default_local_dir=/mnt/local_storage/verl/$EXP_NAME \
    trainer.total_epochs=3 \
    trainer.resume_mode=disable \
    trainer.materialize_data=True \
    +trainer.val_before_train=False $@



