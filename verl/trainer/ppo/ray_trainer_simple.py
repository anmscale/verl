# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simplified version of RayPPOTrainer focusing only on the generation step
"""

import os
from contextlib import contextmanager
from typing import Dict

from codetiming import Timer
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoFuture
from verl.trainer.ppo.ray_trainer import (
    Role, AdvantageEstimator, ResourcePoolManager, RayPPOTrainer
)
import ray

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainerSimple(RayPPOTrainer):
    """
    Simplified version of RayPPOTrainer focusing only on the generation step
    """

    def fit(self, use_future=True):
        """
        Simplified training loop that only focuses on generation using DataProto
        
        Args:
            use_future (bool): Whether to use DataProtoFuture for asynchronous execution
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        print(f"Starting simplified training loop with {'DataProtoFuture' if use_future else 'DataProto'}")
        
        # Process a single batch
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}
            
            # Convert to DataProto
            batch = DataProto.from_single_dict(batch_dict)
            
            # Extract generation inputs
            if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )
            
            with _timer('generation', timing_raw):
                print("Starting generation...")
                
                if use_future:
                    # Simply put the entire DataProto in Ray's object store
                    future = ray.put(gen_batch)
                    
                    # Create a DataProtoFuture with a single future
                    # The worker group will handle the chunking internally
                    gen_batch_future = DataProtoFuture(
                        collect_fn=lambda x: x[0],  # Just return the first (and only) element
                        futures=[future]
                    )
                    # Use DataProtoFuture for asynchronous execution
                    gen_batch_output_future = self.actor_rollout_wg.generate_sequences(gen_batch_future)
                    
                    old_log_prob_future = self.actor_rollout_wg.compute_log_prob(gen_batch_output_future)
                    
                    # When we need the results, we call get()
                    old_log_prob = old_log_prob_future.get()
                    gen_batch_output = gen_batch_output_future.get()
                    breakpoint()
                    
                else:
                    # Original approach using DataProto directly
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    
                
                print("Generation completed!")
            
            # Log some basic metrics
            metrics['generation_time'] = timing_raw['generation']
            metrics['batch_size'] = len(batch)
            
            # Print generated text for the first example
            input_ids = gen_batch.batch['input_ids'][0]
            response_ids = gen_batch_output.batch['responses'][0]
            
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            print(f"Input: {input_text[:100]}...")
            print(f"Generated: {response_text[:100]}...")
            
            # Log metrics
            logger.log(data=metrics, step=0)
            
            # Only process one batch for this simple example
            break
        
        print("Simplified training loop completed") 