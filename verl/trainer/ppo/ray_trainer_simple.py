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
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    Role, AdvantageEstimator, ResourcePoolManager, RayPPOTrainer
)

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainerSimple(RayPPOTrainer):
    """
    Simplified version of RayPPOTrainer focusing only on the generation step
    """

    def fit(self):
        """
        Simplified training loop that only focuses on generation using DataProto
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        print("Starting simplified training loop")
        
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
            
            # Pad to be divisible by dp_size
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
            
            with _timer('generation', timing_raw):
                print("Starting generation...")
                
                # Use the existing DataProto approach
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                
                # Unpad the results
                gen_batch_output = unpad_dataproto(gen_batch_output, pad_size=pad_size)
                
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