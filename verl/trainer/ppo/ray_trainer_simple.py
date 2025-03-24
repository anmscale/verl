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
import uuid
import numpy as np
from contextlib import contextmanager
from typing import Dict

from codetiming import Timer
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoFuture
from verl.trainer.ppo.ray_trainer import (
    Role, AdvantageEstimator, ResourcePoolManager, RayPPOTrainer, apply_kl_penalty, compute_advantage, reduce_metrics, compute_data_metrics, compute_timing_metrics
)
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs

import ray
import torch

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainerSimple(RayPPOTrainer):
    """
    Simplified version of RayPPOTrainer focusing only on the generation step
    """
    
    def _init_resource_pool(self):
        """Initialize resource pool"""
        super()._init_resource_pool()
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Scoring)
        scoring_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Scoring],
            config=self.config,
            reward_fn=self.reward_fn,
            kl_ctrl=self.kl_ctrl,
        )
        self.resource_pool_to_cls[resource_pool]["scoring"] = scoring_cls

    def _init_colocated_worker_groups(self):
        super()._init_colocated_worker_groups()
        
        # Initialize scoring worker group
        self.scoring_wg = self.all_wg["scoring"]
        self.scoring_wg.init_model()
        

    def fit(self):
        """
        Simplified training loop that only focuses on generation using DataProto
        
        Args:
            use_future (bool): Whether to use DataProtoFuture for asynchronous execution
        """
        from verl.utils.tracking import Tracking
        
        self.global_steps = 0
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        print(f"Starting simplified training loop with 'DataProtoFuture'")
        
        # Process a single batch
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}
            
            
            # Convert to DataProto
            batch = DataProto.from_single_dict(batch_dict)
            # Add unique IDs to the batch
            batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                        dtype=object)
            with _timer('step', timing_raw):
                # Generate sequences
                batch = self.actor_rollout_wg.generate_sequences(batch)
                assert isinstance(batch, DataProtoFuture)
                
                # # Repeat to align with repeated responses in rollout (if needed)
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                
                # # Balance the batch for better performance across workers
                # self._balance_batch(batch, metrics=metrics)
                
                # # Track token counts for metrics
                # batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                
                # Compute log probabilities for the generated sequences
                batch = self.actor_rollout_wg.compute_log_prob(batch)
                assert isinstance(batch, DataProtoFuture)
                
                # Compute reference policy log probabilities if available
                if self.use_reference_policy:
                    batch = self.ref_policy_wg.compute_ref_log_prob(batch)
                    assert isinstance(batch, DataProtoFuture)
                    
                # Compute values if using critic
                if self.use_critic:
                    batch = self.critic_wg.compute_values(batch)
                    assert isinstance(batch, DataProtoFuture)

                # Compute reward scores
                batch = self.scoring_wg.compute_token_level_scores(batch)
                assert isinstance(batch, DataProtoFuture)
                
                # Compute advantages
                # TODO: get rid of materialized batch
                batch_materialized = batch.get()
                batch = compute_advantage(batch_materialized,
                                         adv_estimator=self.config.algorithm.adv_estimator,
                                         gamma=self.config.algorithm.gamma,
                                         lam=self.config.algorithm.lam)
                assert isinstance(batch, DataProto)
                training_output = []
                
                # Update critic if using critic
                if self.use_critic:
                    critic_output = self.critic_wg.update_critic(batch)
                    assert isinstance(critic_output, DataProtoFuture)
                    training_output.append(critic_output)
                
                # Update actor
                actor_output = self.actor_rollout_wg.update_actor(batch)
                assert isinstance(actor_output, DataProtoFuture)
                training_output.append(actor_output)
                
                # Get all outputs and reduce metrics
                training_output = [x.get() for x in training_output]
                output_metrics = [reduce_metrics(o.meta_info['metrics']) for o in training_output]
                for om in output_metrics:
                    metrics.update(om)
            
            # Collect metrics
            # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            
            # Log metrics
            logger.log(data=metrics, step=self.global_steps)
            self.global_steps += 1
            
            # Only process three batches for this simple example
            if self.global_steps == 3:
                break
        
        print("Simplified training loop completed") 