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

import time
import torch
import torch.distributed as dist
import logging
import numpy as np
from verl import DataProto



def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        message = f'{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}'

        # if logger is None:
        print(message)
        # else:
        #     logger.log(msg=message, level=level)


def log_cuda_time(head: str, start_event: torch.cuda.Event, end_event: torch.cuda.Event, 
                 start_perf: float = None, logger: logging.Logger = None, 
                 level=logging.DEBUG, rank: int = 0):
    """Log both CUDA event time and perf_counter time.
    
    Args:
        head: Message prefix
        start_event: Starting CUDA event
        end_event: Ending CUDA event
        start_perf: Starting perf_counter time
        logger: Optional logger instance
        level: Logging level
        rank: Rank to log from (None to log from all ranks)
    """
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        torch.cuda.synchronize()
        cuda_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        
        # Get perf_counter timing if provided
        perf_time = None
        if start_perf is not None:
            perf_time = time.perf_counter() - start_perf
            
        # Construct message
        message = f'TIMER: {head}'
        message += f', CUDA time (s): {cuda_time:.4f}'
        if perf_time is not None:
            message += f', perf_counter time (s): {perf_time:.4f}'

        # if logger is None:
        # else:
        #     logger.log(msg=message, level=level)
        print(message)

def log_data_size(head: str, data: DataProto):
    # Calculate total size in kilobytes for tensor batch
    tensor_size_kb = 0
    device = None
    if data.batch is not None:
        for tensor in data.batch.values():
            if isinstance(tensor, torch.Tensor):
                tensor_size_kb += (tensor.element_size() * tensor.numel()) / 1024
                device = tensor.device
                    
    # Calculate size for non-tensor batch (numpy arrays)
    non_tensor_size_kb = 0
    for arr in data.non_tensor_batch.values():
        if isinstance(arr, np.ndarray):
            non_tensor_size_kb += arr.nbytes / 1024
            
    total_size_kb = tensor_size_kb + non_tensor_size_kb
    
    print(f'{head}, {total_size_kb:.2f} KB, '
          f'(tensor: {tensor_size_kb:.2f} KB, non-tensor: {non_tensor_size_kb:.2f} KB), '
          f'device: {device if device is not None else "N/A"}')
