import ray
import torch
import numpy as np
from verl import DataProto
from verl.protocol import DataProtoFuture
from functools import partial

# Initialize Ray
ray.init(ignore_reinit_error=True)

def create_sample_dataproto(batch_size=8):
    """
    Creates a sample DataProto object with exactly the specified batch size
    """
    seq_length = 128
    
    # Common inputs for language models
    input_ids = torch.randint(0, 50000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    position_ids = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, -1)
    
    # Create tensor dictionary
    tensors = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    }
    
    # Create non-tensor data
    raw_prompt_ids = [input_ids[i].tolist() for i in range(batch_size)]
    non_tensors = {
        'raw_prompt_ids': raw_prompt_ids
    }
    
    # Create DataProto using from_dict
    data_proto = DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors
    )
    
    return data_proto

def create_dataproto_future_with_all_futures():
    """
    Creates a single DataProtoFuture with futures for all 8 chunks
    """
    # Number of workers
    num_workers = 8
    
    # Create a sample DataProto with batch size 8
    data_proto = create_sample_dataproto(batch_size=8)
    print(f"Created DataProto with batch size: {len(data_proto)}")
    
    # Split the DataProto into 8 chunks
    data_chunks = data_proto.chunk(num_workers)
    print(f"Split into {len(data_chunks)} chunks of size {len(data_chunks[0])} each")
    
    # Create Ray futures for each chunk
    futures = []
    for chunk in data_chunks:
        future = ray.put(chunk)
        futures.append(future)
    
    # Create a single DataProtoFuture with all 8 futures
    data_future = DataProtoFuture(
        collect_fn=DataProto.concat,  # Function to combine results
        futures=futures
    )
    
    print(f"Created a single DataProtoFuture with {len(futures)} futures")
    
    return data_future

if __name__ == "__main__":
    # Create a DataProtoFuture with all 8 futures
    data_future = create_dataproto_future_with_all_futures()
    
    # Demonstrate how to get the combined result
    print("\nRetrieving the combined result:")
    result = data_future.get()
    print(f"Combined result has batch size: {len(result)}")
    print(f"Result tensor keys: {list(result.batch.keys())}")
    
    # Clean up Ray
    ray.shutdown()