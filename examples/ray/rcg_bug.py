import ray
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from tensordict import TensorDict
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.channel.torch_tensor_type import TorchTensorType


# Simplified DataProto class
@dataclass
class DataProto:
    batch: TensorDict = None

    @classmethod
    def from_dict(cls, tensors):

        tensor_dict = TensorDict(
            source=tensors, batch_size=list(tensors.values())[0].shape[:1]
        )
        return cls(batch=tensor_dict)

    @staticmethod
    def concat(data: List["DataProto"]) -> "DataProto":
        """Concat a list of DataProto. The batch is concatenated among dim=0."""
        batch_lst = [proto.batch for proto in data if proto.batch is not None]

        if len(batch_lst) > 0:
            new_batch = TensorDict.cat(batch_lst, dim=0)
        else:
            new_batch = None

        return DataProto(batch=new_batch)

    def chunk(self, chunks: int) -> Tuple["DataProto"]:
        """Split the batch among dim=0 into chunks."""
        assert self.batch is not None, "Batch is None"
        batch_lst = self.batch.chunk(chunks=chunks, dim=0)

        return tuple(DataProto(batch=batch) for batch in batch_lst)

    def union(self, data_proto: "DataProto") -> "DataProto":
        """Union two DataProto objects by merging their tensor dicts."""
        for key, val in data_proto.batch.items():
            if key not in self.batch.keys():
                self.batch[key] = val
            else:
                assert self.batch[key].equal(
                    val
                ), f"{key} in two DataProto objects are not the same"
        return self


# Mock worker classes
@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank=0):
        self.state = torch.zeros(1, dtype=torch.int64).to("cuda") + rank

    def generate_sequences(self, data: DataProto) -> DataProto:
        # Simulate response generation

        data.batch = data.batch.to("cuda")
        new_input_ids = data.batch["input_ids"] + self.state
        return DataProto.from_dict(
            {
                "input_ids": new_input_ids,
                "old_log_probs": torch.randn_like(new_input_ids, dtype=torch.float),
            }
        )

    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        # Simulate reference policy computation
        data.batch = data.batch.to("cuda")
        ref_log_prob = torch.randn_like(data.batch["input_ids"], dtype=torch.float)
        return DataProto.from_dict({"ref_log_prob": ref_log_prob})

    def compute_values(self, data: DataProto) -> DataProto:
        # Simulate value computation
        data.batch = data.batch.to("cuda")
        values = torch.rand_like(data.batch["input_ids"], dtype=torch.float)
        return DataProto.from_dict({"values": values})

    def compute_advantages(self, data: DataProto) -> DataProto:
        # Simulate advantages computation
        data.batch = data.batch.to("cuda")
        values = data.batch["values"]
        ref_log_prob = data.batch["ref_log_prob"]
        old_log_probs = data.batch["old_log_probs"]

        rewards = torch.rand_like(values)  # Simulate base rewards
        kl_diff = (old_log_probs - ref_log_prob).abs()  # Simple KL penalty
        token_level_rewards = rewards - 0.1 * kl_diff  # Apply KL penalty

        advantages = token_level_rewards - values
        returns = advantages + values

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.batch["token_level_rewards"] = token_level_rewards

        return data

    def update_actor(self, data: DataProto) -> Dict[str, float]:
        # Simulate actor update
        data.batch = data.batch.to("cuda")
        metrics = {"loss": torch.rand_like(data.batch["advantages"]).sum().item()}
        return metrics

    def union(self, *data_protos: DataProto) -> DataProto:
        """Union tensor dicts of multiple DataProto objects"""
        data = data_protos[0]
        for proto in data_protos[1:]:
            data.union(proto)
        return data

    def concat(self, *data_protos: DataProto) -> DataProto:
        """Concatenate DataProto objects sharded on dim=0"""
        data = DataProto.concat(data_protos)
        return data

    def chunk(self, data: DataProto, chunks: int) -> Tuple[DataProto]:
        """Split the batch among dim=0 into chunks."""
        return data.chunk(chunks=chunks)


def gather_to_rank_zero(
    workers: List[Worker], sharded_data: List[DataProto], nccl: bool = False
) -> DataProto:
    assert len(workers) == len(sharded_data)

    type_hinted_data = [sharded_data[0]]
    for data in sharded_data[1:]:
        type_hinted_data.append(
            data.with_type_hint(TorchTensorType(transport="nccl")) if nccl else data
        )
    return workers[0].concat.bind(*type_hinted_data)


def scatter_from_rank_zero(
    workers: List[Worker], data: DataProto, nccl: bool = False
) -> Tuple[DataProto]:
    n_shards = len(workers)
    sharded_data = workers[0].chunk.options(num_returns=n_shards).bind(data, n_shards)
    type_hinted_data = [sharded_data[0]]
    for data in sharded_data[1:]:
        type_hinted_data.append(
            data.with_type_hint(TorchTensorType(transport="nccl")) if nccl else data
        )
    return tuple(type_hinted_data)


def main():
    # Initialize Ray
    ray.init(runtime_env={"env_vars": {"RAY_ADAG_ENABLE_DETECT_DEADLOCK": "0"}})

    # Create 4 worker instances
    workers = [Worker.remote() for _ in range(4)]

    # Build compiled graph
    with InputNode() as graph_input:
        # First stage: generate sequences
        gen_output = [
            worker.generate_sequences.bind(graph_input[i])
            for i, worker in enumerate(workers)
        ]

        # Second stage: parallel ref policy and critic computation
        ref_output = [
            worker.compute_ref_log_prob.bind(gen_output[i])
            for i, worker in enumerate(workers)
        ]
        critic_output = [
            worker.compute_values.bind(gen_output[i])
            for i, worker in enumerate(workers)
        ]

        # Union all the outputs
        union_output = [
            worker.union.bind(gen_output[i], ref_output[i], critic_output[i])
            for i, worker in enumerate(workers)
        ]

        # Third stage: compute advantages on worker 0
        ## Gather
        concat_input = gather_to_rank_zero(workers, union_output, nccl=True)
        ## Compute
        advantage_data = workers[0].compute_advantages.bind(concat_input)
        ## Scatter
        train_data = scatter_from_rank_zero(workers, advantage_data, nccl=True)

        # Fourth stage: update actor on worker 0
        metrics = [
            worker.update_actor.bind(train_data[i]) for i, worker in enumerate(workers)
        ]

        dag = MultiOutputNode(metrics)

    # Compile the graph
    print("Compiling graph...")
    compiled_dag = dag.experimental_compile()
    print("Success!")

    # Execute the graph
    input_data = [
        DataProto.from_dict(
            {
                "input_ids": torch.randint(0, 100, (4, 10)),
            }
        )
        for _ in range(4)
    ]

    print("Executing graph...")
    metrics = ray.get(compiled_dag.execute(*input_data))
    print("Success!")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
