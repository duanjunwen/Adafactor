from typing import Any

import torch
import torch.distributed as dist
from initialize import get_model_parallel_group
from utils import split_tensor_along_last_dim

# torch.Tensor should be a cuda tensor
def _reduce(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # print(f"group {group}")
    # # # All-reduce.
    # output_ = input_.contiguous()
    # print(f"input_ before {input_}")
    dist.all_reduce(input_, group=group)
    # torch.distributed.all_reduce(input_, op=torch.distributed.ReduceOp.SUM)
    # print(f"input_ after {input_}")

    return input_


def _split(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output
