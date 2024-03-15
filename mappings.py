import torch


def _gather(input_: torch.Tensor, group_:torch.distributed.ProcessGroup) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    group = group_

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
