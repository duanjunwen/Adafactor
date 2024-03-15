import os
import time
from prettytable import PrettyTable

import torch
from torch import nn
import torch.distributed as dist

from mappings import _gather
from adafactor import (
    Adafactor, 
    DistributedAdaFactor,
)

from colossalai.tensor.d_tensor import (
    distribute_tensor,
    ShardingSpec,
)
from colossalai.device.device_mesh import DeviceMesh


# dist env
def init_dist():
    rank = int(os.environ['RANK']) 
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    torch.cuda.set_device(local_rank)

def correctness_verify(tensor1: torch.Tensor, tensor2: torch.Tensor):
    return torch.all(tensor1.isclose(tensor2, rtol=1e-05, atol=1e-05, equal_nan=True))
    # return torch.testing.assert_close(tensor1, tensor2,  rtol=1e-05, atol=1e-05, equal_nan=True)
    
def error_idx(tensor1: torch.Tensor, tensor2: torch.Tensor):
    return torch.isclose(tensor1, tensor2, rtol=1e-05, atol=1e-05, equal_nan=True)

def get_time():
    torch.cuda.synchronize()
    return time.time()

def main():
    # ==============================
    # torch distributed init
    # ==============================
    torch.manual_seed(0)
    device = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    tensor_parallel_size = world_size
    init_dist()
    
    # ==============================
    # Param Info
    # ==============================
    H, W = 4096, 4096
    model_base = nn.Linear(H, W).to(device)  
    
    # ==============================
    # Param init
    # ==============================
    # base param
    # layer shape [H, W], then
    # weight [W, H] [4, 2]
    # bias [W]  [4]
    weight, bias = model_base.weight, model_base.bias
    # physical_mesh_id: torch.Tensor[0,1,2,3]
    # logical_mesh_id: (DP size, TP size); WORLD SIZE = DP size, TP size; 2 GPU view as (1,2); 4 GPU view as (1,4) or (2,2)
    device_mesh = DeviceMesh(torch.Tensor([i for i in range(world_size)]), (1, tensor_parallel_size), init_process_group=True)
    sharding_spec = ShardingSpec(dim_size=weight.dim(), dim_partition_dict={weight.dim() - 1: [1]})
    weight_shard = distribute_tensor(weight, device_mesh, sharding_spec)
    
    # local_weight [W*H/N] [4*1]
    # local_bias [W]  [4]
    # flatten param; TP first, then flatten ;
    local_weight_flatten = nn.Parameter(weight_shard.clone().flatten().requires_grad_(True))
    local_bias_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))


    # ==============================
    # Adafactor Base
    # ==============================
    torch.cuda.synchronize()
    optimizer_base = Adafactor([weight, bias])
    optimizer_base.zero_grad()
    weight.grad = torch.rand_like(weight)
    bias.grad = torch.rand_like(bias)
    optimizer_base.step()

    # # ==============================
    # # DistributedAdafactor
    # # ==============================
    torch.cuda.synchronize()
    optimizer_zero2 = DistributedAdaFactor([local_weight_flatten, local_bias_flatten], param_height = W, param_width = H, device_mesh=device_mesh, sharding_spec=sharding_spec)
    optimizer_zero2.zero_grad()
    local_weight_flatten.grad = distribute_tensor(weight.grad, device_mesh, sharding_spec).clone().flatten()
    local_bias_flatten.grad = bias.grad.clone().flatten()
    optimizer_zero2.step()
    
    # ==============================
    # Correctness Verify
    # ==============================
    # tensor parallel & flatten view &gather data
    torch.cuda.synchronize()
    reshape_flatten_weight = local_weight_flatten.view(-1, H // tensor_parallel_size) # reshape
    gather_flatten_weight = _gather(reshape_flatten_weight.data, device_mesh.get_process_group(axis=1)) # gather
    weight_correctness = correctness_verify(weight.data, gather_flatten_weight)
    bias_correctness = correctness_verify(bias.data, local_bias_flatten.data)
    if weight_correctness:
        print(f"Distributed weight correctness Pass")
    else:
        print(f"Distributed weight inclued incorrectness value")
        weight_err_idx = error_idx(weight.data, gather_flatten_weight.data)
        print(f"Distributed weight err idx {weight_err_idx}")
    if bias_correctness:
        print(f"Distributed bias correctness Pass")
    else:
        print(f"Distributed bias inclued incorrectness value")
        bias_err_idx = error_idx(bias.data, local_bias_flatten.data)
        print(f"Distributed bias err idx {bias_err_idx}")

    # ==============================
    # Runtime Test
    # ==============================
    niter = 50
    base_start, base_end, base_runtime = 0, 0, 0
    zero_start, zero_end, zero_runtime, zero_best_runtime = 0, 0, 0, float('inf')
    table = PrettyTable(['Version', 'weight shape', 'bias shape', 'Avg runtime(ms)',
                        'Speed Up Rate', 'Best Speed Up Rate'], float_format='.2')
    for i in range(0, niter):
        # Base Adafactor
        optimizer_base.zero_grad()
        weight.grad = torch.rand_like(weight)
        bias.grad = torch.rand_like(bias)
        base_start = get_time()
        optimizer_base.step()
        base_end = get_time()
        
        # Distributed Adafactor
        optimizer_zero2.zero_grad()
        local_weight_flatten.grad = distribute_tensor(weight.grad, device_mesh, sharding_spec).clone().flatten()
        local_bias_flatten.grad = bias.grad.clone().flatten()
        zero_start = get_time()
        optimizer_zero2.step()
        zero_end = get_time()
    
        reshape_flatten_weight = local_weight_flatten.view(-1, H // tensor_parallel_size) # reshape
        gather_flatten_weight = _gather(reshape_flatten_weight.data, device_mesh.get_process_group(axis=1)) # gather

        torch.cuda.synchronize()
        v3_weight_correctness = correctness_verify(weight.data, gather_flatten_weight)
        v3_bias_correctness = correctness_verify(bias.data, local_bias_flatten.data)
        
        
        print(f"iter {i}")
        if v3_weight_correctness:
            print(f"Distributed weight correctness Pass")
        else:
            print(f"Distributed weight inclued incorrectness value")
            weight_err_idx = error_idx(weight.data, gather_flatten_weight.data)
            print(f"Distributed weight err idx {weight_err_idx}")
                
        if v3_bias_correctness:
            print(f"Distributed bias correctness Pass")
        else:
            print(f"Distributed bias inclued incorrectness value")
            bias_err_idx = error_idx(bias.data, local_bias_flatten.data)
            print(f"Distributed bias err idx {bias_err_idx}")


        base_runtime += base_end - base_start
        zero_runtime  += zero_end - zero_start
        zero_best_runtime= min(zero_best_runtime, zero_runtime)

    table = PrettyTable(['Version', 'weight shape', 'bias shape', 'Avg runtime(ms)',
                        'Speed Up Rate', 'Best Speed Up Rate'], float_format='.2')
    table.add_row(["AdaFactor", weight.shape, bias.shape, (base_runtime / niter) * 10.0**3, None, None])
    table.add_row(["DistributedAdaFactor", weight.shape, bias.shape, (zero_runtime / niter) * 10.0**3, base_runtime/zero_runtime ,base_runtime/zero_best_runtime])
    
    print(table)

if __name__ == "__main__":
    main()