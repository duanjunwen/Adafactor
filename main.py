import os
import time
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.distributed as dist
# import torch.optim as optim
from adafactor import (
    Adafactor, 
    DistributedAdaFactor,
)
import initialize as fs_init
from mappings import (
    _split,
    _gather
)
from prettytable import PrettyTable
from colossalai.tensor.d_tensor import (
    distribute_tensor,
    distribute_tensor_with_customization,
    get_device_mesh,
    get_sharding_spec,
    init_as_dtensor,
    init_tensor_as_customization_distributed,
    is_customized_distributed_tensor,
    is_distributed_tensor,
)


# dist env
def init_dist():
    rank = int(os.environ['RANK']) 
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    fs_init.initialize_model_parallel(model_parallel_size_= world_size)
    torch.cuda.set_device(local_rank)

def correctness_verify(tensor1: torch.Tensor, tensor2: torch.Tensor):
    return torch.all(tensor1.isclose(tensor2, rtol=1e-05, atol=1e-05, equal_nan=True))
    # return torch.testing.assert_close(tensor1, tensor2,  rtol=1e-05, atol=1e-04, equal_nan=True)
    
def error_idx(tensor1: torch.Tensor, tensor2: torch.Tensor):
    return torch.isclose(tensor1, tensor2, rtol=1e-05, atol=1e-04, equal_nan=True)

def get_time():
    torch.cuda.synchronize()
    return time.time()

def main():
    torch.manual_seed(0)
    device = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    init_dist()
    tensor_parallel_size = fs_init.get_model_parallel_world_size() 
    
    H, W = 4096, 4096
    model_base = nn.Linear(H, W).to(device)  
    
    # base param
    # layer shape [H, W], then
    # weight [W, H] [4, 2]
    # bias [W]  [4]
    weight, bias = model_base.weight, model_base.bias
    
    # local_weight [W*H/N] [4*1]
    # local_bias [W]  [4]
    # flatten param; TP first, then flatten ;
    local_weight_flatten = nn.Parameter(_split(weight).clone().flatten().requires_grad_(True))
    local_bias_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))


    # # ==============================
    # # Adafactor Base
    # # ==============================
    torch.cuda.synchronize()
    optimizer_base = Adafactor([weight, bias])
    optimizer_base.zero_grad()
    weight.grad = torch.rand_like(weight)
    bias.grad = torch.rand_like(bias)
    optimizer_base.step()

    # # ==============================
    # # Adafactor Tensor Parallel v0.3
    # # ==============================
    torch.cuda.synchronize()
    optimizer_zero2 = DistributedAdaFactor([local_weight_flatten, local_bias_flatten], param_height = W, param_width = H)
    optimizer_zero2.zero_grad()
    local_weight_flatten.grad = _split(weight.grad).clone().flatten()
    local_bias_flatten.grad = bias.grad.clone().flatten()
    optimizer_zero2.step()
    
    # ==============================
    # Correctness Verify
    # ==============================
    # tensor parallel & flatten view &gather data
    torch.cuda.synchronize()
    reshape_flatten_weight = local_weight_flatten.view(-1, H // tensor_parallel_size) # reshape
    gather_flatten_weight = _gather(reshape_flatten_weight.data) # gather
    weight_correctness = correctness_verify(weight.data, gather_flatten_weight)
    bias_correctness = correctness_verify(bias.data, local_bias_flatten.data)
    if weight_correctness:
        print(f"Distributed weight correctness {weight_correctness}")
    else:
        print(f"Distributed weight inclued incorrectness value")
        weight_err_idx = error_idx(weight.data, gather_flatten_weight.data)
        print(f"Distributed weight err idx {weight_err_idx}")
    if bias_correctness:
        print(f"Distributed bias correctness {bias_correctness}")
    else:
        print(f"Distributed bias inclued incorrectness value")
        bias_err_idx = error_idx(bias.data, local_bias_flatten.data)
        print(f"Distributed bias err idx {bias_err_idx}")

    # # ==============================
    # # Run training epoch
    # # ==============================
    niter = 1
    # runtime test
    base_start, base_end, base_runtime = 0, 0, 0
    zero_start, zero_end, zero_runtime, zero_best_runtime = 0, 0, 0, float('inf')
    table = PrettyTable(['Version', 'weight shape', 'bias shape', 'Avg runtime(ms)',
                        'Speed Up Rate', 'Best Speed Up Rate'], float_format='.2')
    for i in range(0, niter):
        # Base optim
        optimizer_base.zero_grad()
        weight.grad = torch.rand_like(weight)
        bias.grad = torch.rand_like(bias)
        base_start = get_time()
        optimizer_base.step()
        base_end = get_time()
        
        # TP+Zero2 optim
        optimizer_zero2.zero_grad()
        local_weight_flatten.grad = _split(weight.grad).clone().flatten()
        local_bias_flatten.grad = bias.grad.clone().flatten()
        zero_start = get_time()
        optimizer_zero2.step()
        zero_end = get_time()
    
        reshape_flatten_weight = local_weight_flatten.view(-1, H // tensor_parallel_size) # reshape
        gather_flatten_weight = _gather(reshape_flatten_weight.data) # gather
        
     
        torch.cuda.synchronize()
        v3_weight_correctness = correctness_verify(weight.data, gather_flatten_weight)
        v3_bias_correctness = correctness_verify(bias.data, local_bias_flatten.data)
        
        
        print(f"iter {i}")

        if v3_weight_correctness:
            print(f"Distributed weight correctness {v3_weight_correctness}")
        else:
            weight_err_idx = error_idx(weight.data, gather_flatten_weight.data)
            print(f"Distributed weight err idx {weight_err_idx}")
                
        if v3_bias_correctness:
            print(f"Distributed bias correctness {v3_bias_correctness}")
        else:
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