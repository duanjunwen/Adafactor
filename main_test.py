import os
import time
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.distributed as dist
# import torch.optim as optim
from adafactor import (
    Adafactor, 
    AdafactorTP, 
    AdafactorTPv01, 
    AdafactorTPv02,
    AdafactorTPv03,
)
import initialize as fs_init

from mappings import (
    scatter_to_model_parallel_region,
    reduce_from_model_parallel_region,
    gather_from_model_parallel_region,
    _split,
    _gather
)
from prettytable import PrettyTable


# dist env
def init_dist():
    rank = int(os.environ['RANK'])  # 当前进程的全局排名（Global Rank）
    local_rank = int(os.environ['LOCAL_RANK'])  # 表示当前进程在本地节点中的排名（Local Rank）。
    # single node GPU num :LOCAL RANK node0:{LOCAL_RANK0-3}, node1:{LOCAL_RANK4-7}
    world_size = int(os.environ['WORLD_SIZE'])  # 表示分布式训练中总共的进程数。

    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    # model_parallel_size_= world_size，GPU全部用于tensor parallism
    fs_init.initialize_model_parallel(model_parallel_size_= world_size)
    torch.cuda.set_device(local_rank)

def correctness_verify(tensor1: torch.Tensor, tensor2: torch.Tensor):
    # return torch.all(tensor1.eq(tensor2))
    return torch.all(tensor1.isclose(tensor2, rtol=1e-05, atol=1e-04, equal_nan=True))
    # return torch.testing.assert_close(tensor1, tensor2,  rtol=1e-05, atol=1e-04, equal_nan=True)
    
def error_idx(tensor1: torch.Tensor, tensor2: torch.Tensor):
    # return tensor1.eq(tensor2)
    return torch.isclose(tensor1, tensor2, rtol=1e-05, atol=1e-04, equal_nan=True)

# def correctness_info_log(weight_correct :bool, bias_correct:bool):
#     if weight_correctness:
#         print(f"weight correctness {weight_correctness}")
#     else:
#         weight_err_idx = error_idx(weight.data, gather_weight.data)
#         print(f"weight err idx {weight_err_idx}")
#     if bias_correctness:
#         print(f"bias correctness {bias_correctness}")
#     else:
#         bias_err_idx = error_idx(bias.data, local_bias.data)
#         print(f"bias err idx {bias_err_idx}")
#     return 0 

def get_time():
    torch.cuda.synchronize()
    return time.time()

def main():
    torch.manual_seed(0)
    device = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    init_dist()
    tensor_parallel_size = fs_init.get_model_parallel_world_size()  # 可用于tensor parallel的GPU
    
    
    param_list = [((2**i)*64, (2**i)*64) for i  in range(0, 10)]
    # print(param_list) 
    table = PrettyTable(['Version', 'weight shape', 'bias shape', 'Avg runtime(ms)',
                            'Speed Up Rate', 'Best Speed Up Rate'], float_format='.2')
    for (H,W) in param_list:
        # H, W = 4096, 4096

        model_base = nn.Linear(H, W).to(device)  
        
        # base param
        # layer shape [H, W], then
        # weight [W, H] [4, 2]
        # bias [W]  [4]
        weight, bias = model_base.weight, model_base.bias
        # print(f"weight shape {weight.shape} {weight}")
        # print(f"bias shape {bias.shape} {bias}")
        # tensor parallel param 
        
        # local_weight [W, H/N] [4, 1]
        # local_bias [W]  [4]
        local_weight = _split(weight)
        local_weight = nn.Parameter(local_weight.clone().requires_grad_(True))
        local_bias = nn.Parameter(bias.clone().requires_grad_(True))
        # print(f"local_weight shape {local_weight.shape} {local_weight}")
        # print(f"local_bias shape {local_bias.shape} {local_bias}")
        
        # local_weight [W*H/N] [4*1]
        # local_bias [W]  [4]
        # flatten param; TP first, then flatten ;
        local_weight_flatten = nn.Parameter(_split(weight).clone().flatten().requires_grad_(True))
        local_bias_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))
        # print(f"local_weight_flatten shape {local_weight_flatten.shape} {local_weight_flatten}")
        # print(f"local_bias_flatten shape {local_bias_flatten.shape} {local_bias_flatten}")
        
        # print(f"correctness_verify weight {correctness_verify(local_weight.flatten(), local_weight_flatten)}") # step后梯度作为输入不变
        


        # # ==============================
        # # Adafactor Base
        # # ==============================
        # optimizer_base = Adafactor(model_base.parameters(), beta1 = 0.9, weight_decay=0.1)
        # print(f"Before base weight shape {weight.shape} on device {device} {weight.data}")
        torch.cuda.synchronize()
        optimizer_base = Adafactor([weight, bias])
        # optimizer_base = Adafactor([weight, bias], beta1 = 0.9, weight_decay=0.1)
        optimizer_base.zero_grad()
        weight.grad = torch.rand_like(weight)
        bias.grad = torch.rand_like(bias)
        optimizer_base.step()
        # print(f"base weight shape {weight.shape} on device {device} {weight.data}")
        # ==============================
        # Adafactor Tensor Parallel v0.1
        # ==============================
        # print(f"TP weight before {list(model_tp.parameters())[0]}")
        # optimizer_tp = AdafactorTPv01(model_tp.parameters(),  beta1 = 0.9,  weight_decay=0.1, weight_height=H, weight_width=W)
        # loss_fn_tp = loss_fn_base
        # optimizer_tp.zero_grad()
        # y_pred_tp_1 = model_tp(x)
        # loss_tp_1 = loss_fn_tp(y_pred_tp_1, y_true)
        # loss_tp_1.backward()
        # optimizer_tp.step()
        # print(f"TP weight after {list(model_tp.parameters())[0]}")
        
        # ==============================
        # Adafactor Tensor Parallel v0.2
        # ==============================
        torch.cuda.synchronize()
        # print(f"Before tp weight shape {local_weight.shape} on device {device} {local_weight.data}\n")
        optimizer_tp = AdafactorTPv02([local_weight, local_bias])
        # optimizer_tp = AdafactorTPv02([local_weight, local_bias],  beta1 = 0.9,  weight_decay=0.1)
        optimizer_tp.zero_grad()
        local_weight.grad = _split(weight.grad).clone()
        local_bias.grad = bias.grad.clone()
        # print(f"local_weight grad shape {local_weight.grad.shape}{local_weight.grad}")
        optimizer_tp.step()
        # print(f"After tp weight shape {local_weight.shape} on device {device} {local_weight}\n")
        # print(f"After tp bias shape {local_bias.shape} on device {device} {local_bias}\n")

        # # ==============================
        # # Adafactor Tensor Parallel v0.3
        # # ==============================
        torch.cuda.synchronize()
        # print(f"Before flatten weight shape {local_weight_flatten.shape} on device {device} {local_weight_flatten.data}\n")
        optimizer_zero2 = AdafactorTPv03([local_weight_flatten, local_bias_flatten], param_height = W, param_width = H)
        optimizer_zero2.zero_grad()
        local_weight_flatten.grad = _split(weight.grad).clone().flatten()
        local_bias_flatten.grad = bias.grad.clone().flatten()
        optimizer_zero2.step()
        # print(f"correctness_verify weight grad {correctness_verify(local_weight.grad.flatten(), local_weight_flatten.grad)}") # step后梯度作为输入不变
        # print(f"After flatten weight shape {local_weight_flatten.shape} on device {device} {local_weight_flatten.data}\n")
        # print(f"After flatten bias shape {local_bias_flatten.shape} on device {device} {local_bias_flatten.data}\n")

        # ==============================
        # Correctness Verify
        # ==============================
        # tensor parallel gather data
        torch.cuda.synchronize()
        gather_weight = _gather(local_weight.data)
        weight_correctness = correctness_verify(weight.data, gather_weight)
        bias_correctness = correctness_verify(bias.data, local_bias.data)
        # print(f"tp weight shape {gather_weight.shape} on device {device} {gather_weight.data}")
        # if weight_correctness:
        #     print(f"V2 weight correctness {weight_correctness}")
        # else:
        #     weight_err_idx = error_idx(weight.data, gather_weight.data)
        #     print(f"V2 weight err idx {weight_err_idx}")
        # if bias_correctness:
        #     print(f"V2 bias correctness {bias_correctness}")
        # else:
        #     bias_err_idx = error_idx(bias.data, local_bias.data)
        #     print(f"V2 bias err idx {bias_err_idx}")
        
        # # tensor parallel & flatten view &gather data
        torch.cuda.synchronize()
        reshape_flatten_weight = local_weight_flatten.view(-1, H // tensor_parallel_size) # reshape
        gather_flatten_weight = _gather(reshape_flatten_weight.data) # gather
        # print(f"flatten weight shape {gather_flatten_weight.shape} on device {device} {gather_flatten_weight.data}")
        # print(f"tp_weight shape {gather_weight.shape} {gather_weight}")
        # print(f"local weight shape {local_weight.shape} {local_weight}")
        # print(f"gather_flatten_weight shape {gather_flatten_weight.shape} {gather_flatten_weight}")
        # print(f"V3 gather_flatten_weight shape {gather_flatten_weight.shape} {gather_flatten_weight}")
        weight_correctness = correctness_verify(weight.data, gather_flatten_weight)
        bias_correctness = correctness_verify(bias.data, local_bias_flatten.data)
        # if weight_correctness:
        #     print(f"V3 weight correctness {weight_correctness}")
        # else:
        #     print(f"V3 weight inclued incorrectness value")
        #     weight_err_idx = error_idx(weight.data, gather_flatten_weight.data)
        #     print(f"V3 weight err idx {weight_err_idx}")
        # if bias_correctness:
        #     print(f"V3 bias correctness {bias_correctness}")
        # else:
        #     print(f"V3 bias inclued incorrectness value")
        #     bias_err_idx = error_idx(bias.data, local_bias_flatten.data)
        #     print(f"V3 bias err idx {bias_err_idx}")
        
        # print(f"Grad same on {int(os.environ['LOCAL_RANK'])} {torch.all(V2_Global_grad.isclose(V3_Global_grad, rtol=1e-05, atol=1e-04, equal_nan=True))}")
        # print(f"Update same on {int(os.environ['LOCAL_RANK'])} {torch.all(V2_Global_update.isclose(V3_Global_update, rtol=1e-05, atol=1e-04, equal_nan=True))}")
        # print(V2_Global_grad)

        # # ==============================
        # # Run training epoch
        # # ==============================
        niter = 10
        # runtime test
        base_start, base_end, base_runtime = 0, 0, 0
        tp_start, tp_end, tp_runtime, tp_best_runtime = 0, 0, 0, float('inf')
        zero_start, zero_end, zero_runtime, zero_best_runtime = 0, 0, 0, float('inf')
        
        for i in range(0, niter):
            # Base optim
            optimizer_base.zero_grad()
            weight.grad = torch.rand_like(weight)
            bias.grad = torch.rand_like(bias)
            base_start = get_time()
            optimizer_base.step()
            base_end = get_time()
            
            # TP optim
            optimizer_tp.zero_grad()
            local_weight.grad = _split(weight.grad).clone()
            local_bias.grad = bias.grad.clone()
            tp_start = get_time()
            optimizer_tp.step()
            tp_end = get_time()
            gather_weight = _gather(local_weight.data)
            
            # TP+Zero2 optim
            optimizer_zero2.zero_grad()
            local_weight_flatten.grad = _split(weight.grad).clone().flatten()
            local_bias_flatten.grad = bias.grad.clone().flatten()
            zero_start = get_time()
            optimizer_zero2.step()
            zero_end = get_time()
            # gather_weight = _gather(local_weight.data)
            reshape_flatten_weight = local_weight_flatten.view(-1, H // tensor_parallel_size) # reshape
            gather_flatten_weight = _gather(reshape_flatten_weight.data) # gather
            
            torch.cuda.synchronize()
            v2_weight_correctness = correctness_verify(weight.data, gather_weight)
            v2_bias_correctness = correctness_verify(bias.data, local_bias.data)
            
            torch.cuda.synchronize()
            v3_weight_correctness = correctness_verify(weight.data, gather_flatten_weight)
            v3_bias_correctness = correctness_verify(bias.data, local_bias_flatten.data)
            
            # print(f"iter {i} weight.data {weight.data}")
            # print(f"iter {i} gather_weight.data {gather_weight.data}")
            
            # print(f"iter {i}")
            # # v2 tp correctness
            # if v2_weight_correctness:
            #     print(f"v2 weight correctness {v2_weight_correctness}")
            # else:
            #     print(f"base iter {i} weight.data {weight.data}")
            #     print(f"v2 iter {i} gather_weight.data {gather_weight.data}")
            #     print(f"v3 iter {i} gather_flatten_weight.data {gather_flatten_weight.data}")
            #     weight_err_idx = error_idx(weight.data, gather_weight.data)
            #     print(f"v2 weight err idx {weight_err_idx}")
            # if v2_bias_correctness:
            #     print(f"v2 bias correctness {v2_bias_correctness}")
            # else:
            #     bias_err_idx = error_idx(bias.data, local_bias.data)
            #     print(f"v2 bias err idx {bias_err_idx}")
                
                
            # # v3 tp + zero correctness
            # if v3_weight_correctness:
            #     print(f"v3 weight correctness {v3_weight_correctness}")
            # else:
            #     # print(f"iter {i} weight.data {weight.data}")
            #     # print(f"iter {i} gather_weight.data {gather_weight.data}")
            #     weight_err_idx = error_idx(weight.data, gather_flatten_weight.data)
            #     print(f"v3 weight err idx {weight_err_idx}")
                    
            # if v3_bias_correctness:
            #     print(f"v3 bias correctness {v2_bias_correctness}")
            # else:
            #     bias_err_idx = error_idx(bias.data, local_bias_flatten.data)
            #     print(f"v3 bias err idx {bias_err_idx}")


            # print(f"Current base avg runtime {(base_end - base_start) * 10.0**3} ms; Current tp avg runtime {(tp_end - tp_start)*10.0**3} ms; Current zero(tp) avg runtime {(zero_end - zero_start)*10.0**3} ms")
            base_runtime += base_end - base_start
            tp_runtime += tp_end - tp_start
            zero_runtime  += zero_end - zero_start
            tp_best_runtime = min(tp_best_runtime, tp_runtime)
            zero_best_runtime= min(zero_best_runtime, zero_runtime)
        # print(f"v2 base avg runtime {(base_runtime / niter) * 10.0**3} ms; tp avg runtime {(tp_runtime / niter)*10.0**3} ms; Zero avg runtime {(zero_runtime / niter)*10.0**3} ms;\n")
        # print(f"v2 Avg Speed Up Rate {base_runtime/tp_runtime}; v3 Avg Speed Up Rate {base_runtime/zero_runtime};\n")
        # print(f"v2 Best Speed Up Rate {base_runtime/tp_best_runtime}; v3 Best Speed Up Rate {base_runtime/zero_best_runtime};\n")
        
        table.add_row(["Version Base", weight.shape, bias.shape, (base_runtime / niter) * 10.0**3, None, None])
        table.add_row(["Version TP", weight.shape, bias.shape, (tp_runtime / niter)*10.0**3, base_runtime/tp_runtime, base_runtime/tp_best_runtime])
        table.add_row(["Version TP+Zero2", weight.shape, bias.shape, (zero_runtime / niter) * 10.0**3, base_runtime/zero_runtime, base_runtime/zero_best_runtime])
    if device == 1:
        print(table)
        with open('Adafactor_Performance_Test_2GPU.csv', 'w+', newline='') as file:
            file.write(table.get_csv_string())
if __name__ == "__main__":
    main()