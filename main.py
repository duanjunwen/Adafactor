import os
import time
import torch
from torch import nn
import torch.distributed as dist
# import torch.optim as optim
from adafactor import Adafactor, AdafactorTP, AdafactorTP_v0_1, AdafactorTpZero2
import initialize as fs_init

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
    return torch.all(tensor1.eq(tensor2))
    
def error_idx(tensor1: torch.Tensor, tensor2: torch.Tensor):
    return torch.eq(tensor1, tensor2)

def get_time():
    torch.cuda.synchronize()
    return time.time()

def main():
    torch.manual_seed(0)
    device = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    init_dist()
    
    # runtime test
    base_start, base_end, base_runtime = 0, 0, 0
    tp_start, tp_end, tp_runtime = 0, 0, 0
    
    # Data and model Param
    a = 0.1
    b = 1.0
    error = 0.1
    batch_size = 64
    # Weight and bias shape; Weight [8, 8], Bias 
    H, W = 4096, 4096
    # x = torch.arange(1.0, 1.0 + 2 * 2 * 1.0, 1.0).reshape(2, 2)
    # if device == 0:
    #     print(f"x {x}")
    #     print(f"x mean dim -1 {x.mean(dim=-1)}")
    #     print(f"x mean dim -2 {x.mean(dim=-2)}")
    
    # Data
    x = torch.arange(0.1, 0.1 + batch_size * H * 0.1, 0.1).reshape(batch_size, H).cuda(device=device)
    y_true = a * x + b + (torch.randn(batch_size, 1).cuda(device=device) * error) # shape [batch size, H]
    y_true = y_true.cuda(device=device)

    model_base = nn.Linear(H, W).to(device)  
    model_tp = model_base.to(device)
    model_tp_zero2 = model_base.to(device)
    # weight [H*W]
    # bias [W]
    
    # ==============================
    # Adafactor
    # ==============================
    # torch.cuda.synchronize()
    # print("Adafactor")
    # print(f"Base weight before {list(model_base.parameters())[0]}")
    optimizer_base = Adafactor(model_base.parameters(), beta1 = 0.9, weight_decay=0.1)
    loss_fn_base = nn.MSELoss()
    
    optimizer_base.zero_grad()
    y_pred_base = model_base(x)
    loss_base = loss_fn_base(y_pred_base, y_true)
    loss_base.backward()
    optimizer_base.step()
    # print(f"Base weight after {list(model_base.parameters())[0]}")

    # # ==============================
    # # Adafactor Tensor Parallel v0.0
    # # ==============================
    # optimizer_tp = AdafactorTP(model_tp.parameters())
    # loss_fn_tp = loss_fn_basedddd

    # optimizer_tp.zero_grad()
    # y_pred_tp = model_tp(x)
    # loss_tp = loss_fn_tp(y_pred_tp, y_true)
    # loss_tp.backward()
    # optimizer_tp.step()
    
    # ==============================
    # Adafactor Tensor Parallel v0.1
    # ==============================
    # print(f"TP weight before {list(model_tp.parameters())[0]}")
    optimizer_tp = AdafactorTP_v0_1(model_tp.parameters(),  beta1 = 0.9,  weight_decay=0.1, weight_height=H, weight_width=W)
    loss_fn_tp = loss_fn_base

    optimizer_tp.zero_grad()
    y_pred_tp_1 = model_tp(x)
    loss_tp_1 = loss_fn_tp(y_pred_tp_1, y_true)
    loss_tp_1.backward()
    optimizer_tp.step()
    # print(f"TP weight after {list(model_tp.parameters())[0]}")

    # # ==============================
    # # Adafactor Tensor Parallel + Zero Stage 2
    # # ==============================
    # optimizer_tp_zero2 = AdafactorTpZero2(model_tp_zero2.parameters())
    # loss_fn_tp_zero2 = loss_fn_base

    # optimizer_tp_zero2.zero_grad()
    # y_pred_tp_zero2 = model_tp_zero2(x)
    # loss_tp_zero2 = loss_fn_tp_zero2(y_pred_tp_zero2, y_true)
    # loss_tp_zero2.backward()
    # optimizer_tp_zero2.step()

    # ==============================
    # Correctness Verify
    # ==============================
    torch.cuda.synchronize()
    weight_correctness = correctness_verify(list(model_base.parameters())[0].data, list(model_tp.parameters())[0].data)
    bias_correctness = correctness_verify(list(model_base.parameters())[1].data, list(model_tp.parameters())[1].data)
    
    print(f"weight correctness {weight_correctness}")
    print(f"bias correctness {bias_correctness}")
    
    # ==============================
    # Run training epoch
    # ==============================
    niter = 10
    for i in range(0, niter):
        # Base optim
        optimizer_base.zero_grad()
        y_pred_base = model_base(x)
        loss_base = loss_fn_base(y_pred_base, y_true)
        loss_base.backward()
        base_start = get_time()
        optimizer_base.step()
        base_end = get_time()
        
        # TP optim
        optimizer_tp.zero_grad()
        y_pred_tp = model_tp(x)
        loss_tp = loss_fn_tp(y_pred_tp, y_true)
        loss_tp.backward()
        tp_start = get_time()
        optimizer_tp.step()
        tp_end = get_time()
        
        torch.cuda.synchronize()
        weight_correctness = correctness_verify(list(model_base.parameters())[0].data, list(model_tp.parameters())[0].data)
        bias_correctness = correctness_verify(list(model_base.parameters())[1].data, list(model_tp.parameters())[1].data)
        
        print(f"iter {i}")
        if weight_correctness:
            print(f"weight correctness {weight_correctness}")
        else:
            weight_err_idx = error_idx(list(model_base.parameters())[0].data, list(model_tp.parameters())[0].data)
            print(f"bias err idx {weight_err_idx}")
            
        if bias_correctness:
            print(f"bias correctness {bias_correctness}")
        else:
            bias_err_idx = error_idx(list(model_base.parameters())[1].data, list(model_tp.parameters())[1].data)
            print(f"bias err idx {bias_err_idx}")
        
        base_runtime += base_end - base_start
        tp_runtime += tp_end - tp_start
    print(f"base avg runtime {(base_runtime / niter) * 10.0**3} ms; tp avg runtime {(tp_runtime / niter)*10.0**3} ms")
    
if __name__ == "__main__":
    main()