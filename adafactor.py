import math
import torch
from torch import nn
import torch.distributed as dist
from torch.optim import Optimizer

import initialize as fs_init
import os 

from mappings import (
    scatter_to_model_parallel_region,
    reduce_from_model_parallel_region,
    gather_from_model_parallel_region
)

# Adafactor From transformer.optim
class Adafactor(Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://arxiv.org/abs/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
    `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*):
            The external learning rate.
        eps (`Tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults to 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

        - Training without LR warmup or clip_threshold is not recommended.

           - use scheduled LR warm-up to fixed LR
           - use clip_threshold=1.0 (https://arxiv.org/abs/1804.04235)
        - Disable relative updates
        - Use scale_parameter=False
        - Additional optimizer operations like gradient clipping should not be used alongside Adafactor

    Example:

    ```python
    Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    ```

    Others reported the following combination to work well:

    ```python
    Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    ```

    When using `lr=None` with [`Trainer`] you will most likely need to use [`~optimization.AdafactorSchedule`]
    scheduler as following:

    ```python
    from transformers.optimization import Adafactor, AdafactorSchedule

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)
    trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
    ```

    Usage:

    ```python
    # replace AdamW with Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    ```"""

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod         
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # print(f"param_groups:\n {list(self.param_groups)}")
        
        """
        param_groups: Dict
        {
            "params":[weight, bias]
            "lr"
            "eps"
            "clip_threshold"
            "decay_rate"
            "beta1"
            "weight_decay"
            "scale_parameter"
            "relative_step"
            "warmup_init"
        }
        """
        
        for group in self.param_groups:
            # update weight & bias
            for p in group["params"]:
                if p.grad is None:
                    continue
                """
                # grad shape is same as weigh / bias
                """
                grad = p.grad
                # print(f"grad {p.grad}")
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                
                """
                p is weight
                state 
                {'step', 
                'exp_avg_sq_row', 
                'exp_avg_sq_col', 
                'RMS'
                }
                
                p is bias
                state 
                {'step', 
                'exp_avg_sq', 
                'RMS'
                }
                """
                
                state = self.state[p]
                # print(f"state {list(state)}")
                grad_shape = grad.shape
                # print(f"grad_shape {grad_shape}")

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()
                
                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)
                
                # 参数Beta 2
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                # print(f"beta2t {beta2t}")
                update = (grad**2) + group["eps"][0]
                if factored:  
                    # 若使用adafactor
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    
                    # (Line No.5)计算行指数平均
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    # (Line No.6)计算列指数平均
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
                    
                    # (Line No.7)近似计算，提前开根号
                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    # 若使用adam
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                #  (Line No.8)
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss


# Adafactor Tensor parallel v0.0.0
class AdafactorTP(Optimizer):
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        super().__init__(params, defaults)
        self.tensor_parallel_size = fs_init.get_model_parallel_world_size()  # 可用于tensor parallel的GPU
        self.tensor_parallel_group = fs_init.get_model_parallel_group() # 可用于tensor parallel的Group class
        self.localRank = int(os.environ['LOCAL_RANK'])  # 当前GPU id
        self.worldSize = int(os.environ['WORLD_SIZE'])  # 总调用GPU

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod         
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        
        # print(f"Row col shape Before {exp_avg_sq_row.shape, exp_avg_sq_col.shape}")
        # exp_avg_sq_row shape [8]
        # exp_avg_sq_col shape [8]
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        # r_factor shape [8,1]
        # c_factor shape [1,8]
        # print(f"Row col shape After {r_factor.shape, c_factor.shape}")
        return torch.mul(r_factor, c_factor)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # print(f"param_groups:\n {list(self.param_groups)}")
        
        # print(f"Self Param TPsize, localRank, worldSize:{self.tensor_parallel_size, self.localRank, self.worldSize}")
        """
        param_groups: Dict
        {
            "params":[weight, bias]
            "lr"
            "eps"
            "clip_threshold"
            "decay_rate"
            "beta1"
            "weight_decay"
            "scale_parameter"
            "relative_step"
            "warmup_init"
        }
        """
        for group in self.param_groups:
            # update weight & bias
            for p in group["params"]:
                if p.grad is None:
                    continue
                """
                # grad shape is same as weigh / bias
                """
                grad = p.grad.to(self.localRank)
                # print(f"grad {p.grad}")
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                
                """
                p (group["params"]) is weight
                state 
                {'step', 
                'exp_avg_sq_row', 
                'exp_avg_sq_col', 
                'RMS'
                }
                
                p (group["params"]) is bias
                state 
                {'step', 
                'exp_avg_sq', 
                'RMS'
                }
                """
                
                state = self.state[p]
                # print(f"state {list(state)}")
                grad_shape = grad.shape
                # print(f"grad_shape {grad_shape}")

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.to(self.localRank)
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()
                
                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)
                
                # 参数Beta 2
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                # print(f"beta2t {beta2t}")
                
                """
                p (group["params"]) is weight
                update shape [H, W]
                
                p (group["params"]) is bias
                update shape [W]
                """
                update = ((grad**2) + group["eps"][0]).to(self.localRank)
                # print(f"Update shape {update.shape}")
                # torch.cuda.synchronize()
                if factored:  
                    # 若使用adafactor
                    exp_avg_sq_row = state["exp_avg_sq_row"].to(self.localRank)
                    exp_avg_sq_col = state["exp_avg_sq_col"].to(self.localRank)
                    
                    # print(f"Row Type: {type(exp_avg_sq_row)}, shape: {exp_avg_sq_row.shape}")
                    # print(f"Col Type: {type(exp_avg_sq_col)}, shape: {exp_avg_sq_col.shape}")
                    
                    # if (self.localRank == 0):
                    #     print(f"exp_avg_sq_row {exp_avg_sq_row}")
                    #     print(f"exp_avg_sq_row {exp_avg_sq_col}")
                    #     print(f"beta2t {beta2t}")
                    #     print(f"update {update}")
                    #     print(f"update mean dim -1 {update.mean(dim=-1)}")  # 倒数第1个维度
                    #     print(f"update mean dim -2 {update.mean(dim=-2)}")  # 倒数第2个维度
                    #     print(f"alpha {(1.0 - beta2t)}")
                    
                    # ==============================
                    #  Without Tensor Parallel
                    # ==============================
                    # # (Line No.5)计算行指数平均
                    # exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    # # exp_avg_sq_row[8] * val + update[8,8]
                    
                    # # (Line No.6)计算列指数平均
                    # exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
                    
                    # if (self.localRank == 0):
                    #     print(f"exp_avg_sq_row res {exp_avg_sq_row}")
                    #     print(f"exp_avg_sq_col res {exp_avg_sq_col}")
                
                        
                    # ==============================
                    #  Tensor Parallel
                    # ==============================
                    # print(f"device {self.localRank}\n update\n {update}")
                    # update [H, W] split by last dim
                    update_parallel = scatter_to_model_parallel_region(update)
                    # print(f"update shape {update.shape}")
                    # print(f"update_parallel shape {update_parallel.shape}")
                    # print(f"device {self.localRank} update_parallel\n {update_parallel}")
                    
                    # Cal update_row_mean on each device
                    update_row_mean_parallel = update_parallel.mean(dim=-1)
                    
                    # Cal update_col_mean on each device
                    update_col_mean_parallel = update_parallel.mean(dim=-2)
                    
                    # print(f"device {self.localRank} update_row_mean parallel\n {update_row_mean_parallel}")
                    # All-reduce update_row_mean div tp_GPU num , then get mean
                    update_row_mean = reduce_from_model_parallel_region(update_row_mean_parallel).div_(self.tensor_parallel_size)
                    # print(f"device {self.localRank} update_row_mean reduce\n {update_row_mean}")
                    
                    # All-gather update_col_mean
                    update_col_mean = gather_from_model_parallel_region(update_col_mean_parallel)
                    
                    # Cal exp_avg_sq_row
                    exp_avg_sq_row.mul_(beta2t).add_(update_row_mean, alpha=(1.0 - beta2t))
                    
                    # Cal exp_avg_sq_col
                    exp_avg_sq_col.mul_(beta2t).add_(update_col_mean, alpha=(1.0 - beta2t))
                    
                    # if (self.localRank == 0):
                    #     print(f"exp_avg_sq_row res {exp_avg_sq_row}")
                    #     print(f"exp_avg_sq_col res {exp_avg_sq_col}")
                    
                    # # (Line No.7)近似计算，提前开根号
                    # # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    # 若使用adam
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                #  (Line No.8)
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)
                print(f"p_data_fp32 shape:{p_data_fp32.shape}; update shape:{update.shape}")

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss


# Adafactor Tensor parallel v0.1
class AdafactorTPv01(Optimizer):
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        weight_height = None,
        weight_width = None,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        super().__init__(params, defaults)
        self.tensor_parallel_size = fs_init.get_model_parallel_world_size()  # 可用于tensor parallel的GPU
        self.tensor_parallel_group = fs_init.get_model_parallel_group() # 可用于tensor parallel的Group class
        self.localRank = int(os.environ['LOCAL_RANK'])  # 当前GPU id
        self.worldSize = int(os.environ['WORLD_SIZE'])  # 总调用GPU
        
        self.weight_height = weight_height
        self.weight_width = weight_width

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod         
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        
        # print(f"Row col shape Before {exp_avg_sq_row.shape, exp_avg_sq_col.shape}")
        # exp_avg_sq_row shape [8]
        # exp_avg_sq_col shape [8]
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        # r_factor shape [8,1]
        # c_factor shape [1,8]
        # print(f"Row col shape After {r_factor.shape, c_factor.shape}")
        return torch.mul(r_factor, c_factor)

        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # print(f"param_groups:\n {list(self.param_groups)}")
        
        # print(f"Self Param TPsize, localRank, worldSize:{self.tensor_parallel_size, self.localRank, self.worldSize}")
        """
        param_groups: Dict
        {
            "params":[weight, bias]
            "lr"
            "eps"
            "clip_threshold"
            "decay_rate"
            "beta1"
            "weight_decay"
            "scale_parameter"
            "relative_step"
            "warmup_init"
        }
        """
        for group in self.param_groups:
            # update weight & bias
            for p in group["params"]:
                if p.grad is None:
                    continue
                """
                # grad shape is same as weigh / bias
                """
                grad = p.grad.to(self.localRank)
                # print(f"grad {p.grad}")
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                
                """
                p (group["params"]) is weight
                state 
                {'step', 
                'exp_avg_sq_row', 
                'exp_avg_sq_col', 
                'RMS'
                }
                
                p (group["params"]) is bias
                state 
                {'step', 
                'exp_avg_sq', 
                'RMS'
                }
                """
                
                state = self.state[p]
                # print(f"state {list(state)}")
                grad_shape = grad.shape
                # print(f"grad_shape {grad_shape}")

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.to(self.localRank)
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()
                
                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)
                
                # 参数Beta 2
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                # print(f"{state['step'], group['decay_rate'], math.pow(state['step'], group['decay_rate'])}")
                
                """
                p (group["params"]) is weight
                update shape [H, W]
                
                p (group["params"]) is bias
                update shape [W]
                """
                # update = ((grad**2) + group["eps"][0]).to(self.localRank)
                # print(f"Update shape {update.shape}")
                # torch.cuda.synchronize()
                if factored:  
                    # split grad alone last dim 
                    """
                    grad_parallel: [H, W/N]
                    update_parallel: [H, W/N]
                    exp_avg_sq_row: [H]
                    exp_avg_sq_col: [W]
                    update_row_mean_parallel:[H]
                    update_col_mean_parallel:[W/N])
                    
                    """
                    # if (self.localRank == 0):
                    #     print(f"Grad: {grad}")
                    #     print(f"Grad shape: {grad.shape}")
                    
                    # shift idx for weight
                    # split weight alone last dim
                    col_start = self.weight_width // self.tensor_parallel_size * self.localRank
                    col_end = self.weight_width // self.tensor_parallel_size * (self.localRank + 1)
                    row_start = 0
                    row_end = self.weight_height
                    
                    grad_parallel = scatter_to_model_parallel_region(grad) 
                    # print(f"Grad_parallel: {grad_parallel}")
                    # print(f"Grad shape: {grad_parallel.shape}")
                    update_parallel = ((grad_parallel**2) + group["eps"][0]).to(self.localRank)
                    # print(f"update_parallel: {update_parallel}")
                    # print(f"update_parallel shape: {update_parallel.shape}")
                    exp_avg_sq_row = state["exp_avg_sq_row"].to(self.localRank)
                    exp_avg_sq_col = state["exp_avg_sq_col"].to(self.localRank)
                    update_row_mean_parallel = update_parallel.mean(dim=-1)
                    update_col_mean_parallel = update_parallel.mean(dim=-2)
                    # print(f"update_row_mean_parallel: {update_row_mean_parallel}")
                    # print(f"update_row_mean_parallel shape: {update_row_mean_parallel.shape}")
                    # print(f"update_col_mean_parallel: {update_col_mean_parallel}")
                    # print(f"update_col_mean_parallel shape: {update_col_mean_parallel.shape}")
                    update_row_mean = reduce_from_model_parallel_region(update_row_mean_parallel).div_(self.tensor_parallel_size)
                    # if (self.localRank == 0):
                    #     print(f"grad_parallel shape {grad_parallel.shape}")
                    #     print(f"update_parallel shape {update_parallel.shape}")
                    #     print(f"exp_avg_sq_row shape {exp_avg_sq_row.shape}")
                    #     print(f"exp_avg_sq_col shape {exp_avg_sq_col.shape}")
                    #     print(f"update_row_mean_parallel shape {update_row_mean_parallel.shape}")
                    #     print(f"update_col_mean_parallel shape {update_col_mean_parallel.shape}")
                    #     print(f"update_row_mean shape {update_row_mean.shape}")
                    
                    # Cal exp_avg_sq_row
                    exp_avg_sq_row.mul_(beta2t).add_(update_row_mean, alpha=(1.0 - beta2t))
                    # print(f"exp_avg_sq_row: {exp_avg_sq_row.mul_(beta2t)}")

                    # Cal exp_avg_sq_col
                    col_offset_start = self.weight_width // self.tensor_parallel_size * self.localRank
                    col_offset_end = self.weight_width // self.tensor_parallel_size * (self.localRank + 1)
                    # print(f"update_col_mean_parallel: {update_col_mean_parallel}")
                    # print(f"update_col_mean_parallel shape: {update_col_mean_parallel.shape}")
                    """
                    update_col_mean_parallel all Good
                    """
                    # print(f"update_col_mean_parallel : {update_col_mean_parallel.mul_(beta2t)}")
                    exp_avg_sq_col[col_offset_start:col_offset_end].mul_(beta2t).add_(update_col_mean_parallel, alpha=(1.0 - beta2t))
                    
                    
                    
                    """
                    update_col_mean_parallel all zero
                    """
                    # print(f"update_col_mean_parallel: {update_col_mean_parallel}")
                    # print(f"update_col_mean_parallel shape: {update_col_mean_parallel.shape}")

                    # exp_avg_sq_col[col_offset_start:col_offset_end] = update_col_mean_parallel
                    # print(f"update_parallel: {update_parallel}")
                    # print(f"update_parallel shape: {update_parallel.shape}")
                    # print(f"exp_avg_sq_row: {exp_avg_sq_row}")
                    # print(f"exp_avg_sq_row shape: {exp_avg_sq_row.shape}")
                    # print(f"update_col_mean_parallel: {update_col_mean_parallel}")
                    # print(f"update_col_mean_parallel shape: {update_col_mean_parallel.shape}")

                    update_parallel = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col[col_offset_start:col_offset_end])
                    # update_parallel = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update_parallel.mul_(grad_parallel)
                    update_parallel.div_((self._rms(update_parallel) / group["clip_threshold"]).clamp_(min=1.0))
                    update_parallel.mul_(lr)

                    # update = gather_from_model_parallel_region(update_parallel)

                    if use_first_moment:
                        # exp_avg = state["exp_avg"]
                        # print(f"exp_avg shape: {exp_avg.shape}, update shape {update.shape}, update_parallel shape {update_parallel.shape}")
                        # exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                        # update = exp_avg
                        
                        exp_avg_parallel = state["exp_avg"][row_start:row_end, col_start:col_end]
                        # print(f"update_parallel shape {update_parallel.shape}; exp_avg_parallel shape {exp_avg_parallel.shape}")
                        exp_avg_parallel.mul_(group["beta1"]).add_(update_parallel, alpha=(1 - group["beta1"]))
                        update_parallel = exp_avg_parallel

                    if group["weight_decay"] != 0:
                        p_data_fp32_parallel = p_data_fp32[row_start:row_end, col_start:col_end]
                        p_data_fp32_parallel.add_(p_data_fp32_parallel, alpha=(-group["weight_decay"] * lr))
                        # p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))
                        # print(f"p_data_fp32 shape {p_data_fp32.shape}")
                        
                    # p_data_fp32.add_(-update)
                    p_data_fp32_parallel.add_(-update_parallel)
                    # print(f"p_data_fp32 shape:{p_data_fp32.shape}; update shape:{update.shape}")

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p[row_start:row_end, col_start:col_end].copy_(p_data_fp32_parallel)

                else:
                    # Adam
                    # bias or other 1dim param use Adam
                    update = ((grad**2) + group["eps"][0]).to(self.localRank)
                    # print(f"update shape {update.shape}, grad shape {grad.shape}")
                    exp_avg_sq = state["exp_avg_sq"]
                    # print(f"exp_avg_sq shape {exp_avg_sq.shape}")
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)
                    # RMS均方根
                    update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                    update.mul_(lr)

                    if use_first_moment:
                        exp_avg = state["exp_avg"]
                        exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                        update = exp_avg

                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                    p_data_fp32.add_(-update)
                    # print(f"p_data_fp32 shape:{p_data_fp32.shape}; update shape:{update.shape}")

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p.copy_(p_data_fp32)
                    
                    # col_start = self.weight_width // self.tensor_parallel_size * self.localRank
                    # col_end = self.weight_width // self.tensor_parallel_size * (self.localRank + 1)
                    # grad_parallel = grad[col_start, col_end]
                    # update_parallel = ((grad_parallel**2) + group["eps"][0])
                    # exp_avg_sq_parallel = state["exp_avg_sq"][col_start, col_end]
                    # exp_avg_sq_parallel.mul_(beta2t).add_(update_parallel, alpha=(1.0 - beta2t))
                    # update_parallel = exp_avg_sq_parallel.rsqrt().mul_(grad_parallel)
                    # update_parallel
                    

        return loss


# Adafactor Tensor parallel v0.2(input split as Weight [H, W/N])
class AdafactorTPv02(Optimizer):
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod         
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        
        # print(f"Row col shape Before {exp_avg_sq_row.shape, exp_avg_sq_col.shape}")
        # exp_avg_sq_row shape [8]
        # exp_avg_sq_col shape [8]
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        # r_factor shape [8,1]
        # c_factor shape [1,8]
        # print(f"Row col shape After {r_factor.shape, c_factor.shape}")
        return torch.mul(r_factor, c_factor)

        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # print(f"param_groups:\n {list(self.param_groups)}")
        
        # print(f"Self Param TPsize, localRank, worldSize:{self.tensor_parallel_size, self.localRank, self.worldSize}")
        """
        param_groups: Dict
        {
            "params":[weight, bias]
            "lr"
            "eps"
            "clip_threshold"
            "decay_rate"
            "beta1"
            "weight_decay"
            "scale_parameter"
            "relative_step"
            "warmup_init"
        }
        """
        # print(f"self.param_groups", self.param_groups,"\n")
        for group in self.param_groups:
            # update weight & bias
            for p in group["params"]:
                # print(f"p.grad {p.grad}\n")
                if p.grad is None:
                    continue
                """
                # grad shape is same as weigh / bias
                """
                grad = p.grad.to(self.localRank)
                # print(f"grad {p.grad}")
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                
                """
                p (group["params"]) is weight
                state 
                {'step', 
                'exp_avg_sq_row', 
                'exp_avg_sq_col', 
                'RMS'
                }
                
                p (group["params"]) is bias
                state 
                {'step', 
                'exp_avg_sq', 
                'RMS'
                }
                """
                state = self.state[p]
                # print(f"state {list(state)}")
                grad_shape = grad.shape
                # print(f"grad_shape {grad_shape}")

                factored, use_first_moment = self._get_options(group, grad_shape)
                print(f"factor {factored}, use_first_moment { use_first_moment}")
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.to(self.localRank)
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()
                
                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)
                
                # 参数Beta 2
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                # print(f"{state['step'], group['decay_rate'], math.pow(state['step'], group['decay_rate'])}")
                
                update = (grad**2) + group["eps"][0]
                if factored:  
                    # 若使用adafactor
                    # print(f"factor\n")
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    
                    # (Line No.5)计算行指数平均
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    # (Line No.6)计算列指数平均
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
                    
                    # (Line No.7)近似计算，提前开根号
                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    # 若使用adam
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                #  (Line No.8)
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss

