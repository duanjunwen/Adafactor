import math
import os 
import torch
from torch.optim import Optimizer
import torch.distributed as dist
# import initialize as fs_init
from mappings import _gather


# Adafactor From transformer.optim
class Adafactor(Optimizer):
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
        param_height = None,
        param_width = None
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
        self.param_height = param_height
        self.param_width = param_width

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
                # print(f"base p.grad {p.grad}\n")
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
                # state["RMS"] = self._rms(p_data_fp32)
                # print(f"RMS base {state['RMS']}")
                # if factored:
                #    print(f"v0 device {0} RMS {state['RMS']}")
                lr = self._get_lr(group, state)
                
                # 参数Beta 2
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                # print(f"beta2t {beta2t}")
                update = (grad**2) + group["eps"][0]
                # if int(os.environ['LOCAL_RANK']) == 0:
                #     print(f"update {update}")
                #     # print(f"grad {grad}")
                    
                if factored:  
                    # 若使用adafactor
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    
                    # (Line No.5)计算行指数平均
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    # (Line No.6)计算列指数平均
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
                    
                    if int(os.environ['LOCAL_RANK']) == 0:
                        print(f"exp_avg_sq_row {exp_avg_sq_row}")
                    #     print(f"exp_avg_sq_col {exp_avg_sq_col}")

                    # (Line No.7)近似计算，提前开根号
                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    # if int(os.environ['LOCAL_RANK']) == 0:
                    #     print(f"update {update}")
                    update.mul_(grad)
                else:
                    # 若使用adam
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                # if factored and int(os.environ['LOCAL_RANK']) == 0:  
                #     print(f"v0 device {int(os.environ['LOCAL_RANK'])} shape {update.shape} update {update}")

                #  (Line No.8)
                # update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))
                
                # if factored and int(os.environ['LOCAL_RANK']) == 0:  
                #     print(f"v0 device {int(os.environ['LOCAL_RANK'])} shape {update.shape} update {update}")

                
                p_data_fp32.add_(-update)
                # if factored:  
                #     print(f"v0 device {int(os.environ['LOCAL_RANK'])} shape {p_data_fp32.shape} p_data_fp32 {p_data_fp32}")


                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss

# DistributedAdaFactor (with Tensor parallel and Zero stage 2)
class DistributedAdaFactor(Optimizer):
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
        param_height = None,
        param_width = None,
        device_mesh = None,
        sharding_spec = None
    ):
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
            "param_height": param_height,
            "param_width": param_width,
            "device_mesh": device_mesh,
            "sharding_spec": sharding_spec
        }
        super().__init__(params, defaults)
        # self.tensor_parallel_size = fs_init.get_model_parallel_world_size() 
        # self.tensor_parallel_group = fs_init.get_model_parallel_group()
        self.tensor_parallel_size = device_mesh._physical_mesh_id.shape[0]
        self.tensor_parallel_group = device_mesh.get_process_group(axis=1) # "Expected row process group"
        # print(f"tensor_parallel_group {self.tensor_parallel_group}")
        self.localRank = int(os.environ['LOCAL_RANK']) 
        self.worldSize = int(os.environ['WORLD_SIZE']) 
        self.param_height = param_height # H
        self.param_width = param_width  # W
        self.param_height_parallel = self.param_height // self.tensor_parallel_size # H/N
        self.param_width_parallel = self.param_width // self.tensor_parallel_size # W/N
        # print(f"sharding_spec {sharding_spec}")
        self.sharding_spec = sharding_spec
        # print(f"self.sharding_spec {self.sharding_spec.sharding_sequence[0]} {self.sharding_spec.sharding_sequence[1]}")
        # print(f"Deter1 {self.sharding_spec.sharding_sequence[0] == 'R'} {self.sharding_spec.sharding_sequence[0] != 'S1'}")
        # # print(f"Deter2 {self.sharding_spec.sharding_sequence[-1].startwith('S')} ")
        

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
    def _get_options(param_group, param_shape, tensor_parallel_size):
        # tensor_parallel_size = fs_init.get_model_parallel_world_size() 
        # tensor_parallel_size = self.tensor_parallel_size._physical_mesh_id.shape[0]
        # print(f"param_shape {param_shape}")
        param_shape_flatten = list(param_shape)[0]
        check_shape = param_group['param_height'] * param_group['param_width'] // tensor_parallel_size 
        factored = (len(param_shape) >= 2) or  (param_shape_flatten >= check_shape)
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod         
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        print(f"row mean {exp_avg_sq_row.mean(dim=-1, keepdim=True)}")
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
                state = self.state[p]  # always empty
                grad_shape = grad.shape
                factored, use_first_moment = self._get_options(group, grad_shape, self.tensor_parallel_size)
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        if self.sharding_spec.sharding_sequence[0] == 'R': # Col Parallel
                            state["exp_avg_sq_row"] = torch.zeros(self.param_height).to(grad)  # [H:4096]
                            state["exp_avg_sq_col"] = torch.zeros(self.param_width_parallel).to(grad)  # [W/N:2048]
                        
                        if self.sharding_spec.sharding_sequence[-1] == 'R': # Row Parallel
                            state["exp_avg_sq_row"] = torch.zeros(self.param_height_parallel).to(grad)  # [H/N:2048]
                            state["exp_avg_sq_col"] = torch.zeros(self.param_width).to(grad)  # [W:4096]
                            
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
                lr = self._get_lr(group, state)
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:  
                    # ==============================
                    # First Dim is R, Last Dim is S1 means split dim -1  ---> 
                    # Means Coloum Parallel ---> sq_row need Do (col) Reduce
                    # ==============================
                    if self.sharding_spec.sharding_sequence[0] == 'R': 
                        update_reshape = update.view(-1, self.param_width_parallel)
                        # print(f"grad {grad}")
                        grad_reshape = grad.view(-1, self.param_width_parallel)
                        exp_avg_sq_row = state["exp_avg_sq_row"] # [H]
                        exp_avg_sq_col = state["exp_avg_sq_col"] # [W/N]
                        # self.sharding_spec.sharding_sequence[0]
                        exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                        exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                        dist.all_reduce(exp_avg_sq_row, group=self.tensor_parallel_group)
                        exp_avg_sq_row.div_(self.tensor_parallel_size)
                        update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                        update_reshape.mul_(grad_reshape)
                        update = update_reshape.flatten()
                    # ==============================
                    # Last Dim is R, First Dim is S1 means split dim 0  --->
                    # Means Row Parallel ---> sq_col need Do (row) Reduce
                    # ==============================
                    elif self.sharding_spec.sharding_sequence[-1] == 'R':
                        update_reshape = update.view(-1, self.param_width)
                        grad_reshape = grad.view(-1, self.param_width)
                        exp_avg_sq_row = state["exp_avg_sq_row"] # [H/N]
                        exp_avg_sq_col = state["exp_avg_sq_col"] # [W]
                        exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                        exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                        # reduce col
                        dist.all_reduce(exp_avg_sq_col, group=self.tensor_parallel_group)
                        exp_avg_sq_col.div_(self.tensor_parallel_size)
                        # gather row
                        exp_avg_sq_row_gather = _gather(exp_avg_sq_row, self.tensor_parallel_group)
                        
                        # print(f"dist exp_avg_sq_row device {self.localRank} {exp_avg_sq_row}") # row correct
                        print(f"dist exp_avg_sq_row_gather device {self.localRank} {exp_avg_sq_row_gather}") # row correct
                        # print(f"dist exp_avg_sq_col device {self.localRank} {exp_avg_sq_col}")
                        update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                        # print(f"update_reshape device {self.localRank}  {update_reshape}")
                        update_reshape.mul_(grad_reshape)
                        update = update_reshape.flatten()

                        
                    #     # print(f"Row Parallel")
                    #     dist.all_reduce(exp_avg_sq_col, group=self.tensor_parallel_group)
                    #     exp_avg_sq_col.div_(self.tensor_parallel_size)
                    #     print(f"dist exp_avg_sq_row device {self.localRank} {exp_avg_sq_row}")

                    
                    # update = update_reshape.flatten()
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                # (Line No.8) RMS
                # update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)
            
                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))
                
                p_data_fp32.add_(-update).flatten()
                
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)
                
        return loss
