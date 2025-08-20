import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import time 
from basic_modules import *
import basic_modules
class Toymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(3,16)
        self.norm = nn.LayerNorm(16)
        self.ln2 = nn.Linear(16,1)
    def forward(self, x):
        x = self.ln1(x)
        x = self.norm(x)
        x = self.ln2(x)
        return x
def loss_func(x):
    return torch.sum(x**2)
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29100"
    dist.init_process_group(backend=backend,rank=rank,world_size=world_size)
def sync_model_params(model: nn.Module):
    for param in model.parameters():
        dist.all_reduce(param,op=dist.ReduceOp.SUM,async_op=False)
        param.data = param.data / dist.get_world_size()
class My_DDP(nn.Module):
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handlers = []
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                parameter.register_post_accumulate_grad_hook(self.sync_model_grad_async)
    def sync_model_grad_async(self, parameter):
        handler = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handlers.append((handler, parameter))
    def finish_gradient_synchronization(self):
        for handler in self.handlers:
            handler[0].wait()
            param = handler[1]
            if param.grad is not None:
                param.grad.data = param.grad.data / dist.get_world_size()
    def forward(self, x):
        return self.model(x)
def distributed(rank, world_size, backend, warmup):
    setup(rank, world_size, backend)
    if warmup == True and rank == 0:
        print("warmup......")
    if warmup == False and rank == 0:
        print("training......")
    if backend == 'nccl':
        device = f"cuda:{rank}"
    else:
        device = 'cpu'
    # model = Toymodel().to(device=device)
    # 1. prepare model
    model = My_transformer_lm(vocab_size=10000, context_length=256, d_model=1024, num_layers=24, num_heads=16, d_ff=4096, rope_theta=10000)
    model.to(device)
    with torch.no_grad():
        sync_model_params(model)
    model = My_DDP(model)

    optimizer = My_AdamW(model.parameters(), lr=0.00001)
    batched_data_x = torch.randint(0,10000,(1,256)).to(device)
    batched_data_y = torch.zeros_like(batched_data_x)
    if rank == 0 and warmup is not True:
        full_time = []
    for _ in (range(100)):
        if rank == 0 and warmup is not True:
            start = time.time()
        output = model(batched_data_x)
        loss = My_cross_entropy(output, batched_data_y)
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()
        if rank == 0 and warmup is not True:
            end = time.time()
            full_time.append(end - start)
            # print(f"loss:{loss.item()}")
    if rank == 0 and warmup is not True:
        print(f"average full time: {sum(full_time) / len(full_time)}")

def ddp_train(backend, process, warmup):
    world_size = process
    mp.spawn(fn=distributed, args=(world_size,backend,warmup),nprocs=world_size,join=True)
if __name__ == "__main__":
    type = 'ddp'
    if type == 'single':
        model = Toymodel().to('mps')
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
        batched_x = torch.rand((8, 1024, 3),dtype=torch.float32,device='mps',requires_grad=True)
        for _ in range(100):
            output = model(batched_x)
            loss = loss_func(output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss:{loss.item()}")
    elif type == 'ddp':
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            backend = 'nccl'  # 使用NCCL后端进行GPU训练
            process = 2       # 使用2个GPU
        else:
            backend = 'gloo'  # 如果GPU不够，回退到CPU训练
            process = 2
        ddp_train(backend, process, warmup=True)
        ddp_train(backend, process, warmup=False)
    
