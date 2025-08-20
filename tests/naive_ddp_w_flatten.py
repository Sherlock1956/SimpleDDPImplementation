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
def sync_model_grad(model: nn.Module,flatten):
    grads = []
    if flatten == True:
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad)
        flattened_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM,async_op=False)
        flattened_grads = flattened_grads / dist.get_world_size()
        unflattened_grad = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                param.grad = unflattened_grad[i]
    else:
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad)
                dist.all_reduce(param.grad,op=dist.ReduceOp.SUM,async_op=False)
                param.grad.data = param.grad.data / dist.get_world_size()
def distributed(rank, world_size, backend, warmup, flatten):
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
    # optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
    # batched_x = torch.rand((8, 1024, 3),dtype=torch.float32,requires_grad=True,device=device)
    optimizer = My_AdamW(model.parameters(), lr=0.00001)
    batched_data_x = torch.randint(0,10000,(1,256)).to(device)
    batched_data_y = torch.zeros_like(batched_data_x)
    if rank == 0 and warmup is not True:
        full_time = []
        sync_time = []
    for _ in (range(100)):
        if rank == 0 and warmup is not True:
            start = time.time()
        output = model(batched_data_x)
        loss = My_cross_entropy(output, batched_data_y)
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        if rank == 0 and warmup is not True:
            sync_start = time.time()
        with torch.no_grad():
            sync_model_grad(model,flatten)
            torch.cuda.synchronize()
        if rank == 0 and warmup is not True:
            sync_end = time.time()
        optimizer.step()
        if rank == 0 and warmup is not True:
            end = time.time()
            full_time.append(end - start)
            sync_time.append(sync_end - sync_start)
            # print(f"loss:{loss.item()}")
    if rank == 0 and warmup is not True:
        print(f"use flatten: {flatten}")
        print(f"average full time: {sum(full_time) / len(full_time)}")
        print(f"average sync time: {sum(sync_time) / len(sync_time)}")
    # 确保所有进程同步后再退出
    dist.barrier()
def ddp_train(backend, process, warmup, flatten=True):
    world_size = process
    mp.spawn(fn=distributed, args=(world_size,backend,warmup,flatten),nprocs=world_size,join=True)
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
        ddp_train(backend, process, warmup=False, flatten=True)
        ddp_train(backend, process, warmup=False, flatten=False)
    
