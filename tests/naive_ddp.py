import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch.distributed as dist
import time 
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
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend,rank=rank,world_size=world_size)
def sync_model_params(model: nn.Module):
    for param in model.parameters():
        dist.all_reduce(param,op=dist.ReduceOp.SUM,async_op=False)
        param.data = param.data / dist.get_world_size()
def sync_model_grad(model: nn.Module):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad,op=dist.ReduceOp.SUM,async_op=False)
            param.grad.data = param.grad.data / dist.get_world_size()
def distributed(rank, world_size, backend):
    setup(rank, world_size, backend)
    if backend == 'nccl':
        device = f"cuda:{rank}"
    else:
        device = 'cpu'
    model = Toymodel().to(device=device)
    sync_model_params(model)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
    batched_x = torch.rand((8, 1024, 3),dtype=torch.float32,requires_grad=True,device=device)
    if rank == 0:
        full_time = []
        sync_time = []
    for _ in range(5000):
        if rank == 0:
            start = time.time()
        output = model(batched_x)
        loss = loss_func(output)
        optimizer.zero_grad()
        loss.backward()
        if rank == 0:
            sync_start = time.time()
        sync_model_grad(model)
        if rank == 0:
            sync_end = time.time()
        optimizer.step()
        if rank == 0:
            end = time.time()
            full_time.append(end - start)
            sync_time.append(end - start - (sync_end - sync_start))
            print(f"loss:{loss.item()}")
    if rank == 0:
        print(f"average full time: {sum(full_time) / len(full_time)}")
        print(f"average sync time: {sum(sync_time) / len(sync_time)}")
def ddp_train(backend, process):
    world_size = process
    mp.spawn(fn=distributed, args=(world_size,backend),nprocs=world_size,join=True)
if __name__ == "__main__":
    type = 'ddp'
    if type == 'single':
        model = Toymodel().to('mps')
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.00001)
        batched_x = torch.rand((8, 1024, 3),dtype=torch.float32,device='mps',requires_grad=True)
        for _ in range(5000):
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
        ddp_train(backend, process)
    
