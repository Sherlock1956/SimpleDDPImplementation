import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp # 多进程，并行

def setup(rank, world_size):
    # 让四个进程属于同一个process group，设置相同的MASTER_ADDR, MASTER_PORT，也即shared master
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # gloo可以用作cpu-only的后端，nccl一般用于有gpu的分布式训练
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    """
    如果是在多GPU训练中，需要保证不同的进程使用不同的GPU。
    方法1: 使用torch.cuda.set_device(rank)，然后再使用tensor.to("cuda")的时候就会自动放到指定的GPU上
    方法2: 使用device = f"cuda:{rank}"然后使用tensor.to(device)即可
    """

def distributed_demo(rank, world_size):
    # mp.spawn会自动传入rank参数，args中的参数作为后续参数传入
    # rank为0的进程就是master进程，帮助其他进程完成数据收集，计算与分发，但是计算过程是分布式计算，进行多轮数据传输和局部求和。
    # 与指导书中Collective communication operations like all-reduce operate on each process in the process group一致
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    # async_op=False 意味着程序会在all-reduce操作完全结束后才继续执行，确保数据一致性。
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)
    # join=True代表主进程会阻塞等待，直到所有4个子进程都执行完毕