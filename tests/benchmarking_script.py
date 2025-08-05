from basic_modules import *
import basic_modules
import torch
from adapters import *
from config import config
import numpy as np
from tqdm import tqdm
import timeit
import torch.cuda.nvtx as nvtx
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(q, k, v, mask=None, softmax=torch.softmax):
    # q: ([8, 12, 256, 64]), k: ([8, 12, 256, 64]), v: ([8, 12, 256, 64])
    d_k = q.shape[-1]
    with nvtx.range("computing attention scores"):
        attention = q @ k.transpose(-1,-2) / d_k ** 0.5
    if mask is not None:
        attention = attention.masked_fill(~mask, float('-inf'))
    with nvtx.range("computing softmax"):
        result = softmax(attention,dim=-1)
    with nvtx.range("final matmul"):
        result = result @ v
    return result
def benchmark(d_model, d_ff, num_layers, num_heads, size):
    basic_modules.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    device = config['device']
    # 1. prepare model
    transformer_lm = My_transformer_lm(vocab_size=config['vocab_size'], context_length=config['context_length'], d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=config['rope_theta'])
    transformer_lm.to(device)
    trainable_parameters = sum(param.numel() for param in transformer_lm.parameters() if param.requires_grad)
    # 2. prepare optimizer
    optimizer = My_AdamW(transformer_lm.parameters(), lr=config['max_lr'])
    batched_data_x = torch.randint(0,10000,(8,256)).to(config['device'])
    batched_data_y = batched_data_x + 1

    # 定义前向传播函数（只包括模型推理和损失计算）
    def forward_pass_only():
        """纯前向传播函数 - 只包括模型推理和损失计算"""
        optimizer.zero_grad()  # 清零梯度
        output = transformer_lm(batched_data_x)
        loss = My_cross_entropy(output, batched_data_y)
        torch.cuda.synchronize()
        return loss

    # 定义反向传播函数（只包括梯度计算）
    def backward_pass():
        """纯反向传播函数 - 只包括梯度计算"""
        # 每次都重新计算前向传播，然后只测量反向传播时间
        optimizer.zero_grad()
        output = transformer_lm(batched_data_x)
        loss = My_cross_entropy(output, batched_data_y)
        loss.backward()
        torch.cuda.synchronize()

    # 定义反向传播函数（只包括梯度计算）
    def backward_pass_full():
        """纯反向传播函数 - 只包括梯度计算"""
        # 每次都重新计算前向传播，然后只测量反向传播时间
        optimizer.zero_grad()
        output = transformer_lm(batched_data_x)
        loss = My_cross_entropy(output, batched_data_y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # 使用timeit测试前向传播时间
    print("=== 性能测试 ===")
    print("测试配置:")
    print(f"- 设备: {config['device']}")
    print(f"- d_model: {d_model}")
    print(f"- d_ff: {d_ff}")
    print(f"- num_layers: {num_layers}")
    print(f"- num_heads: {num_heads}")
    print(f"- model size: {size}")
    print(f"- trainable_parameters: {trainable_parameters/1e9:.6f}B")
    print()

    # 测试前向传播时间 (重复10次取平均)
    forward_pass_time = []
    for i in range(15):
        # 使用NVTX标记预热和测试阶段
        if i > 5:
            nvtx.range_push(f"forward_pass_test_{i}")
            forward_pass_time.append(timeit.timeit(forward_pass_only, number=1))
            nvtx.range_pop()
        else:
            nvtx.range_push(f"forward_pass_warmup_{i}")
            timeit.timeit(forward_pass_only, number=1)
            nvtx.range_pop()
    forward_pass_time = np.array(forward_pass_time)

    # 测试反向传播时间 (包含前向传播，但主要测量反向传播)
    backward_pass_time = []
    for i in range(25):
        if i > 15:
            nvtx.range_push(f"backward_pass_test_{i}")
            backward_pass_time.append(timeit.timeit(backward_pass, number=1))
            nvtx.range_pop()
        else:
            nvtx.range_push(f"backward_pass_warmup_{i}")
            timeit.timeit(backward_pass, number=1)
            nvtx.range_pop()
    backward_pass_time = np.array(backward_pass_time)

    # 测试反向传播时间 (包含前向传播，但主要测量反向传播，包含优化器更新)
    backward_opt_pass_time = []
    for i in range(25):
        if i > 15:
            nvtx.range_push(f"backward_pass_opt_test_{i}")
            backward_opt_pass_time.append(timeit.timeit(backward_pass_full, number=1))
            nvtx.range_pop()
        else:
            nvtx.range_push(f"backward_pass_opt_warmup_{i}")
            timeit.timeit(backward_pass_full, number=1)
            nvtx.range_pop()
    backward_opt_pass_time = np.array(backward_opt_pass_time)
    print(f"前向传播平均时间: {forward_pass_time.mean():.6f} 秒")
    print(f"前向+反向传播时间: {backward_pass_time.mean():.6f} 秒")
    print(f"纯反向传播时间(估算): {(backward_pass_time.mean() - forward_pass_time.mean()):.4f} 秒")
    print(f"完整一步更新时间: {(backward_opt_pass_time.mean()):.4f} 秒")

if __name__ == "__main__":
    model_size_dict = {
        "d_model":[768, 1024, 1280, 1600, 2560],
        "d_ff":[3072, 4096, 5120, 6400, 10240],
        "num_layers":[12, 24, 36, 48, 32],
        "num_heads":[12, 16, 20, 25, 32],
        "size":['small','medium','large','xl','2.7B']
    }
    for i in range(1):
        d_model = model_size_dict['d_model'][i]
        d_ff = model_size_dict['d_ff'][i]
        num_layers = model_size_dict['num_layers'][i]
        num_heads = model_size_dict['num_heads'][i]
        size = model_size_dict['size'][i]
        benchmark(d_model, d_ff, num_layers, num_heads, size)
    