# Benchmarking

run on 4090D autodl platform

| 模型规模 | 设备 | d_model | d_ff | num_layers | num_heads | 可训练参数量 (B) | 前向传播时间 (s) | 前向+反向时间 (s) | 纯反向时间 (估算, s) |
| :------- | ---- | :------ | ---- | ---------- | --------- | ---------------- | ---------------- | ----------------- | -------------------- |
| small    | cuda | 768     | 3072 | 12         | 12        | 0.128625         | 0.035008         | 0.076579          | 0.0416               |
| medium   | cuda | 1024    | 4096 | 24         | 16        | 0.423183         | 0.092815         | 0.216040          | 0.1232               |
| large    | cuda | 1280    | 5120 | 36         | 20        | 0.969412         | 0.187382         | 0.447217          | 0.2598               |

# nsys_profile  

如无特殊说明，本节及后续章节模型大小如下：
- d_model: 768
- d_ff: 3072
- num_layers: 12
- num_heads: 12
- model size: small
- trainable_parameters: 0.128625B

主要学习nsight system的基本使用，包括用nvtx标注感兴趣代码位置，用nsys收集数据，用nsight system打开，察看耗时等等

前向传播平均时间: 0.148052 秒
前向+反向传播时间: 0.439438 秒
纯反向传播时间(估算): 0.2914 秒
完整一步更新时间: 0.5072 秒

前向传播平均时间: 0.077418 秒
前向+反向传播时间: 0.208083 秒
纯反向传播时间(估算): 0.1307 秒
完整一步更新时间: 0.2676 秒

(a) 在cpu上的前向传播和反向传播的nvtx标记和timeit测试出来的基本一致

(b)在单次前向传播过程中，调用次数最多的是ampere_sgemm_...这个矩阵乘法kernel，一次前向传播调用了85次（small size），如果包含前向和反向传播，也是这个kernel执行次数最多，次数也是85次

![image-20250805130558198](C:\Users\Sherlock\AppData\Roaming\Typora\typora-user-images\image-20250805130558198.png)

![image-20250805130821730](C:\Users\Sherlock\AppData\Roaming\Typora\typora-user-images\image-20250805130821730.png)

(c)除了矩阵乘法之后，elementwise_kernel也占了不少的计算，例如加法、乘法、ReLU、Sigmoid 等。这种 kernel 在深度学习中大量出现，因为很多激活函数、loss 函数、张量变换都属于逐元素操作。

(d)测试一轮完整的前向传播+反向传播+优化器更新，占比最多的还是矩阵乘法，但是占比稍微下降，因为优化器更新中有大量的逐元素的更新操作，相比仅前向传播的66.4%占比下降到47.3%

![image-20250805132049031](C:\Users\Sherlock\AppData\Roaming\Typora\typora-user-images\image-20250805132049031.png)

(e)测试一次前向传播中一个attention中的矩阵乘法和softmax的时间消耗之比和两者FLOPS之比，观察是否有什么现象。

computing attention scores:

​	time: 350us

​	FLOPS: 8 * 12 * 256 * 64 * 256 * 2 + 8 * 12 * 256 * 256 = 811,597,824

computing softmax: 

​	time: 161us

​	FLOPS: 8   ×   12   ×   256   ×   256   ×   3 = 18,874,368

final matmul: 

​	time: 350us

​	FLOPS: 8 * 12 * 256 * 64 * 256 * 2 = 805,306,368

final matmul和computing softmax的FLOPS比例：42.7

final matmul和computing softmax的Time比例：2.2

![image-20250805135156228](C:\Users\Sherlock\AppData\Roaming\Typora\typora-user-images\image-20250805135156228.png)

**结论：虽然 softmax 的 FLOPS 很小，但由于其 memory-bound、本身结构复杂、指令混合率差，导致执行时间并不成比例地小。相比之下，GEMM 虽然 FLOPS 巨大，但因为其是 compute-bound 且高度优化，反而执行时间较短或与 softmax 相近。**

## nvtx用法(import torch.cuda.nvtx as nvtx)

**用法1：使用装饰器**

​	在一个函数前面加上@nvtx.range("")可以将这个函数标注为nvtx范围

**用法2：with语法**

​	在需要标注的代码块前面上使用with nvtx.range("")

**用法3：push & pop**

​	在合适的地方使用nvtx.range_push("")和nvtx.range_pop()

```python
# 用法1,2代码示例
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

# 用法3代码示例
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
```

**注意：同一个代码范围内如果有多个nvtx标注，在nsight system中则会显示为重叠的nvtx，是正常情况**

# mixed_precision_accumulation  

```python
import torch
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(s)
s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
```

代码运行结果：
```BASH
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```

说明在累加时低精度会导入累加误差较大，累加过程需要使用高精度

# benchmarking_mixed_precision  

全精度：

前向传播平均时间: 0.148052 秒
前向+反向传播时间: 0.439438 秒
纯反向传播时间(估算): 0.2914 秒
完整一步更新时间: 0.5072 秒

混合精度：

前向传播平均时间: 0.077418 秒
前向+反向传播时间: 0.208083 秒
纯反向传播时间(估算): 0.1307 秒
完整一步更新时间: 0.2676 秒

(a)不同层在autocast中是否会被下降精度到fp16

模型参数（model parameters）：

- **数据类型：FP32**
- **原因：** PyTorch 默认在混合精度训练中保留参数为 FP32，以避免梯度更新时的数值不稳定。

第一层线性层输出（`self.fc1(x)` 的输出）：

- **数据类型：FP16**
- **原因：** 线性层是计算密集型操作，`autocast` 会将其输入和输出自动转换为 FP16 来加速计算。

LayerNorm 层的输出（`self.ln(x)` 的输出）：

- **数据类型：FP32**
- **原因：** `LayerNorm` 对数值稳定性要求高，`autocast` 默认将其保持在 FP32。

模型最终输出 / logits（`self.fc2(x)` 的输出）：

- **数据类型：FP16**
- **原因：** `fc2` 是线性层，处于 `autocast` 范围内，因此其输出为 FP16。

Loss 值：

- **数据类型：FP32**
- **原因：** 为了数值稳定性，loss 计算通常在 FP32 中进行。比如 `nn.CrossEntropyLoss` 在混合精度时也会保留 FP32。

模型参数的梯度（gradients）：

- **数据类型：FP32**
- **原因：** 即使前向和部分反向计算使用 FP16，梯度仍然会累积到 FP32 参数上，因此是 FP32。若使用 `GradScaler`，它也会在更新权重前自动将梯度 unscale。

(b)为什么layer norm会保持FP32

layer norm需要计算方差，均值，对数值敏感，而fp16动态范围小，所以保持fp32。如果使用BF16，动态范围较大，就可以使用。

# memory_profiling  

用torch.cuda.memory._record_memory_history(*max_entries*=1000000)来记录模型运行时显存使用情况

(a)分别测试仅前向传播和一步完整的更新

全精度

![image-20250808170929398](C:\Users\Sherlock\AppData\Roaming\Typora\typora-user-images\image-20250808170929398.png)

![image-20250808170942366](C:\Users\Sherlock\AppData\Roaming\Typora\typora-user-images\image-20250808170942366.png)

混合精度

![image-20250808164939819](C:\Users\Sherlock\AppData\Roaming\Typora\typora-user-images\image-20250808164939819.png)

![image-20250808164951970](C:\Users\Sherlock\AppData\Roaming\Typora\typora-user-images\image-20250808164951970.png)

(b)内存峰值是什么？

先上升后下降是因为计算图是逐渐构建的过程，在前向传播过程中，计算到的变量会逐渐加入计算图，占用显存资源，使用完之后中间变量会逐渐被释放。完整一步的显存占用比仅前向传播多是因为backward操作本身也会创建一些变量占用内存，下降之后保持在2G左右，推测是因为pytorch并不会直接将释放的显存归还显卡，而是缓存一部分供后续使用。

(c)如上图，在仅前向传播中，混合精度峰值显存占用从3.8G下降到2.9G

(e)调节detail这一项，占用显存最多的是哪个部分，这个暂时还不太会看，不过显示的很多都是basic_modules.py中的一个forward函数，大小都是24MB左右，如果是看单个最大的话，峰值处的计算cross entropy loss分配了78MB，是最大的，不太确定是不是这个意思。

# pytorch_attention 

| d_model | seq_len | 100 forward passes (ms) | 100 backward passes (ms) | Memory reserved (MB) | Memory allocated (MB) |
| ------- | ------- | ----------------------- | ------------------------ | -------------------- | --------------------- |
| 16      | 256     | 16.39                   | 60.58                    | 42.00                | 19.12                 |
| 16      | 1024    | 104.65                  | 332.99                   | 174.00               | 51.75                 |
| 16      | 4096    | 1574.61                 | 4940.63                  | 2090.00              | 542.25                |
| 16      | 8192    | 56328.37                | 123112.23                | 10282.00             | 2092.25               |
| 16      | 16384   | OOM                     | OOM                      | OOM                  | OOM                   |
| 32      | 256     | 19.13                   | 63.78                    | 44.00                | 20.00                 |
| 32      | 1024    | 112.35                  | 359.50                   | 178.00               | 55.25                 |
| 32      | 4096    | 1667.90                 | 5109.14                  | 2602.00              | 556.25                |
| 32      | 8192    | 54828.48                | 115598.80                | 10342.00             | 2120.25               |
| 32      | 16384   | OOM                     | OOM                      | OOM                  | OOM                   |
| 64      | 256     | 16.35                   | 59.71                    | 46.00                | 21.75                 |
| 64      | 1024    | 116.11                  | 370.95                   | 170.00               | 62.25                 |
| 64      | 4096    | 1884.75                 | 5561.23                  | 2662.00              | 584.25                |
| 64      | 8192    | 52725.00                | 115790.84                | 10346.00             | 2176.25               |
| 64      | 16384   | OOM                     | OOM                      | OOM                  | OOM                   |
| 128     | 256     | 22.58                   | 58.24                    | 50.00                | 25.25                 |
| 128     | 1024    | 136.35                  | 423.52                   | 202.00               | 76.25                 |
| 128     | 4096    | 2295.73                 | 6524.06                  | 2666.00              | 640.25                |
| 128     | 8192    | 53845.41                | 118617.78                | 10410.00             | 2288.25               |
| 128     | 16384   | OOM                     | OOM                      | OOM                  | OOM                   |

两点观察：

- 时间消耗与seq_len成平方比，dim增加对时间消耗影响不是很大
- 在8192情况下Memory reserved超过了显卡最大显存，说明使用了Memory reserved