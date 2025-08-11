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

测试普通的attention在不同数据规模下的耗时

| d_model | seq_len | Forward (100 passes) | Backward (100 passes) | Memory Reserved | Memory Allocated |
| ------- | ------- | -------------------- | --------------------- | --------------- | ---------------- |
| 16      | 256     | 9.80 ms              | 75.05 ms              | 42.00 MB        | 19.12 MB         |
| 16      | 1024    | 26.56 ms             | 113.99 ms             | 174.00 MB       | 51.75 MB         |
| 16      | 4096    | 381.23 ms            | 1176.45 ms            | 2090.00 MB      | 542.25 MB        |
| 16      | 8192    | 1453.46 ms           | 4420.82 ms            | 10282.00 MB     | 2092.25 MB       |
| 16      | 16384   | OOM                  | OOM                   | OOM             | OOM              |
| 32      | 256     | 20.58 ms             | 81.62 ms              | 44.00 MB        | 20.00 MB         |
| 32      | 1024    | 28.02 ms             | 90.45 ms              | 178.00 MB       | 55.25 MB         |
| 32      | 4096    | 385.21 ms            | 1185.48 ms            | 2602.00 MB      | 556.25 MB        |
| 32      | 8192    | 1473.76 ms           | 4477.36 ms            | 10342.00 MB     | 2120.25 MB       |
| 32      | 16384   | OOM                  | OOM                   | OOM             | OOM              |
| 64      | 256     | 14.52 ms             | 93.63 ms              | 46.00 MB        | 21.75 MB         |
| 64      | 1024    | 28.66 ms             | 99.31 ms              | 170.00 MB       | 62.25 MB         |
| 64      | 4096    | 391.25 ms            | 1205.10 ms            | 2662.00 MB      | 584.25 MB        |
| 64      | 8192    | 1488.09 ms           | 4545.52 ms            | 10346.00 MB     | 2176.25 MB       |
| 64      | 16384   | OOM                  | OOM                   | OOM             | OOM              |
| 128     | 256     | 9.78 ms              | 50.73 ms              | 50.00 MB        | 25.25 MB         |
| 128     | 1024    | 29.00 ms             | 91.72 ms              | 202.00 MB       | 76.25 MB         |
| 128     | 4096    | 426.99 ms            | 1289.84 ms            | 2666.00 MB      | 640.25 MB        |
| 128     | 8192    | 1624.00 ms           | 4840.41 ms            | 10410.00 MB     | 2288.25 MB       |
| 128     | 16384   | OOM                  | OOM                   | OOM             | OOM              |

重点观察：

- 时间消耗与seq_len成平方比，dim增加对时间消耗影响不是很大

# torch_compile  

使用torch.compile(model)进行模型优化，测试速度明显变快，reserved memory也变少了

| d_model | seq_len | Forward (100 passes) | Backward (100 passes) | Memory Reserved | Memory Allocated |
| ------- | ------- | -------------------- | --------------------- | --------------- | ---------------- |
| 16      | 256     | 31.73 ms             | 96.26 ms              | 42.00 MB        | 19.12 MB         |
| 16      | 1024    | 39.53 ms             | 121.09 ms             | 110.00 MB       | 51.75 MB         |
| 16      | 4096    | 327.78 ms            | 892.84 ms             | 1066.00 MB      | 542.25 MB        |
| 16      | 8192    | 1403.92 ms           | 3694.73 ms            | 4158.00 MB      | 2092.25 MB       |
| 16      | 16384   | 3892.81 ms           | 11068.14 ms           | 16742.00 MB     | 8264.25 MB       |
| 32      | 256     | 28.41 ms             | 83.87 ms              | 44.00 MB        | 20.00 MB         |
| 32      | 1024    | 89.37 ms             | 191.02 ms             | 116.00 MB       | 55.25 MB         |
| 32      | 4096    | 436.34 ms            | 1064.99 ms            | 1086.00 MB      | 556.25 MB        |
| 32      | 8192    | 1121.31 ms           | 3005.46 ms            | 6246.00 MB      | 2120.25 MB       |
| 32      | 16384   | OOM                  | OOM                   | OOM             | OOM              |
| 64      | 256     | 35.62 ms             | 102.22 ms             | 46.00 MB        | 21.75 MB         |
| 64      | 1024    | 92.68 ms             | 194.52 ms             | 106.00 MB       | 62.25 MB         |
| 64      | 4096    | 443.34 ms            | 1085.61 ms            | 1638.00 MB      | 584.25 MB        |
| 64      | 8192    | 1142.18 ms           | 3069.71 ms            | 6282.00 MB      | 2176.25 MB       |
| 64      | 16384   | OOM                  | OOM                   | OOM             | OOM              |
| 128     | 256     | 29.24 ms             | 88.36 ms              | 52.00 MB        | 25.25 MB         |
| 128     | 1024    | 94.58 ms             | 199.20 ms             | 126.00 MB       | 76.25 MB         |
| 128     | 4096    | 477.61 ms            | 1165.66 ms            | 1674.00 MB      | 640.25 MB        |
| 128     | 8192    | 1302.41 ms           | 3403.81 ms            | 6378.00 MB      | 2288.25 MB       |
| 128     | 16384   | OOM                  | OOM                   | OOM             | OOM              |

# Flash attention学习笔记

[推荐观看b站视频](https://www.bilibili.com/video/BV1UT421k7rA/?spm_id_from=333.1391.0.0&vd_source=cacd898e44cd6114d93337514538a038)

# flash_forward

(a)用pytorch实现flash attention2的操作

理解flash attention的操作之后再写代码就不困难了，写好之后可以先与普通的attention操作进行简单对比

```python
  seq_len_list = [256, 1024, 4096, 8192, 16384]
  for _ in range(5):
      self_attention(torch.rand(1, 4096, 64),torch.rand(1, 4096, 64),torch.rand(1, 4096, 64))
  Q = torch.rand(8, 16384, 64)
  K = torch.rand(8, 16384, 64)
  V = torch.rand(8, 16384, 64)
  attention_pt = self_attention(Q, K, V)
  attention_flash = apply_flash_atn_pt(Q, K, V)
  assert torch.allclose(attention_flash, attention_pt, rtol=1e-5)
  for seq_len in seq_len_list:
      Q = torch.rand(8, seq_len, 64)
      K = torch.rand(8, seq_len, 64)
      V = torch.rand(8, seq_len, 64)
      pytorch_time = timeit.timeit(lambda: self_attention(Q, K, V), number=1)
      flash_time = timeit.timeit(lambda: apply_flash_atn_pt(Q, K, V), number=1)
      print(f"Seq_len: {seq_len}, PyTorch time: {pytorch_time:.4f}s, Flash time: {flash_time:.4f}s")
```

运行结果如下：

```bash
Seq_len: 256, PyTorch time: 0.0002s, Flash time: 0.0018s
Seq_len: 1024, PyTorch time: 0.0002s, Flash time: 0.0130s
Seq_len: 4096, PyTorch time: 0.0002s, Flash time: 0.1810s
Seq_len: 8192, PyTorch time: 0.0002s, Flash time: 0.6894s
Seq_len: 16384, PyTorch time: 0.0956s, Flash time: 2.7439s
```

发现flash attention的pytorch版本比pytorch本身的实现慢很多，因为有很多IO加载和更多的变量操作，这是符合常理的。
