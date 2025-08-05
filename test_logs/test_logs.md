# Benchmarking

run on 4090D autodl platform

=== 性能测试 ===
测试配置:

- 设备: cuda
- d_model: 768
- d_ff: 3072
- num_layers: 12
- num_heads: 12
- model size: small
- trainable_parameters: 0.128625B

前向传播平均时间: 0.035008 秒
前向+反向传播时间: 0.076579 秒
纯反向传播时间(估算): 0.0416 秒
=== 性能测试 ===
测试配置:

- 设备: cuda
- d_model: 1024
- d_ff: 4096
- num_layers: 24
- num_heads: 16
- model size: medium
- trainable_parameters: 0.423183B

前向传播平均时间: 0.092815 秒
前向+反向传播时间: 0.216040 秒
纯反向传播时间(估算): 0.1232 秒
=== 性能测试 ===
测试配置:

- 设备: cuda
- d_model: 1280
- d_ff: 5120
- num_layers: 36
- num_heads: 20
- model size: large
- trainable_parameters: 0.969412B

前向传播平均时间: 0.187382 秒
前向+反向传播时间: 0.447217 秒
纯反向传播时间(估算): 0.2598 秒
=== 性能测试 ===
测试配置:
- 设备: cuda
- d_model: 1600
- d_ff: 6400
- num_layers: 48
- num_heads: 25
- model size: xl
- trainable_parameters: 1.998235B

# nsys_profile  

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