# Benchmarking

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