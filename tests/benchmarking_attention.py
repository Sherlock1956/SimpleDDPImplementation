import torch
import torch.nn as nn
import time
import itertools
from contextlib import contextmanager
try:
    from flash_attention_modules import *
except:
    from .flash_attention_modules import *

# Context manager for timing
@contextmanager
def timer(name: str):
    start = time.time()
    yield
    torch.cuda.synchronize()  # Ensure GPU finishes
    print(f"{name} took {(time.time() - start)*1000:.2f} ms")
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
class Attention_triton(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V):
        return apply_flash_atn_triton(Q, K, V, is_causal=False)

# Simple scaled dot-product attention (no masking, no multi-head)
# def attention(Q, K, V):
#     d_k = Q.size(-1)
#     scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5
#     attn = torch.softmax(scores, dim=-1)
#     return torch.matmul(attn, V)

# Warm-up function
def warmup(steps=10, model=None):
    for _ in range(steps):
        with torch.no_grad():
            _ = model(torch.randn(1, 1024, 64, device='cuda'),
                          torch.randn(1, 1024, 64, device='cuda'),
                          torch.randn(1, 1024, 64, device='cuda'))
        torch.cuda.synchronize()

# Benchmark function
def benchmark_attention(d_model, seq_len, model):
    print(f"\nBenchmarking d_model={d_model}, seq_len={seq_len}...")

    # Allocate inputs
    Q = torch.randn(1, seq_len, d_model, device='cuda', requires_grad=True)
    K = torch.randn(1, seq_len, d_model, device='cuda', requires_grad=True)
    V = torch.randn(1, seq_len, d_model, device='cuda', requires_grad=True)

    # Warm-up forward
    for _ in range(10):
        out = model(Q, K, V)
        out.sum().backward()
        Q.grad.zero_(); K.grad.zero_(); V.grad.zero_()
    torch.cuda.synchronize()

    # === Forward timing ===
    with timer("100 forward passes"):
        for _ in range(100):
            out = model(Q, K, V)
            torch.cuda.synchronize()

    # === Memory measurement before backward ===
    mem_reserved = torch.cuda.memory_reserved()
    mem_allocated = torch.cuda.memory_allocated()
    print(f"Memory reserved: {mem_reserved / 1024**2:.2f} MB")
    print(f"Memory allocated: {mem_allocated / 1024**2:.2f} MB")

    # === Backward timing ===
    with timer("100 backward passes"):
        for _ in range(100):
            out = model(Q, K, V)
            out.sum().backward()
            Q.grad.zero_(); K.grad.zero_(); V.grad.zero_()
            torch.cuda.synchronize()

# Main script
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    torch.cuda.empty_cache()
    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Model
    model = Attention()
    model = torch.compile(model)

    # Warm up GPU
    print("Warming up...")
    warmup(model=model)

    # Parameters
    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]

    # Cartesian product
    for d_model, seq_len in itertools.product(d_models, seq_lengths):
        try:
            # with torch.no_grad():
            #     torch.cuda.reset_peak_memory_stats()
            #     torch.cuda.empty_cache()
            benchmark_attention(d_model, seq_len, model=model)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error for d_model={d_model}, seq_len={seq_len}")
                torch.cuda.empty_cache()
            else:
                print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print("\nBenchmarking complete.")