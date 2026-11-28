import time
import torch

N = 2048
a_cpu = torch.randn(N, N)
b_cpu = torch.randn(N, N)

_ = a_cpu @ b_cpu
start_cpu = time.perf_counter()
c_cpu = a_cpu @ b_cpu
cpu_time = time.perf_counter() - start_cpu
print(f"CPU ({N}x{N} matmul): {cpu_time * 1000:.2f} ms")

if torch.cuda.is_available():
  a_gpu = a_cpu.to("cuda")
  b_gpu = b_cpu.to("cuda")

  for _ in range(5):
    _ = a_gpu @ b_gpu
  torch.cuda.synchronize()

  start_gpu = time.perf_counter()
  c_gpu = a_gpu @ b_gpu
  torch.cuda.synchronize()  
  gpu_time = time.perf_counter() - start_gpu

  print(f"GPU ({N}x{N} matmul): {gpu_time * 1000:.2f} ms")
  print(f"Speedup: {cpu_time / gpu_time:.1f}x")
else:
  print("CUDA GPU not available on this machine; run in Google Colab with a T4 GPU to compare.")