import torch
from torch.utils.cpp_extension import load

ext = load(
    name='gemm_kernels',
    sources=[
        'main.cu',
        'naive_scalar.cu',
        # 'wmma_naive.cu',    # Wk2
        'shared_tile.cu',   # Wk3
        # 'cutlass_gemm.cu',  # Wk4
        # 'cute_gemm.cu',     # Wk5
    ],
    extra_cuda_cflags=['-arch=sm_90', '-O3'],
    extra_ldflags=['-lcublas'],
    verbose=True
)

M, K, N = 2048, 4096, 16384
torch.manual_seed(42)
A     = torch.randn(M, K, dtype=torch.float16, device='cuda') * 0.1
B     = torch.randn(K, N, dtype=torch.float16, device='cuda') * 0.02
C_ref = torch.mm(A.float(), B.float())

def benchmark(fn, label, n=100):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record()
    torch.cuda.synchronize()
    ms     = s.elapsed_time(e) / n
    tflops = 2 * M * N * K / (ms * 1e9)
    err    = (fn().float() - C_ref).abs().max().item()
    print(f"[{label:<20}] {ms:7.3f} ms | {tflops:6.1f} TFLOPS | "
          f"L-inf {err:.4f} | {'PASS' if err < 0.05 else 'FAIL'}")
    return ms

print(f"\n=== FFN GEMM  A({M}×{K}) @ B({K}×{N}) = C({M}×{N})  FP16→FP32 ===\n")
ms_ref = benchmark(lambda: ext.cublas_gemm(A, B),  "cuBLAS (ref)")
ms_naive = benchmark(lambda: ext.naive_scalar(A, B), "Naive Scalar")
ms_shared_tilling = benchmark(lambda: ext.shared_tile(A, B), "Shared Tilling")
# ms = benchmark(lambda: ext.wmma_naive(A, B),   "WMMA Naive")    # Wk2
# ms = benchmark(lambda: ext.cutlass_gemm(A, B), "CUTLASS")       # Wk4
# ms = benchmark(lambda: ext.cute_gemm(A, B),    "CuTe")          # Wk5

print(f"\nNaive Scalar: {100 * ms_ref / ms_naive:.1f}% of cuBLAS")
print(f"\nShared Tilling: {100 * ms_ref / ms_shared_tilling:.1f}% of cuBLAS")