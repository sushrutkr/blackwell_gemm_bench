import torch
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

CUTLASS_INCLUDE = os.path.join(os.path.dirname(__file__), 'cutlass/include')

print(f"\n=== Compiling ===\n")

ext = load(
    name='gemm_kernels',
    sources=[
        'main.cu',
        'naive_scalar.cu',
        'shared_tile.cu',
        'cute_gemm.cu',
        # 'warp_specialize.cu',  # kernel 4
        # 'tiled_epilogue.cu',   # kernel 5
        # 'collective_mma.cu',   # kernel 6
        # 'persistent_kernel.cu',# kernel 7
        # 'dedicated_epilogue_warpgroup.cu', # kernel 8
        # 'grid_tilling.cu',     # kernel 9
    ],
    extra_cuda_cflags=[
        '-arch=sm_90a',
        # '-O3',
        '-std=c++17',
        '-lineinfo',
        '-lcuda',
        f'-I{CUTLASS_INCLUDE}',
        # '-ccbin=g++',
        # '-DCUTLASS_ARCH_MMA_SM120A_ENABLED=1',
      
    ],
    extra_ldflags=['-lcublas'],
    verbose=True
)

M, K, N = 2048, 4096, 16384
torch.manual_seed(42)
A     = torch.randn(M, K, dtype=torch.float16, device='cuda') * 0.1
B     = torch.randn(K, N, dtype=torch.float16, device='cuda') * 0.02
C_ref = torch.mm(A.float(), B.float())

print(f"\n=== Starting ===\n")

def benchmark(fn, label, n=100):
    try:
        out = fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n): fn()
        e.record()
        torch.cuda.synchronize()
        ms     = s.elapsed_time(e) / n
        tflops = 2 * M * N * K / (ms * 1e9)
        err    = (out.float() - C_ref).abs().max().item()
        status = 'PASS' if err < 0.05 else 'FAIL'
        print(f"[{label:<35}] {ms:7.3f} ms | {tflops:6.1f} TFLOPS | "
              f"L-inf {err:.4f} | {status}")
        return ms
    except Exception as ex:
        print(f"[{label:<35}] ERROR: {ex}")
        return None

print(f"\n=== FFN GEMM  A({M}×{K}) @ B({K}×{N}) = C({M}×{N})  FP16→FP32 ===\n")
Bt = B.T.contiguous()

ms_ref              = benchmark(lambda: ext.cublas_gemm(A, B),              "cuBLAS (ref)")
ms_naive            = benchmark(lambda: ext.naive_scalar(A, B),             "0. Naive Scalar")
ms_shared           = benchmark(lambda: ext.shared_tile(A, B),              "1. Shared Tile")
ms_cute             = benchmark(lambda: ext.cute_gemm(A, Bt),               "3. CuTe")
# ms_warp_spec        = benchmark(lambda: ext.warp_specialized_gemm(A, B),    "4. Warp Specialization SM100")
# ms_tiled_epi        = benchmark(lambda: ext.tiled_epilogue_gemm(A, B),      "5. Tiled Epilogue SM100")
# ms_collective       = benchmark(lambda: ext.collective_mma_gemm(A, B),      "6. Collective 2CTA MMA")
# ms_persistent       = benchmark(lambda: ext.persistent_gemm(A, B),          "7. Persistent Kernel")
# ms_ded_epi          = benchmark(lambda: ext.dedicated_epilogue_gemm(A, B),  "8. Dedicated Epilogue Warpgroup")
# ms_grid             = benchmark(lambda: ext.grid_tiling_gemm(A, B),         "9. Grid Tiling (Snake)")
ms_wgmma_tma = None
# ms_cute = None
ms_warp_spec = ms_tiled_epi = None
ms_collective = ms_persistent = ms_ded_epi = ms_grid = None

print(f"\n{'─'*75}")
print(f"{'Kernel':<35} {'vs cuBLAS':>10}")
print(f"{'─'*75}")
for label, ms in [
    ("0. Naive Scalar",               ms_naive),
    ("1. Shared Tile",                ms_shared),
    ("2. Raw PTX WGMMA+TMA",          ms_wgmma_tma),
    ("3. CuTe SM90",                  ms_cute),
    ("4. Warp Specialization SM100",  ms_warp_spec),
    ("5. Tiled Epilogue SM100",       ms_tiled_epi),
    ("6. Collective 2CTA MMA",        ms_collective),
    ("7. Persistent Kernel",          ms_persistent),
    ("8. Dedicated Epilogue Warpgroup", ms_ded_epi),
    ("9. Grid Tiling (Snake)",        ms_grid),
]:
    if ms is not None and ms_ref is not None:
        print(f"{label:<35} {100*ms_ref/ms:>9.1f}%")
    else:
        print(f"{label:<35} {'N/A':>10}")
