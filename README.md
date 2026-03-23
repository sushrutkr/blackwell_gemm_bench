# Blackwell GEMM Benchmark

Progressive GEMM optimization on **NVIDIA RTX PRO 6000 Blackwell (sm_90)**,

Inspired by the Pallas matmul progressive optimization table from JAX docs.
Implemented in **CUDA + WMMA + CUTLASS + CuTe** with a PyTorch interface.

---

## Results

| Kernel           | Time (ms) | TFLOPS | % cuBLAS | L-inf Error |
|------------------|-----------|--------|----------|-------------|
| Naive Scalar     | ~800      | ~0.7   | ~0.2%    | PASS        |
| WMMA Naive       | TBD       | TBD    | TBD      | TBD         |
| Shared Tile      | TBD       | TBD    | TBD      | TBD         |
| CUTLASS          | TBD       | TBD    | TBD      | TBD         |
| CuTe             | TBD       | TBD    | TBD      | TBD         |
| **cuBLAS (ref)** | ~0.8      | ~345   | 100%     | PASS        |

> Results on RTX PRO 6000 Blackwell (96GB GDDR7, sm_90). Updated weekly.

---

## Why this GEMM?

The FFN projection (`gate_proj` / `up_proj`)

- `d_model = 4096`
- `d_ff    = 16384` (4× expansion)
- `seq     = 2048`  (typical prefill batch)

---

## Kernel Progression

Each kernel adds one hardware trick:

