# Blackwell GEMM Benchmark

Progressive GEMM optimization on **NVIDIA RTX PRO 6000 Blackwell**,

Inspired by the Pallas matmul progressive optimization table from JAX docs.
Implemented in **CUDA + WMMA + CUTLASS + CuTe** with a PyTorch interface.

---

## Results

| Kernel           | Time (ms) | TFLOPS | % cuBLAS | L-inf Error |
|------------------|-----------|--------|----------|-------------|
| Naive Scalar     | 55.804    | 4.9    | 1.8%     | PASS        |
| Shared Tile      | 34.656    | 7.9    | 3.0%     | PASS        |
| WMMA Naive       | TBD       | TBD    | TBD      | TBD         |
| CUTLASS          | TBD       | TBD    | TBD      | TBD         |
| CuTe             | TBD       | TBD    | TBD      | TBD         |
| **cuBLAS (ref)** | 1.029     | 267.0  | 100%     | PASS        |

> Results on RTX PRO 6000 Blackwell (96GB GDDR7).

---

## Why this GEMM?

The FFN projection (`gate_proj` / `up_proj`)

- `d_model = 4096`
- `d_ff    = 16384` (4× expansion)
- `seq     = 2048`  (typical prefill batch)

---

## Kernel Progression

Each kernel adds one hardware trick:

