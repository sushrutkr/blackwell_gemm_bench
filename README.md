# Blackwell GEMM Benchmark

Progressive GEMM optimization on **NVIDIA RTX Pro 6000 Blackwell**.

This repository tracks the evolution of matrix multiplication from textbook SIMT (Single Instruction, Multiple Thread) CUDA kernels all the way up to advanced optimization using CuTe DSL. The problem is challenging and blend concepts from Hopper and Blackwell programming model as the present GPU is not a complete Blackwell and lacks things like TMEM.

---

## 🚀 The Roadmap to cuBLAS Performance

### Phase 1: The Traditional SIMT Era

* **Naive Scalar (Global Memory):**
* **Shared Memory Tiling:**

### Phase 2: Basic Tensor Core

* **CuTe Baseline:** 
* **CuTe TMA Async:** 

### Phase 3: Blackwell-Native Async

* **Warp Specialization:** 
* **Tiled Epilogue:** 
* **Collective (2CTA) MMA:** 
* **Persistent Kernel:** 
* **Dedicated Epilogue Warpgroup:** 
* **Grid Tiling (Rasterization):** 
---

## 📊 Benchmark Results
> **Target Shape:** FFN Projection (`m = 2048`, `n = 16384`, `k = 4096`)

| Step | Kernel | Time (ms) | TFLOPS | % cuBLAS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Naive Scalar** | 55.804 | 4.9 | 1.8% | ✅ PASS |
| **1** | **Shared Tile** | 34.656 | 7.9 | 3.0% | ✅ PASS |
| **3** | **CuTe Baseline** | 4.528 | 60.7 | 22.8% | ✅ PASS |
| **4** | **Warp Specialization** | TBD | TBD | TBD | ⏳ Pending |
| **5** | **Tiled Epilogue** | TBD | TBD | TBD | ⏳ Pending |
| **6** | **Collective MMA** | TBD | TBD | TBD | ⏳ Pending |
| **7** | **Persistent Kernel** | TBD | TBD | TBD | ⏳ Pending |
| **8** | **Dedicated Epilogue Warpgroup** | TBD | TBD | TBD | ⏳ Pending |
| **9** | **Grid Tiling** | TBD | TBD | TBD | ⏳ Pending |
| **Ref** | **cuBLAS** | **~1.029** | **267.0** | **100%** | ✅ PASS |

---

## 🛠️ Build Instructions

This project uses **NVIDIA CUTLASS** as a git submodule, but only for the **CuTe** headers (`cutlass/include/cute/**`).  
High‑level CUTLASS GEMM templates (e.g. `GemmUniversal`, `GemmUniversalAdapter`) are **not** used; all kernels are written directly with CuTe + CUDA to build the async state machines manually.

### 1. Clone the repository with submodules

```bash
git clone --recursive https://github.com/sushrutkr/blackwell_gemm_bench.git
cd blackwell_gemm_bench
# If you cloned without --recursive:
# git submodule update --init --recursive
```