# Blackwell GEMM Benchmark

Progressive GEMM optimization on **NVIDIA RTX Blackwell 6000 architecture**.

This repository tracks the evolution of matrix multiplication from textbook SIMT (Single Instruction, Multiple Thread) CUDA kernels all the way up to **Blackwell-native async CUDA** using Tensor Memory Accelerator (TMA), Warp Group Matrix Multiply Accumulate (WGMMA), and 5th-generation Tensor Core instructions (`tcgen05`) with on-chip Tensor Memory (TMEM) via **CuTe**.

---

## 🚀 The Roadmap to cuBLAS Performance

Modern NVIDIA GPUs do not scale performance by having threads calculate memory addresses faster. They scale by using specialized hardware units and orchestrating them asynchronously. We start with the classics to see where they bottleneck, and then move into the modern stack.

### Phase 1: The Traditional SIMT Era

* **0. Naive Scalar (Global Memory):** The textbook 3-for-loop implementation. Every thread computes one output element, reading directly from Global Memory. Bottlenecked instantly by memory bandwidth.
* **1. Shared Memory Tiling:** Using Shared Memory (SMEM) as a user-managed L1 cache. Threads cooperatively load a tile of A and B into SMEM, synchronize (`__syncthreads()`), and compute.

### Phase 2: Basic Tensor Core

* **3. CuTe Baseline:** Replacing raw PTX with CuTe's `make_tma_copy` and `make_tiled_mma` (`SM90` atoms) to elegantly map logical coordinates to physical memory. Same hardware, cleaner code.

### Phase 3: Blackwell-Native Async

* **4. Warp Specialization:** 
* **5. Tiled Epilogue:** 
* **6. Collective (2CTA) MMA:** 
* **7. Persistent Kernel:** 
* **8. Dedicated Epilogue Warpgroup:** 
* **9. Grid Tiling (Rasterization):** 
---

## 📊 Benchmark Results

> **Hardware:** NVIDIA RTX Blackwell 6000
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

This project relies on the **CUTLASS 3.x/4.x** library strictly for the **CuTe** headers. High-level CUTLASS templates (`GemmUniversal`) are *not* used in the kernel source code, as the goal is to build the async state machines from scratch.

### 1. Clone the repository with submodules
```bash
git clone --recursive https://github.com/sushrutkr/blackwell_gemm_bench.git
cd blackwell_gemm_bench
```