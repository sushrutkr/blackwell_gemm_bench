# Blackwell GEMM Benchmark

Progressive GEMM optimization on **NVIDIA RTX 6000 Ada / Blackwell architectures**, inspired by the Pallas matmul progressive optimization table from the JAX documentation.

This repository tracks the evolution of matrix multiplication from textbook SIMT (Single Instruction, Multiple Thread) CUDA kernels all the way up to **Asynchronous CUDA** using Tensor Memory Accelerator (TMA) and Warp Group Matrix Multiply Accumulate (WGMMA) via **CuTe**.

---

## 🚀 The Roadmap to cuBLAS Performance

Modern NVIDIA GPUs do not scale performance by having threads calculate memory addresses faster. They scale by using specialized hardware units and orchestrating them asynchronously. We start with the classics to see where they bottleneck, and then move into the modern stack.

### Phase 1: The Traditional SIMT Era
* **0. Naive Scalar (Global Memory):** The textbook 3-for-loop implementation. Every thread computes one output element, reading directly from Global Memory. Bottlenecked instantly by memory bandwidth.
* **1. Shared Memory Tiling:** Using Shared Memory (SMEM) as a user-managed L1 cache. Threads cooperatively load a tile of A and B into SMEM, synchronize (`__syncthreads()`), and compute. 

### Phase 2: The Hopper/Blackwell Async Era
* **2. The "Raw" Baseline (WGMMA + TMA):** Bypassing standard memory loads. Writing device code using raw inline PTX assembly (`cp.async.bulk.tensor` and `wgmma.mma_async`). 
* **3. Version 0: The CuTe Baseline:** Replacing raw PTX with CuTe's `make_tma_copy` and `make_tiled_mma` to elegantly map logical coordinates to physical memory.
* **4. Version 1: Warp Specialization:** Splitting the threadblock into **Producer** warps (TMA loads) and **Consumer** warps (WGMMA math) to overlap memory and compute.
* **5. Version 2: Tiled Epilogue:** Breaking up the final accumulator write-back to Global Memory into smaller chunks using TMA Stores to prevent pipeline stalls.
* **6. Version 3: Collective (2CTA) MMA:** Utilizing **Threadblock Clusters** and TMA Multicast to load a single tile of data into the SMEM of multiple neighboring blocks.
* **7. Version 4: Persistent Kernel:** Bypassing the hardware scheduler by launching a fixed grid of SMs that pull work from an atomic tile queue.
* **8. Version 5: Dedicated Epilogue Warpgroup:** Three-way warp specialization (Producer, Math, Epilogue) using double-buffered Tensor Memory (TMEM) accumulators.
* **9. Version 6: Grid Tiling (Rasterization):** Modifying the persistent atomic work queue to dish out output tiles in 2D spatial chunks to maximize L2 cache hit rates.

---

## 📊 Benchmark Results

> **Hardware:** NVIDIA RTX 6000 Ada / Blackwell (96GB GDDR)  
> **Target Shape:** FFN Projection (`m = 2048`, `n = 16384`, `k = 4096`)

| Step | Kernel | Time (ms) | TFLOPS | % cuBLAS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Naive Scalar** | 55.804 | 4.9 | 1.8% | ✅ PASS |
| **1** | **Shared Tile** | 34.656 | 7.9 | 3.0% | ✅ PASS |
| **2** | **Raw PTX (WGMMA+TMA)** | TBD | TBD | TBD | 🚧 WIP |
| **3** | **CuTe Baseline** | TBD | TBD | TBD | ⏳ Pending |
| **4** | **Warp Specialization** | TBD | TBD | TBD | ⏳ Pending |
| **5** | **Tiled Epilogue** | TBD | TBD | TBD | ⏳ Pending |
| **6** | **Collective MMA** | TBD | TBD | TBD | ⏳ Pending |
| **7** | **Persistent Kernel** | TBD | TBD | TBD | ⏳ Pending |
| **8** | **Dedicated Epilogue** | TBD | TBD | TBD | ⏳ Pending |
| **9** | **Grid Tiling** | TBD | TBD | TBD | ⏳ Pending |
| **Ref**| **cuBLAS** | **~1.029** | **267.0** | **100%** | ✅ PASS |

---

## 🛠️ Build Instructions

This project relies on the **CUTLASS 3.x/4.x** library strictly for the **CuTe** headers. High-level CUTLASS templates (`GemmUniversal`) are *not* used in the kernel source code, as the goal is to build the async state machines from scratch.

### 1. Clone the repository with submodules
To get the CuTe headers, you must clone the repo recursively to fetch the CUTLASS submodule:
```bash
git clone --recursive [https://github.com/sushrutkr/blackwell_gemm_bench.git](https://github.com/sushrutkr/blackwell_gemm_bench.git)
cd blackwell_gemm_bench
