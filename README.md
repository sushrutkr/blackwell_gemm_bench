# Blackwell GEMM Benchmark

Progressive GEMM optimization on **NVIDIA RTX Blackwell 6000 architecture**.

This repository tracks the evolution of matrix multiplication from textbook SIMT (Single Instruction, Multiple Thread) CUDA kernels all the way up to **Blackwell-native async CUDA** using Tensor Memory Accelerator (TMA), Warp Group Matrix Multiply Accumulate (WGMMA), and 5th-generation Tensor Core instructions (`tcgen05`) with on-chip Tensor Memory (TMEM) via **CuTe**.

---

## 🚀 The Roadmap to cuBLAS Performance

Modern NVIDIA GPUs do not scale performance by having threads calculate memory addresses faster. They scale by using specialized hardware units and orchestrating them asynchronously. We start with the classics to see where they bottleneck, and then move into the modern stack.

### Phase 1: The Traditional SIMT Era

* **0. Naive Scalar (Global Memory):** The textbook 3-for-loop implementation. Every thread computes one output element, reading directly from Global Memory. Bottlenecked instantly by memory bandwidth.
* **1. Shared Memory Tiling:** Using Shared Memory (SMEM) as a user-managed L1 cache. Threads cooperatively load a tile of A and B into SMEM, synchronize (`__syncthreads()`), and compute.

### Phase 2: Hopper-Compatible Async (SM90 atoms)

* **2. Raw PTX Baseline (WGMMA + TMA):** Bypassing standard memory loads entirely. Writing device code using raw inline PTX assembly (`cp.async.bulk.tensor` and `wgmma.mma_async`). Accumulator lives in registers.
* **3. CuTe Baseline:** Replacing raw PTX with CuTe's `make_tma_copy` and `make_tiled_mma` (`SM90` atoms) to elegantly map logical coordinates to physical memory. Same hardware, cleaner code.

### Phase 3: Blackwell-Native Async (SM100 atoms + TMEM)

* **4. Warp Specialization:** Splitting the threadblock into a **Producer** warp (TMA loads) and a **Consumer** warp (`tcgen05.mma` math) with double-buffered SMEM slots to overlap memory and compute. Accumulator moves from registers into **TMEM**.
* **5. Tiled Epilogue:** Breaking up the final accumulator write-back into `EPILOGUE_TILE_N`-wide chunks, pipelining TMEM→SMEM with SMEM→GMEM via TMA stores to prevent epilogue stalls.
* **6. Collective (2CTA) MMA:** Utilizing **Threadblock Clusters** and TMA Multicast so two CTAs collaboratively load operands and issue a single collective `tcgen05.mma`, doubling arithmetic intensity.
* **7. Persistent Kernel:** Bypassing the hardware scheduler by launching a fixed grid of SMs that pull output tiles from an atomic counter. Amortizes block initialization and overlaps epilogue with next tile's compute.
* **8. Dedicated Epilogue Warpgroup:** Splitting the threadblock into two warpgroups — **wg0** owns TMA loads + `tcgen05.mma`, **wg1** owns the TMEM→GMEM epilogue — with double-buffered TMEM accumulators so both run concurrently.
* **9. Grid Tiling (Rasterization):** Replacing the linear tile traversal order with a **planar snake** pattern so neighboring CTAs share L2 cache lines, maximizing L2 hit rate across the output matrix.

---

## 📊 Benchmark Results

> **Hardware:** NVIDIA RTX Blackwell 6000
> **Target Shape:** FFN Projection (`m = 2048`, `n = 16384`, `k = 4096`)

| Step | Kernel | Time (ms) | TFLOPS | % cuBLAS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Naive Scalar** | 55.804 | 4.9 | 1.8% | ✅ PASS |
| **1** | **Shared Tile** | 34.656 | 7.9 | 3.0% | ✅ PASS |
| **2** | **Raw PTX (WGMMA+TMA)** | TBD | TBD | TBD | 🚧 WIP |
| **3** | **CuTe Baseline (SM90)** | TBD | TBD | TBD | ⏳ Pending |
| **4** | **Warp Specialization (SM100)** | TBD | TBD | TBD | ⏳ Pending |
| **5** | **Tiled Epilogue (SM100)** | TBD | TBD | TBD | ⏳ Pending |
| **6** | **Collective MMA (SM100)** | TBD | TBD | TBD | ⏳ Pending |
| **7** | **Persistent Kernel (SM100)** | TBD | TBD | TBD | ⏳ Pending |
| **8** | **Dedicated Epilogue Warpgroup (SM100)** | TBD | TBD | TBD | ⏳ Pending |
| **9** | **Grid Tiling (SM100)** | TBD | TBD | TBD | ⏳ Pending |
| **Ref** | **cuBLAS** | **~1.029** | **267.0** | **100%** | ✅ PASS |

---

## 🛠️ Build Instructions

This project relies on the **CUTLASS 3.x/4.x** library strictly for the **CuTe** headers. High-level CUTLASS templates (`GemmUniversal`) are *not* used in the kernel source code, as the goal is to build the async state machines from scratch.

### 1. Clone the repository with submodules
```bash
git clone --recursive https://github.com/sushrutkr/blackwell_gemm_bench.git
cd blackwell_gemm_bench
```