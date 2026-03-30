#include <torch/extension.h>
#include <cuda_fp16.h>
#include "kernels.h"

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

using namespace cute;

#define TM 128
#define TN 256
#define TK 64
#define MAX_CONCURRENT_STEPS 2
#define EPILOGUE_TILE_N 64
#define CLUSTER_SIZE 2

__global__ __launch_bounds__(128, 1)
void collective_mma_kernel_sm100(
    const half_t* A, const half_t* B, float* C,
    int M, int K, int N)
{
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, 1));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, 1));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, 1));

    int block_in_cluster = blockIdx.x % CLUSTER_SIZE;
    bool is_lead         = (block_in_cluster == 0);
    int warpIdx          = threadIdx.x / 32;

    int cluster_tile_m = CLUSTER_SIZE * TM;
    int cluster_tile_n = CLUSTER_SIZE * TN;

    int cluster_id = blockIdx.x / CLUSTER_SIZE;
    int m_start    = blockIdx.y * cluster_tile_m + block_in_cluster * TM;
    int n_start    = cluster_id  * cluster_tile_n;

    auto tiled_mma = make_tiled_mma(
        SM100_MMA_F16BF16_2x1SM_SS<half_t, half_t, float,
                                    TM, TN,
                                    UMMA::Major::K, UMMA::Major::K>{});

    __shared__ half_t  smem_A[MAX_CONCURRENT_STEPS][TM][TK];
    __shared__ half_t  smem_B[MAX_CONCURRENT_STEPS][TK][TN];
    __shared__ float   acc_smem[2][TM][EPILOGUE_TILE_N];
    __shared__ uint64_t load_barriers[MAX_CONCURRENT_STEPS];
    __shared__ uint64_t consumed_barriers[MAX_CONCURRENT_STEPS];
    __shared__ uint64_t mma_done_barrier;

    cute::TmemAllocator tmem_alloc{};
    auto tC = tmem_alloc.allocate(
        make_shape(Int<TM>{}, Int<TN>{}),
        make_stride(Int<1>{}, Int<TM>{}),
        float{});

    if (threadIdx.x == 0) {
        for (int i = 0; i < MAX_CONCURRENT_STEPS; i++) {
            initialize_barrier(load_barriers[i],     1);
            initialize_barrier(consumed_barriers[i], 1);
        }
        initialize_barrier(mma_done_barrier, 1);
    }
    cute::cluster_sync();

    auto blk_shape = make_shape(Int<TM>{}, Int<TN>{}, Int<TK>{});
    auto gA_tile   = local_tile(gA, select<0,2>(blk_shape),
                         make_coord(m_start / TM, _, 0));
    auto gB_tile   = local_tile(gB, select<2,1>(blk_shape),
                         make_coord(_, n_start / TN, 0));
    auto gC_tile   = local_tile(gC, select<0,1>(blk_shape),
                         make_coord(m_start / TM, n_start / TN));

    int phases = K / TK;

    // warp 0 — producer: each CTA loads its half via TMA multicast
    if (warpIdx == 0) {
        auto tma_copy_A = make_tma_copy_multicast(SM90_TMA_LOAD_MULTICAST{},
            gA, smem_A[0], CLUSTER_SIZE);
        auto tma_copy_B = make_tma_copy_multicast(SM90_TMA_LOAD_MULTICAST{},
            gB, smem_B[0], CLUSTER_SIZE);

        for (int phase = 0; phase < phases; phase++) {
            int slot = phase % MAX_CONCURRENT_STEPS;

            if (phase >= MAX_CONCURRENT_STEPS)
                wait_barrier(consumed_barriers[slot], (phase / MAX_CONCURRENT_STEPS) & 1);

            auto sA = make_tensor(make_smem_ptr(&smem_A[slot][0][0]),
                make_shape(Int<TM>{}, Int<TK>{}), make_stride(Int<TK>{}, Int<1>{}));
            auto sB = make_tensor(make_smem_ptr(&smem_B[slot][0][0]),
                make_shape(Int<TK>{}, Int<TN>{}), make_stride(Int<TN>{}, Int<1>{}));

            if (block_in_cluster == 0)
                copy(tma_copy_A, gA_tile(_, _, phase), sA, load_barriers[slot]);
            else
                copy(tma_copy_B, gB_tile(_, _, phase), sB, load_barriers[slot]);
        }
    }

    // warp 1, lead block only — consumer fires collective tcgen05.mma
    if (warpIdx == 1 && is_lead) {
        auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

        for (int phase = 0; phase < phases; phase++) {
            int slot = phase % MAX_CONCURRENT_STEPS;

            wait_barrier(load_barriers[slot], phase & 1);

            auto sA = make_tensor(make_smem_ptr(&smem_A[slot][0][0]),
                make_shape(Int<TM>{}, Int<TK>{}), make_stride(Int<TK>{}, Int<1>{}));
            auto sB = make_tensor(make_smem_ptr(&smem_B[slot][0][0]),
                make_shape(Int<TK>{}, Int<TN>{}), make_stride(Int<TN>{}, Int<1>{}));

            auto tAgA = thr_mma.partition_A(sA);
            auto tBgB = thr_mma.partition_B(sB);

            gemm(tiled_mma, tC, tAgA, tBgB, tC);
            arrive_barrier(consumed_barriers[slot]);
        }
        tiled_mma.commit(mma_done_barrier);
    }

    __syncthreads();
    wait_barrier(mma_done_barrier, 0);

    // epilogue — each CTA writes its own TM rows
    auto tmem_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tC);

    int epilogue_phases = TN / EPILOGUE_TILE_N;
    for (int ni = 0; ni < epilogue_phases; ni++) {
        auto acc_smem_ni = make_tensor(
            make_smem_ptr(&acc_smem[ni % 2][0][0]),
            make_shape(Int<TM>{}, Int<EPILOGUE_TILE_N>{}),
            make_stride(Int<EPILOGUE_TILE_N>{}, Int<1>{}));

        auto gC_ni = local_tile(gC_tile,
            make_shape(Int<TM>{}, Int<EPILOGUE_TILE_N>{}),
            make_coord(0, ni));

        auto tC_ni = local_tile(tC,
            make_shape(Int<TM>{}, Int<EPILOGUE_TILE_N>{}),
            make_coord(0, ni));

        if (ni >= 1)
            tma_store_wait<1>();

        copy(tmem_copy, tC_ni, acc_smem_ni);
        tma_store_fence();
        copy(SM90_TMA_STORE{}, acc_smem_ni, gC_ni);
        tma_store_arrive();
    }
    tma_store_wait<0>();

    tmem_alloc.release_allocation_lock();
    cute::cluster_sync();
}

torch::Tensor collective_mma_gemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    int cluster_tile_m = CLUSTER_SIZE * TM;
    int cluster_tile_n = CLUSTER_SIZE * TN;

    cudaLaunchConfig_t config = {};
    config.gridDim  = dim3((N / cluster_tile_n) * CLUSTER_SIZE, M / cluster_tile_m);
    config.blockDim = dim3(128);

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {CLUSTER_SIZE, 1, 1};
    config.attrs    = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config,
        collective_mma_kernel_sm100,
        (half_t*)A.data_ptr<at::Half>(),
        (half_t*)B.data_ptr<at::Half>(),
        C.data_ptr<float>(), M, K, N);

    cudaDeviceSynchronize();
    return C;
}