#include <torch/extension.h>
#include <cuda_fp16.h>
#include "kernels.h"

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define TM 64
#define TN 128
#define TK 16
#define MAX_CONCURRENT_STEPS 2
#define EPILOGUE_TILE_N 32
#define CLUSTER_SIZE 2

__global__ __launch_bounds__(128, 1)
void persistent_kernel(
    const __half* A, const __half* B, float* C,
    int M, int K, int N,
    int m_iters, int n_iters,
    int* tile_counter)
{
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, 1));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, 1));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, 1));

    int block_in_cluster = blockIdx.x % CLUSTER_SIZE;
    bool is_lead         = (block_in_cluster == 0);

    __shared__ __half smem_A[MAX_CONCURRENT_STEPS][TM][TK];
    __shared__ __half smem_B[MAX_CONCURRENT_STEPS][TK][TN];
    __shared__ float  acc_smem[2][TM][EPILOGUE_TILE_N];
    __shared__ uint64_t load_barriers[MAX_CONCURRENT_STEPS];
    __shared__ uint64_t consumed_barriers[MAX_CONCURRENT_STEPS];
    __shared__ uint64_t mma_done_barrier;
    __shared__ int tile_idx_smem;

    auto tiled_mma = make_tiled_mma(
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{},
        Layout<Shape<_1,_1,_1>>{});

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    int warpIdx  = threadIdx.x / 32;
    int phases   = K / TK;

    if (threadIdx.x == 0) {
        for (int i = 0; i < MAX_CONCURRENT_STEPS; i++) {
            initialize_barrier(load_barriers[i],     1);
            initialize_barrier(consumed_barriers[i], 1);
        }
        initialize_barrier(mma_done_barrier, 1);
    }
    __syncthreads();

    int local_index = 0;

    // persistent loop — each cluster grabs tiles from the atomic counter
    while (true) {
        if (threadIdx.x == 0 && is_lead)
            tile_idx_smem = atomicAdd(tile_counter, 1);
        __syncthreads();

        int tile_idx = tile_idx_smem;
        if (tile_idx >= m_iters * n_iters) break;

        int m_index = tile_idx / n_iters;
        int n_index = tile_idx % n_iters;
        int m_start = m_index * CLUSTER_SIZE * TM + block_in_cluster * TM;
        int n_start = n_index * CLUSTER_SIZE * TN;

        auto blk_shape = make_shape(Int<TM>{}, Int<TN>{}, Int<TK>{});
        auto gA_tile   = local_tile(gA, select<0,2>(blk_shape),
                             make_coord(m_start / TM, _, 0));
        auto gB_tile   = local_tile(gB, select<2,1>(blk_shape),
                             make_coord(_, n_start / TN, 0));
        auto gC_tile   = local_tile(gC, select<0,1>(blk_shape),
                             make_coord(m_start / TM, n_start / TN));

        auto acc = partition_fragment_C(tiled_mma, select<0,1>(blk_shape));
        clear(acc);

        // warp 0 — producer
        if (warpIdx == 0) {
            auto tma_copy_A = make_tma_copy_multicast(SM90_TMA_LOAD_MULTICAST{},
                gA, smem_A[0], CLUSTER_SIZE);
            auto tma_copy_B = make_tma_copy_multicast(SM90_TMA_LOAD_MULTICAST{},
                gB, smem_B[0], CLUSTER_SIZE);

            for (int phase = 0; phase < phases; phase++) {
                int slot = phase % MAX_CONCURRENT_STEPS;

                // on first tile skip wait for phases < MAX_CONCURRENT_STEPS
                // on subsequent tiles always wait — smem was used by prev tile
                if (phase >= MAX_CONCURRENT_STEPS || local_index > 0)
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

        // warp 1, lead block only — consumer
        if (warpIdx == 1 && is_lead) {
            for (int phase = 0; phase < phases; phase++) {
                int slot = phase % MAX_CONCURRENT_STEPS;

                wait_barrier(load_barriers[slot], phase & 1);

                auto sA = make_tensor(make_smem_ptr(&smem_A[slot][0][0]),
                    make_shape(Int<TM>{}, Int<TK>{}), make_stride(Int<TK>{}, Int<1>{}));
                auto sB = make_tensor(make_smem_ptr(&smem_B[slot][0][0]),
                    make_shape(Int<TK>{}, Int<TN>{}), make_stride(Int<TN>{}, Int<1>{}));

                auto tAgA = thr_mma.partition_A(sA);
                auto tBgB = thr_mma.partition_B(sB);

                gemm(tiled_mma, acc, tAgA, tBgB, acc);
                warpgroup_arrive();
                warpgroup_wait<0>();

                arrive_barrier(consumed_barriers[slot]);
            }
            arrive_barrier(mma_done_barrier);
        }

        __syncthreads();

        wait_barrier(mma_done_barrier, 0);

        int epilogue_phases = TN / EPILOGUE_TILE_N;
        for (int ni = 0; ni < epilogue_phases; ni++) {
            auto acc_smem_ni = make_tensor(
                make_smem_ptr(&acc_smem[ni % 2][0][0]),
                make_shape(Int<TM>{}, Int<EPILOGUE_TILE_N>{}),
                make_stride(Int<EPILOGUE_TILE_N>{}, Int<1>{}));

            auto gC_ni = local_tile(gC_tile,
                make_shape(Int<TM>{}, Int<EPILOGUE_TILE_N>{}),
                make_coord(0, ni));

            if (ni >= 1)
                tma_store_wait<1>();

            auto acc_ni = local_tile(acc,
                make_shape(Int<TM>{}, Int<EPILOGUE_TILE_N>{}),
                make_coord(0, ni));

            copy(acc_ni, acc_smem_ni);
            tma_store_fence();
            copy(SM90_TMA_STORE{}, acc_smem_ni, gC_ni);
            tma_store_arrive();
        }
        tma_store_wait<0>();

        // wait for async TMEM loads to complete before WGMMA can overwrite
        // TMEM on the next tile — mirrors plgpu.wait_load_tmem() from JAX
        warpgroup_wait<0>();

        local_index++;
    }
}

torch::Tensor persistent_gemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    int cluster_tile_m = CLUSTER_SIZE * TM;
    int cluster_tile_n = CLUSTER_SIZE * TN;
    int m_iters        = M / cluster_tile_m;
    int n_iters        = N / cluster_tile_n;

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int num_clusters = num_sms / CLUSTER_SIZE;

    int* tile_counter;
    cudaMalloc(&tile_counter, sizeof(int));
    cudaMemset(tile_counter, 0, sizeof(int));

    dim3 grid(num_clusters * CLUSTER_SIZE, 1);
    dim3 block(128);

    cudaLaunchConfig_t config = {};
    config.gridDim  = grid;
    config.blockDim = block;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {CLUSTER_SIZE, 1, 1};
    config.attrs    = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config,
        persistent_kernel,
        reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), M, K, N,
        m_iters, n_iters, tile_counter);

    cudaDeviceSynchronize();
    cudaFree(tile_counter);
    return C;
}