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

__global__ __launch_bounds__(128, 1)
void cute_gemm_kernel(
    const __half* A, const __half* B, float* C,
    int M, int K, int N)
{
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, 1));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, 1));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, 1));

    auto blk_shape = make_shape(Int<TM>{}, Int<TN>{}, Int<TK>{});
    auto blk_coord = make_coord(blockIdx.y, blockIdx.x, _);

    auto gA_tile = local_tile(gA, select<0,2>(blk_shape), select<0,2>(blk_coord));
    auto gB_tile = local_tile(gB, select<2,1>(blk_shape), select<2,1>(blk_coord));
    auto gC_tile = local_tile(gC, select<0,1>(blk_shape), select<0,1>(blk_coord));

    __shared__ __half smem_A[TM][TK];
    __shared__ __half smem_B[TK][TN];

    auto sA = make_tensor(make_smem_ptr(&smem_A[0][0]), make_shape(Int<TM>{}, Int<TK>{}), make_stride(Int<TK>{}, Int<1>{}));
    auto sB = make_tensor(make_smem_ptr(&smem_B[0][0]), make_shape(Int<TK>{}, Int<TN>{}), make_stride(Int<TN>{}, Int<1>{}));

    auto tma_copy_A = make_tma_copy(SM90_TMA_LOAD{}, gA, sA);
    auto tma_copy_B = make_tma_copy(SM90_TMA_LOAD{}, gB, sB);

    auto tiled_mma = make_tiled_mma(
        SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{},
        Layout<Shape<_1,_1,_1>>{});

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto tAgA = thr_mma.partition_A(sA);
    auto tBgB = thr_mma.partition_B(sB);
    auto tCgC = thr_mma.partition_C(gC_tile);

    auto acc = partition_fragment_C(tiled_mma, select<0,1>(blk_shape));
    clear(acc);

    auto [tma_load_A, tma_load_B] = tma_partition(
        tma_copy_A, tma_copy_B, threadIdx.x, sA, sB);

    __shared__ uint64_t mbar;
    if (threadIdx.x == 0)
        initialize_barrier(mbar, 1);
    __syncthreads();

    int phases = K / TK;
    for(int phase = 0; phase < phases; phase++) {
        if (threadIdx.x == 0) {
            copy(tma_copy_A, tma_load_A(_, phase), sA, mbar);
            copy(tma_copy_B, tma_load_B(_, phase), sB, mbar);
        }

        wait_barrier(mbar, phase & 1);
        __syncwarp();

        gemm(tiled_mma, acc, tAgA, tBgB, acc);

        warpgroup_arrive();
        warpgroup_wait<0>();
    }

    copy(acc, tCgC);
}

torch::Tensor cute_gemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    cute_gemm_kernel<<<dim3(N/TN, M/TM), 128>>>(
        reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), M, K, N);

    cudaDeviceSynchronize();
    return C;
}