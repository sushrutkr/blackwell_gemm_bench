#include <torch/extension.h>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void cute_gemm_kernel(
    const cutlass::half_t* Aptr, 
    const cutlass::half_t* Bptr, 
    float* Cptr, 
    int M, int N, int K) 
{
    Tensor gA = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor gB = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, Int<1>{})); 
    Tensor gC = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, Int<1>{}));

    using bM = Int<128>;
    using bN = Int<128>;
    using bK = Int<32>;

    Tensor gA_tile = local_tile(gA, make_shape(bM{}, bK{}), make_coord(blockIdx.y, _)); 
    Tensor gB_tile = local_tile(gB, make_shape(bN{}, bK{}), make_coord(blockIdx.x, _)); 
    Tensor gC_tile = local_tile(gC, make_shape(bM{}, bN{}), make_coord(blockIdx.y, blockIdx.x));

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    TiledMMA tiled_mma = make_tiled_mma(mma_op{}, make_layout(Shape<_4, _2, _1>{})); // 256 threads
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    Tensor tAgA = thr_mma.partition_A(gA_tile); // (MMA, MMA_M, MMA_K, num_k_tiles)
    Tensor tBgB = thr_mma.partition_B(gB_tile); 
    Tensor tCgC = thr_mma.partition_C(gC_tile); 

    Tensor tCrA = thr_mma.make_fragment_A(tAgA(_,_,_,0));
    Tensor tCrB = thr_mma.make_fragment_B(tBgB(_,_,_,0));
    Tensor acc  = thr_mma.make_fragment_C(tCgC);
    clear(acc);

    int num_tiles = size<3>(tAgA);
    for (int k = 0; k < num_tiles; ++k) {
        copy(tAgA(_,_,_,k), tCrA);
        copy(tBgB(_,_,_,k), tCrB);
        
        gemm(tiled_mma, tCrA, tCrB, acc);
    }

    copy(acc, tCgC);
}

torch::Tensor cute_gemm(torch::Tensor A, torch::Tensor B) {
    // B was transposed in Python, so its shape is (N, K)
    int M = A.size(0), K = A.size(1), N = B.size(0); 
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 grid(N / 128, M / 128);
    dim3 block(256); 

    cute_gemm_kernel<<<grid, block>>>(
        (cutlass::half_t*)A.data_ptr(), 
        (cutlass::half_t*)B.data_ptr(), 
        C.data_ptr<float>(), 
        M, N, K
    );
    
    return C;
}