#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include "kernels.h"

#define TM 64
#define TN 128
#define TK 16
#define NUM_ACCUM ((TM * TN) / 128)

static CUtensorMap make_2d_tma_desc(void* ptr, uint32_t rows, uint32_t cols,
    uint32_t tile_rows, uint32_t tile_cols, CUtensorMapDataType dtype) {
    CUtensorMap desc{};
    uint64_t dims[2]    = { cols, rows };
    uint64_t strides[1] = { (uint64_t)cols * sizeof(__half) };
    uint32_t tile_sz[2] = { tile_cols, tile_rows };
    cuTensorMapEncodeTiled(&desc, dtype, 2, ptr, dims, strides, tile_sz,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

__global__ __launch_bounds__(128, 1)
void wgmma_tma_kernel(const CUtensorMap* tma_A, const CUtensorMap* tma_B,
    float* C, int M, int K, int N) {

    __shared__ __half smem_A[TM][TK];
    __shared__ __half smem_B[TK][TN];
    __shared__ uint64_t mbar;

    int m_start = blockIdx.y * TM;
    int n_start = blockIdx.x * TN;

    if (threadIdx.x == 0)
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(1));
    __syncthreads();

    float acc[NUM_ACCUM] = {0.f};

    int phases = K / TK;
    for(int phase = 0; phase < phases; phase++) {
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive_expect_tx.shared.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)),
                   "r"((uint32_t)(sizeof(smem_A) + sizeof(smem_B))));

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared(smem_A)), "l"(tma_A),
                   "r"((int32_t)(phase * TK)), "r"((int32_t)m_start),
                   "r"((uint32_t)__cvta_generic_to_shared(&mbar)) : "memory");

            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)__cvta_generic_to_shared(smem_B)), "l"(tma_B),
                   "r"((int32_t)n_start), "r"((int32_t)(phase * TK)),
                   "r"((uint32_t)__cvta_generic_to_shared(&mbar)) : "memory");
        }

        asm volatile(
            "{\n .reg .pred p;\n"
            "$retry_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
            "@!p bra $retry_%=;\n }\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(phase & 1));
        __syncwarp();

        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        uint32_t smem_A_addr = __cvta_generic_to_shared(smem_A);
        uint32_t smem_B_addr = __cvta_generic_to_shared(smem_B);

        asm volatile(
            "wgmma.mma_sync.aligned.m64n128k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
            " %64, %65, 1, 1, 1, 0, 1;"
            : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
              "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
              "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
              "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
              "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
              "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
              "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
              "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]),
              "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]),
              "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]),
              "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]),
              "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]),
              "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]),
              "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]),
              "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]),
              "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
            : "r"(smem_A_addr), "r"(smem_B_addr));

        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }

    int warpIdx = threadIdx.x / 32;
    int laneIdx = threadIdx.x % 32;
    #pragma unroll
    for (int i = 0; i < NUM_ACCUM; i++) {
        int row = m_start + warpIdx * 16 + (laneIdx / 4) + (i / 8) * 8;
        int col = n_start + (laneIdx % 4) * 2 + (i % 8) * 16;
        if (row < M && col     < N) C[row * N + col]     = acc[i];
        if (row < M && col + 1 < N) C[row * N + col + 1] = acc[i];
    }
}

torch::Tensor wgmma_tma_gemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    CUtensorMap tma_A = make_2d_tma_desc(A.data_ptr(), M, K, TM, TK,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
    CUtensorMap tma_B = make_2d_tma_desc(B.data_ptr(), K, N, TK, TN,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16);

    CUtensorMap *d_tma_A, *d_tma_B;
    cudaMalloc(&d_tma_A, sizeof(CUtensorMap));
    cudaMalloc(&d_tma_B, sizeof(CUtensorMap));
    cudaMemcpy(d_tma_A, &tma_A, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tma_B, &tma_B, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    wgmma_tma_kernel<<<dim3(N/TN, M/TM), 128>>>(
        d_tma_A, d_tma_B, C.data_ptr<float>(), M, K, N);

    cudaDeviceSynchronize();
    cudaFree(d_tma_A);
    cudaFree(d_tma_B);
    return C;
}