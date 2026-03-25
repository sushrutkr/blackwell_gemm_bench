#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "kernels.h"

// ── cuBLAS baseline ──────────────────────────────────────────────
static cublasHandle_t handle = nullptr;
void init_cublas() { if (!handle) cublasCreate(&handle); }

torch::Tensor cublas_gemm(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N},
                 torch::dtype(torch::kFloat32).device(torch::kCUDA));
    init_cublas();
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, B.data_ptr<at::Half>(), CUDA_R_16F, N,
                         A.data_ptr<at::Half>(), CUDA_R_16F, K,
                 &beta,  C.data_ptr<float>(),    CUDA_R_32F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return C;
}

// ── PYBIND ───────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublas_gemm",  &cublas_gemm,  "cuBLAS baseline");
    m.def("naive_scalar", &naive_scalar, "Naive scalar");
    // m.def("wmma_naive",   &wmma_naive,   "WMMA naive");    // Wk2
    m.def("shared_tile",  &shared_tile,  "Shared tile");
    // m.def("cutlass_gemm", &cutlass_gemm, "CUTLASS");       // Wk4
    // m.def("cute_gemm",    &cute_gemm,    "CuTe");          // Wk5
}