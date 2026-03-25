#include <torch/extension.h>
#include <cuda_fp16.h>
#include "kernels.h"

__global__ void naive_scalar_kernel(
        const __half* A, const __half* B, float* C,
        int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += __half2float(A[row * K + k]) *
               __half2float(B[k   * N + col]);
    C[row * N + col] = acc;
}

torch::Tensor naive_scalar(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N},
                 torch::dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_scalar_kernel<<<grid, block>>>(
        reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), M, K, N);
    cudaDeviceSynchronize();
    return C;
}