#include <torch/extension.h>
#include <cuda_fp16.h>
#include "kernels.h"

#define tile_size 32
#define coarse_factor 8

__global__ void shared_tile_matmul(
    const __half* A, const __half* B, float* C,
    int M, int K, int N){
    
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int colStart = threadIdx.x + blockIdx.x * blockDim.x * coarse_factor;

    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    __shared__ __half shA[tile_size][tile_size];
    __shared__ __half shB[tile_size][tile_size];

    float acc[coarse_factor] = {0.0f};

    int phases = (K + tile_size - 1)/tile_size;

    for(int phase = 0; phase < phases; phase++){
        int colA = tidX + tile_size * phase;
        int rowB = tidY + tile_size * phase;
        
        shA[tidY][tidX] = (row < M && colA < K) ? A[row*K + colA] : __float2half(0.0f);
        __syncthreads();

        for(int c = 0; c < coarse_factor; c++){
            int col = colStart + c * tile_size;

            shB[tidY][tidX] = (rowB < K && col < N) ? B[rowB*N + col] : __float2half(0.0f);
            __syncthreads();

            for(int i = 0; i < tile_size; i++){
                acc[c] += __half2float(shA[tidY][i]) * __half2float(shB[i][tidX]);
            }
            __syncthreads();
        }
    }

    for(int c = 0; c < coarse_factor; c++){
        int col = colStart + c * tile_size;

        if(row < M && col < N){
            C[row*N + col] = acc[c];
        }
    }
}


torch::Tensor shared_tile(torch::Tensor A, torch::Tensor B){
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M,N},
                            torch::dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 block(tile_size, tile_size);
    dim3 grid((N + block.x * coarse_factor - 1)/(block.x * coarse_factor), (M + block.y - 1)/block.y);

    shared_tile_matmul<<<grid,block>>>(
        reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), M, K, N);
    cudaDeviceSynchronize();
    return C;
}