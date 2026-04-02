#pragma once
#include <torch/extension.h>

torch::Tensor naive_scalar         (torch::Tensor A, torch::Tensor B);
torch::Tensor shared_tile          (torch::Tensor A, torch::Tensor B);
torch::Tensor cute_gemm            (torch::Tensor A, torch::Tensor B);
// torch::Tensor warp_specialized_gemm(torch::Tensor A, torch::Tensor B);
// torch::Tensor tiled_epilogue_gemm  (torch::Tensor A, torch::Tensor B);
// torch::Tensor collective_mma_gemm  (torch::Tensor A, torch::Tensor B);
// torch::Tensor persistent_gemm      (torch::Tensor A, torch::Tensor B);
// torch::Tensor dedicated_epilogue_gemm(torch::Tensor A, torch::Tensor B);
// torch::Tensor grid_tiling_gemm     (torch::Tensor A, torch::Tensor B);