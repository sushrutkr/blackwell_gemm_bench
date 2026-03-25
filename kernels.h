#pragma once
#include <torch/extension.h>

torch::Tensor naive_scalar (torch::Tensor A, torch::Tensor B);
torch::Tensor wmma_naive   (torch::Tensor A, torch::Tensor B);
torch::Tensor shared_tile  (torch::Tensor A, torch::Tensor B);
torch::Tensor cutlass_gemm (torch::Tensor A, torch::Tensor B);
torch::Tensor cute_gemm    (torch::Tensor A, torch::Tensor B);