#include <torch/extension.h>
#include <ATen/autocast_mode.h>
at::Tensor cached_bf16(const at::Tensor& x) { return at::autocast::cached_cast(at::kBFloat16, x, at::kCUDA); }
