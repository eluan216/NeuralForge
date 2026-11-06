#pragma once
#include <cstddef>
namespace neuralforge {
using AxpyKernel = void (*)(float*, const float*, float, std::size_t);
/// Selected once, without -march=native. Scalar fallback is always available.
AxpyKernel selected_axpy() noexcept;
const char* inference_backend() noexcept;
}
