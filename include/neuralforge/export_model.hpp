#pragma once

#include <filesystem>

#include "neuralforge/tensor.hpp"

namespace neuralforge {

/// Write a NEURALFORGE_MLP 1 text model readable by MLPModel::load.
/// Weight layouts match the training/export convention:
///   W1: (input, hidden), b1: (hidden), W2: (hidden, output), b2: (output).
void export_mlp_model(const std::filesystem::path& path,
                      const Tensor& w1, const Tensor& b1,
                      const Tensor& w2, const Tensor& b2);

} // namespace neuralforge
