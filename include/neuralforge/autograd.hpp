#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "neuralforge/tensor.hpp"

namespace neuralforge {

/// Differentiable tensor wrapper with reverse-mode automatic differentiation.
///
/// Leaf variables are created with requires_grad=true. Differentiable ops
/// (matmul, add, relu, sum, …) build a dynamic computation graph. Calling
/// backward() on a scalar output accumulates ∂L/∂leaf into each leaf's grad().
class Variable {
public:
    Variable() = default;

    /// Own a tensor; set requires_grad to record ops for reverse-mode AD.
    explicit Variable(Tensor data, bool requires_grad = false);

    [[nodiscard]] const Tensor& data() const;
    [[nodiscard]] Tensor& data();

    /// Gradient buffer (same shape as data). Empty until backward runs.
    [[nodiscard]] const Tensor& grad() const;
    [[nodiscard]] Tensor& grad();

    [[nodiscard]] bool requires_grad() const noexcept;
    [[nodiscard]] bool is_leaf() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return !impl_; }

    /// Zero the gradient buffer (allocates zeros if needed).
    void zero_grad();

    /// Reverse-mode AD. *this must be a scalar (size == 1).
    /// Accumulates gradients into all upstream leaves that require grad.
    void backward();

    struct Node;
    struct Impl;

    [[nodiscard]] const std::shared_ptr<Impl>& impl() const noexcept { return impl_; }
    [[nodiscard]] std::shared_ptr<Impl>& impl() noexcept { return impl_; }

private:
    std::shared_ptr<Impl> impl_;
};

struct Variable::Impl {
    Tensor data;
    Tensor grad;
    bool requires_grad{false};
    bool is_leaf{true};
    std::shared_ptr<Node> grad_fn;
};

struct Variable::Node {
    std::vector<std::shared_ptr<Impl>> inputs;
    std::function<void(const Tensor&)> backward;
};

[[nodiscard]] Variable matmul(const Variable& a, const Variable& b);
[[nodiscard]] Variable add(const Variable& a, const Variable& b);
[[nodiscard]] Variable relu(const Variable& x);
[[nodiscard]] Variable sum(const Variable& x);

[[nodiscard]] Tensor finite_difference_grad(
    const Variable& leaf,
    const std::function<Variable(const Variable&)>& scalar_fn,
    float eps = 1e-3f);

} // namespace neuralforge
