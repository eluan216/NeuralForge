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

    /// Zero the gradient buffer (allocates zeros if needed).
    void zero_grad();

    /// Reverse-mode AD. *this must be a scalar (size == 1).
    /// Accumulates gradients into all upstream leaves that require grad.
    void backward();

    // Internal graph plumbing (ops in autograd.cpp).
    struct Node;
    struct Impl;
    [[nodiscard]] const std::shared_ptr<Impl>& impl() const noexcept { return impl_; }

private:
    std::shared_ptr<Impl> impl_;

    friend Variable matmul(const Variable& a, const Variable& b);
    friend Variable add(const Variable& a, const Variable& b);
    friend Variable relu(const Variable& x);
    friend Variable sum(const Variable& x);
};

struct Variable::Impl {
    Tensor data;
    Tensor grad;
    bool requires_grad{false};
    bool is_leaf{true};
    std::shared_ptr<Node> grad_fn;  // non-null for non-leaves that need backward
};

struct Variable::Node {
    std::vector<std::shared_ptr<Impl>> inputs;
    // Given ∂L/∂output, accumulate into inputs' .grad and schedule their grad_fns.
    std::function<void(const Tensor&)> backward;
};

/// C = A @ B  (2-D only). Gradients: dA = dC @ Bᵀ, dB = Aᵀ @ dC.
[[nodiscard]] Variable matmul(const Variable& a, const Variable& b);

/// Element-wise add. Supports (M×N)+(M×N) and (M×N)+(N) bias broadcast.
[[nodiscard]] Variable add(const Variable& a, const Variable& b);

/// ReLU: max(0, x). Gradient is the unit step (0 at exactly 0).
[[nodiscard]] Variable relu(const Variable& x);

/// Sum of all elements → scalar. Used as a simple reduction for tests / loss.
[[nodiscard]] Variable sum(const Variable& x);

/// Finite-difference gradient for a scalar function f: R^{n} → R.
/// `f` receives a flat parameter vector of length n and returns a scalar Variable
/// (or rather: we take leaf x, run the user-built graph, call sum if needed).
/// Returns a Tensor with the same shape as `x.data()`.
[[nodiscard]] Tensor finite_difference_grad(
    const Variable& leaf,
    const std::function<Variable(const Variable&)>& scalar_fn,
    float eps = 1e-3f);

} // namespace neuralforge
