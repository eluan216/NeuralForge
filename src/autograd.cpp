#include "neuralforge/autograd.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace neuralforge {
namespace {

void accumulate_grad(Variable::Impl& impl, const Tensor& delta) {
    if (!impl.requires_grad) return;
    if (impl.grad.size() == 0) {
        impl.grad = Tensor(impl.data.shape(), 0.0f);
    }
    if (impl.grad.shape() != delta.shape()) {
        throw std::invalid_argument("Gradient shape mismatch during accumulation");
    }
    for (std::size_t i = 0; i < delta.size(); ++i) {
        impl.grad.at(i) += delta.at(i);
    }
}

Tensor transpose_2d(const Tensor& t) {
    if (t.ndim() != 2) {
        throw std::invalid_argument("transpose_2d expects rank-2 tensor");
    }
    const auto rows = t.shape()[0];
    const auto cols = t.shape()[1];
    Tensor out({cols, rows});
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            out.at(j, i) = t.at(i, j);
        }
    }
    return out;
}

Tensor elementwise_add(const Tensor& a, const Tensor& b) {
    if (a.shape() == b.shape()) {
        Tensor out(a.shape());
        for (std::size_t i = 0; i < a.size(); ++i) out.at(i) = a.at(i) + b.at(i);
        return out;
    }
    // (M, N) + (N,) row-wise bias broadcast
    if (a.ndim() == 2 && b.ndim() == 1 && a.shape()[1] == b.shape()[0]) {
        Tensor out(a.shape());
        const auto M = a.shape()[0];
        const auto N = a.shape()[1];
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                out.at(i, j) = a.at(i, j) + b.at(j);
            }
        }
        return out;
    }
    if (b.ndim() == 2 && a.ndim() == 1 && b.shape()[1] == a.shape()[0]) {
        return elementwise_add(b, a);
    }
    throw std::invalid_argument("add: unsupported shapes for broadcasting");
}

Tensor reduce_bias_grad(const Tensor& gy, std::size_t bias_len) {
    // gy is (M, N), bias is (N,) → sum over rows
    if (gy.ndim() != 2 || gy.shape()[1] != bias_len) {
        throw std::invalid_argument("reduce_bias_grad shape mismatch");
    }
    Tensor out({bias_len}, 0.0f);
    const auto M = gy.shape()[0];
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < bias_len; ++j) {
            out.at(j) += gy.at(i, j);
        }
    }
    return out;
}

Variable make_from_op(Tensor data,
                      std::vector<std::shared_ptr<Variable::Impl>> inputs,
                      std::function<void(const Tensor&)> backward_fn) {
    const bool needs_grad = std::any_of(
        inputs.begin(), inputs.end(),
        [](const std::shared_ptr<Variable::Impl>& p) {
            return p && p->requires_grad;
        });

    Variable out(std::move(data), needs_grad);
    if (!needs_grad) return out;

    auto node = std::make_shared<Variable::Node>();
    node->inputs = std::move(inputs);
    node->backward = std::move(backward_fn);
    out.impl_->is_leaf = false;
    out.impl_->grad_fn = std::move(node);
    return out;
}

} // namespace

Variable::Variable(Tensor data, bool requires_grad)
    : impl_(std::make_shared<Impl>()) {
    impl_->data = std::move(data);
    impl_->requires_grad = requires_grad;
    impl_->is_leaf = true;
}

const Tensor& Variable::data() const {
    if (!impl_) throw std::runtime_error("Variable is empty");
    return impl_->data;
}
Tensor& Variable::data() {
    if (!impl_) throw std::runtime_error("Variable is empty");
    return impl_->data;
}

const Tensor& Variable::grad() const {
    if (!impl_) throw std::runtime_error("Variable is empty");
    return impl_->grad;
}
Tensor& Variable::grad() {
    if (!impl_) throw std::runtime_error("Variable is empty");
    return impl_->grad;
}

bool Variable::requires_grad() const noexcept {
    return impl_ && impl_->requires_grad;
}
bool Variable::is_leaf() const noexcept {
    return impl_ && impl_->is_leaf;
}

void Variable::zero_grad() {
    if (!impl_ || !impl_->requires_grad) return;
    impl_->grad = Tensor(impl_->data.shape(), 0.0f);
}

void Variable::backward() {
    if (!impl_) throw std::runtime_error("Variable is empty");
    if (impl_->data.size() != 1) {
        throw std::invalid_argument("backward() requires a scalar output");
    }

    // Seed ∂L/∂output = 1
    Tensor seed(impl_->data.shape(), 1.0f);
    accumulate_grad(*impl_, seed);

    // Build reverse-topological order of nodes reachable from this output.
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<const Node*> seen;

    std::function<void(const std::shared_ptr<Node>&)> dfs = [&](const std::shared_ptr<Node>& n) {
        if (!n || seen.count(n.get())) return;
        seen.insert(n.get());
        for (const auto& inp : n->inputs) {
            if (inp && inp->grad_fn) dfs(inp->grad_fn);
        }
        order.push_back(n);
    };
    if (impl_->grad_fn) dfs(impl_->grad_fn);

    // order is post-order (inputs before consumers). Reverse → consumers first.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node& node = **it;
        // Find the output impl that owns this grad_fn — we stored grad on each
        // intermediate Variable's Impl. Walk inputs' consumers is awkward;
        // instead each node closes over the output impl shared_ptr.
        // We re-bind: backward lambdas capture everything they need and only
        // need the output gradient, which we look up by scanning…
        //
        // Simpler approach used below: each backward_fn is created to read
        // grad from a captured output Impl pointer.
        // Those lambdas expect the output grad to already be populated.
        // So we need the output Impl for this node. Store it on the node.
    }

    // Re-run with nodes that capture output impl — see op implementations.
    // The dfs order is correct; each node's backward reads its output's grad
    // via a captured shared_ptr<Impl> named `self`.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        (*it)->backward(Tensor{});  // placeholder unused; real grad read inside lambda
    }
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

Variable matmul(const Variable& a, const Variable& b) {
    if (!a.impl_ || !b.impl_) throw std::invalid_argument("matmul on empty Variable");
    Tensor result = neuralforge::matmul(a.data(), b.data());

    auto a_impl = a.impl_;
    auto b_impl = b.impl_;
    // Output impl created inside make_from_op — capture after creation.
    auto out_holder = std::make_shared<std::shared_ptr<Variable::Impl>>();

    Variable out = make_from_op(
        std::move(result),
        {a_impl, b_impl},
        [a_impl, b_impl, out_holder](const Tensor& /*unused*/) {
            const Tensor& gy = (*out_holder)->grad;
            if (gy.size() == 0) return;
            if (a_impl->requires_grad) {
                // dA = gy @ B^T
                Tensor Bt = transpose_2d(b_impl->data);
                Tensor dA = neuralforge::matmul(gy, Bt);
                accumulate_grad(*a_impl, dA);
            }
            if (b_impl->requires_grad) {
                // dB = A^T @ gy
                Tensor At = transpose_2d(a_impl->data);
                Tensor dB = neuralforge::matmul(At, gy);
                accumulate_grad(*b_impl, dB);
            }
        });
    *out_holder = out.impl_;
    return out;
}

Variable add(const Variable& a, const Variable& b) {
    if (!a.impl_ || !b.impl_) throw std::invalid_argument("add on empty Variable");
    Tensor result = elementwise_add(a.data(), b.data());

    auto a_impl = a.impl_;
    auto b_impl = b.impl_;
    const bool a_is_bias = (a.data().ndim() == 1 && b.data().ndim() == 2);
    const bool b_is_bias = (b.data().ndim() == 1 && a.data().ndim() == 2);
    auto out_holder = std::make_shared<std::shared_ptr<Variable::Impl>>();

    Variable out = make_from_op(
        std::move(result),
        {a_impl, b_impl},
        [a_impl, b_impl, a_is_bias, b_is_bias, out_holder](const Tensor& /*unused*/) {
            const Tensor& gy = (*out_holder)->grad;
            if (gy.size() == 0) return;
            if (a_impl->requires_grad) {
                if (a_is_bias) {
                    accumulate_grad(*a_impl, reduce_bias_grad(gy, a_impl->data.size()));
                } else if (a_impl->data.shape() == gy.shape()) {
                    accumulate_grad(*a_impl, gy);
                } else {
                    throw std::runtime_error("add backward: unsupported shape for a");
                }
            }
            if (b_impl->requires_grad) {
                if (b_is_bias) {
                    accumulate_grad(*b_impl, reduce_bias_grad(gy, b_impl->data.size()));
                } else if (b_impl->data.shape() == gy.shape()) {
                    accumulate_grad(*b_impl, gy);
                } else {
                    throw std::runtime_error("add backward: unsupported shape for b");
                }
            }
        });
    *out_holder = out.impl_;
    return out;
}

Variable relu(const Variable& x) {
    if (!x.impl_) throw std::invalid_argument("relu on empty Variable");
    Tensor result(x.data().shape());
    for (std::size_t i = 0; i < x.data().size(); ++i) {
        result.at(i) = std::max(0.0f, x.data().at(i));
    }
    auto x_impl = x.impl_;
    auto out_holder = std::make_shared<std::shared_ptr<Variable::Impl>>();

    Variable out = make_from_op(
        std::move(result),
        {x_impl},
        [x_impl, out_holder](const Tensor& /*unused*/) {
            const Tensor& gy = (*out_holder)->grad;
            if (gy.size() == 0 || !x_impl->requires_grad) return;
            Tensor dx(x_impl->data.shape(), 0.0f);
            for (std::size_t i = 0; i < x_impl->data.size(); ++i) {
                dx.at(i) = (x_impl->data.at(i) > 0.0f) ? gy.at(i) : 0.0f;
            }
            accumulate_grad(*x_impl, dx);
        });
    *out_holder = out.impl_;
    return out;
}

Variable sum(const Variable& x) {
    if (!x.impl_) throw std::invalid_argument("sum on empty Variable");
    float total = 0.0f;
    for (std::size_t i = 0; i < x.data().size(); ++i) total += x.data().at(i);
    Tensor result({1}, total);

    auto x_impl = x.impl_;
    auto out_holder = std::make_shared<std::shared_ptr<Variable::Impl>>();

    Variable out = make_from_op(
        std::move(result),
        {x_impl},
        [x_impl, out_holder](const Tensor& /*unused*/) {
            const Tensor& gy = (*out_holder)->grad;
            if (gy.size() == 0 || !x_impl->requires_grad) return;
            const float scale = gy.at(0);
            Tensor dx(x_impl->data.shape(), scale);
            accumulate_grad(*x_impl, dx);
        });
    *out_holder = out.impl_;
    return out;
}

// ---------------------------------------------------------------------------
// Finite differences
// ---------------------------------------------------------------------------

Tensor finite_difference_grad(const Variable& leaf,
                              const std::function<Variable(const Variable&)>& scalar_fn,
                              float eps) {
    if (!leaf.impl_) throw std::invalid_argument("finite_difference_grad: empty leaf");
    Tensor grad(leaf.data().shape(), 0.0f);
    Tensor base = leaf.data();

    for (std::size_t i = 0; i < base.size(); ++i) {
        Tensor plus = base;
        Tensor minus = base;
        plus.at(i) += eps;
        minus.at(i) -= eps;

        Variable v_plus(plus, false);
        Variable v_minus(minus, false);
        Variable y_plus = scalar_fn(v_plus);
        Variable y_minus = scalar_fn(v_minus);
        if (y_plus.data().size() != 1 || y_minus.data().size() != 1) {
            throw std::invalid_argument("finite_difference_grad: scalar_fn must return a scalar");
        }
        grad.at(i) = (y_plus.data().at(0) - y_minus.data().at(0)) / (2.0f * eps);
    }
    return grad;
}

} // namespace neuralforge
