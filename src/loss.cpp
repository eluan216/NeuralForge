#include "neuralforge/loss.hpp"

#include <cmath>
#include <stdexcept>

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

} // namespace

Variable mse_loss(const Variable& prediction, const Variable& target) {
    if (prediction.empty() || target.empty()) {
        throw std::invalid_argument("mse_loss: empty Variable");
    }
    if (prediction.data().shape() != target.data().shape()) {
        throw std::invalid_argument("mse_loss: prediction and target shapes must match");
    }
    if (prediction.data().size() == 0) {
        throw std::invalid_argument("mse_loss: empty tensors");
    }

    const auto n = prediction.data().size();
    float sum_sq = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float p = prediction.data().at(i);
        const float t = target.data().at(i);
        if (!std::isfinite(p) || !std::isfinite(t)) {
            throw std::invalid_argument("mse_loss: non-finite values");
        }
        const float d = p - t;
        sum_sq += d * d;
    }
    const float mean = sum_sq / static_cast<float>(n);
    Tensor out({1}, mean);

    auto pred_impl = prediction.impl();
    auto out_holder = std::make_shared<std::shared_ptr<Variable::Impl>>();

    const bool needs_grad = pred_impl && pred_impl->requires_grad;
    Variable result(std::move(out), needs_grad);
    if (!needs_grad) return result;

    // Capture target values (constant) and n for backward.
    Tensor target_copy = target.data();
    auto node = std::make_shared<Variable::Node>();
    node->inputs = {pred_impl};
    node->backward = [pred_impl, out_holder, target_copy, n](const Tensor&) {
        const Tensor& gy = (*out_holder)->grad;
        if (gy.size() == 0 || !pred_impl->requires_grad) return;
        const float scale = gy.at(0) * (2.0f / static_cast<float>(n));
        Tensor dx(pred_impl->data.shape());
        for (std::size_t i = 0; i < n; ++i) {
            dx.at(i) = scale * (pred_impl->data.at(i) - target_copy.at(i));
        }
        accumulate_grad(*pred_impl, dx);
    };
    result.impl()->is_leaf = false;
    result.impl()->grad_fn = std::move(node);
    *out_holder = result.impl();
    return result;
}

} // namespace neuralforge
