#include "neuralforge/loss.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

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

Variable cross_entropy_loss(const Variable& logits,
                            const std::vector<std::size_t>& targets) {
    if (logits.empty()) {
        throw std::invalid_argument("cross_entropy_loss: empty logits");
    }
    if (logits.data().ndim() != 2) {
        throw std::invalid_argument("cross_entropy_loss: logits must be rank-2 (batch, classes)");
    }
    const auto batch = logits.data().shape()[0];
    const auto classes = logits.data().shape()[1];
    if (classes == 0) {
        throw std::invalid_argument("cross_entropy_loss: need at least one class");
    }
    if (targets.size() != batch) {
        throw std::invalid_argument("cross_entropy_loss: targets size must equal batch");
    }
    if (batch == 0) {
        throw std::invalid_argument("cross_entropy_loss: empty batch");
    }

    for (std::size_t i = 0; i < batch; ++i) {
        if (targets[i] >= classes) {
            throw std::invalid_argument("cross_entropy_loss: target class out of range");
        }
        for (std::size_t j = 0; j < classes; ++j) {
            if (!std::isfinite(logits.data().at(i, j))) {
                throw std::invalid_argument("cross_entropy_loss: non-finite logits");
            }
        }
    }

    // Forward: mean_i [ logsumexp(logits_i) - logits_i[y_i] ]
    double total = 0.0;
    for (std::size_t i = 0; i < batch; ++i) {
        float max_v = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < classes; ++j) {
            max_v = std::max(max_v, logits.data().at(i, j));
        }
        double sum_exp = 0.0;
        for (std::size_t j = 0; j < classes; ++j) {
            sum_exp += std::exp(static_cast<double>(logits.data().at(i, j) - max_v));
        }
        const double logsumexp = static_cast<double>(max_v) + std::log(sum_exp);
        total += logsumexp - static_cast<double>(logits.data().at(i, targets[i]));
    }
    const float mean = static_cast<float>(total / static_cast<double>(batch));
    Tensor out({1}, mean);

    auto logits_impl = logits.impl();
    auto out_holder = std::make_shared<std::shared_ptr<Variable::Impl>>();
    const bool needs_grad = logits_impl && logits_impl->requires_grad;
    Variable result(std::move(out), needs_grad);
    if (!needs_grad) return result;

    std::vector<std::size_t> targets_copy = targets;
    auto node = std::make_shared<Variable::Node>();
    node->inputs = {logits_impl};
    node->backward = [logits_impl, out_holder, targets_copy, batch, classes](const Tensor&) {
        const Tensor& gy = (*out_holder)->grad;
        if (gy.size() == 0 || !logits_impl->requires_grad) return;

        // ∂L/∂logits = (1/B) * gy * (softmax(logits) - one_hot(y))
        const float scale = gy.at(0) / static_cast<float>(batch);
        Tensor dx(logits_impl->data.shape(), 0.0f);
        for (std::size_t i = 0; i < batch; ++i) {
            float max_v = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < classes; ++j) {
                max_v = std::max(max_v, logits_impl->data.at(i, j));
            }
            double sum_exp = 0.0;
            for (std::size_t j = 0; j < classes; ++j) {
                sum_exp += std::exp(static_cast<double>(logits_impl->data.at(i, j) - max_v));
            }
            for (std::size_t j = 0; j < classes; ++j) {
                const float soft = static_cast<float>(
                    std::exp(static_cast<double>(logits_impl->data.at(i, j) - max_v)) / sum_exp);
                const float oh = (j == targets_copy[i]) ? 1.0f : 0.0f;
                dx.at(i, j) = scale * (soft - oh);
            }
        }
        accumulate_grad(*logits_impl, dx);
    };
    result.impl()->is_leaf = false;
    result.impl()->grad_fn = std::move(node);
    *out_holder = result.impl();
    return result;
}

} // namespace neuralforge
