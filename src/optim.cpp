#include "neuralforge/optim.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace neuralforge {

SGD::SGD(float learning_rate) : lr_(learning_rate) {
    if (!(learning_rate > 0.0f) || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("SGD: learning_rate must be finite and > 0");
    }
}

void SGD::step(std::vector<Variable>& parameters) {
    for (Variable& p : parameters) {
        if (p.empty() || !p.requires_grad()) continue;
        if (p.grad().size() == 0) {
            throw std::runtime_error("SGD::step: parameter has empty gradient; call backward() first");
        }
        if (p.grad().shape() != p.data().shape()) {
            throw std::runtime_error("SGD::step: grad shape does not match parameter");
        }
        for (std::size_t i = 0; i < p.data().size(); ++i) {
            const float g = p.grad().at(i);
            if (!std::isfinite(g)) {
                throw std::runtime_error("SGD::step: non-finite gradient");
            }
            const float updated = p.data().at(i) - lr_ * g;
            if (!std::isfinite(updated)) {
                throw std::runtime_error("SGD::step: non-finite parameter update");
            }
            p.data().at(i) = updated;
        }
    }
}

void SGD::zero_grad(std::vector<Variable>& parameters) {
    for (Variable& p : parameters) {
        if (!p.empty()) p.zero_grad();
    }
}

Adam::Adam(float learning_rate, float beta1, float beta2, float eps)
    : lr_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {
    if (!(lr_ > 0.0f) || !std::isfinite(lr_)) {
        throw std::invalid_argument("Adam: learning_rate must be finite and > 0");
    }
    if (!(beta1_ >= 0.0f && beta1_ < 1.0f) || !std::isfinite(beta1_)) {
        throw std::invalid_argument("Adam: beta1 must be in [0, 1)");
    }
    if (!(beta2_ >= 0.0f && beta2_ < 1.0f) || !std::isfinite(beta2_)) {
        throw std::invalid_argument("Adam: beta2 must be in [0, 1)");
    }
    if (!(eps_ > 0.0f) || !std::isfinite(eps_)) {
        throw std::invalid_argument("Adam: eps must be finite and > 0");
    }
}

void Adam::step(std::vector<Variable>& parameters) {
    // Validate all gradients first so a failure never partially updates params.
    struct Pending {
        Variable* p;
        const void* key;
    };
    std::vector<Pending> pending;
    pending.reserve(parameters.size());

    for (Variable& p : parameters) {
        if (p.empty() || !p.requires_grad()) continue;
        if (p.grad().size() == 0) {
            throw std::runtime_error("Adam::step: parameter has empty gradient; call backward() first");
        }
        if (p.grad().shape() != p.data().shape()) {
            throw std::runtime_error("Adam::step: grad shape does not match parameter");
        }
        for (std::size_t i = 0; i < p.grad().size(); ++i) {
            if (!std::isfinite(p.grad().at(i))) {
                throw std::runtime_error("Adam::step: non-finite gradient");
            }
        }
        pending.push_back({&p, static_cast<const void*>(p.impl().get())});
    }

    ++t_;
    const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (const Pending& item : pending) {
        Variable& p = *item.p;
        auto& st = state_[item.key];
        if (st.m.size() == 0) {
            st.m = Tensor(p.data().shape(), 0.0f);
            st.v = Tensor(p.data().shape(), 0.0f);
        }
        if (st.m.shape() != p.data().shape()) {
            throw std::runtime_error("Adam::step: moment state shape mismatch (parameter replaced?)");
        }

        for (std::size_t i = 0; i < p.data().size(); ++i) {
            const float g = p.grad().at(i);
            st.m.at(i) = beta1_ * st.m.at(i) + (1.0f - beta1_) * g;
            st.v.at(i) = beta2_ * st.v.at(i) + (1.0f - beta2_) * (g * g);
            const float m_hat = st.m.at(i) / bc1;
            const float v_hat = st.v.at(i) / bc2;
            const float updated = p.data().at(i) - lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            if (!std::isfinite(updated)) {
                throw std::runtime_error("Adam::step: non-finite parameter update");
            }
            p.data().at(i) = updated;
        }
    }
}

void Adam::zero_grad(std::vector<Variable>& parameters) {
    for (Variable& p : parameters) {
        if (!p.empty()) p.zero_grad();
    }
}

} // namespace neuralforge
