#include "neuralforge/optim.hpp"

#include <cmath>
#include <stdexcept>

namespace neuralforge {

SGD::SGD(float learning_rate) : lr_(learning_rate) {
    if (!(learning_rate > 0.0f) || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("SGD: learning_rate must be finite and > 0");
    }
}

void SGD::step(const std::vector<Variable>& parameters) {
    for (const Variable& p : parameters) {
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
            // Leaf parameters own their storage; mutate in place for the next forward.
            const_cast<Tensor&>(p.data()).at(i) = updated;
        }
    }
}

void SGD::zero_grad(const std::vector<Variable>& parameters) {
    for (const Variable& p : parameters) {
        if (!p.empty()) {
            const_cast<Variable&>(p).zero_grad();
        }
    }
}

} // namespace neuralforge
