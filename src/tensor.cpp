#include "neuralforge/tensor.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace neuralforge {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Tensor::Tensor(std::vector<size_type> shape)
    : shape_(std::move(shape)) {
    allocate();
}

Tensor::Tensor(std::vector<size_type> shape, value_type fill_value)
    : shape_(std::move(shape)) {
    allocate();
    std::fill(data_->begin(), data_->end(), fill_value);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<value_type>> rows) {
    if (rows.size() == 0) {
        return;
    }
    const size_type n_rows = static_cast<size_type>(rows.size());
    const size_type n_cols = static_cast<size_type>(rows.begin()->size());

    for (const auto& row : rows) {
        if (row.size() != n_cols) {
            throw std::invalid_argument("All rows must have the same length");
        }
    }

    shape_ = {n_rows, n_cols};
    allocate();

    size_type idx = 0;
    for (const auto& row : rows) {
        for (value_type v : row) {
            (*data_)[idx++] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

Tensor::size_type Tensor::compute_size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), size_type{1}, std::multiplies<>());
}

void Tensor::allocate() {
    const size_type n = compute_size();
    data_ = std::make_shared<std::vector<value_type>>(n);
}

void Tensor::check_rank(size_type expected) const {
    if (ndim() != expected) {
        throw std::invalid_argument(
            "Expected rank-" + std::to_string(expected) +
            " tensor, got rank-" + std::to_string(ndim()));
    }
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

Tensor::value_type& Tensor::at(size_type index) {
    if (index >= size()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return (*data_)[index];
}

const Tensor::value_type& Tensor::at(size_type index) const {
    if (index >= size()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return (*data_)[index];
}

Tensor::value_type& Tensor::at(size_type row, size_type col) {
    check_rank(2);
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("2-D tensor index out of range");
    }
    return (*data_)[row * shape_[1] + col];
}

const Tensor::value_type& Tensor::at(size_type row, size_type col) const {
    check_rank(2);
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("2-D tensor index out of range");
    }
    return (*data_)[row * shape_[1] + col];
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

Tensor Tensor::zeros(std::vector<size_type> shape) {
    return Tensor(std::move(shape), 0.0f);
}

Tensor Tensor::ones(std::vector<size_type> shape) {
    return Tensor(std::move(shape), 1.0f);
}

Tensor Tensor::full(std::vector<size_type> shape, value_type value) {
    return Tensor(std::move(shape), value);
}

// ---------------------------------------------------------------------------
// Debugging
// ---------------------------------------------------------------------------

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_type i = 0; i < shape_.size(); ++i) {
        if (i) oss << ", ";
        oss << shape_[i];
    }
    oss << "], data=[";

    const size_type n = size();
    const size_type preview = std::min(n, size_type{8});
    for (size_type i = 0; i < preview; ++i) {
        if (i) oss << ", ";
        oss << (*data_)[i];
    }
    if (n > preview) oss << ", ...";
    oss << "])";
    return oss.str();
}

} // namespace neuralforge
