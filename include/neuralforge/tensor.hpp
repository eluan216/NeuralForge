#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuralforge {

/**
 * @brief Multi-dimensional array (tensor) with contiguous row-major storage.
 *
 * Design goals for Weeks 1–2:
 *  - RAII ownership of data
 *  - Clear shape / strides
 *  - Safe element access
 *  - Foundation for matrix multiplication and later graph execution
 */
class Tensor {
public:
    using value_type = float;
    using size_type  = std::size_t;

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Empty tensor
    Tensor() = default;

    /// Create a tensor with given shape, uninitialized (caller must fill)
    explicit Tensor(std::vector<size_type> shape);

    /// Create a tensor with given shape, filled with a constant value
    Tensor(std::vector<size_type> shape, value_type fill_value);

    /// Create from nested initializer list (2-D convenience)
    Tensor(std::initializer_list<std::initializer_list<value_type>> rows);

    // Copy / move
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // ------------------------------------------------------------------
    // Shape & size
    // ------------------------------------------------------------------

    [[nodiscard]] const std::vector<size_type>& shape() const noexcept { return shape_; }
    [[nodiscard]] size_type ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] size_type size() const noexcept { return data_ ? data_->size() : 0; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    // ------------------------------------------------------------------
    // Data access
    // ------------------------------------------------------------------

    [[nodiscard]] value_type* data() noexcept { return data_ ? data_->data() : nullptr; }
    [[nodiscard]] const value_type* data() const noexcept { return data_ ? data_->data() : nullptr; }

    /// Element access with bounds checking (throws on out-of-range)
    [[nodiscard]] value_type& at(size_type index);
    [[nodiscard]] const value_type& at(size_type index) const;

    /// 2-D convenience accessors (throws if not rank-2)
    [[nodiscard]] value_type& at(size_type row, size_type col);
    [[nodiscard]] const value_type& at(size_type row, size_type col) const;

    // ------------------------------------------------------------------
    // Factory helpers
    // ------------------------------------------------------------------

    static Tensor zeros(std::vector<size_type> shape);
    static Tensor ones(std::vector<size_type> shape);
    static Tensor full(std::vector<size_type> shape, value_type value);

    // ------------------------------------------------------------------
    // Debugging
    // ------------------------------------------------------------------

    [[nodiscard]] std::string to_string() const;

private:
    std::vector<size_type> shape_;
    std::shared_ptr<std::vector<value_type>> data_;

    void allocate();
    [[nodiscard]] size_type compute_size() const;
    void check_rank(size_type expected) const;
};

// Free functions -------------------------------------------------------------

/**
 * @brief Matrix multiplication: C = A @ B
 *
 * A must be (M, K), B must be (K, N). Result is (M, N).
 * Uses a straightforward triple-loop implementation (baseline for later optimization).
 */
Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace neuralforge
