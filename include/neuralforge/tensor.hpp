#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuralforge {

class Tensor {
public:
    using value_type = float;
    using size_type = std::size_t;

    Tensor() = default;
    explicit Tensor(std::vector<size_type> shape);
    Tensor(std::vector<size_type> shape, value_type fill_value);
    Tensor(std::initializer_list<std::initializer_list<value_type>> rows);

    // Value semantics: copying a Tensor copies its values; moving transfers them.
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    [[nodiscard]] const std::vector<size_type>& shape() const noexcept { return shape_; }
    [[nodiscard]] size_type ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] size_type size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] value_type* data() noexcept { return data_.data(); }
    [[nodiscard]] const value_type* data() const noexcept { return data_.data(); }

    [[nodiscard]] value_type& at(size_type index);
    [[nodiscard]] const value_type& at(size_type index) const;
    [[nodiscard]] value_type& at(size_type row, size_type col);
    [[nodiscard]] const value_type& at(size_type row, size_type col) const;

    static Tensor zeros(std::vector<size_type> shape);
    static Tensor ones(std::vector<size_type> shape);
    static Tensor full(std::vector<size_type> shape, value_type value);

    [[nodiscard]] std::string to_string() const;

private:
    std::vector<size_type> shape_;
    std::vector<value_type> data_;

    void allocate();
    [[nodiscard]] size_type compute_size() const;
    void check_rank(size_type expected) const;
};

/// Outer tile traversal for cache-blocked matmul.
enum class MatMulTileOrder {
    Ikj,  ///< i0 → k0 → j0 (stream B panels; C tiles revisited)
    Ijk,  ///< i0 → j0 → k0 (keep C tile resident while reducing over k)
};

/// Triple-loop baseline (i-k-j). Kept for correctness checks and A/B benchmarks.
[[nodiscard]] Tensor matmul_naive(const Tensor& a, const Tensor& b);

/// Cache-blocked matmul (default production path via matmul).
/// block_size tiles the (M,K,N) loops; order selects outer tile traversal.
[[nodiscard]] Tensor matmul_blocked(const Tensor& a, const Tensor& b,
                                    std::size_t block_size = 64,
                                    MatMulTileOrder order = MatMulTileOrder::Ijk);

/// Default matmul: currently delegates to matmul_blocked.
[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace neuralforge
