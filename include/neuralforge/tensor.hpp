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

Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace neuralforge
