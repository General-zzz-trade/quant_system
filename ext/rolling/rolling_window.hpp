#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

class RollingWindow {
public:
    explicit RollingWindow(int size) : size_(size), head_(0), count_(0), sum_(0.0), sumsq_(0.0) {
        if (size <= 0)
            throw std::invalid_argument("size must be positive");
        buf_.resize(size, 0.0);
    }

    void push(double x) noexcept {
        if (count_ < size_) {
            buf_[count_] = x;
            ++count_;
        } else {
            double old = buf_[head_];
            sum_ -= old;
            sumsq_ -= old * old;
            buf_[head_] = x;
            head_ = (head_ + 1) % size_;
        }
        sum_ += x;
        sumsq_ += x * x;
    }

    bool full() const noexcept { return count_ == size_; }
    int n() const noexcept { return count_; }
    int get_size() const noexcept { return size_; }

    std::optional<double> mean() const noexcept {
        if (count_ == 0) return std::nullopt;
        return sum_ / count_;
    }

    std::optional<double> variance() const noexcept {
        if (count_ == 0) return std::nullopt;
        double mu = sum_ / count_;
        double v = sumsq_ / count_ - mu * mu;
        return std::max(v, 0.0);
    }

    std::optional<double> std_dev() const noexcept {
        auto v = variance();
        if (!v) return std::nullopt;
        return std::sqrt(*v);
    }

private:
    int size_;
    int head_;
    int count_;
    double sum_;
    double sumsq_;
    std::vector<double> buf_;
};
