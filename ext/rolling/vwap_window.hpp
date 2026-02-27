#pragma once

#include <optional>
#include <stdexcept>
#include <vector>

/**
 * O(1) rolling VWAP window.
 *
 * Maintains running sum(price*volume) and sum(volume) with circular buffer
 * eviction — same pattern as RollingWindow but tracking two sums.
 */
class VWAPWindow {
public:
    explicit VWAPWindow(int size)
        : size_(size), head_(0), count_(0), sum_pv_(0.0), sum_v_(0.0) {
        if (size <= 0)
            throw std::invalid_argument("size must be positive");
        prices_.resize(size, 0.0);
        volumes_.resize(size, 0.0);
    }

    void push(double price, double volume) noexcept {
        if (count_ < size_) {
            prices_[count_] = price;
            volumes_[count_] = volume;
            ++count_;
        } else {
            double old_p = prices_[head_];
            double old_v = volumes_[head_];
            sum_pv_ -= old_p * old_v;
            sum_v_ -= old_v;
            prices_[head_] = price;
            volumes_[head_] = volume;
            head_ = (head_ + 1) % size_;
        }
        sum_pv_ += price * volume;
        sum_v_ += volume;
    }

    bool full() const noexcept { return count_ == size_; }
    int n() const noexcept { return count_; }
    int get_size() const noexcept { return size_; }

    std::optional<double> vwap() const noexcept {
        if (count_ == 0 || sum_v_ <= 0.0) return std::nullopt;
        return sum_pv_ / sum_v_;
    }

    double sum_pv() const noexcept { return sum_pv_; }
    double sum_v() const noexcept { return sum_v_; }

private:
    int size_;
    int head_;
    int count_;
    double sum_pv_;
    double sum_v_;
    std::vector<double> prices_;
    std::vector<double> volumes_;
};
