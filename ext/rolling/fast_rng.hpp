#pragma once

#include <cstdint>

// xorshift64* PRNG — fast, good quality, no external deps.
struct FastRNG {
    uint64_t state;
    explicit FastRNG(uint64_t seed) : state(seed ^ 0x9E3779B97F4A7C15ULL) {}

    uint64_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    int randint(int n) {
        return static_cast<int>(next() % static_cast<uint64_t>(n));
    }

    // Gaussian via Box-Muller (for parametric MC)
    double gauss(double mu, double sigma) {
        // Generate two uniform [0,1) and apply Box-Muller
        double u1 = (static_cast<double>(next()) + 1.0) / 18446744073709551617.0;
        double u2 = (static_cast<double>(next()) + 1.0) / 18446744073709551617.0;
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
        return mu + sigma * z;
    }
};
