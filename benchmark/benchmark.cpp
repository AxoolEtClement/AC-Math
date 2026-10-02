#include <nanobench.h>
#include "mathLibCPP/Vec4.h"
#include "mathLibSIMD/include/Vec4.hpp"

int main() {
    // 1. Initialisation pour mathLibCPP (Standard)
    Maths::Vec4<float> v1_cpp{ 1.0f, 2.0f, 3.0f, 4.0f };
    Maths::Vec4<float> v2_cpp{ 4.0f, 5.0f, 6.0f, 7.0f };

    // 2. Initialisation pour mathLibSIMD (SSE4.1)
    simd::Vec4 v1_simd{ 1.0f, 2.0f, 3.0f, 4.0f };
    simd::Vec4 v2_simd{ 4.0f, 5.0f, 6.0f, 7.0f };

    float scalar = 2.5f;

    // --- WARM UP ---
    {
        ankerl::nanobench::doNotOptimizeAway(v1_cpp + v2_cpp);
        ankerl::nanobench::doNotOptimizeAway(v1_simd + v2_simd);
        ankerl::nanobench::doNotOptimizeAway(v1_cpp * scalar);
        ankerl::nanobench::doNotOptimizeAway(v1_simd * scalar);
    }

    // --- BENCHMARK 1 : Addition de vecteurs (+) ---
    {
        ankerl::nanobench::Bench bench;
        bench.epochs(50)
             .minEpochIterations(1'100'000) // Augmenté pour stabiliser
             .relative(true);

        bench.run("+ vec4 (Standard C++)", [&] {
            ankerl::nanobench::doNotOptimizeAway(v1_cpp + v2_cpp);
        });

        bench.run("+ vec4 (SIMD)", [&] {
            ankerl::nanobench::doNotOptimizeAway(v1_simd + v2_simd);
        });
    }

    // --- BENCHMARK 2 : Multiplication par un scalaire (* float) ---
    {
        ankerl::nanobench::Bench bench;
        bench.epochs(50)
             .minEpochIterations(1'100'000) // Augmenté pour stabiliser
             .relative(true);

        bench.run("* scalar (Standard C++)", [&] {
            ankerl::nanobench::doNotOptimizeAway(v1_cpp * scalar);
        });

        bench.run("* scalar (SIMD)", [&] {
            ankerl::nanobench::doNotOptimizeAway(v1_simd * scalar);
        });
    }

    return 0;
}