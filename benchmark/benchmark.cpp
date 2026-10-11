// =====================================================================================
//  BenchmarkMath.cpp : High-precision benchmarking using nanobench
// =====================================================================================

#include <nanobench.h>

#include "mathLibCPP/Vec3.h"
#include "mathLibCPP/Matrix4x4.h"
#include "mathLibSIMD/include/Vec3f.hpp"
#include "mathLibSIMD/include/Mat4x4.hpp"
#include "mathLibSIMD/include/Vec4SOA.hpp"
#include "mathLibSIMD/include/LayoutConversions.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <cstdlib>

// External ASM function for Dot Product
extern "C" float Dot_Asm_x64(const void* lhs, const void* rhs);

namespace
{
    // --- Configuration ---
    enum class BenchSize { Small = 128, Medium = 4096, Large = 131072 };

    ankerl::nanobench::Bench MakeBench()
    {
        return ankerl::nanobench::Bench{}
            .epochs(25)
            .warmup(1000)
            .minEpochIterations(5000)
            .minEpochTime(std::chrono::milliseconds(100));
    }

    struct SystemInfo {
        const char* cpu = "x86_64 (SIMD SSE2)";
        const char* compiler = "MSVC";
        const char* options = "/O2 /arch:SSE2 /fp:precise";
    };

    bool Matches(const char* selected, const char* value)
    {
        return std::string(selected) == "all" || std::string(selected) == value;
    }

    bool IsSize(const char* value)
    {
        return Matches(value, "small") || Matches(value, "medium") ||
            Matches(value, "large");
    }

    // --- Data Generation ---
    struct BenchData {
        std::vector<simd::Vec3f> vecA;
        std::vector<simd::Vec3f> vecB;
        std::vector<float> dotResults;
        std::vector<simd::Vec3f> normResults;
        std::vector<simd::Vec3f> transResults;
        simd::Matrix4x4 transformMat;

        explicit BenchData(std::size_t count, uint32_t seed = 42) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

            vecA.reserve(count);
            vecB.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                vecA.emplace_back(dist(rng), dist(rng), dist(rng));
                vecB.emplace_back(dist(rng), dist(rng), dist(rng));
            }

            dotResults.resize(count);
            normResults.resize(count);
            transResults.resize(count);
            transformMat = simd::Matrix4x4::Translation(simd::Vec3f{10.0f, -5.0f, 2.0f});
        }
    };

    // --- ASM Wrapper ---
    void AsmBatchDot(const simd::Vec3f* a, const simd::Vec3f* b, float* res, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            res[i] = Dot_Asm_x64(&a[i], &b[i]);
        }
    }
}

int main(int argc, char** argv) {
    const char* selectedOperation = argc > 1 ? argv[1] : "all";
    const char* selectedVersion = "all";
    const char* selectedSize = "all";

    // Keep the previous [operation] [size] syntax valid while accepting the
    // required [operation] [version] [size] form.


    if (argc > 3) {
        selectedVersion = argv[2];
        selectedSize = argv[3];
    } else if (argc > 2) {
        if (IsSize(argv[2])) {
            selectedSize = argv[2];
        } else {
            selectedVersion = argv[2];
        }
    }

    if (!Matches(selectedOperation, "dot") &&
        !Matches(selectedOperation, "normalize") &&
        !Matches(selectedOperation, "transform")) {
        std::cerr << "Usage: benchmark [all|dot|normalize|transform] "
                     "[all|reference|simd|asm] [all|small|medium|large]\n";
        return EXIT_FAILURE;
    }
    if (!Matches(selectedVersion, "reference") &&
        !Matches(selectedVersion, "simd") &&
        !Matches(selectedVersion, "asm")) {
        std::cerr << "Usage: benchmark [all|dot|normalize|transform] "
                     "[all|reference|simd|asm] [all|small|medium|large]\n";
        return EXIT_FAILURE;
    }
    if (!IsSize(selectedSize)) {
        std::cerr << "Usage: benchmark [all|dot|normalize|transform] "
                     "[all|reference|simd|asm] [all|small|medium|large]\n";
        return EXIT_FAILURE;
    }

    SystemInfo sys;
    std::cout << "================================================================\n";
    std::cout << " MATH LIBRARY BENCHMARK (nanobench)\n";
    std::cout << "================================================================\n";
    std::cout << "CPU:      " << sys.cpu << "\n";
    std::cout << "Compiler:  " << sys.compiler << "\n";
    std::cout << "Options:   " << sys.options << "\n";
    std::cout << "================================================================\n\n";

    const std::vector<std::pair<std::string, std::size_t>> sizes = {
        {"Small",  (std::size_t)BenchSize::Small},
        {"Medium", (std::size_t)BenchSize::Medium},
        {"Large",  (std::size_t)BenchSize::Large}
    };

    for (const auto& [sizeName, count] : sizes) {
        const std::string sizeKey = sizeName == "Small" ? "small"
            : sizeName == "Medium" ? "medium" : "large";
        if (!Matches(selectedSize, sizeKey.c_str()))
            continue;

        std::cout << "--- Lot Size: " << sizeName << " (" << count << ") ---\n";
        BenchData data(count);

        std::vector<simd::Vec4> aos(count);
        std::vector<simd::Vec4SOA> soa((count + 3) / 4);
        for (std::size_t i = 0; i < count; ++i)
            aos[i] = simd::Vec4{data.vecA[i].getX(), data.vecA[i].getY(),
                                data.vecA[i].getZ(), 1.0f};
        std::vector<simd::Vec4> aosRoundTrip(count);

        std::cout << "\n[AoS/SoA conversion cost]\n";
        MakeBench().run("ToSOA", [&] {
            simd::ToSOA(aos.data(), soa.data(), count);
            return soa[0].getX();
        });
        MakeBench().run("ToAOS", [&] {
            simd::ToAOS(soa.data(), aosRoundTrip.data(), count);
            return aosRoundTrip[0].getX();
        });

        // 1. DOT PRODUCT
        if (Matches(selectedOperation, "dot")) {
            std::cout << "\n[Dot Product]\n";

            if (Matches(selectedVersion, "reference")) {
                MakeBench().run("Reference (Scalar)", [&] {
                    simd::BatchDotReference(data.vecA.data(), data.vecB.data(), data.dotResults.data(), count);
                    return data.dotResults[0];
                });
            }

            if (Matches(selectedVersion, "simd")) {
                MakeBench().run("SIMD (SSE2)", [&] {
                    simd::BatchDot(data.vecA.data(), data.vecB.data(), data.dotResults.data(), count);
                    return data.dotResults[0];
                });
            }

            if (Matches(selectedVersion, "asm")) {
                MakeBench().run("ASM (x64)", [&] {
                    AsmBatchDot(data.vecA.data(), data.vecB.data(), data.dotResults.data(), count);
                    return data.dotResults[0];
                });
            }
        }

        // 2. NORMALIZATION
        if (Matches(selectedOperation, "normalize")) {
            std::cout << "\n[Normalization]\n";

            if (Matches(selectedVersion, "reference")) {
                MakeBench().run("Reference (Scalar)", [&] {
                    simd::BatchNormalizeReference(data.vecA.data(), data.normResults.data(), count);
                    return data.normResults[0].getX();
                });
            }

            if (Matches(selectedVersion, "simd")) {
                MakeBench().run("SIMD (SSE2)", [&] {
                    simd::BatchNormalize(data.vecA.data(), data.normResults.data(), count);
                    return data.normResults[0].getX();
                });
            }
        }

        // 3. TRANSFORM POINT
        if (Matches(selectedOperation, "transform")) {
            std::cout << "\n[Transform Point]\n";

            if (Matches(selectedVersion, "reference")) {
                MakeBench().run("Reference (Scalar)", [&] {
                    simd::BatchTransformPointReference(data.transformMat, data.vecA.data(), data.transResults.data(), count);
                    return data.transResults[0].getX();
                });
            }

            if (Matches(selectedVersion, "simd")) {
                MakeBench().run("SIMD (SSE2)", [&] {
                    simd::BatchTransformPoint(data.transformMat, data.vecA.data(), data.transResults.data(), count);
                    return data.transResults[0].getX();
                });
            }
        }
        std::cout << "\n----------------------------------------------------------------\n";
    }

    return 0;
}
