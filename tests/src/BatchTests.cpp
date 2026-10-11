#include <gtest/gtest.h>
#include "mathLibSIMD/include/Vec3f.hpp"
#include "mathLibSIMD/include/Vec4.hpp"
#include "mathLibSIMD/include/Vec4SOA.hpp"
#include "mathLibSIMD/include/Mat4x4.hpp"
#include "mathLibSIMD/include/LayoutConversions.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <random>

namespace SimdBatchTests {
    using V3 = simd::Vec3f;
    using V4 = simd::Vec4;
    using V4S = simd::Vec4SOA;

    namespace {
        constexpr float kTol = 1e-5f;
        void ExpectVec3Near(const V3& expected, const V3& actual) {
            EXPECT_NEAR(expected.getX(), actual.getX(), kTol);
            EXPECT_NEAR(expected.getY(), actual.getY(), kTol);
            EXPECT_NEAR(expected.getZ(), actual.getZ(), kTol);
        }
        void ExpectVec4Near(const V4& expected, const V4& actual) {
            EXPECT_NEAR(expected.getX(), actual.getX(), kTol);
            EXPECT_NEAR(expected.getY(), actual.getY(), kTol);
            EXPECT_NEAR(expected.getZ(), actual.getZ(), kTol);
            EXPECT_NEAR(expected.getW(), actual.getW(), kTol);
        }
    }

    // B1-B5 Non-Regression
    TEST(BatchTests, BugFixNonRegression) {
        // B1: Dot summation
        V3 a{1, 2, 3}, b{4, 5, 6};
        EXPECT_NEAR(a.Dot(b), 1*4 + 2*5 + 3*6, kTol);

        // B3: Angle zero vector
        EXPECT_THROW(V3::Zero.Angle(V3::UnitX), std::domain_error);

        // B5: Batch Normalize zero vector (no NaN, no exception)
        V3 zero_vec = V3::Zero;
        V3 out;
        simd::BatchNormalize(&zero_vec, &out, 1);
        EXPECT_EQ(out.getX(), 0.f);
        EXPECT_FALSE(std::isnan(out.getX()));
    }

    // S1: Batch Processings
    void TestBatchDot(std::size_t count) {
        std::vector<V3> a(count), b(count);
        std::vector<float> res(count);
        for(std::size_t i=0; i<count; ++i) {
            a[i] = V3{(float)i, 1.0f, 2.0f};
            b[i] = V3{1.0f, (float)i, 3.0f};
        }
        simd::BatchDot(a.data(), b.data(), res.data(), count);
        std::vector<float> reference(count);
        simd::BatchDotReference(a.data(), b.data(), reference.data(), count);
        for(std::size_t i=0; i<count; ++i) {
            EXPECT_NEAR(res[i], (float)i * 1.0f + 1.0f * (float)i + 2.0f * 3.0f, kTol);
            EXPECT_NEAR(res[i], reference[i], kTol);
        }
    }

    TEST(BatchTests, DotEdgeSizes) {
        for (auto size : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 100u}) {
            TestBatchDot(size);
        }
    }

    void TestBatchNormalize(std::size_t count) {
        std::vector<V3> in(count), out(count);
        for(std::size_t i=0; i<count; ++i) in[i] = V3{1.0f, 2.0f, 3.0f};
        simd::BatchNormalize(in.data(), out.data(), count);
        std::vector<V3> reference(count);
        simd::BatchNormalizeReference(in.data(), reference.data(), count);
        for(std::size_t i=0; i<count; ++i) {
            EXPECT_NEAR(out[i].Magnitude(), 1.0f, kTol);
            ExpectVec3Near(reference[i], out[i]);
        }
    }

    TEST(BatchTests, NormalizeEdgeSizes) {
        for (auto size : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 100u}) {
            TestBatchNormalize(size);
        }
    }

    TEST(BatchTests, TransformPointEdgeSizes) {
        simd::Matrix4x4 mat = simd::Matrix4x4::Translation(V3{10, 20, 30});
        std::vector<V3> in(17, V3{1, 2, 3});
        std::vector<V3> out(17);
        simd::BatchTransformPoint(mat, in.data(), out.data(), 17);
        std::vector<V3> reference(17);
        simd::BatchTransformPointReference(mat, in.data(), reference.data(), 17);
        for(auto& v : out) ExpectVec3Near(V3{11, 22, 33}, v);
        for(std::size_t i = 0; i < out.size(); ++i)
            ExpectVec3Near(reference[i], out[i]);
    }

    // S2: Layout Conversions
    TEST(BatchTests, AoS_SoA_RoundTrip) {
        std::vector<V4> aos_in(17);
        for(std::size_t i=0; i<17; ++i) aos_in[i] = V4{(float)i, (float)i+1, (float)i+2, (float)i+3};

        std::vector<V4S> soa( (17 + 3) / 4 );
        simd::ToSOA(aos_in.data(), soa.data(), 17);

        std::vector<V4> aos_out(17);
        simd::ToAOS(soa.data(), aos_out.data(), 17);

        for(std::size_t i=0; i<17; ++i) ExpectVec4Near(aos_in[i], aos_out[i]);
    }

    // Canary test for overflow
    TEST(BatchTests, CanaryOverflow) {
        const std::size_t count = 4;
        std::vector<float> buffer(count + 1, 0.123f); // canary at end
        std::vector<V3> a(count, V3::UnitX), b(count, V3::UnitX);
        simd::BatchDot(a.data(), b.data(), buffer.data(), count);
        EXPECT_EQ(buffer[count], 0.123f);
    }
}
