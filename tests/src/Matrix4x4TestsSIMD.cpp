#include <gtest/gtest.h>
#include "mathLibSIMD/include/Mat4x4.hpp"
#include "mathLibSIMD/include/Vec3f.hpp"
#include "mathLibSIMD/include/Vec4.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace SimdMatrix4x4Tests
{
    using M = simd::Matrix4x4;
    using V = simd::Vec4;
    using V3 = simd::Vec3f;

    namespace
    {
        // Helpers locaux : accès via operator()(r, c), getX()/getY()/getZ()/getW().
        constexpr float kEpsilon = 1e-5f;

        void ExpectNear(float expected, float actual)
        {
            const float tolerance = kEpsilon * std::max(1.0f, std::abs(expected));
            EXPECT_NEAR(expected, actual, tolerance);
        }

        void ExpectVecNear(const V& expected, const V& actual)
        {
            ExpectNear(expected.getX(), actual.getX());
            ExpectNear(expected.getY(), actual.getY());
            ExpectNear(expected.getZ(), actual.getZ());
            ExpectNear(expected.getW(), actual.getW());
        }

        void ExpectVec3Near(const V3& expected, const V3& actual)
        {
            ExpectNear(expected.getX(), actual.getX());
            ExpectNear(expected.getY(), actual.getY());
            ExpectNear(expected.getZ(), actual.getZ());
        }

        void ExpectMatNear(const M& expected, const M& actual)
        {
            for (std::size_t r = 0; r < 4; ++r)
            {
                for (std::size_t c = 0; c < 4; ++c)
                {
                    SCOPED_TRACE(testing::Message() << "element (" << r << ", " << c << ")");
                    ExpectNear(expected(r, c), actual(r, c));
                }
            }
        }

        M Make(const std::array<float, 16>& elements)
        {
            return M(elements);
        }
    }

    TEST(Matrix4x4TestsSIMD, DefaultAndFactoryAreIdentity)
    {
        const M expected = Make({ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
        ExpectMatNear(expected, M{});
        ExpectMatNear(expected, M::Identity());
        ExpectMatNear(Make({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }), M::Zero());
    }

    TEST(Matrix4x4TestsSIMD, ConstructorAndIndexUseRowMajorOrder)
    {
        M matrix = Make({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        ExpectNear(2.0f, matrix(0, 1));
        ExpectNear(5.0f, matrix(1, 0));
        matrix(1, 0) = 99.0f;
        const M& view = matrix;
        ExpectNear(99.0f, view(1, 0));
        EXPECT_THROW((void)matrix(4, 0), std::out_of_range);
        EXPECT_THROW((void)view(0, 4), std::out_of_range);
    }

    TEST(Matrix4x4TestsSIMD, ArithmeticHasIndependentExpectedValues)
    {
        const M a = Make({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        const M b = Make({ -1, 2, -2, 1, -3, 0, 3, -1, 2, -2, 1, -3, 0, 3, -1, 2 });
        ExpectMatNear(Make({ 0, 4, 1, 5, 2, 6, 10, 7, 11, 8, 12, 9, 13, 17, 14, 18 }), a + b);
        ExpectMatNear(Make({ 2, 0, 5, 3, 8, 6, 4, 9, 7, 12, 10, 15, 13, 11, 16, 14 }), a - b);
        ExpectMatNear(Make({ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32 }), a * 2.0f);
    }

    TEST(Matrix4x4TestsSIMD, MatrixProductHasIndependentExpectedValues)
    {
        const M a = Make({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        const M b = Make({ -1, 2, -2, 1, -3, 0, 3, -1, 2, -2, 1, -3, 0, 3, -1, 2 });
        const M expected = Make({ -1, 8, 3, -2, -9, 20, 7, -6, -17, 32, 11, -10, -25, 44, 15, -14 });
        ExpectMatNear(expected, a * b);
        EXPECT_TRUE(a * b != b * a);
    }

    TEST(Matrix4x4TestsSIMD, MultiplyAssignSupportsSelfAliasing)
    {
        M matrix = Make({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        EXPECT_TRUE(&(matrix *= matrix) == &matrix);
        ExpectMatNear(Make({ 90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600 }), matrix);
    }

    TEST(Matrix4x4TestsSIMD, MatrixVectorUsesColumnVectorConvention)
    {
        const M matrix = Make({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        const V vector{ 1, 2, 3, 4 };
        ExpectVecNear(V{ 30, 70, 110, 150 }, matrix * vector);
    }

    TEST(Matrix4x4TestsSIMD, TransposeMovesEveryElement)
    {
        const M matrix = Make({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        ExpectMatNear(Make({ 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 }), matrix.Transpose());
    }

    TEST(Matrix4x4TestsSIMD, EqualityChecksEveryElement)
    {
        const M a;
        for (std::size_t r = 0; r < 4; ++r)
        {
            for (std::size_t c = 0; c < 4; ++c)
            {
                M b = a;
                b(r, c) += 1.0f;
                EXPECT_TRUE(a != b);
                EXPECT_FALSE(a == b);
            }
        }
        EXPECT_TRUE(a == a);
        EXPECT_FALSE(a != a);
    }

    TEST(Matrix4x4TestsSIMD, ScaleHasKnownResult)
    {
        const M scale = M::Scale(V3{ 2, 3, 4 });
        ExpectMatNear(Make({ 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1 }), scale);
        ExpectNear(24.0f, scale.Determinant());
    }

    TEST(Matrix4x4TestsSIMD, RotationXIsRightHanded)
    {
        const M rotation = M::RotationX(std::numbers::pi_v<float> / 2.0f);
        ExpectVecNear(V{ 0, 0, 1, 0 }, rotation * V{ 0, 1, 0, 0 });
        ExpectNear(1.0f, rotation.Determinant());
    }

    TEST(Matrix4x4TestsSIMD, RotationYIsRightHanded)
    {
        const M rotation = M::RotationY(std::numbers::pi_v<float> / 2.0f);
        ExpectVecNear(V{ 1, 0, 0, 0 }, rotation * V{ 0, 0, 1, 0 });
        ExpectNear(1.0f, rotation.Determinant());
    }

    TEST(Matrix4x4TestsSIMD, RotationZIsRightHanded)
    {
        const M rotation = M::RotationZ(std::numbers::pi_v<float> / 2.0f);
        ExpectVecNear(V{ 0, 1, 0, 0 }, rotation * V{ 1, 0, 0, 0 });
        ExpectNear(1.0f, rotation.Determinant());
    }

    TEST(Matrix4x4TestsSIMD, InverseUsesPivotingAndKnownExpectedValues)
    {
        const M matrix = Make({ 0, 2, 0, 0, 1, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
        const M expected = Make({ -1.5f, 1, 0, 0, 0.5f, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
        const M actual = matrix.Inverse();
        ExpectMatNear(expected, actual);
        ExpectMatNear(M::Identity(), matrix * actual);
        ExpectMatNear(M::Identity(), actual * matrix);
        ExpectNear(-2.0f, matrix.Determinant());
    }

    TEST(Matrix4x4TestsSIMD, DenseInverseMatchesRationalFixture)
    {
        const M matrix = Make({ 4, 1, -1, 1, 2, 5, 2, 0, 1, -1, 6, -1, 0, 2, 0, 7 });
        const M expected = Make({
             44.0f / 175.0f, -1.0f / 35.0f,   9.0f / 175.0f, -1.0f / 35.0f,
             -2.0f / 25.0f,   1.0f / 5.0f,   -2.0f / 25.0f,   0.0f,
             -9.0f / 175.0f,  1.0f / 35.0f,  26.0f / 175.0f,  1.0f / 35.0f,
              4.0f / 175.0f, -2.0f / 35.0f,   4.0f / 175.0f,  1.0f / 7.0f });
        ExpectMatNear(expected, matrix.Inverse());
    }

    TEST(Matrix4x4TestsSIMD, InverseRejectsSingularAndNonFinite)
    {
        EXPECT_THROW((void)M::Zero().Inverse(), std::domain_error);
        M duplicate;
        for (std::size_t c = 0; c < 4; ++c)
        {
            duplicate(1, c) = duplicate(0, c);
        }
        ExpectNear(0.0f, duplicate.Determinant());
        EXPECT_THROW((void)duplicate.Inverse(), std::domain_error);
        M invalid;
        invalid(0, 0) = std::numeric_limits<float>::infinity();
        EXPECT_THROW((void)invalid.Inverse(), std::domain_error);
        invalid(0, 0) = std::numeric_limits<float>::quiet_NaN();
        EXPECT_THROW((void)invalid.Inverse(), std::domain_error);
    }

    TEST(Matrix4x4TestsSIMD, InverseValidatesTolerance)
    {
        EXPECT_THROW((void)M {}.Inverse(-1.0f), std::invalid_argument);
        EXPECT_THROW((void)M {}.Inverse(1.0f), std::invalid_argument);
        EXPECT_THROW((void)M {}.Inverse(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    }

    TEST(Matrix4x4TestsSIMD, InverseToleranceCanRejectNearDependentRows)
    {
        M matrix;
        matrix(0, 0) = 1.0f;
        matrix(0, 1) = 1.0f;
        matrix(1, 0) = 1.0f;
        matrix(1, 1) = 1.0f + std::numeric_limits<float>::epsilon();
        EXPECT_THROW((void)matrix.Inverse(), std::domain_error);
    }

    TEST(Matrix4x4TestsSIMD, InverseHandlesDifferentRowScales)
    {
        M matrix;
        matrix(0, 0) = 0.0001f;
        matrix(1, 1) = 10000.0f;
        M expected;
        expected(0, 0) = 10000.0f;
        expected(1, 1) = 0.0001f;
        ExpectMatNear(expected, matrix.Inverse());
    }

    TEST(Matrix4x4TestsSIMD, TranslationAffectsPointsButNotDirections)
    {
        const M matrix = M::Translation(V3{ 10, 20, 30 });
        const V3 input{ 1, 2, 3 };
        ExpectVec3Near(V3{ 11, 22, 33 }, matrix.TransformPoint(input));
        ExpectVec3Near(input, matrix.TransformDirection(input));
    }

    TEST(Matrix4x4TestsSIMD, CompositionAppliesRightMatrixFirst)
    {
        const M translate = M::Translation(V3{ 10, 20, 30 });
        const M scale = M::Scale(V3{ 2, 3, 4 });
        const V3 point{ 1, 2, 3 };
        ExpectVec3Near(V3{ 12, 26, 42 }, (translate * scale).TransformPoint(point));
        ExpectVec3Near(V3{ 22, 66, 132 }, (scale * translate).TransformPoint(point));
    }

    TEST(Matrix4x4TestsSIMD, AffineHelpersRejectPerspective)
    {
        M matrix;
        matrix(3, 2) = 1.0f;
        EXPECT_THROW((void)matrix.TransformPoint(V3{ 1, 2, 3 }), std::domain_error);
        EXPECT_THROW((void)matrix.TransformDirection(V3{ 1, 2, 3 }), std::domain_error);
    }

    TEST(Matrix4x4TestsSIMD, CopyIsIndependent)
    {
        const M original = Make({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        M copy = original;
        copy(1, 1) = 99.0f;
        ExpectMatNear(Make({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }), original);
        ExpectNear(99.0f, copy(1, 1));
    }

    TEST(Matrix4x4TestsSIMD, IsSixteenByteAligned)
    {
        // 4 lignes __m128 : 4 * 16 octets = 64 octets, alignées sur 16.
        static_assert(alignof(M) == 16);
        static_assert(sizeof(M) == 64);
        const M m;
        EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(&m) % 16u);
    }
}