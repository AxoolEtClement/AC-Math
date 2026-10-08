#include <gtest/gtest.h>
#include "mathLibSIMD/include/Mat3.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace SimdMatrix3x3Tests
{
    using M = simd::Matrix3x3;
    using V = simd::Vec3f;

    namespace
    {
        // Helpers locaux : TestHelpers.h cible Maths::Matrix3x3<T> (membres publics),
        // ici on passe par operator()(r, c) et getX()/getY()/getZ().
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
        }

        void ExpectMatNear(const M& expected, const M& actual)
        {
            for (std::size_t r = 0; r < 3; ++r)
            {
                for (std::size_t c = 0; c < 3; ++c)
                {
                    SCOPED_TRACE(testing::Message() << "element (" << r << ", " << c << ")");
                    ExpectNear(expected(r, c), actual(r, c));
                }
            }
        }

        M Make(const std::array<float, 9>& elements)
        {
            return M(elements);
        }
    }

    TEST(Matrix3x3TestsSIMD, DefaultAndFactoryAreIdentity)
    {
        const M expected = Make({1, 0, 0, 0, 1, 0, 0, 0, 1});
        ExpectMatNear(expected, M{});
        ExpectMatNear(expected, M::Identity());
        ExpectMatNear(Make({0, 0, 0, 0, 0, 0, 0, 0, 0}), M::Zero());
    }

    TEST(Matrix3x3TestsSIMD, ConstructorAndIndexUseRowMajorOrder)
    {
        M matrix = Make({1, 2, 3, 4, 5, 6, 7, 8, 9});
        ExpectNear(2.0f, matrix(0, 1));
        ExpectNear(4.0f, matrix(1, 0));
        matrix(1, 0) = 99.0f;
        const M& view = matrix;
        ExpectNear(99.0f, view(1, 0));
        EXPECT_THROW((void)matrix(3, 0), std::out_of_range);
        EXPECT_THROW((void)view(0, 3), std::out_of_range);
    }

    TEST(Matrix3x3TestsSIMD, ArithmeticHasIndependentExpectedValues)
    {
        const M a = Make({1, 2, 3, 4, 5, 6, 7, 8, 9});
        const M b = Make({-1, 2, -2, 1, -3, 0, 3, -1, 2});
        ExpectMatNear(Make({0, 4, 1, 5, 2, 6, 10, 7, 11}), a + b);
        ExpectMatNear(Make({2, 0, 5, 3, 8, 6, 4, 9, 7}), a - b);
        ExpectMatNear(Make({2, 4, 6, 8, 10, 12, 14, 16, 18}), a * 2.0f);
    }

    TEST(Matrix3x3TestsSIMD, MatrixProductHasIndependentExpectedValues)
    {
        const M a = Make({1, 2, 3, 4, 5, 6, 7, 8, 9});
        const M b = Make({-1, 2, -2, 1, -3, 0, 3, -1, 2});
        const M expected = Make({10, -7, 4, 19, -13, 4, 28, -19, 4});
        ExpectMatNear(expected, a * b);
        EXPECT_TRUE(a * b != b * a);
    }

    TEST(Matrix3x3TestsSIMD, MultiplyAssignSupportsSelfAliasing)
    {
        M matrix = Make({1, 2, 3, 4, 5, 6, 7, 8, 9});
        EXPECT_TRUE(&(matrix *= matrix) == &matrix);
        ExpectMatNear(Make({30, 36, 42, 66, 81, 96, 102, 126, 150}), matrix);
    }

    TEST(Matrix3x3TestsSIMD, MatrixVectorUsesColumnVectorConvention)
    {
        const M matrix = Make({1, 2, 3, 4, 5, 6, 7, 8, 9});
        const V vector{1, 2, 3};
        ExpectVecNear(V{14, 32, 50}, matrix * vector);
    }

    TEST(Matrix3x3TestsSIMD, TransposeMovesEveryElement)
    {
        const M matrix = Make({1, 2, 3, 4, 5, 6, 7, 8, 9});
        ExpectMatNear(Make({1, 4, 7, 2, 5, 8, 3, 6, 9}), matrix.Transpose());
    }

    TEST(Matrix3x3TestsSIMD, EqualityChecksEveryElement)
    {
        const M a;
        for (std::size_t r = 0; r < 3; ++r)
        {
            for (std::size_t c = 0; c < 3; ++c)
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

    TEST(Matrix3x3TestsSIMD, ScaleHasKnownResult)
    {
        const M scale = M::Scale(V{2, 3, 4});
        ExpectMatNear(Make({2, 0, 0, 0, 3, 0, 0, 0, 4}), scale);
        ExpectNear(24.0f, scale.Determinant());
    }

    TEST(Matrix3x3TestsSIMD, RotationXIsRightHanded)
    {
        const M rotation = M::RotationX(std::numbers::pi_v<float> / 2.0f);
        ExpectVecNear(V{0, 0, 1}, rotation * V{0, 1, 0});
        ExpectNear(1.0f, rotation.Determinant());
    }

    TEST(Matrix3x3TestsSIMD, RotationYIsRightHanded)
    {
        const M rotation = M::RotationY(std::numbers::pi_v<float> / 2.0f);
        ExpectVecNear(V{1, 0, 0}, rotation * V{0, 0, 1});
        ExpectNear(1.0f, rotation.Determinant());
    }

    TEST(Matrix3x3TestsSIMD, RotationZIsRightHanded)
    {
        const M rotation = M::RotationZ(std::numbers::pi_v<float> / 2.0f);
        ExpectVecNear(V{0, 1, 0}, rotation * V{1, 0, 0});
        ExpectNear(1.0f, rotation.Determinant());
    }

    TEST(Matrix3x3TestsSIMD, InverseUsesPivotingAndKnownExpectedValues)
    {
        const M matrix = Make({0, 2, 0, 1, 3, 0, 0, 0, 1});
        const M expected = Make({-1.5f, 1, 0, 0.5f, 0, 0, 0, 0, 1});
        const M actual = matrix.Inverse();
        ExpectMatNear(expected, actual);
        ExpectMatNear(M::Identity(), matrix * actual);
        ExpectMatNear(M::Identity(), actual * matrix);
        ExpectNear(-2.0f, matrix.Determinant());
    }

    TEST(Matrix3x3TestsSIMD, DenseInverseMatchesRationalFixture)
    {
        const M matrix = Make({4, 1, -1, 2, 5, 2, 1, -1, 6});
        const M expected = Make({
             32.0f / 125.0f, -1.0f / 25.0f,   7.0f / 125.0f,
             -2.0f / 25.0f,   1.0f / 5.0f,   -2.0f / 25.0f,
             -7.0f / 125.0f,  1.0f / 25.0f,  18.0f / 125.0f});
        ExpectMatNear(expected, matrix.Inverse());
    }

    TEST(Matrix3x3TestsSIMD, InverseRejectsSingularAndNonFinite)
    {
        EXPECT_THROW((void)M::Zero().Inverse(), std::domain_error);
        M duplicate;
        for (std::size_t c = 0; c < 3; ++c)
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

    TEST(Matrix3x3TestsSIMD, InverseValidatesTolerance)
    {
        EXPECT_THROW((void)M{}.Inverse(-1.0f), std::invalid_argument);
        EXPECT_THROW((void)M{}.Inverse(1.0f), std::invalid_argument);
        EXPECT_THROW((void)M{}.Inverse(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
    }

    TEST(Matrix3x3TestsSIMD, InverseToleranceCanRejectNearDependentRows)
    {
        M matrix;
        matrix(0, 0) = 1.0f;
        matrix(0, 1) = 1.0f;
        matrix(1, 0) = 1.0f;
        matrix(1, 1) = 1.0f + std::numeric_limits<float>::epsilon();
        EXPECT_THROW((void)matrix.Inverse(), std::domain_error);
    }

    TEST(Matrix3x3TestsSIMD, InverseHandlesDifferentRowScales)
    {
        M matrix;
        matrix(0, 0) = 0.0001f;
        matrix(1, 1) = 10000.0f;
        M expected;
        expected(0, 0) = 10000.0f;
        expected(1, 1) = 0.0001f;
        ExpectMatNear(expected, matrix.Inverse());
    }

    TEST(Matrix3x3TestsSIMD, CopyIsIndependent)
    {
        const M original = Make({1, 2, 3, 4, 5, 6, 7, 8, 9});
        M copy = original;
        copy(1, 1) = 99.0f;
        ExpectMatNear(Make({1, 2, 3, 4, 5, 6, 7, 8, 9}), original);
        ExpectNear(99.0f, copy(1, 1));
    }

    TEST(Matrix3x3TestsSIMD, IsSixteenByteAligned)
    {
        // 3 lignes __m128 : 3 * 16 octets, alignées sur 16.
        static_assert(alignof(M) == 16);
        static_assert(sizeof(M) == 48);
        const M m;
        EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(&m) % 16u);
    }
}