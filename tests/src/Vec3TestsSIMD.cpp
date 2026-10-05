#include <gtest/gtest.h>
#include "mathLibSIMD/include/Vec3f.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace SimdVec3Tests
{
    using V = simd::Vec3f;

    namespace
    {
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
    }

    TEST(Vec3TestsSIMD, DefaultConstructorIsZero)
    {
        const V actual;
        ExpectVecNear(V{ 0, 0, 0 }, actual);
    }

    TEST(Vec3TestsSIMD, CopyIsIndependent)
    {
        const V original{ 1, 2, 3 };
        V copy = original;
        copy.setX(99.0f);
        ExpectVecNear(V{ 1, 2, 3 }, original);
        EXPECT_EQ(99.0f, copy.getX());
    }

    TEST(Vec3TestsSIMD, GettersReturnComponentsInOrder)
    {
        const V v{ 1, 2, 3 };
        EXPECT_EQ(1.0f, v.getX());
        EXPECT_EQ(2.0f, v.getY());
        EXPECT_EQ(3.0f, v.getZ());
    }

    TEST(Vec3TestsSIMD, SettersOnlyModifyTheirComponent)
    {
        V v{ 1, 2, 3 };
        v.setX(10.0f);
        ExpectVecNear(V{ 10, 2, 3 }, v);
        v.setY(20.0f);
        ExpectVecNear(V{ 10, 20, 3 }, v);
        v.setZ(30.0f);
        ExpectVecNear(V{ 10, 20, 30 }, v);
    }

    TEST(Vec3TestsSIMD, AdditionHasKnownResult)
    {
        const V a{ 1, 2, 3 };
        const V b{ 2, 3, 4 };
        ExpectVecNear(V{ 3, 5, 7 }, a + b);
    }

    TEST(Vec3TestsSIMD, SubtractionHasKnownResult)
    {
        const V a{ 1, 2, 3 };
        const V b{ 2, 3, 4 };
        ExpectVecNear(V{ -1, -1, -1 }, a - b);
    }

    TEST(Vec3TestsSIMD, ComponentProductHasKnownResult)
    {
        const V a{ 1, 2, 3 };
        const V b{ 2, 3, 4 };
        ExpectVecNear(V{ 2, 6, 12 }, a * b);
    }

    TEST(Vec3TestsSIMD, ComponentDivisionHasKnownResult)
    {
        const V a{ 2, 6, 12 };
        const V b{ 2, 3, 4 };
        ExpectVecNear(V{ 1, 2, 3 }, a / b);
    }

    TEST(Vec3TestsSIMD, ComponentDivisionByZeroComponentThrows)
    {
        const V a{ 1, 2, 3 };
        EXPECT_THROW((void)(a / V{ 0, 1, 1 }), std::domain_error);
        EXPECT_THROW((void)(a / V{ 1, 0, 1 }), std::domain_error);
        EXPECT_THROW((void)(a / V{ 1, 1, 0 }), std::domain_error);
        EXPECT_THROW((void)(a / 0.0f), std::domain_error);
    }

    TEST(Vec3TestsSIMD, ScalarOperationsAndNegation)
    {
        const V a{ 1, 2, 3 };
        ExpectVecNear(V{ -1, -2, -3 }, -a);
        ExpectVecNear(V{ 2, 4, 6 }, a * 2.0f);
        ExpectVecNear(V{ 2, 4, 6 }, 2.0f * a);
        ExpectVecNear(a, (a * 2.0f) / 2.0f);
    }

    TEST(Vec3TestsSIMD, CompoundOperatorsReturnSelfAndSupportAliasing)
    {
        V a{ 1, 2, 3 };
        EXPECT_TRUE(&(a += a) == &a);
        ExpectVecNear(V{ 2, 4, 6 }, a);
        a -= V{ 1, 2, 3 };
        a *= a;
        ExpectVecNear(V{ 1, 4, 9 }, a);
        a /= V{ 1, 2, 3 };
        a *= 2.0f;
        a /= 2.0f;
        ExpectVecNear(V{ 1, 2, 3 }, a);
    }

    TEST(Vec3TestsSIMD, DivisionByZeroThrowsWithoutPartialMutation)
    {
        const V original{ 1, 2, 3 };
        V actual = original;
        EXPECT_THROW(actual /= 0.0f, std::domain_error);
        ExpectVecNear(original, actual);
        EXPECT_THROW((actual /= V{ 0, 1, 1 }), std::domain_error);
        ExpectVecNear(original, actual);
        EXPECT_THROW((actual /= V{ 1, 0, 1 }), std::domain_error);
        ExpectVecNear(original, actual);
        EXPECT_THROW((actual /= V{ 1, 1, 0 }), std::domain_error);
        ExpectVecNear(original, actual);
    }

    TEST(Vec3TestsSIMD, EqualityChecksEveryComponentExactly)
    {
        const V a{ 1, 2, 3 };
        EXPECT_TRUE(a == a);
        EXPECT_FALSE(a != a);

        V changedx = a;
        changedx.setX(changedx.getX() + 1.0f);
        EXPECT_TRUE(changedx != a);
        EXPECT_FALSE(changedx == a);

        V changedy = a;
        changedy.setY(changedy.getY() + 1.0f);
        EXPECT_TRUE(changedy != a);
        EXPECT_FALSE(changedy == a);

        V changedz = a;
        changedz.setZ(changedz.getZ() + 1.0f);
        EXPECT_TRUE(changedz != a);
        EXPECT_FALSE(changedz == a);

        V nan = a;
        nan.setX(std::numeric_limits<float>::quiet_NaN());
        EXPECT_FALSE(nan == nan);
    }

    TEST(Vec3TestsSIMD, DotHasKnownResult)
    {
        const V a{ 1, 2, 3 };
        const V b{ 2, 3, 4 };
        ExpectNear(20.0f, a.Dot(b));
    }

    TEST(Vec3TestsSIMD, MagnitudeHasKnownResult)
    {
        const V a{ 3, 4, 0 };
        ExpectNear(25.0f, a.MagnitudeSquared());
        ExpectNear(5.0f, a.Magnitude());
        ExpectNear(0.0f, V{}.Magnitude());
    }

    TEST(Vec3TestsSIMD, NormalizeReturnsNewUnitVector)
    {
        const V a{ 3, 4, 0 };
        const V actual = a.Normalize();
        ExpectVecNear(V{ 0.6f, 0.8f, 0.0f }, actual);
        ExpectNear(1.0f, actual.Magnitude());
        ExpectVecNear(V{ 3, 4, 0 }, a);
    }

    TEST(Vec3TestsSIMD, NormalizeRejectsZeroAndNonFinite)
    {
        EXPECT_THROW((void)V {}.Normalize(), std::domain_error);
        V invalid{ 1, 2, 3 };
        invalid.setX(std::numeric_limits<float>::infinity());
        EXPECT_THROW((void)invalid.Normalize(), std::domain_error);
        invalid.setX(std::numeric_limits<float>::quiet_NaN());
        EXPECT_THROW((void)invalid.Normalize(), std::domain_error);
    }

    TEST(Vec3TestsSIMD, NormalizeHandlesVeryLargeAndSmallFiniteValues)
    {
        const float large = std::numeric_limits<float>::max();
        const float small = std::numeric_limits<float>::min();
        const V expected = V{ 1, 1, 1 } / std::sqrt(3.0f);
        ExpectVecNear(expected, V{ large, large, large }.Normalize());
        ExpectVecNear(expected, V{ small, small, small }.Normalize());
    }

    TEST(Vec3TestsSIMD, DistanceHasKnownResult)
    {
        const V a{ 1, 2, 3 };
        const V b{ 4, 6, 3 };
        ExpectNear(25.0f, a.DistanceSquared(b));
        ExpectNear(5.0f, a.Distance(b));
    }

    TEST(Vec3TestsSIMD, AngleUsesRadians)
    {
        ExpectNear(std::numbers::pi_v<float> / 2.0f, V::UnitX.Angle(V::UnitY));
        ExpectNear(std::numbers::pi_v<float>, V::UnitX.Angle(-V::UnitX));
        ExpectNear(0.0f, V::UnitX.Angle(V::UnitX));
        EXPECT_THROW((void)V::Zero.Angle(V::UnitX), std::domain_error);
        EXPECT_THROW((void)V::UnitX.Angle(V::Zero), std::domain_error);
    }

    TEST(Vec3TestsSIMD, LerpSupportsEndpointsMidpointAndExtrapolation)
    {
        const V a{ 0, 0, 0 };
        const V b{ 2, 4, 6 };
        ExpectVecNear(a, V::Lerp(a, b, 0.0f));
        ExpectVecNear(b, V::Lerp(a, b, 1.0f));
        ExpectVecNear(V{ 1, 2, 3 }, V::Lerp(a, b, 0.5f));
        ExpectVecNear(V{ 4, 8, 12 }, V::Lerp(a, b, 2.0f));
    }

    TEST(Vec3TestsSIMD, MinMaxAreComponentWise)
    {
        const V a{ 1, -4, 7 };
        const V b{ 3, -2, 5 };
        ExpectVecNear(V{ 1, -4, 5 }, V::Min(a, b));
        ExpectVecNear(V{ 3, -2, 7 }, V::Max(a, b));
    }

    TEST(Vec3TestsSIMD, NamedConstantsHaveExpectedComponents)
    {
        ExpectVecNear(V{ 0, 0, 0 }, V::Zero);
        ExpectVecNear(V{ 1, 1, 1 }, V::One);
        ExpectVecNear(V{ 1, 0, 0 }, V::UnitX);
        ExpectVecNear(V{ 0, 1, 0 }, V::UnitY);
        ExpectVecNear(V{ 0, 0, 1 }, V::UnitZ);
    }

    TEST(Vec3TestsSIMD, CrossUsesRightHandedOrientation)
    {
        ExpectVecNear(V{ 0, 0, 1 }, V::UnitX.Cross(V::UnitY));
        ExpectVecNear(V{ 0, 0, -1 }, V::UnitY.Cross(V::UnitX));
        const V a{ 1, 2, 3 };
        const V b{ 4, 5, 6 };
        ExpectVecNear(V{ -3, 6, -3 }, a.Cross(b));
        ExpectVecNear(V::Zero, a.Cross(a));
    }

    TEST(Vec3TestsSIMD, IsSixteenByteAligned)
    {
        static_assert(alignof(V) == 16);
        static_assert(sizeof(V) == 16);
        const V v{ 1, 2, 3 };
        EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(&v) % 16u);
    }
}