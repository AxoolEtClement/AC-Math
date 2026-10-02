#include <gtest/gtest.h>
#include "mathLibSIMD/include/Vec4.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
 
namespace SimdVec4Tests
{
    using V = simd::Vec4;
 
    namespace
    {
        // Helpers locaux : TestHelpers.h cible Maths::Vec4<T> (membres .x/.y/.z/.w),
        // ici on passe par getX()/getY()/getZ()/getW().
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
    }
 
    TEST(Vec4TestsSIMD, DefaultConstructorIsZero)
    {
        const V actual;
        ExpectVecNear(V{0, 0, 0, 0}, actual);
    }
 
    TEST(Vec4TestsSIMD, CopyIsIndependent)
    {
        const V original{1, 2, 3, 4};
        V copy = original;
        copy.setX(99.0f);
        ExpectVecNear(V{1, 2, 3, 4}, original);
        EXPECT_EQ(99.0f, copy.getX());
    }
 
    TEST(Vec4TestsSIMD, GettersReturnComponentsInOrder)
    {
        const V v{1, 2, 3, 4};
        EXPECT_EQ(1.0f, v.getX());
        EXPECT_EQ(2.0f, v.getY());
        EXPECT_EQ(3.0f, v.getZ());
        EXPECT_EQ(4.0f, v.getW());
    }
 
    TEST(Vec4TestsSIMD, SettersOnlyModifyTheirComponent)
    {
        V v{1, 2, 3, 4};
        v.setX(10.0f);
        ExpectVecNear(V{10, 2, 3, 4}, v);
        v.setY(20.0f);
        ExpectVecNear(V{10, 20, 3, 4}, v);
        v.setZ(30.0f);
        ExpectVecNear(V{10, 20, 30, 4}, v);
        v.setW(40.0f);
        ExpectVecNear(V{10, 20, 30, 40}, v);
    }
 
    TEST(Vec4TestsSIMD, ConstructFromM128PreservesLaneOrder)
    {
        // _mm_set_ps prend les arguments de w vers x
        const V actual(_mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f));
        ExpectVecNear(V{1, 2, 3, 4}, actual);
    }
 
    TEST(Vec4TestsSIMD, StoreConvertsM128ToVec4)
    {
        const V actual = V::store(_mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f));
        ExpectVecNear(V{1, 2, 3, 4}, actual);
    }
 
    TEST(Vec4TestsSIMD, AdditionHasKnownResult)
    {
        const V a{1, 2, 3, 4};
        const V b{2, 3, 4, 5};
        ExpectVecNear(V{3, 5, 7, 9}, a + b);
    }
 
    TEST(Vec4TestsSIMD, SubtractionHasKnownResult)
    {
        const V a{1, 2, 3, 4};
        const V b{2, 3, 4, 5};
        ExpectVecNear(V{-1, -1, -1, -1}, a - b);
    }
 
    TEST(Vec4TestsSIMD, ComponentProductHasKnownResult)
    {
        const V a{1, 2, 3, 4};
        const V b{2, 3, 4, 5};
        ExpectVecNear(V{2, 6, 12, 20}, a * b);
    }
 
    TEST(Vec4TestsSIMD, ComponentDivisionHasKnownResult)
    {
        const V a{2, 6, 12, 20};
        const V b{2, 3, 4, 5};
        ExpectVecNear(V{1, 2, 3, 4}, a / b);
    }
 
    TEST(Vec4TestsSIMD, ComponentDivisionByZeroComponentThrows)
    {
        const V a{1, 2, 3, 4};
        EXPECT_THROW((void)(a / V{0, 1, 1, 1}), std::domain_error);
        EXPECT_THROW((void)(a / V{1, 0, 1, 1}), std::domain_error);
        EXPECT_THROW((void)(a / V{1, 1, 0, 1}), std::domain_error);
        EXPECT_THROW((void)(a / V{1, 1, 1, 0}), std::domain_error);
        EXPECT_THROW((void)(a / 0.0f), std::domain_error);
    }
 
    TEST(Vec4TestsSIMD, ScalarOperationsAndNegation)
    {
        const V a{1, 2, 3, 4};
        ExpectVecNear(V{-1, -2, -3, -4}, -a);
        ExpectVecNear(V{2, 4, 6, 8}, a * 2.0f);
        ExpectVecNear(V{2, 4, 6, 8}, 2.0f * a);
        ExpectVecNear(a, (a * 2.0f) / 2.0f);
    }
 
    TEST(Vec4TestsSIMD, CompoundOperatorsReturnSelfAndSupportAliasing)
    {
        V a{1, 2, 3, 4};
        EXPECT_TRUE(&(a += a) == &a);
        ExpectVecNear(V{2, 4, 6, 8}, a);
        a -= V{1, 2, 3, 4};
        a *= a;
        ExpectVecNear(V{1, 4, 9, 16}, a);
        a /= V{1, 2, 3, 4};
        a *= 2.0f;
        a /= 2.0f;
        ExpectVecNear(V{1, 2, 3, 4}, a);
    }
 
    TEST(Vec4TestsSIMD, DivisionByZeroThrowsWithoutPartialMutation)
    {
        const V original{1, 2, 3, 4};
        V actual = original;
        EXPECT_THROW(actual /= 0.0f, std::domain_error);
        ExpectVecNear(original, actual);
        EXPECT_THROW((actual /= V{0, 1, 1, 1}), std::domain_error);
        ExpectVecNear(original, actual);
        EXPECT_THROW((actual /= V{1, 0, 1, 1}), std::domain_error);
        ExpectVecNear(original, actual);
        EXPECT_THROW((actual /= V{1, 1, 0, 1}), std::domain_error);
        ExpectVecNear(original, actual);
        EXPECT_THROW((actual /= V{1, 1, 1, 0}), std::domain_error);
        ExpectVecNear(original, actual);
    }
 
    TEST(Vec4TestsSIMD, EqualityChecksEveryComponentExactly)
    {
        const V a{1, 2, 3, 4};
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
 
        V changedw = a;
        changedw.setW(changedw.getW() + 1.0f);
        EXPECT_TRUE(changedw != a);
        EXPECT_FALSE(changedw == a);
 
        V nan = a;
        nan.setX(std::numeric_limits<float>::quiet_NaN());
        EXPECT_FALSE(nan == nan);
    }
 
    TEST(Vec4TestsSIMD, DotHasKnownResult)
    {
        const V a{1, 2, 3, 4};
        const V b{2, 3, 4, 5};
        ExpectNear(40.0f, a.Dot(b));
    }
 
    TEST(Vec4TestsSIMD, MagnitudeHasKnownResult)
    {
        const V a{3, 4, 0, 0};
        ExpectNear(25.0f, a.MagnitudeSquared());
        ExpectNear(5.0f, a.Magnitude());
        ExpectNear(0.0f, V{}.Magnitude());
    }
 
    TEST(Vec4TestsSIMD, NormalizeReturnsNewUnitVector)
    {
        const V a{3, 4, 0, 0};
        const V actual = a.Normalize();
        ExpectVecNear(V{0.6f, 0.8f, 0.0f, 0.0f}, actual);
        ExpectNear(1.0f, actual.Magnitude());
        ExpectVecNear(V{3, 4, 0, 0}, a);
    }
 
    TEST(Vec4TestsSIMD, NormalizeRejectsZeroAndNonFinite)
    {
        EXPECT_THROW((void)V{}.Normalize(), std::domain_error);
        V invalid{1, 2, 3, 4};
        invalid.setX(std::numeric_limits<float>::infinity());
        EXPECT_THROW((void)invalid.Normalize(), std::domain_error);
        invalid.setX(std::numeric_limits<float>::quiet_NaN());
        EXPECT_THROW((void)invalid.Normalize(), std::domain_error);
    }
 
    TEST(Vec4TestsSIMD, NormalizeHandlesVeryLargeAndSmallFiniteValues)
    {
        const float large = std::numeric_limits<float>::max();
        const float small = std::numeric_limits<float>::min();
        const V expected = V{1, 1, 1, 1} / std::sqrt(4.0f);
        ExpectVecNear(expected, V{large, large, large, large}.Normalize());
        ExpectVecNear(expected, V{small, small, small, small}.Normalize());
    }
 
    TEST(Vec4TestsSIMD, DistanceHasKnownResult)
    {
        const V a{1, 2, 3, 3};
        const V b{4, 6, 3, 3};
        ExpectNear(25.0f, a.DistanceSquared(b));
        ExpectNear(5.0f, a.Distance(b));
    }
 
    TEST(Vec4TestsSIMD, AngleUsesRadians)
    {
        ExpectNear(std::numbers::pi_v<float> / 2.0f, V::UnitX.Angle(V::UnitY));
        ExpectNear(std::numbers::pi_v<float>, V::UnitX.Angle(-V::UnitX));
        ExpectNear(0.0f, V::UnitX.Angle(V::UnitX));
        EXPECT_THROW((void)V::Zero.Angle(V::UnitX), std::domain_error);
        EXPECT_THROW((void)V::UnitX.Angle(V::Zero), std::domain_error);
    }
 
    TEST(Vec4TestsSIMD, LerpSupportsEndpointsMidpointAndExtrapolation)
    {
        const V a{0, 0, 0, 0};
        const V b{2, 4, 6, 8};
        ExpectVecNear(a, V::Lerp(a, b, 0.0f));
        ExpectVecNear(b, V::Lerp(a, b, 1.0f));
        ExpectVecNear(V{1, 2, 3, 4}, V::Lerp(a, b, 0.5f));
        ExpectVecNear(V{4, 8, 12, 16}, V::Lerp(a, b, 2.0f));
    }
 
    TEST(Vec4TestsSIMD, MinMaxAreComponentWise)
    {
        const V a{1, -4, 7, 2};
        const V b{3, -2, 5, -8};
        ExpectVecNear(V{1, -4, 5, -8}, V::Min(a, b));
        ExpectVecNear(V{3, -2, 7, 2}, V::Max(a, b));
    }
 
    TEST(Vec4TestsSIMD, NamedConstantsHaveExpectedComponents)
    {
        ExpectVecNear(V{0, 0, 0, 0}, V::Zero);
        ExpectVecNear(V{1, 1, 1, 1}, V::One);
        ExpectVecNear(V{1, 0, 0, 0}, V::UnitX);
        ExpectVecNear(V{0, 1, 0, 0}, V::UnitY);
        ExpectVecNear(V{0, 0, 1, 0}, V::UnitZ);
        ExpectVecNear(V{0, 0, 0, 1}, V::UnitW);
    }
 
    TEST(Vec4TestsSIMD, IsSixteenByteAligned)
    {
        static_assert(alignof(V) == 16);
        static_assert(sizeof(V) == 16);
        const V v{1, 2, 3, 4};
        EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(&v) % 16u);
    }
}
