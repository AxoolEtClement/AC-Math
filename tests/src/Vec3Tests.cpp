#include <gtest/gtest.h>
#include "TestHelpers.h"
#include "mathLibCPP/Vec3.h"
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace TestHelpers;

namespace MathStarterTests
{
    template <typename T>
    class Vec3Tests : public ::testing::Test {};

    using FloatingPointTypes = ::testing::Types<float, double>;
    TYPED_TEST_SUITE(Vec3Tests, FloatingPointTypes);

    TYPED_TEST(Vec3Tests, DefaultConstructorIsZero)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V actual;
        VectorNear(V{0, 0, 0}, actual);
    }

    TYPED_TEST(Vec3Tests, ComponentsAndCopyAreIndependent)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V original{1, 2, 3};
        V copy = original;
        copy.x = T{99};
        VectorNear(V{1, 2, 3}, original);
        EXPECT_TRUE(copy.x == T{99});
    }

    TYPED_TEST(Vec3Tests, ExplicitConversionPreservesValues)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const Maths::Vec3<double> source{1.5, 2.5, 3.5};
        const V actual(source);
        VectorNear(V{T{1.5}, T{2.5}, T{3.5}}, actual);
    }

    TYPED_TEST(Vec3Tests, AdditionHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{1, 2, 3};
        const V b = V{2, 3, 4};
        const V actual = a + b;
        VectorNear(V{3, 5, 7}, actual);
    }

    TYPED_TEST(Vec3Tests, SubtractionHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{1, 2, 3};
        const V b = V{2, 3, 4};
        const V actual = a - b;
        VectorNear(V{-1, -1, -1}, actual);
    }

    TYPED_TEST(Vec3Tests, ComponentProductHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{1, 2, 3};
        const V b = V{2, 3, 4};
        const V actual = a * b;
        VectorNear(V{2, 6, 12}, actual);
    }

    TYPED_TEST(Vec3Tests, ComponentDivisionHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{2, 6, 12};
        const V b = V{2, 3, 4};
        const V actual = a / b;
        VectorNear(V{1, 2, 3}, actual);
    }

    TYPED_TEST(Vec3Tests, ScalarOperationsAndNegation)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{1, 2, 3};
        VectorNear(V{-1, -2, -3}, -a);
        VectorNear(V{2, 4, 6}, a * T{2});
        VectorNear(V{2, 4, 6}, T{2} * a);
        VectorNear(a, (a * T{2}) / T{2});
    }

    TYPED_TEST(Vec3Tests, CompoundOperatorsReturnSelfAndSupportAliasing)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        V a = V{1, 2, 3};
        EXPECT_TRUE(&(a += a) == &a);
        VectorNear(V{2, 4, 6}, a);
        a -= V{1, 2, 3};
        a *= a;
        VectorNear(V{1, 4, 9}, a);
        a /= V{1, 2, 3};
        a *= T{2};
        a /= T{2};
        VectorNear(V{1, 2, 3}, a);
    }

    TYPED_TEST(Vec3Tests, DivisionByZeroThrowsWithoutPartialMutation)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V original = V{1, 2, 3};
        V actual = original;
        EXPECT_THROW(actual /= T{0}, std::domain_error);
        VectorNear(original, actual);
        EXPECT_THROW((actual /= V{0, 1, 1}), std::domain_error);
        VectorNear(original, actual);
        EXPECT_THROW((actual /= V{1, 0, 1}), std::domain_error);
        VectorNear(original, actual);
        EXPECT_THROW((actual /= V{1, 1, 0}), std::domain_error);
        VectorNear(original, actual);
    }

    TYPED_TEST(Vec3Tests, EqualityChecksEveryComponentExactly)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{1, 2, 3};
        EXPECT_TRUE(a == a);
        V changedx = a;
        changedx.x += T{1};
        EXPECT_TRUE(changedx != a);
        V changedy = a;
        changedy.y += T{1};
        EXPECT_TRUE(changedy != a);
        V changedz = a;
        changedz.z += T{1};
        EXPECT_TRUE(changedz != a);
        V nan = a;
        nan.x = std::numeric_limits<T>::quiet_NaN();
        EXPECT_FALSE(nan == nan);
    }

    TYPED_TEST(Vec3Tests, DotHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{1, 2, 3};
        const V b = V{2, 3, 4};
        Near(T{20}, a.Dot(b));
    }

    TYPED_TEST(Vec3Tests, MagnitudeHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{3, 4, 0};
        Near(T{25}, a.MagnitudeSquared());
        Near(T{5}, a.Magnitude());
        Near(T{0}, V{}.Magnitude());
    }

    TYPED_TEST(Vec3Tests, NormalizeReturnsNewUnitVector)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{3, 4, 0};
        const V actual = a.Normalize();
        VectorNear(V{T{0.6}, T{0.8}, T{0}}, actual);
        Near(T{1}, actual.Magnitude());
        VectorNear(V{3, 4, 0}, a);
    }

    TYPED_TEST(Vec3Tests, NormalizeRejectsZeroAndNonFinite)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        EXPECT_THROW((void)V{}.Normalize(), std::domain_error);
        V invalid = V{1, 2, 3};
        invalid.x = std::numeric_limits<T>::infinity();
        EXPECT_THROW((void)invalid.Normalize(), std::domain_error);
        invalid.x = std::numeric_limits<T>::quiet_NaN();
        EXPECT_THROW((void)invalid.Normalize(), std::domain_error);
    }

    TYPED_TEST(Vec3Tests, NormalizeHandlesVeryLargeAndSmallFiniteValues)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const T large = std::numeric_limits<T>::max();
        const T small = std::numeric_limits<T>::min();
        const V expected = V{T{1}, T{1}, T{1}} / std::sqrt(T{3});
        VectorNear(expected, V{large, large, large}.Normalize());
        VectorNear(expected, V{small, small, small}.Normalize());
    }

    TYPED_TEST(Vec3Tests, DistanceHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{1, 2, 3};
        const V b = V{4, 6, 3};
        Near(T{25}, a.DistanceSquared(b));
        Near(T{5}, a.Distance(b));
    }

    TYPED_TEST(Vec3Tests, AngleUsesRadians)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        Near(std::numbers::pi_v<T> / T{2}, V::UnitX.Angle(V::UnitY));
        Near(std::numbers::pi_v<T>, V::UnitX.Angle(-V::UnitX));
        Near(T{0}, V::UnitX.Angle(V::UnitX));
        EXPECT_THROW((void)V::Zero.Angle(V::UnitX), std::domain_error);
    }

    TYPED_TEST(Vec3Tests, LerpSupportsEndpointsMidpointAndExtrapolation)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{0, 0, 0};
        const V b = V{2, 4, 6};
        VectorNear(a, V::Lerp(a, b, T{0}));
        VectorNear(b, V::Lerp(a, b, T{1}));
        VectorNear(V{1, 2, 3}, V::Lerp(a, b, T{0.5}));
        VectorNear(V{4, 8, 12}, V::Lerp(a, b, T{2}));
    }

    TYPED_TEST(Vec3Tests, MinMaxAreComponentWise)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        const V a = V{1, -4, 7};
        const V b = V{3, -2, 5};
        VectorNear(V{1, -4, 5}, V::Min(a, b));
        VectorNear(V{3, -2, 7}, V::Max(a, b));
    }

    TYPED_TEST(Vec3Tests, NamedConstantsHaveExpectedComponents)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        VectorNear(V{0, 0, 0}, V::Zero);
        VectorNear(V{1, 1, 1}, V::One);
        VectorNear(V{1, 0, 0}, V::UnitX);
        VectorNear(V{0, 1, 0}, V::UnitY);
        VectorNear(V{0, 0, 1}, V::UnitZ);
    }

    TYPED_TEST(Vec3Tests, CrossUsesRightHandedOrientation)
    {
        using T = TypeParam;
        using V = Maths::Vec3<T>;
        VectorNear(V{0, 0, 1}, V::UnitX.Cross(V::UnitY));
        VectorNear(V{0, 0, -1}, V::UnitY.Cross(V::UnitX));
        const V a{1, 2, 3};
        const V b{4, 5, 6};
        VectorNear(V{-3, 6, -3}, a.Cross(b));
        VectorNear(V::Zero, a.Cross(a));
    }
}
