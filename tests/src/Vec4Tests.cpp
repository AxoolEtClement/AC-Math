#include <gtest/gtest.h>
#include "TestHelpers.h"
#include "mathLibCPP/Vec4.h"
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace TestHelpers;

namespace MathStarterTests
{
    template <typename T>
    class Vec4Tests : public ::testing::Test {};

    using FloatingPointTypes = ::testing::Types<float, double>;
    TYPED_TEST_SUITE(Vec4Tests, FloatingPointTypes);

    TYPED_TEST(Vec4Tests, DefaultConstructorIsZero)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V actual;
        VectorNear(V{0, 0, 0, 0}, actual);
    }

    TYPED_TEST(Vec4Tests, ComponentsAndCopyAreIndependent)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V original{1, 2, 3, 4};
        V copy = original;
        copy.x = T{99};
        VectorNear(V{1, 2, 3, 4}, original);
        EXPECT_TRUE(copy.x == T{99});
    }

    TYPED_TEST(Vec4Tests, ExplicitConversionPreservesValues)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const Maths::Vec4<double> source{1.5, 2.5, 3.5, 4.5};
        const V actual(source);
        VectorNear(V{T{1.5}, T{2.5}, T{3.5}, T{4.5}}, actual);
    }

    TYPED_TEST(Vec4Tests, AdditionHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{1, 2, 3, 4};
        const V b = V{2, 3, 4, 5};
        const V actual = a + b;
        VectorNear(V{3, 5, 7, 9}, actual);
    }

    TYPED_TEST(Vec4Tests, SubtractionHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{1, 2, 3, 4};
        const V b = V{2, 3, 4, 5};
        const V actual = a - b;
        VectorNear(V{-1, -1, -1, -1}, actual);
    }

    TYPED_TEST(Vec4Tests, ComponentProductHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{1, 2, 3, 4};
        const V b = V{2, 3, 4, 5};
        const V actual = a * b;
        VectorNear(V{2, 6, 12, 20}, actual);
    }

    TYPED_TEST(Vec4Tests, ComponentDivisionHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{2, 6, 12, 20};
        const V b = V{2, 3, 4, 5};
        const V actual = a / b;
        VectorNear(V{1, 2, 3, 4}, actual);
    }

    TYPED_TEST(Vec4Tests, ScalarOperationsAndNegation)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{1, 2, 3, 4};
        VectorNear(V{-1, -2, -3, -4}, -a);
        VectorNear(V{2, 4, 6, 8}, a * T{2});
        VectorNear(V{2, 4, 6, 8}, T{2} * a);
        VectorNear(a, (a * T{2}) / T{2});
    }

    TYPED_TEST(Vec4Tests, CompoundOperatorsReturnSelfAndSupportAliasing)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        V a = V{1, 2, 3, 4};
        EXPECT_TRUE(&(a += a) == &a);
        VectorNear(V{2, 4, 6, 8}, a);
        a -= V{1, 2, 3, 4};
        a *= a;
        VectorNear(V{1, 4, 9, 16}, a);
        a /= V{1, 2, 3, 4};
        a *= T{2};
        a /= T{2};
        VectorNear(V{1, 2, 3, 4}, a);
    }

    TYPED_TEST(Vec4Tests, DivisionByZeroThrowsWithoutPartialMutation)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V original = V{1, 2, 3, 4};
        V actual = original;
        EXPECT_THROW(actual /= T{0}, std::domain_error);
        VectorNear(original, actual);
        EXPECT_THROW((actual /= V{0, 1, 1, 1}), std::domain_error);
        VectorNear(original, actual);
        EXPECT_THROW((actual /= V{1, 0, 1, 1}), std::domain_error);
        VectorNear(original, actual);
        EXPECT_THROW((actual /= V{1, 1, 0, 1}), std::domain_error);
        VectorNear(original, actual);
        EXPECT_THROW((actual /= V{1, 1, 1, 0}), std::domain_error);
        VectorNear(original, actual);
    }

    TYPED_TEST(Vec4Tests, EqualityChecksEveryComponentExactly)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{1, 2, 3, 4};
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
        V changedw = a;
        changedw.w += T{1};
        EXPECT_TRUE(changedw != a);
        V nan = a;
        nan.x = std::numeric_limits<T>::quiet_NaN();
        EXPECT_FALSE(nan == nan);
    }

    TYPED_TEST(Vec4Tests, DotHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{1, 2, 3, 4};
        const V b = V{2, 3, 4, 5};
        Near(T{40}, a.Dot(b));
    }

    TYPED_TEST(Vec4Tests, MagnitudeHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{3, 4, 0, 0};
        Near(T{25}, a.MagnitudeSquared());
        Near(T{5}, a.Magnitude());
        Near(T{0}, V{}.Magnitude());
    }

    TYPED_TEST(Vec4Tests, NormalizeReturnsNewUnitVector)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{3, 4, 0, 0};
        const V actual = a.Normalize();
        VectorNear(V{T{0.6}, T{0.8}, T{0}, T{0}}, actual);
        Near(T{1}, actual.Magnitude());
        VectorNear(V{3, 4, 0, 0}, a);
    }

    TYPED_TEST(Vec4Tests, NormalizeRejectsZeroAndNonFinite)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        EXPECT_THROW((void)V{}.Normalize(), std::domain_error);
        V invalid = V{1, 2, 3, 4};
        invalid.x = std::numeric_limits<T>::infinity();
        EXPECT_THROW((void)invalid.Normalize(), std::domain_error);
        invalid.x = std::numeric_limits<T>::quiet_NaN();
        EXPECT_THROW((void)invalid.Normalize(), std::domain_error);
    }

    TYPED_TEST(Vec4Tests, NormalizeHandlesVeryLargeAndSmallFiniteValues)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const T large = std::numeric_limits<T>::max();
        const T small = std::numeric_limits<T>::min();
        const V expected = V{T{1}, T{1}, T{1}, T{1}} / std::sqrt(T{4});
        VectorNear(expected, V{large, large, large, large}.Normalize());
        VectorNear(expected, V{small, small, small, small}.Normalize());
    }

    TYPED_TEST(Vec4Tests, DistanceHasKnownResult)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{1, 2, 3, 3};
        const V b = V{4, 6, 3, 3};
        Near(T{25}, a.DistanceSquared(b));
        Near(T{5}, a.Distance(b));
    }

    TYPED_TEST(Vec4Tests, AngleUsesRadians)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        Near(std::numbers::pi_v<T> / T{2}, V::UnitX.Angle(V::UnitY));
        Near(std::numbers::pi_v<T>, V::UnitX.Angle(-V::UnitX));
        Near(T{0}, V::UnitX.Angle(V::UnitX));
        EXPECT_THROW((void)V::Zero.Angle(V::UnitX), std::domain_error);
    }

    TYPED_TEST(Vec4Tests, LerpSupportsEndpointsMidpointAndExtrapolation)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{0, 0, 0, 0};
        const V b = V{2, 4, 6, 8};
        VectorNear(a, V::Lerp(a, b, T{0}));
        VectorNear(b, V::Lerp(a, b, T{1}));
        VectorNear(V{1, 2, 3, 4}, V::Lerp(a, b, T{0.5}));
        VectorNear(V{4, 8, 12, 16}, V::Lerp(a, b, T{2}));
    }

    TYPED_TEST(Vec4Tests, MinMaxAreComponentWise)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        const V a = V{1, -4, 7, 2};
        const V b = V{3, -2, 5, -8};
        VectorNear(V{1, -4, 5, -8}, V::Min(a, b));
        VectorNear(V{3, -2, 7, 2}, V::Max(a, b));
    }

    TYPED_TEST(Vec4Tests, NamedConstantsHaveExpectedComponents)
    {
        using T = TypeParam;
        using V = Maths::Vec4<T>;
        VectorNear(V{0, 0, 0, 0}, V::Zero);
        VectorNear(V{1, 1, 1, 1}, V::One);
        VectorNear(V{1, 0, 0, 0}, V::UnitX);
        VectorNear(V{0, 1, 0, 0}, V::UnitY);
        VectorNear(V{0, 0, 1, 0}, V::UnitZ);
        VectorNear(V{0, 0, 0, 1}, V::UnitW);
    }
}
