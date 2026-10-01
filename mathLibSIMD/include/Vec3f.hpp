#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <immintrin.h>

namespace simd
{
    // Scalar teaching baseline. No SIMD, custom alignment or forced inlining.
    class Vec3f
    {
    public:
        float x;
        float y;
        float z;

        Vec3f();
        Vec3f(float _x, float _y, float _z);

        explicit Vec3f(const Vec3f& other);

        Vec3f operator+(const Vec3f& rhs) const;
        Vec3f operator-(const Vec3f& rhs) const;
        Vec3f operator*(const Vec3f& rhs) const;
        Vec3f operator/(const Vec3f& rhs) const;
        Vec3f operator-() const;
        Vec3f operator*(float scalar) const;
        Vec3f operator/(float scalar) const;
        Vec3f& operator+=(const Vec3f& rhs);
        Vec3f& operator-=(const Vec3f& rhs);
        Vec3f& operator*=(const Vec3f& rhs);
        Vec3f& operator/=(const Vec3f& rhs);
        Vec3f& operator*=(T scalar);
        Vec3f& operator/=(T scalar);
        bool operator==(const Vec3f& rhs) const;
        bool operator!=(const Vec3f& rhs) const;

        float Dot(const Vec3f& rhs) const;
        Vec3 Cross(const Vec3f& rhs) const;
        float MagnitudeSquared() const;
        float Magnitude() const;
        // Returns a new vector. Throws domain_error for zero or non-finite input.
        Vec3f Normalize() const;
        float DistanceSquared(const Vec3f& rhs) const;
        float Distance(const Vec3f& rhs) const;
        float Angle(const Vec3f& rhs) const;
        static Vec3f Lerp(const Vec3f& a, const Vec3f& b, float t);
        static Vec3f Min(const Vec3f& a, const Vec3f& b);
        static Vec3f Max(const Vec3f& a, const Vec3f& b);

        static const Vec3f Zero;
        static const Vec3f One;
        static const Vec3f UnitX;
        static const Vec3f UnitY;
        static const Vec3f UnitZ;

    private:
        inline __m128 Load128(const Vec3f& v);
        inline Vec3f Store128(__m128 val);
    };
}
