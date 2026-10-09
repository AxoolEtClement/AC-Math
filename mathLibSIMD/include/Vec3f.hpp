#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <immintrin.h>

namespace simd
{
    class Matrix3x3; // Forward declaration to avoid circular dependency with Vec3f.
    class alignas(16) Vec3f
    {
    public:
        

        Vec3f();
        Vec3f(float x, float y, float z);

        Vec3f(const Vec3f&) = default;
        Vec3f& operator=(const Vec3f&) = default;
        Vec3f(__m128 data);

        float getX() const;
        float getY() const;
        float getZ() const;

        void setX(float value);
        void setY(float value);
        void setZ(float value);

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
        Vec3f& operator*=(float scalar);
        Vec3f& operator/=(float scalar);
        bool operator==(const Vec3f& rhs) const;
        bool operator!=(const Vec3f& rhs) const;

        float Dot(const Vec3f& rhs) const;
        Vec3f Cross(const Vec3f& rhs) const;
        float MagnitudeSquared() const;
        float Magnitude() const;
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

        friend class Matrix3x3;

    private:
        __m128 _data;
    };

    Vec3f operator*(float scalar, const Vec3f& vector);
}
