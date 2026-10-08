#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <immintrin.h>

namespace simd
{
    class alignas(16) Vec4SOA
    {
    public:
        Vec4SOA();
        Vec4SOA(const float x[4], const float y[4], const float z[4], const float w[4]);
        Vec4SOA(const Vec4SOA& other);
        Vec4SOA(__m128 x, __m128 y, __m128 z, __m128 w);

        __m128 getX() const;
        __m128 getY() const;
        __m128 getZ() const;
        __m128 getW() const;

        void setX(__m128 value);
        void setY(__m128 value);
        void setZ(__m128 value);
        void setW(__m128 value);

        Vec4SOA operator+(const Vec4SOA& rhs) const;
        Vec4SOA operator-(const Vec4SOA& rhs) const;
        Vec4SOA operator*(const Vec4SOA& rhs) const;
        Vec4SOA operator/(const Vec4SOA& rhs) const;
        Vec4SOA operator-() const;
        Vec4SOA operator*(float scalar) const;
        Vec4SOA operator/(float scalar) const;
        Vec4SOA& operator+=(const Vec4SOA& rhs);
        Vec4SOA& operator-=(const Vec4SOA& rhs);
        Vec4SOA& operator*=(const Vec4SOA& rhs);
        Vec4SOA& operator/=(const Vec4SOA& rhs);
        Vec4SOA& operator*=(float scalar);
        Vec4SOA& operator/=(float scalar);
        bool operator==(const Vec4SOA& rhs) const;
        bool operator!=(const Vec4SOA& rhs) const;

        __m128 Dot(const Vec4SOA& rhs) const;
        __m128 MagnitudeSquared() const;
        __m128 Magnitude() const;
        Vec4SOA Normalize() const;
        __m128 DistanceSquared(const Vec4SOA& rhs) const;
        __m128 Distance(const Vec4SOA& rhs) const;
        static Vec4SOA Lerp(const Vec4SOA& a, const Vec4SOA& b, float t);
        static Vec4SOA Min(const Vec4SOA& a, const Vec4SOA& b);
        static Vec4SOA Max(const Vec4SOA& a, const Vec4SOA& b);

        static const Vec4SOA Zero;
        static const Vec4SOA One;

    private:
        __m128 _dataX;
        __m128 _dataY;
        __m128 _dataZ;
        __m128 _dataW;
    };

    Vec4SOA operator*(float scalar, const Vec4SOA& vector);
}