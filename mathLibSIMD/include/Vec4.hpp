#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <immintrin.h>



namespace simd
{
    class Matrix4x4;
    class alignas(16) Vec4
    {
    public:

        Vec4();
        Vec4(float _x, float _y, float _z, float _w);
        Vec4(const Vec4& other);
        Vec4(__m128 data);

        float getX() const;
        float getY() const;
        float getZ() const;
        float getW() const;
        void setX(float value);
        void setY(float value);
        void setZ(float value);
        void setW(float value);
        //static __m128 load(const Vec4& v);
        static Vec4 store(__m128 val);

        Vec4 operator+(const Vec4& rhs) const;
        Vec4 operator-(const Vec4& rhs) const;
        Vec4 operator*(const Vec4& rhs) const;
        Vec4 operator/(const Vec4& rhs) const;
        Vec4 operator-() const;
        Vec4 operator*(float scalar) const;
        Vec4 operator/(float scalar) const;
        Vec4& operator+=(const Vec4& rhs);
        Vec4& operator-=(const Vec4& rhs);
        Vec4& operator*=(const Vec4& rhs);
        Vec4& operator/=(const Vec4& rhs);
        Vec4& operator*=(float scalar);
        Vec4& operator/=(float scalar);
        bool operator==(const Vec4& rhs) const;
        bool operator!=(const Vec4& rhs) const;

        float Dot(const Vec4& rhs) const;
        float MagnitudeSquared() const;
        float Magnitude() const;
        // Returns a new vector. Throws domain_error for zero or non-finite input.
        Vec4 Normalize() const;
        float DistanceSquared(const Vec4& rhs) const;
        float Distance(const Vec4& rhs) const;
        float Angle(const Vec4& rhs) const;
        static Vec4 Lerp(const Vec4& a, const Vec4& b, float t);
        static Vec4 Min(const Vec4& a, const Vec4& b);
        static Vec4 Max(const Vec4& a, const Vec4& b);

        static const Vec4 Zero;
        static const Vec4 One;
        static const Vec4 UnitX;
        static const Vec4 UnitY;
        static const Vec4 UnitZ;
        static const Vec4 UnitW;

        friend class Matrix4x4;

    private:
        __m128 _data;
    };
    Vec4 operator*(float scalar, const Vec4& vector);
    
}