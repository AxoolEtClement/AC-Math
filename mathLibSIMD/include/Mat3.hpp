#pragma once

#include "Vec3f.hpp"
#include <array>
#include <limits>
#include <utility>
#include <immintrin.h>

namespace simd
{
    // Row-major storage, column vectors: result = matrix * vector.
    // A * B applies B first. Angles are radians, right-handed rotations.
    class Matrix3x3
    {
    public:
        

        Matrix3x3(); // Identity.
        explicit Matrix3x3(const std::array<float, 9>& elements);
        static Matrix3x3 Identity();
        static Matrix3x3 Zero();
        float& operator()(std::size_t row, std::size_t column);
        const float& operator()(std::size_t row, std::size_t column) const;
        Matrix3x3 operator+(const Matrix3x3& rhs) const;
        Matrix3x3 operator-(const Matrix3x3& rhs) const;
        Matrix3x3 operator*(const Matrix3x3& rhs) const;
        Vec3f operator*(const Vec3f& rhs) const;
        Matrix3x3 operator*(float scalar) const;
        Matrix3x3& operator*=(const Matrix3x3& rhs);
        bool operator==(const Matrix3x3& rhs) const;
        bool operator!=(const Matrix3x3& rhs) const;
        Matrix3x3 Transpose() const;
        float Determinant() const;
        // Gauss-Jordan with scaled partial pivoting. Rejects singular/ill-conditioned input.
        Matrix3x3 Inverse(float relativeTolerance = 64.0f * std::numeric_limits<float>::epsilon()) const;
        static Matrix3x3 Scale(const Vec3f& scale);
        static Matrix3x3 RotationX(float radians);
        static Matrix3x3 RotationY(float radians);
        static Matrix3x3 RotationZ(float radians);

    private:
        std::array<__m128, 3> _data; // Each row is a __m128, with the last element unused.
    };
}