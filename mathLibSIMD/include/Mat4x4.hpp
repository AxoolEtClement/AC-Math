#pragma once

#include "Vec4.hpp"
#include "Vec3f.hpp"
#include <array>
#include <limits>
#include <utility>
#include <immintrin.h>

namespace simd
{
    // Row-major storage, column vectors: result = matrix * vector.
    // A * B applies B first. Angles are radians, right-handed rotations.
    class Matrix4x4
    {
    public:
        Matrix4x4(); // Identity.
        explicit Matrix4x4(const std::array<float, 16>& elements);
        static Matrix4x4 Identity();
        static Matrix4x4 Zero();
        float& operator()(std::size_t row, std::size_t column);
        const float& operator()(std::size_t row, std::size_t column) const;
        Matrix4x4 operator+(const Matrix4x4& rhs) const;
        Matrix4x4 operator-(const Matrix4x4& rhs) const;
        Matrix4x4 operator*(const Matrix4x4& rhs) const;
        Vec4 operator*(const Vec4& rhs) const;
        Matrix4x4 operator*(float scalar) const;
        Matrix4x4& operator*=(const Matrix4x4& rhs);
        bool operator==(const Matrix4x4& rhs) const;
        bool operator!=(const Matrix4x4& rhs) const;
        Matrix4x4 Transpose() const;
        float Determinant() const;
        // Gauss-Jordan with scaled partial pivoting. Rejects singular/ill-conditioned input.
        Matrix4x4 Inverse(float relativeTolerance = float{ 64 } *std::numeric_limits<float>::epsilon()) const;
        static Matrix4x4 Scale(const Vec3f& scale);
        static Matrix4x4 RotationX(float radians);
        static Matrix4x4 RotationY(float radians);
        static Matrix4x4 RotationZ(float radians);
        static Matrix4x4 Translation(const Vec3f& offset);
        Vec3f TransformPoint(const Vec3f& point) const; // Affine only, w = 1.
        Vec3f TransformDirection(const Vec3f& direction) const; // Affine only, w = 0.

    private:
        std::array<__m128, 4> _data;
    };
}
