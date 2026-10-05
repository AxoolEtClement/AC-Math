#include "Mat3.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace simd
{

    Matrix3x3::Matrix3x3() : _data{}
    {
        _data[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
        _data[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
        _data[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
    }


    Matrix3x3::Matrix3x3(const std::array<float, 9>& elements) : _data{}
    {
        for (std::size_t row = 0; row < 3; ++row)
        {
            _data[row] = _mm_set_ps(0.0f,
                                    elements[row * 3 + 2],
                                    elements[row * 3 + 1],
                                    elements[row * 3 + 0]);
        }
    }


    Matrix3x3 Matrix3x3::Identity()
    {
        return Matrix3x3();
    }


    Matrix3x3 Matrix3x3::Zero()
    {
        return Matrix3x3(std::array<float, 9>{});
    }


    float& Matrix3x3::operator()(std::size_t row, std::size_t column)
    {
        if (row >= 3 || column >= 3)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return _data[row][column];
    }


    const float& Matrix3x3::operator()(std::size_t row, std::size_t column) const
    {
        if (row >= 3 || column >= 3)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return _data[row][column];
    }


    Matrix3x3 Matrix3x3::operator+(const Matrix3x3& rhs) const
    {
        Matrix3x3 result = Zero();
        for (std::size_t row = 0; row < 3; ++row)
        {
            result._data[row] = _mm_add_ps(_data[row], rhs._data[row]);
        }
        return result;
    }


    Matrix3x3 Matrix3x3::operator-(const Matrix3x3& rhs) const
    {
        Matrix3x3 result = Zero();
        for (std::size_t row = 0; row < 3; ++row)
        {
            result._data[row] = _mm_sub_ps(_data[row], rhs._data[row]);
        }
        return result;
    }


    Matrix3x3 Matrix3x3::operator*(const Matrix3x3& rhs) const
    {
        Matrix3x3 result = Zero();
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                result._data[row][column] =
                    _data[row][0] * rhs._data[0][column] +
                    _data[row][1] * rhs._data[1][column] +
                    _data[row][2] * rhs._data[2][column];
            }
        }
        return result;
    }


    Vec3f Matrix3x3::operator*(const Vec3f& rhs) const
    {
        // Masque 0x71 : multiplie les lanes 0..2 et écrit la somme dans la lane 0,
        // celle que lit _mm_cvtss_f32, pour les trois lignes.
        Vec3f result;
        result.setX(_mm_cvtss_f32(_mm_dp_ps(_data[0], rhs._data, 0x71)));
        result.setY(_mm_cvtss_f32(_mm_dp_ps(_data[1], rhs._data, 0x71)));
        result.setZ(_mm_cvtss_f32(_mm_dp_ps(_data[2], rhs._data, 0x71)));
        return result;
    }


    Matrix3x3 Matrix3x3::operator*(float scalar) const
    {
        Matrix3x3 result = Zero();
        __m128 scalarVec = _mm_set1_ps(scalar);
        for (std::size_t row = 0; row < 3; ++row)
        {
            result._data[row] = _mm_mul_ps(_data[row], scalarVec);
        }
        return result;
    }


    Matrix3x3& Matrix3x3::operator*=(const Matrix3x3& rhs)
    {
        // operator* construit un résultat séparé : sûr même si &rhs == this.
        *this = *this * rhs;
        return *this;
    }


    bool Matrix3x3::operator==(const Matrix3x3& rhs) const
    {
        for (std::size_t row = 0; row < 3; ++row)
        {
            if (_mm_movemask_ps(_mm_cmpeq_ps(_data[row], rhs._data[row])) != 0xF)
            {
                return false;
            }
        }
        return true;
    }


    bool Matrix3x3::operator!=(const Matrix3x3& rhs) const
    {
        return !(*this == rhs);
    }


    Matrix3x3 Matrix3x3::Transpose() const
    {
        __m128 r0 = _data[0];
        __m128 r1 = _data[1];
        __m128 r2 = _data[2];
        __m128 r3 = _mm_setzero_ps();
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

        Matrix3x3 result;
        result._data[0] = r0;
        result._data[1] = r1;
        result._data[2] = r2;
        return result;
    }


    float Matrix3x3::Determinant() const
    {
        Matrix3x3 working = *this;

        float determinant = working._data[0][0] * (working._data[1][1] * working._data[2][2] - working._data[1][2] * working._data[2][1]) -
                            working._data[0][1] * (working._data[1][0] * working._data[2][2] - working._data[1][2] * working._data[2][0]) +
                            working._data[0][2] * (working._data[1][0] * working._data[2][1] - working._data[1][1] * working._data[2][0]);
        return determinant;
    }


    Matrix3x3 Matrix3x3::Inverse(float relativeTolerance) const
    {
        if (!std::isfinite(relativeTolerance) || relativeTolerance < 0.0f || relativeTolerance >= 1.0f)
        {
            throw std::invalid_argument("Inverse tolerance must be finite and in [0, 1)");
        }
        Matrix3x3 left = *this;
        Matrix3x3 right;
        float scales[3]{};
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                if (!std::isfinite(left._data[row][column]))
                {
                    throw std::domain_error("Cannot invert a non-finite matrix");
                }
                scales[row] = std::max(scales[row], std::abs(left._data[row][column]));
            }
            if (scales[row] == 0.0f)
            {
                throw std::domain_error("Cannot invert a singular matrix");
            }
        }
        for (std::size_t column = 0; column < 3; ++column)
        {
            std::size_t pivot = column;
            for (std::size_t row = column + 1; row < 3; ++row)
            {
                if (std::abs(left._data[row][column]) / scales[row] >
                    std::abs(left._data[pivot][column]) / scales[pivot])
                {
                    pivot = row;
                }
            }
            if (std::abs(left._data[pivot][column]) / scales[pivot] <= relativeTolerance)
            {
                throw std::domain_error("Matrix is singular or too ill-conditioned");
            }
            for (std::size_t j = 0; j < 3; ++j)
            {
                std::swap(left._data[column][j], left._data[pivot][j]);
                std::swap(right._data[column][j], right._data[pivot][j]);
            }
            std::swap(scales[column], scales[pivot]);
            const float diagonal = left._data[column][column];
            for (std::size_t j = 0; j < 3; ++j)
            {
                left._data[column][j] /= diagonal;
                right._data[column][j] /= diagonal;
            }
            for (std::size_t row = 0; row < 3; ++row)
            {
                if (row == column)
                {
                    continue;
                }
                const float factor = left._data[row][column];
                for (std::size_t j = 0; j < 3; ++j)
                {
                    left._data[row][j] -= factor * left._data[column][j];
                    right._data[row][j] -= factor * right._data[column][j];
                }
            }
        }
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                if (!std::isfinite(right._data[row][column]))
                {
                    throw std::domain_error("Inverse is not representable");
                }
            }
        }
        return right;
    }


    Matrix3x3 Matrix3x3::Scale(const Vec3f& scale)
    {
        Matrix3x3 result;
        result._data[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, scale.getX());
        result._data[1] = _mm_set_ps(0.0f, 0.0f, scale.getY(), 0.0f);
        result._data[2] = _mm_set_ps(0.0f, scale.getZ(), 0.0f, 0.0f);
        return result;
    }


    Matrix3x3 Matrix3x3::RotationX(float radians)
    {
        Matrix3x3 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);
        result._data[1] = _mm_set_ps(0.0f, -sine, cosine, 0.0f);
        result._data[2] = _mm_set_ps(0.0f, cosine, sine, 0.0f);
        return result;
    }


    Matrix3x3 Matrix3x3::RotationY(float radians)
    {
        Matrix3x3 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);
        result._data[0] = _mm_set_ps(0.0f, sine, 0.0f, cosine);
        result._data[2] = _mm_set_ps(0.0f, cosine, 0.0f, -sine);
        return result;
    }


    Matrix3x3 Matrix3x3::RotationZ(float radians)
    {
        Matrix3x3 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);
        result._data[0] = _mm_set_ps(0.0f, 0.0f, -sine, cosine);
        result._data[1] = _mm_set_ps(0.0f, 0.0f, cosine, sine);
        return result;
    }

}