#include "Mat3.hpp"

namespace simd
{
    
    Matrix3x3::Matrix3x3() : values{}
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            values[i][i] = T{1};
        }
    }

    
    Matrix3x3::Matrix3x3(const std::array<T, 9>& elements) : values{}
    {
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                values[row][column] = elements[row * 3 + column];
            }
        }
    }

    
    Matrix3x3 Matrix3x3::Identity()
    {
        return Matrix3x3();
    }

    
    Matrix3x3 Matrix3x3::Zero()
    {
        return Matrix3x3(std::array<T, 9>{});
    }

    
    float& Matrix3x3::operator()(std::size_t row, std::size_t column)
    {
        if (row >= 3 || column >= 3)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return values[row][column];
    }

    
    const float& Matrix3x3::operator()(std::size_t row, std::size_t column) const
    {
        if (row >= 3 || column >= 3)
        {
            throw std::out_of_range("Matrix index is out of range");
        }
        return values[row][column];
    }

    
    Matrix3x3 Matrix3x3::operator+(const Matrix3x3& rhs) const
    {
        Matrix3x3 result = Zero();
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                result.values[row][column] = values[row][column] + rhs.values[row][column];
            }
        }
        return result;
    }

    
    Matrix3x3 Matrix3x3::operator-(const Matrix3x3& rhs) const
    {
        Matrix3x3 result = Zero();
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                result.values[row][column] = values[row][column] - rhs.values[row][column];
            }
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
                for (std::size_t k = 0; k < 3; ++k)
                {
                    result.values[row][column] += values[row][k] * rhs.values[k][column];
                }
            }
        }
        return result;
    }

    
    Vec3 Matrix3x3::operator*(const Vec3<T>& rhs) const
    {
        return {
            values[0][0] * rhs.x + values[0][1] * rhs.y + values[0][2] * rhs.z,
            values[1][0] * rhs.x + values[1][1] * rhs.y + values[1][2] * rhs.z,
            values[2][0] * rhs.x + values[2][1] * rhs.y + values[2][2] * rhs.z
        };
    }

    
    Matrix3x3 Matrix3x3::operator*(T scalar) const
    {
        Matrix3x3 result = Zero();
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                result.values[row][column] = values[row][column] * scalar;
            }
        }
        return result;
    }

    
    Matrix3x3& Matrix3x3::operator*=(const Matrix3x3& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    
    bool Matrix3x3::operator==(const Matrix3x3& rhs) const
    {
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                if (values[row][column] != rhs.values[row][column])
                {
                    return false;
                }
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
        Matrix3x3 result = Zero();
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                result.values[column][row] = values[row][column];
            }
        }
        return result;
    }

    
    float Matrix3x3::Determinant() const
    {
        Matrix3x3 working = *this;
        float determinant = 1.0f;
        for (std::size_t column = 0; column < 3; ++column)
        {
            std::size_t pivot = column;
            for (std::size_t row = column + 1; row < 3; ++row)
            {
                if (std::abs(working.values[row][column]) > std::abs(working.values[pivot][column]))
                {
                    pivot = row;
                }
            }
            if (working.values[pivot][column] == 0.0f)
            {
                return 0.0f;
            }
            if (pivot != column)
            {
                for (std::size_t j = 0; j < 3; ++j)
                {
                    std::swap(working.values[pivot][j], working.values[column][j]);
                }
                determinant = -determinant;
            }
            const float diagonal = working.values[column][column];
            determinant *= diagonal;
            for (std::size_t row = column + 1; row < 3; ++row)
            {
                const float factor = working.values[row][column] / diagonal;
                for (std::size_t j = column + 1; j < 3; ++j)
                {
                    working.values[row][j] -= factor * working.values[column][j];
                }
            }
        }
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
                if (!std::isfinite(left.values[row][column]))
                {
                    throw std::domain_error("Cannot invert a non-finite matrix");
                }
                scales[row] = std::max(scales[row], std::abs(left.values[row][column]));
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
                if (std::abs(left.values[row][column]) / scales[row] >
                    std::abs(left.values[pivot][column]) / scales[pivot])
                {
                    pivot = row;
                }
            }
            if (std::abs(left.values[pivot][column]) / scales[pivot] <= relativeTolerance)
            {
                throw std::domain_error("Matrix is singular or too ill-conditioned");
            }
            for (std::size_t j = 0; j < 3; ++j)
            {
                std::swap(left.values[column][j], left.values[pivot][j]);
                std::swap(right.values[column][j], right.values[pivot][j]);
            }
            std::swap(scales[column], scales[pivot]);
            const float diagonal = left.values[column][column];
            for (std::size_t j = 0; j < 3; ++j)
            {
                left.values[column][j] /= diagonal;
                right.values[column][j] /= diagonal;
            }
            for (std::size_t row = 0; row < 3; ++row)
            {
                if (row == column)
                {
                    continue;
                }
                const float factor = left.values[row][column];
                for (std::size_t j = 0; j < 3; ++j)
                {
                    left.values[row][j] -= factor * left.values[column][j];
                    right.values[row][j] -= factor * right.values[column][j];
                }
            }
        }
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                if (!std::isfinite(right.values[row][column]))
                {
                    throw std::domain_error("Inverse is not representable");
                }
            }
        }
        return right;
    }

    
    Matrix3x3 Matrix3x3::Scale(const Vec3& scale)
    {
        Matrix3x3 result;
        result.values[0][0] = scale.x;
        result.values[1][1] = scale.y;
        result.values[2][2] = scale.z;
        return result;
    }

    
    Matrix3x3 Matrix3x3::RotationX(float radians)
    {
        Matrix3x3 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);
        result.values[1][1] = cosine;
        result.values[2][2] = cosine;
        result.values[1][2] = -sine;
        result.values[2][1] = sine;
        return result;
    }

    
    Matrix3x3 Matrix3x3::RotationY(float radians)
    {
        Matrix3x3 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);
        result.values[2][2] = cosine;
        result.values[0][0] = cosine;
        result.values[2][0] = -sine;
        result.values[0][2] = sine;
        return result;
    }

    
    Matrix3x3 Matrix3x3::RotationZ(float radians)
    {
        Matrix3x3 result;
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);
        result.values[0][0] = cosine;
        result.values[1][1] = cosine;
        result.values[0][1] = -sine;
        result.values[1][0] = sine;
        return result;
    }

}