#include "Vec3f.hpp"

namespace simd {
    Vec3f::Vec3f() : _data{ _mm_setzero_ps() }
    {
    }

    Vec3f::Vec3f(float x, float y, float z) : _data{ _mm_set_ps(0.f, z, y, x) }
    {
    }

    Vec3f::Vec3f(const Vec3f& other) : _data{ other._data }
    {
    }

    Vec3f::Vec3f(__m128 data) : _data{ data }
    {
    }

    float Vec3f::getX() const { return _mm_cvtss_f32(_data); }
    float Vec3f::getY() const { return _mm_cvtss_f32(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(1, 1, 1, 1))); }
    float Vec3f::getZ() const { return _mm_cvtss_f32(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(2, 2, 2, 2))); }

    // _mm_insert_ps(dst, src, imm8) : bits 5:4 = lane de destination, bits 7:6 = lane source (0 ici)
    void Vec3f::setX(float value) { _data = _mm_move_ss(_data, _mm_set_ss(value)); }
    void Vec3f::setY(float value) { _data = _mm_insert_ps(_data, _mm_set_ss(value), 0x10); }
    void Vec3f::setZ(float value) { _data = _mm_insert_ps(_data, _mm_set_ss(value), 0x20); }

    Vec3f Vec3f::operator+(const Vec3f& rhs) const
    {
        return { _mm_add_ps(_data, rhs._data) };
    }

    Vec3f Vec3f::operator-(const Vec3f& rhs) const
    {
        return { _mm_sub_ps(_data, rhs._data) };
    }

    Vec3f Vec3f::operator*(const Vec3f& rhs) const
    {
        return { _mm_mul_ps(_data, rhs._data) };
    }

    Vec3f Vec3f::operator/(const Vec3f& rhs) const
    {
        if (rhs.getX() == 0.f || rhs.getY() == 0.f || rhs.getZ() == 0.f)
        {
            throw std::domain_error("Cannot divide by a zero component");
        }

        return { _mm_div_ps(_data, rhs._data) };
    }

    Vec3f Vec3f::operator-() const
    {
        return { _mm_sub_ps(_mm_setzero_ps(), _data) };
    }

    Vec3f Vec3f::operator*(float scalar) const
    {
        return { _mm_mul_ps(_data, _mm_set1_ps(scalar)) };
    }

    Vec3f Vec3f::operator/(float scalar) const
    {
        if (scalar == 0.f)
        {
            throw std::domain_error("Cannot divide by zero");
        }

        return { _mm_div_ps(_data, _mm_set1_ps(scalar)) };
    }

    Vec3f& Vec3f::operator+=(const Vec3f& rhs)
    {
        _data = _mm_add_ps(_data, rhs._data);
        return *this;
    }

    Vec3f& Vec3f::operator-=(const Vec3f& rhs)
    {
        _data = _mm_sub_ps(_data, rhs._data);
        return *this;
    }

    Vec3f& Vec3f::operator*=(const Vec3f& rhs)
    {
        _data = _mm_mul_ps(_data, rhs._data);
        return *this;
    }

    Vec3f& Vec3f::operator/=(const Vec3f& rhs)
    {
        if (rhs.getX() == 0.f || rhs.getY() == 0.f || rhs.getZ() == 0.f)
        {
            throw std::domain_error("Cannot divide by a zero component");
        }
        _data = _mm_div_ps(_data, rhs._data);
        return *this;
    }

    Vec3f& Vec3f::operator*=(float scalar)
    {
        _data = _mm_mul_ps(_data, _mm_set1_ps(scalar));
        return *this;
    }

    Vec3f& Vec3f::operator/=(float scalar)
    {
        if (scalar == 0.f)
        {
            throw std::domain_error("Cannot divide by zero");
        }
        _data = _mm_div_ps(_data, _mm_set_ps1(scalar));
        return *this;
    }

    bool Vec3f::operator==(const Vec3f& rhs) const
    {
        return _mm_movemask_ps(_mm_cmpeq_ps(_data, rhs._data)) == 0xF;
    }

    bool Vec3f::operator!=(const Vec3f& rhs) const
    {
        return _mm_movemask_ps(_mm_cmpeq_ps(_data, rhs._data)) != 0xF;
    }

    float Vec3f::Dot(const Vec3f& rhs) const
    {
        return _mm_cvtss_f32(_mm_dp_ps(_data, rhs._data, 0x7F));
    }

    Vec3f Vec3f::Cross(const Vec3f& rhs) const
    {
        return { _mm_sub_ps(
            _mm_mul_ps(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(rhs._data, rhs._data, _MM_SHUFFLE(3, 1, 0, 2))), 
            _mm_mul_ps(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(rhs._data, rhs._data, _MM_SHUFFLE(3, 0, 2, 1)))
        ) };


        //return { y * rhs.z - z * rhs.y,
        //        z * rhs.x - x * rhs.z,
        //        x * rhs.y - y * rhs.x };
    }

    float Vec3f::MagnitudeSquared() const
    {
        return _mm_cvtss_f32(_mm_dp_ps(_data, _data, 0x7F));
    }

    float Vec3f::Magnitude() const
    {
        return _mm_cvtss_f32(_mm_sqrt_ps(_mm_dp_ps(_data, _data, 0x7F)));
    }

    Vec3f Vec3f::Normalize() const
    {
        if (!std::isfinite(this->getX()) || !std::isfinite(this->getY()) || !std::isfinite(this->getZ()))
        {
            throw std::domain_error("Cannot normalize non-finite components");
        }

        // Scaling avoids overflowing/underflowing the squared magnitude.
        const float scale = std::max({ std::abs(this->getX()), std::abs(this->getY()), std::abs(this->getZ()) });
        if (scale == 0.f)
        {
            throw std::domain_error("Cannot normalize the zero vector");
        }

        __m128 scaled = _mm_div_ps(_data, _mm_set_ps1(scale));
        return { _mm_div_ps(scaled, _mm_sqrt_ps(_mm_dp_ps(scaled, scaled, 0x7F))) };
    }

    float Vec3f::DistanceSquared(const Vec3f& rhs) const
    {
        __m128 diff{ _mm_sub_ps(_data, rhs._data) };
        return _mm_cvtss_f32(_mm_dp_ps(diff, diff, 0x71));
    }

    float Vec3f::Distance(const Vec3f& rhs) const
    {
        __m128 diff{ _mm_sub_ps(_data, rhs._data) };
        return _mm_cvtss_f32(_mm_sqrt_ps(_mm_dp_ps(diff, diff, 0x71)));
    }

    float Vec3f::Angle(const Vec3f& rhs) const
    {
        __m128 dot_vec = _mm_dp_ps(_data, rhs._data, 0x71);
        __m128 mag_vec = _mm_sqrt_ps(_mm_mul_ps(_mm_dp_ps(_data, _data, 0x71), _mm_dp_ps(rhs._data, rhs._data, 0x71)));

        float dot = _mm_cvtss_f32(dot_vec);
        float mag = _mm_cvtss_f32(mag_vec);

        if (mag < 1e-6f)
        {
            throw std::domain_error("Cannot compute angle with a zero vector.");
        }

        float cos_theta = std::clamp(dot / mag, -1.0f, 1.0f);

        return std::acos(cos_theta);
    }

    Vec3f Vec3f::Lerp(const Vec3f& a, const Vec3f& b, float t)
    {
        return { _mm_add_ps(a._data, _mm_mul_ps(_mm_sub_ps(b._data, a._data), _mm_set_ps1(t))) };
    }

    Vec3f Vec3f::Min(const Vec3f& a, const Vec3f& b)
    {
        return { _mm_min_ps(a._data, b._data) };
    }

    Vec3f Vec3f::Max(const Vec3f& a, const Vec3f& b)
    {
        return { _mm_max_ps(a._data, b._data) };
    }

    const Vec3f Vec3f::Zero{ 0, 0, 0 };

    const Vec3f Vec3f::One{ 1, 1, 1 };

    const Vec3f Vec3f::UnitX{ 1, 0, 0 };

    const Vec3f Vec3f::UnitY{ 0, 1, 0 };

    const Vec3f Vec3f::UnitZ{ 0, 0, 1 };

    Vec3f operator*(float scalar, const Vec3f& vector)
    {
        return vector * scalar;
    }
}
