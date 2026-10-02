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
        if (rhs.x == 0.f || rhs.y == 0.f || rhs.z == 0.f)
        {
            throw std::domain_error("Cannot divide by a zero component");
        }

        return { _mm_div_ps(_data, rhs._data) };
    }

    Vec3f Vec3f::operator-() const
    {
        return { _mm_div_ps(_mm_setzero_ps(), _data) };
    }

    Vec3f Vec3f::operator*(float scalar) const
    {
        return { _mm_mul_ps(_data, _mm_set_ps1(scalar)) };
    }

    Vec3f Vec3f::operator/(float scalar) const
    {
        if (scalar == 0.f)
        {
            throw std::domain_error("Cannot divide by zero");
        }

        return { _mm_div_ps(_data, _mm_set_ps1(scalar)) };
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
        _data = _mm_div_ps(_data, rhs._data);
        return *this;
    }

    Vec3f& Vec3f::operator*=(float scalar)
    {
        _data = _mm_mul_ps(_data, _mm_set_ps1(scalar));
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
        return _mm_cvtss_f32(_mm_dp_ps(_data, rhs._data));
    }

    Vec3f Vec3f::Cross(const Vec3f& rhs) const
    {
        return { _mm_sub_ps(
            _mm_mul_ps(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(_data, _data, _MM_SHUFFLE(3, 1, 0, 2))), 
            _mm_mul_ps(_mm_shuffle_ps(_data, _data, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(_data, _data, _MM_SHUFFLE(3, 0, 2, 1)))
        ) };


        //return { y * rhs.z - z * rhs.y,
        //        z * rhs.x - x * rhs.z,
        //        x * rhs.y - y * rhs.x };
    }

    float Vec3f::MagnitudeSquared() const
    {
        return _mm_cvtss_f32(_mm_dp_ps(_data, _data));
    }

    float Vec3f::Magnitude() const
    {
        return _mm_cvtss_f32(_mm_sqrt_ps(_mm_dp_ps(_data, _data)));
    }

    Vec3f Vec3f::Normalize() const
    {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            throw std::domain_error("Cannot normalize non-finite components");
        }

        // Scaling avoids overflowing/underflowing the squared magnitude.
        const float scale = std::max({ std::abs(x), std::abs(y), std::abs(z) });
        if (scale == 0.f)
        {
            throw std::domain_error("Cannot normalize the zero vector");
        }

        __m128 scaled = _mm_div_ps(_data, _mm_set_ps1(scalar));
        return { _mm_div_ps(scaled, _mm_sqrt_ps(_mm_dp_ps(scaled, scaled))) };
    }

    float Vec3f::DistanceSquared(const Vec3f& rhs) const
    {
        __m128 diff{ _mm_sub_ps(_data, rhs._data) };
        return _mm_cvtss_f32(_mm_dp_ps(diff, diff));
    }

    float Vec3f::Distance(const Vec3f& rhs) const
    {
        __m128 diff{ _mm_sub_ps(_data, rhs._data) };
        return _mm_cvtss_f32(_mm_sqrt_ps(_mm_dp_ps(diff, diff)));
    }

    float Vec3f::Angle(const Vec3f& rhs) const
    {
        const float cosine = Normalize().Dot(rhs.Normalize());
        return std::acos(std::clamp(cosine, -1.f, 1.f));
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

    inline __m128 Vec3f::Load128(const Vec3f& v)
    {
        return _mm_setr_ps(v.x, v.y, v.z, 0.f);
    }

    inline Vec3f Vec3f::Store128(__m128 val)
    {
        float components[4];

        _mm_storeu_ps(components, val);

        return {
            components[0],
            components[1],
            components[2]
        };
    }
}
