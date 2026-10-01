#include "Vec3f.hpp"

namespace simd {
    Vec3f::Vec3f() : x(0), y(0), z(0)
    {

    }

    Vec3f::Vec3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z)
    {

    }

    Vec3f::Vec3f(const Vec3f& other) : x(other.x), y(other.y), z(other.z)
    {

    }

    Vec3f Vec3f::operator+(const Vec3f& rhs) const
    {
        return Store128(_mm_add_ps(Load128(this*), Load128(rhs));
    }

    Vec3f Vec3f::operator-(const Vec3f& rhs) const
    {
        return Store128(_mm_sub_ps(Load128(this*), Load128(rhs));
    }

    Vec3f Vec3f::operator*(const Vec3f& rhs) const
    {
        return Store128(_mm_mul_ps(Load128(this*), Load128(rhs));
    }

    Vec3f Vec3f::operator/(const Vec3f& rhs) const
    {
        if (rhs.x == 0.f || rhs.y == 0.f || rhs.z == 0.f)
        {
            throw std::domain_error("Cannot divide by a zero component");
        }

        return Store128(_mm_div_ps(Load128(this*), Load128(rhs));
    }

    Vec3f Vec3f::operator-() const
    {
        return Store128(_mm_div_ps(_mm_load_ps1(0.f), Load128(this*));
    }

    Vec3f Vec3f::operator*(float scalar) const
    {
        return Store128(_mm_mul_ps(Load128(this*), _mm_set1_ps(scalar)));
    }

    Vec3f Vec3f::operator/(float scalar) const
    {
        if (scalar == 0.f)
        {
            throw std::domain_error("Cannot divide by zero");
        }

        return Store128(_mm_div_ps(Load128(this*), _mm_set1_ps(scalar)));
    }

    Vec3f& Vec3f::operator+=(const Vec3f& rhs)
    {
        *this = Store128(_mm_add_ps(Load128(this*), Load128(rhs));
        return *this;
    }

    Vec3f& Vec3f::operator-=(const Vec3f& rhs)
    {
        *this = Store128(_mm_sub_ps(Load128(this*), Load128(rhs));
        return *this;
    }

    Vec3f& Vec3f::operator*=(const Vec3f& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    Vec3f& Vec3f::operator/=(const Vec3f& rhs)
    {
        *this = *this / rhs;
        return *this;
    }

    Vec3f& Vec3f::operator*=(float scalar)
    {
        *this = *this * scalar;
        return *this;
    }

    Vec3f& Vec3f::operator/=(float scalar)
    {
        *this = *this / scalar;
        return *this;
    }

    bool Vec3f::operator==(const Vec3f& rhs) const
    {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }

    bool Vec3f::operator!=(const Vec3f& rhs) const
    {
        return !(*this == rhs);
    }

    T Vec3f::Dot(const Vec3f& rhs) const
    {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    Vec3f Vec3f::Cross(const Vec3f& rhs) const
    {
        return { y * rhs.z - z * rhs.y,
                z * rhs.x - x * rhs.z,
                x * rhs.y - y * rhs.x };
    }

    float Vec3f::MagnitudeSquared() const
    {
        return Dot(*this);
    }

    float Vec3f::Magnitude() const
    {
        return std::hypot(x, y, z);
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

        const Vec3f scaled = *this / scale;
        return scaled / scaled.Magnitude();
    }

    float Vec3f::DistanceSquared(const Vec3f& rhs) const
    {
        return (*this - rhs).MagnitudeSquared();
    }

    float Vec3f::Distance(const Vec3f& rhs) const
    {
        return (*this - rhs).Magnitude();
    }

    float Vec3f::Angle(const Vec3f& rhs) const
    {
        const float cosine = Normalize().Dot(rhs.Normalize());
        return std::acos(std::clamp(cosine, -1.f, 1.f));
    }

    Vec3f Vec3f::Lerp(const Vec3f& a, const Vec3f& b, float t)
    {
        return a * (1.f - t) + b * t;
    }

    Vec3f Vec3f::Min(const Vec3f& a, const Vec3f& b)
    {
        return { std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) };
    }

    Vec3f Vec3f::Max(const Vec3f& a, const Vec3f& b)
    {
        return { std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) };
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
