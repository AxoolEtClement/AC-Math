/**
 * @file Vec4.hpp
 * @brief Implémentation d'un vecteur 4D aligné pour les calculs SIMD.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <immintrin.h>

namespace simd
{
    class Matrix4x4;
    /**
     * @class Vec4
     * @brief Vecteur 4D utilisant le stockage SIMD (SSE).
     *
     * Stockage aligné sur 16 octets via alignas(16).
     * Les données sont stockées dans un registre __m128.
     */
    class alignas(16) Vec4
    {
    public:

        Vec4();
        Vec4(float _x, float _y, float _z, float _w);
        Vec4(const Vec4&) = default;
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
        Vec4& operator=(const Vec4& ) = default;

        /**
         * @brief Calcule le produit scalaire entre deux vecteurs.
         * @param[in] rhs Le vecteur opposé.
         * @return Le résultat scalaire.
         */
        float Dot(const Vec4& rhs) const;

        /**
         * @brief Calcule le carré de la magnitude.
         * @return \f$ x^2 + y^2 + z^2 + w^2 \f$
         */
        float MagnitudeSquared() const;

        /**
         * @brief Calcule la magnitude du vecteur.
         * @return La norme L2 du vecteur.
         */
        float Magnitude() const;

        /**
         * @brief Normalise le vecteur.
         * @return Un nouveau vecteur de norme 1.
         * @throws std::domain_error si le vecteur est nul ou contient des valeurs non finies.
         */
        Vec4 Normalize() const;

        float DistanceSquared(const Vec4& rhs) const;
        float Distance(const Vec4& rhs) const;

        /**
         * @brief Calcule l'angle entre deux vecteurs.
         * @param[in] rhs Le second vecteur.
         * @return L'angle en radians.
         * @throws std::domain_error si l'un des vecteurs est nul.
         */
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

        friend class Vec4SOA;
        friend class Matrix4x4;
        friend void ToSOA(const Vec4* input, Vec4SOA* output, std::size_t count);

    private:
        __m128 _data;
    };
    Vec4 operator*(float scalar, const Vec4& vector);
}
