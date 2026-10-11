/**
 * @file Vec3f.hpp
 * @brief Implémentation d'un vecteur 3D aligné pour les calculs SIMD.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <immintrin.h>

namespace simd
{
    class Matrix3x3; // Forward declaration to avoid circular dependency with Vec3f.
    /**
     * @class Vec3f
     * @brief Vecteur 3D utilisant le stockage SIMD (SSE).
     *
     * Stockage aligné sur 16 octets via alignas(16).
     * Les données sont stockées dans un registre __m128.
     *
     * @see Matrix4x4 pour les conventions de coordonnées.
     */
    class alignas(16) Vec3f
    {
    public:
        

        Vec3f();
        Vec3f(float x, float y, float z);

        Vec3f(const Vec3f&) = default;
        Vec3f& operator=(const Vec3f&) = default;
        Vec3f(__m128 data);

        float getX() const;
        float getY() const;
        float getZ() const;

        void setX(float value);
        void setY(float value);
        void setZ(float value);

        Vec3f operator+(const Vec3f& rhs) const;
        Vec3f operator-(const Vec3f& rhs) const;
        Vec3f operator*(const Vec3f& rhs) const;
        Vec3f operator/(const Vec3f& rhs) const;
        Vec3f operator-() const;
        Vec3f operator*(float scalar) const;
        Vec3f operator/(float scalar) const;
        Vec3f& operator+=(const Vec3f& rhs);
        Vec3f& operator-=(const Vec3f& rhs);
        Vec3f& operator*=(const Vec3f& rhs);
        Vec3f& operator/=(const Vec3f& rhs);
        Vec3f& operator*=(float scalar);
        Vec3f& operator/=(float scalar);
        bool operator==(const Vec3f& rhs) const;
        bool operator!=(const Vec3f& rhs) const;

        /**
         * @brief Calcule le produit scalaire entre deux vecteurs.
         *
         * Formule : \f$ \vec{a} \cdot \vec{b} = \sum a_i b_i \f$
         * @param[in] rhs Le vecteur avec lequel effectuer le produit.
         * @return Le résultat scalaire.
         */
        float Dot(const Vec3f& rhs) const;
        Vec3f Cross(const Vec3f& rhs) const;
        float MagnitudeSquared() const;
        float Magnitude() const;
        /**
         * @brief Normalise le vecteur.
         *
         * Si le vecteur est nul (magnitude proche de 0), renvoie un vecteur nul.
         * @return Un nouveau vecteur de norme 1, ou Vec3f::Zero si le vecteur d'entrée est nul.
         * @throws std::domain_error si le vecteur est nul (dépend de l'implémentation interne).
         */
        Vec3f Normalize() const;
        float DistanceSquared(const Vec3f& rhs) const;
        float Distance(const Vec3f& rhs) const;
        /**
         * @brief Calcule l'angle entre deux vecteurs.
         *
         * @param[in] rhs Le second vecteur.
         * @return L'angle en radians.
         * @throws std::domain_error si l'un des vecteurs est nul.
         */
        float Angle(const Vec3f& rhs) const;
        static Vec3f Lerp(const Vec3f& a, const Vec3f& b, float t);
        static Vec3f Min(const Vec3f& a, const Vec3f& b);
        static Vec3f Max(const Vec3f& a, const Vec3f& b);

        static const Vec3f Zero;
        static const Vec3f One;
        static const Vec3f UnitX;
        static const Vec3f UnitY;
        static const Vec3f UnitZ;

        friend class Matrix3x3;

    private:
        __m128 _data;
    };

    /**
     * @brief Calcule les produits scalaires d'un lot de Vec3f.
     *
     * Le lot peut être vide et sa taille n'a pas besoin d'être multiple de
     * quatre. Les éléments restants sont traités sans accès hors limites.
     *
     * @param[in] a Premier tableau de vecteurs.
     * @param[in] b Second tableau de vecteurs.
     * @param[out] results Tableau de @p count résultats.
     * @param[in] count Nombre de vecteurs.
     */
    void BatchDot(const Vec3f* a, const Vec3f* b, float* results, std::size_t count);

    /**
     * @brief Normalise un lot de Vec3f.
     *
     * Un vecteur nul produit Vec3f::Zero. Les lots vides et les restes
     * inférieurs à quatre éléments sont pris en charge.
     *
     * @param[in] input Tableau source.
     * @param[out] output Tableau de @p count vecteurs normalisés.
     * @param[in] count Nombre de vecteurs.
     */
    void BatchNormalize(const Vec3f* input, Vec3f* output, std::size_t count);

    /**
     * @brief Référence scalaire de BatchDot pour les comparaisons.
     * @param[in] a Premier tableau de vecteurs.
     * @param[in] b Second tableau de vecteurs.
     * @param[out] results Tableau de résultats.
     * @param[in] count Nombre de vecteurs.
     */
    void BatchDotReference(const Vec3f* a, const Vec3f* b,
                           float* results, std::size_t count);

    /**
     * @brief Référence scalaire de BatchNormalize pour les comparaisons.
     * @param[in] input Tableau source.
     * @param[out] output Tableau destination.
     * @param[in] count Nombre de vecteurs.
     */
    void BatchNormalizeReference(const Vec3f* input, Vec3f* output,
                                 std::size_t count);

    Vec3f operator*(float scalar, const Vec3f& vector);
}
