/**
 * @file LayoutConversions.hpp
 * @brief Fonctions de conversion entre les layouts AoS et SoA.
 */

#pragma once

#include <cstddef>

namespace simd
{
    class Vec4;
    class Vec4SOA;

    /**
     * @brief Convertit un tableau de vecteurs Vec4 AoS en blocs Vec4SOA.
     *
     * Chaque bloc contient au plus quatre vecteurs. Les lanes inutilisées du
     * dernier bloc sont remplies avec zéro et ne sont jamais reconverties.
     *
     * @param[in] input Tableau source, de taille @p count.
     * @param[out] output Tableau de @c (count + 3) / 4 blocs.
     * @param[in] count Nombre de vecteurs à convertir.
     */
    void ToSOA(const Vec4* input, Vec4SOA* output, std::size_t count);

    /**
     * @brief Convertit des blocs Vec4SOA en tableau de vecteurs Vec4 AoS.
     *
     * @param[in] input Tableau source de blocs SoA.
     * @param[out] output Tableau de taille @p count.
     * @param[in] count Nombre de vecteurs à produire.
     */
    void ToAOS(const Vec4SOA* input, Vec4* output, std::size_t count);
}
