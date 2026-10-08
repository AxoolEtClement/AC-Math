# Rapport : bibliothèque mathématique C++ vs SIMD

## Table des matières

- [Introduction et contexte](#introduction-et-contexte)
- [Méthodologie de test](#méthodologie-de-test)
- [Mesures et tests](#mesures-et-tests)
- [Conclusion](#conclusion)

---

## Introduction et contexte

Dans le cadre de notre troisième année en moteur, nous avons pour projet fil rouge de réaliser un moteur physique.

Notre premier objectif est de réaliser une bibliothèque mathématique la plus optimisée possible, afin de poser de bonnes fondations pour ce moteur. Nous sommes partis d'une bibliothèque existante en C++ et nous l'avons convertie en SIMD afin de comparer les performances.

Ce rapport présente nos mesures de performances ainsi que d'autres axes d'amélioration.

---

## Méthodologie de test

### Objectif

Comparer `mathLibCPP` (version scalaire, référence) et `mathLibSIMD` (SSE4.1) sur `Vec3`, `Vec4` et `Matrix3x3`.

### Outils

| Rôle | Outil |
|:--|:--|
| Validation de la correction | Google Test |
| Micro-benchmarks | nanobench |
| Langage | C++20 |
| Compilateur et options | -msse4.1 -lto |
| Processeur | ryzen5 4800 |
| Système d'exploitation | nobaraLinux |

### Validation de la correction

- Les tests unitaires de la version scalaire ont été portés sur la version SIMD, avec les mêmes valeurs attendues et une tolérance relative de 1e-5.
- Ils couvrent aussi les cas limites : division par zéro, valeurs non finies, matrice singulière, indices hors bornes, ainsi que l'alignement sur 16 octets.
- Ils ont révélé 6 bugs dans `Mat3.cpp`, corrigés avant toute mesure.

### Protocole de mesure

- Chaque benchmark est une lambda générique exécutée **telle quelle** sur les deux implémentations : même code, mêmes données.
- Les entrées sont aléatoires à l'exécution (graine fixable), pour éviter un calcul à la compilation. Une barrière mémoire et `doNotOptimizeAway` empêchent le compilateur de supprimer le calcul ou de le sortir de la boucle.
- Chaque mesure comprend 1000 itérations d'échauffement, puis 25 époques de 10 ms. La valeur retenue est la **médiane**.
- Environ 100 comparaisons sont réalisées, réparties en deux familles :
  - **Tests unitaires** : une opération sur des données en cache (coût intrinsèque de l'opération).
  - **Tests par lots (Batch)** : la même opération sur 1024 vecteurs (256 matrices), avec un temps ramené **par élément** (cache, boucle, taille des objets).

### Métriques

- **ns/op** : temps par opération (par élément pour les lots).
- **Speedup** = temps scalaire / temps SIMD (supérieur à 1 : le SIMD est plus rapide).
- Un écart est jugé significatif au-delà de ±10 %, ce qui est nettement au-dessus du bruit de mesure (`err%` ≤ 2 %).
- Les résultats sont synthétisés par la moyenne géométrique des speedups de chaque groupe.

### Limites

- Plancher de mesure d'environ 0,5 ns par itération.
- Certaines fonctions n'utilisent pas le même algorithme (norme : `std::hypot` contre racine du produit scalaire ; déterminant : pivot de Gauss contre cofacteurs). Ces écarts ne reflètent pas uniquement l'effet du SIMD.
- Les rotations sont dominées par le coût de `sin` et `cos`.
- Un seul processeur et un seul compilateur : les résultats ne sont pas généralisables.

---

## Mesures et tests

### Comment lire les tableaux

| Colonne | Signification |
|:--|:--|
| C++ (ns/op) | Temps de la version scalaire |
| SIMD (ns/op) | Temps de la version SIMD |
| Speedup | Temps C++ / temps SIMD |
| Verdict | ✅ plus rapide (≥ 1,10×) · ❌ plus lent (≤ 0,90×) · ➖ équivalent (entre les deux) |

### Vec3

| Opération | C++ (ns/op) | SIMD (ns/op) | Speedup | Verdict |
|:--|--:|--:|--:|:--|
| `construct (x,y,z)` | 0,67 | 0,80 | 0,84× | ➖ équivalent |
| `read x+y+z` | 0,74 | 0,74 | 1,01× | ➖ équivalent |
| `write x,y,z` | 0,67 | 1,01 | 0,67× | ❌ plus lent |
| `a + b` | 1,11 | 0,63 | 1,76× | ✅ plus rapide |
| `a - b` | 1,11 | 0,63 | 1,76× | ✅ plus rapide |
| `a * b` (composantes) | 1,11 | 0,63 | 1,76× | ✅ plus rapide |
| `a / b` (composantes) | 1,76 | 1,46 | 1,21× | ✅ plus rapide |
| `-a` | 0,84 | 0,50 | 1,67× | ✅ plus rapide |
| `a * scalaire` | 0,88 | 0,61 | 1,44× | ✅ plus rapide |
| `scalaire * a` | 0,88 | 0,61 | 1,44× | ✅ plus rapide |
| `a / scalaire` | 1,76 | 0,88 | 2,00× | ✅ plus rapide |
| `a += b` | 1,11 | 0,63 | 1,76× | ✅ plus rapide |
| `a -= b` | 1,11 | 0,63 | 1,76× | ✅ plus rapide |
| `a *= b` | 1,11 | 0,63 | 1,76× | ✅ plus rapide |
| `a /= b` | 1,76 | 1,46 | 1,21× | ✅ plus rapide |
| `a *= scalaire` | 0,88 | 0,62 | 1,44× | ✅ plus rapide |
| `a /= scalaire` | 1,76 | 0,88 | 2,00× | ✅ plus rapide |
| `a == a'` (égaux, pire cas) | 1,43 | 0,59 | 2,43× | ✅ plus rapide |
| `a == b` (différents) | 0,59 | 0,59 | 1,00× | ➖ équivalent |
| `a != a'` (égaux) | 1,43 | 0,59 | 2,43× | ✅ plus rapide |
| `Dot` | 1,43 | 1,51 | 0,94× | ➖ équivalent |
| `Cross` | 1,76 | 0,88 | 2,00× | ✅ plus rapide |
| `MagnitudeSquared` | 0,76 | 1,26 | 0,60× | ❌ plus lent |
| `Magnitude` | 4,23 | 1,39 | 3,05× | ✅ plus rapide |
| `Normalize` | 8,12 | 4,48 | 1,81× | ✅ plus rapide |
| `DistanceSquared` | 1,43 | 1,26 | 1,13× | ✅ plus rapide |
| `Distance` | 4,23 | 1,39 | 3,05× | ✅ plus rapide |
| `Angle` | 35,23 | 15,30 | 2,30× | ✅ plus rapide |
| `Lerp` | 1,52 | 0,82 | 1,86× | ✅ plus rapide |
| `Min` | 1,11 | 0,62 | 1,80× | ✅ plus rapide |
| `Max` | 1,11 | 0,61 | 1,80× | ✅ plus rapide |

### Vec4

| Opération | C++ (ns/op) | SIMD (ns/op) | Speedup | Verdict |
|:--|--:|--:|--:|:--|
| `construct (x,y,z,w)` | 0,50 | 0,50 | 1,00× | ➖ équivalent |
| `read x+y+z+w` | 1,01 | 1,01 | 1,00× | ➖ équivalent |
| `write x,y,z,w` | 0,50 | 1,25 | 0,40× | ❌ plus lent |
| `a + b` | 0,59 | 0,63 | 0,93× | ➖ équivalent |
| `a - b` | 0,59 | 0,63 | 0,93× | ➖ équivalent |
| `a * b` (composantes) | 0,59 | 0,63 | 0,93× | ➖ équivalent |
| `a / b` (composantes) | 2,39 | 1,93 | 1,24× | ✅ plus rapide |
| `-a` | 0,50 | 0,50 | 1,00× | ➖ équivalent |
| `a * scalaire` | 0,59 | 0,62 | 0,95× | ➖ équivalent |
| `scalaire * a` | 0,59 | 0,62 | 0,96× | ➖ équivalent |
| `a / scalaire` | 1,01 | 0,88 | 1,14× | ✅ plus rapide |
| `a += b` | 0,59 | 0,63 | 0,93× | ➖ équivalent |
| `a -= b` | 0,59 | 0,63 | 0,93× | ➖ équivalent |
| `a *= b` | 0,59 | 0,63 | 0,93× | ➖ équivalent |
| `a /= b` | 2,40 | 1,94 | 1,24× | ✅ plus rapide |
| `a *= scalaire` | 0,59 | 0,62 | 0,95× | ➖ équivalent |
| `a /= scalaire` | 1,01 | 0,89 | 1,14× | ✅ plus rapide |
| `a == a'` (égaux, pire cas) | 1,85 | 0,76 | 2,44× | ✅ plus rapide |
| `a == b` (différents) | 0,59 | 0,76 | 0,78× | ❌ plus lent |
| `a != a'` (égaux) | 1,85 | 0,59 | 3,14× | ✅ plus rapide |
| `Dot` | 1,68 | 1,51 | 1,11× | ✅ plus rapide |
| `MagnitudeSquared` | 1,01 | 1,26 | 0,80× | ❌ plus lent |
| `Magnitude` | 10,43 | 1,38 | 7,54× | ✅ plus rapide |
| `Normalize` | 18,53 | 4,91 | 3,78× | ✅ plus rapide |
| `DistanceSquared` | 1,85 | 1,26 | 1,47× | ✅ plus rapide |
| `Distance` | 10,95 | 1,39 | 7,91× | ✅ plus rapide |
| `Angle` | 52,67 | 18,57 | 2,84× | ✅ plus rapide |
| `Lerp` | 0,93 | 0,82 | 1,13× | ✅ plus rapide |
| `Min` | 0,59 | 0,62 | 0,96× | ➖ équivalent |
| `Max` | 0,59 | 0,62 | 0,96× | ➖ équivalent |

### Matrix3x3

| Opération | C++ (ns/op) | SIMD (ns/op) | Speedup | Verdict |
|:--|--:|--:|--:|:--|
| `default ctor` | 11,39 | 0,76 | 15,08× | ✅ plus rapide |
| `Identity()` | 11,38 | 0,76 | 15,07× | ✅ plus rapide |
| `Zero()` | 0,76 | 0,76 | 1,00× | ➖ équivalent |
| construction depuis `array<9>` | 0,90 | 2,01 | 0,45× | ❌ plus lent |
| `operator()` lecture ×9 | 2,01 | 2,01 | 1,00× | ➖ équivalent |
| `operator()` écriture ×9 | 0,76 | 13,33 | 0,06× | ❌ plus lent |
| `A + B` | 1,51 | 1,50 | 1,01× | ➖ équivalent |
| `A - B` | 1,51 | 1,51 | 1,00× | ➖ équivalent |
| `A * B` | 4,65 | 8,41 | 0,55× | ❌ plus lent |
| `A * vec3` | 3,04 | 4,78 | 0,64× | ❌ plus lent |
| `A * scalaire` | 1,13 | 1,11 | 1,02× | ➖ équivalent |
| `A *= B` | 4,16 | 12,43 | 0,33× | ❌ plus lent |
| `A == A'` (égales, pire cas) | 4,12 | 1,43 | 2,88× | ✅ plus rapide |
| `A == B` (différentes) | 1,01 | 1,26 | 0,80× | ❌ plus lent |
| `A != A'` (égales) | 4,11 | 1,43 | 2,88× | ✅ plus rapide |
| `Transpose` | 0,98 | 1,26 | 0,78× | ❌ plus lent |
| `Determinant` | 14,35 | 2,02 | 7,12× | ✅ plus rapide |
| `Inverse` | 53,64 | 65,82 | 0,82× | ❌ plus lent |
| `Inverse(A) * vec3` (résolution) | 54,63 | 67,51 | 0,81× | ❌ plus lent |
| `Scale(vec3)` | 9,82 | 0,76 | 12,99× | ✅ plus rapide |
| `RotationX` | 11,85 | 7,85 | 1,51× | ✅ plus rapide |
| `RotationY` | 11,51 | 6,77 | 1,70× | ✅ plus rapide |
| `RotationZ` | 8,59 | 6,97 | 1,23× | ✅ plus rapide |
| `Rz * Ry * Rx` (construction + composition) | 32,37 | 35,75 | 0,91× | ➖ équivalent |
| `(Scale * Rz) * vec3` (pipeline) | 19,61 | 21,11 | 0,93× | ➖ équivalent |
| `A * B * vec3` | 8,56 | 20,13 | 0,43× | ❌ plus lent |

### Traitements par lots (Batch)

Temps rapporté **par élément** (1024 vecteurs ou 256 matrices par lot).

| Opération | C++ (ns/élément) | SIMD (ns/élément) | Speedup | Verdict |
|:--|--:|--:|--:|:--|
| `[Vec3] out[i] = a[i] + b[i]` | 0,30 | 0,56 | 0,54× | ❌ plus lent |
| `[Vec3] out[i] = a[i] * scalaire` | 0,19 | 0,40 | 0,48× | ❌ plus lent |
| `[Vec3]` somme des `Dot(a[i], b[i])` | 0,76 | 1,52 | 0,50× | ❌ plus lent |
| `[Vec3] out[i] = Cross(a[i], b[i])` | 0,91 | 0,84 | 1,08× | ➖ équivalent |
| `[Vec3] out[i] = Normalize(a[i])` | 8,82 | 4,96 | 1,78× | ✅ plus rapide |
| `[Vec3] out[i] = Lerp(a[i], b[i], t)` | 0,32 | 0,56 | 0,57× | ❌ plus lent |
| `[Vec4] out[i] = a[i] + b[i]` | 0,40 | 0,56 | 0,70× | ❌ plus lent |
| `[Vec4] out[i] = a[i] * scalaire` | 0,33 | 0,38 | 0,85× | ❌ plus lent |
| `[Vec4]` somme des `Dot(a[i], b[i])` | 0,85 | 1,52 | 0,56× | ❌ plus lent |
| `[Vec4] out[i] = Normalize(a[i])` | 20,46 | 5,49 | 3,72× | ✅ plus rapide |
| `[Vec4] out[i] = Lerp(a[i], b[i], t)` | 0,52 | 0,56 | 0,93× | ➖ équivalent |
| `[Vec4] out[i] = Min(a[i], b[i])` | 0,40 | 0,55 | 0,72× | ❌ plus lent |
| `[Mat3] out[i] = A * v[i]` (1 matrice, N vecteurs) | 0,73 | 4,79 | 0,15× | ❌ plus lent |
| `[Mat3] out[i] = A[i] * v[i]` | 3,02 | 4,81 | 0,63× | ❌ plus lent |
| `[Mat3] out[i] = A[i] * B[i]` | 9,50 | 8,47 | 1,12× | ✅ plus rapide |
| `[Mat3] out[i] = Transpose(A[i])` | 1,15 | 1,26 | 0,91× | ➖ équivalent |
| `[Mat3]` somme des `Determinant(A[i])` | 15,34 | 2,46 | 6,25× | ✅ plus rapide |
| `[Mat3] out[i] = Inverse(A[i])` | 53,67 | 65,90 | 0,81× | ❌ plus lent |

### Synthèse

#### Moyenne géométrique du speedup par groupe

| Groupe | Speedup moyen |
|:--|--:|
| Vec3 | 1,55× |
| Vec4 | 1,30× |
| Matrix3x3 | 1,24× |
| Batch | 0,85× |
| **Global** | **1,26×** |

#### Les 10 cas où le SIMD est le plus lent

| Opération | Speedup |
|:--|--:|
| Matrix3x3 : `operator()` écriture ×9 | 0,06× |
| Batch : `[Mat3] out[i] = A * v[i]` (1 matrice, N vecteurs) | 0,15× |
| Matrix3x3 : `A *= B` | 0,33× |
| Vec4 : écriture `x,y,z,w` | 0,40× |
| Matrix3x3 : `A * B * vec3` | 0,43× |
| Matrix3x3 : construction depuis `array<9>` | 0,45× |
| Batch : `[Vec3] out[i] = a[i] * scalaire` | 0,48× |
| Batch : `[Vec3]` somme des `Dot(a[i], b[i])` | 0,50× |
| Batch : `[Vec3] out[i] = a[i] + b[i]` | 0,54× |
| Matrix3x3 : `A * B` | 0,55× |

#### Les 10 plus gros gains du SIMD

| Opération | Speedup |
|:--|--:|
| Matrix3x3 : `default ctor` | 15,08× |
| Matrix3x3 : `Identity()` | 15,07× |
| Matrix3x3 : `Scale(vec3)` | 12,99× |
| Vec4 : `Distance` | 7,91× |
| Vec4 : `Magnitude` | 7,54× |
| Matrix3x3 : `Determinant` | 7,12× |
| Batch : `[Mat3]` somme des `Determinant(A[i])` | 6,25× |
| Vec4 : `Normalize` | 3,78× |
| Batch : `[Vec4] out[i] = Normalize(a[i])` | 3,72× |
| Vec4 : `a != a'` (égaux) | 3,14× |

---

## Conclusion

En conclusion, utiliser le SIMD n'est pas toujours pertinent à notre niveau : nous avons donc remis en version C++ standard les fonctions qui présentaient des régressions.

Cependant, le SIMD a significativement amélioré les performances de certaines fonctions, ce qui nous permettra certainement de mieux exploiter la machine qui exécute notre code, afin de fournir les meilleures performances possibles.