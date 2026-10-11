# AC-Math

Bibliothèque mathématique C++ comparant une référence scalaire à des kernels
SIMD explicites SSE/SSE2 pour le projet ASM/SIMD.

## Fonctionnalités

- `Vec3f`, `Vec4` et `Matrix4x4` indépendants du rendu et du gameplay ;
- stockage SIMD `__m128`, alignement 16 octets ;
- trois traitements batch :
  - `BatchDot` ;
  - `BatchNormalize` ;
  - `BatchTransformPoint` ;
- versions scalaires de référence pour les comparaisons ;
- conversions AoS/SoA pour `Vec4` ;
- fonction ASM x64 `Dot_Asm_x64` ;
- tests GoogleTest et benchmark nanobench ;
- documentation Doxygen.

## Conventions mémoire

`Matrix4x4` est stockée en row-major et utilise des vecteurs colonnes :

```text
résultat = matrice × vecteur
```

Dans une composition `A * B`, `B` est appliquée avant `A`. Les points
utilisent `w = 1`, sans division perspective. Les kernels batch traitent
quatre lanes à la fois et terminent les lots incomplets par une boucle
scalaire bornée.

Les conversions `ToSOA` et `ToAOS` organisent les lanes comme suit :

```text
x = [x0, x1, x2, x3]
y = [y0, y1, y2, y3]
z = [z0, z1, z2, z3]
w = [w0, w1, w2, w3]
```

Les détails sont disponibles dans
[`conventions.md`](conventions.md).

## Prérequis

- Windows x64 ;
- Visual Studio avec outils C++ et MASM x64 ;
- CMake 3.25 ou supérieur ;
- Ninja ou un générateur Visual Studio ;
- connexion réseau lors de la première configuration pour récupérer
  GoogleTest et nanobench.

## Compiler en Release x64

Depuis un terminal développeur Visual Studio :

```powershell
cmake --preset default
cmake --build build --config Release
```

Le preset `default` configure une cible Ninja x64 en Release. Les exécutables
produits sont notamment :

```text
build\tests\run_tests.exe
build\benchmark\benchmark.exe
```

## Exécuter les tests

```powershell
ctest --test-dir build -C Release --output-on-failure
```

La campagne validée contient 264 tests. Elle couvre les résultats connus,
vecteurs nuls, valeurs non finies, tailles de lots limites, alignement,
identité, translation, rotation, composition et comparaison référence/SIMD.

## Exécuter le benchmark

Le benchmark utilise la graine reproductible `42`, prépare les données avant
la mesure, effectue 1000 itérations d'échauffement et répète 25 epochs.
Les options affichées sont `/O2 /arch:SSE2 /fp:precise`. Les tailles sont :

| Nom | Éléments |
|:--|--:|
| `small` | 128 |
| `medium` | 4096 |
| `large` | 131072 |

Syntaxe :

```powershell
build\benchmark\benchmark.exe [operation] [version] [size]
```

Valeurs acceptées :

- opérations : `all`, `dot`, `normalize`, `transform` ;
- versions : `all`, `reference`, `simd`, `asm` (`asm` concerne le produit
  scalaire) ;
- tailles : `all`, `small`, `medium`, `large`.

Exemples :

```powershell
build\benchmark\benchmark.exe all all
build\benchmark\benchmark.exe dot medium
build\benchmark\benchmark.exe dot simd large
build\benchmark\benchmark.exe normalize large
build\benchmark\benchmark.exe transform small
```

La forme historique `[operation] [size]` reste acceptée ; la forme complète
permet de sélectionner explicitement la version mesurée.

Le benchmark mesure aussi séparément `ToSOA` et `ToAOS`. Les allocations,
l'affichage et la conversion ne sont pas inclus dans le temps du kernel.
Lorsque le format SoA est utilisé en production, le coût des conversions doit
être ajouté au coût total.

Les résultats de la dernière campagne sont dans
[`benchmark-results.txt`](benchmark-results.txt) et sont résumés dans
[`rapport.md`](rapport.md). La campagne montre un gain d'environ `2,5x` pour
la transformation de points, un gain modéré pour la normalisation et une
régression du produit scalaire SIMD avec le stockage AoS actuel.

## Générer la documentation

```powershell
doxygen Doxyfile
```

La sortie est générée dans `docs/doxygen`. Les headers documentent les
conventions, les contraintes d'alignement, les cas limites et le
comportement des vecteurs nuls.

## Assembleur x64

La fonction MASM se trouve dans
[`mathLibSIMD/src/dotProd.asm`](mathLibSIMD/src/dotProd.asm). Elle est appelée
depuis le benchmark et comparée aux versions scalaire et SIMD.

Le rapport indique les deux extraits de désassemblage à joindre à la remise :
la référence C++ optimisée et le kernel SSE explicite.

Un profilage CPU Visual Studio a également été réalisé. La capture
[`Profilage CPU.png`](Profilage%20CPU.png) montre une session de `2 min 11 s`
et les cinq fonctions les plus coûteuses. Visual Studio affichait des
adresses de `benchmark.dll` plutôt que les noms C++ ; le rapport conserve donc
ces résultats sans attribuer arbitrairement une adresse à `BatchDot`.

## Limites et usage de l'IA

Le projet ne fournit pas de moteur physique, de rendu, de GPU,
multithreading, AVX ou AoSoA. Les résultats doivent être interprétés sur la
machine de mesure et ne constituent pas une garantie universelle.

Une IA a été utilisée comme assistant pour la revue de conformité, la
documentation Doxygen, l'analyse des benchmarks et l'identification de
corrections minimales.
