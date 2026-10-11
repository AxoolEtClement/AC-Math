# Rapport technique — AC-Math

## 1. Objet et périmètre

Ce projet compare une bibliothèque mathématique C++ de référence à une
implémentation SIMD explicite SSE/SSE2. Le périmètre du cahier des charges
porte sur trois traitements par lots de `Vec3f` :

1. produit scalaire de deux tableaux ;
2. normalisation d'un tableau de vecteurs ;
3. transformation de points 3D par une matrice affine `Matrix4x4`.

Les extensions AVX, AoSoA, GPU et multithreading ne font pas partie de cette
campagne.

## 2. Conventions et implémentation

`Vec3f` et `Vec4` stockent leurs composantes dans un registre `__m128` et
sont alignés sur 16 octets. `Matrix4x4` utilise un stockage row-major et des
vecteurs colonnes :

```text
résultat = matrice × vecteur
```

Dans une composition `A * B`, la matrice `B` est appliquée avant `A`.
`TransformPoint` utilise `w = 1` et ne réalise pas de division perspective.

Les kernels SIMD chargent quatre vecteurs sous forme de lanes :

```text
x = [x0, x1, x2, x3]
y = [y0, y1, y2, y3]
z = [z0, z1, z2, z3]
```

Les tailles non multiples de quatre sont terminées par une boucle scalaire
bornée. Un lot vide ou un pointeur nul est ignoré. Pour la normalisation,
un vecteur de longueur inférieure au seuil `1e-12` produit `Vec3f::Zero`.

La conversion AoS/SoA de `Vec4` est disponible avec `ToSOA` et `ToAOS`. Son
coût est mesuré séparément du coût des kernels.

## 3. Protocole de mesure

| Paramètre | Valeur |
|:--|:--|
| Système | Windows x64 |
| Compilateur | MSVC |
| Options affichées | `/O2 /arch:SSE2 /fp:precise` |
| Jeu d'instructions | SSE2 pour les kernels batch |
| Bibliothèque de mesure | nanobench |
| Graine aléatoire | `42` |
| Échauffement | 1000 itérations |
| Répétitions | 25 epochs |
| Tailles | 128, 4096 et 131072 éléments |

Les données et les buffers sont préparés avant chaque mesure. Les résultats
sont consommés par la lambda de benchmark afin d'empêcher la suppression du
calcul. Les conversions AoS/SoA, l'allocation initiale et l'affichage des
résultats ne sont pas inclus dans les kernels batch.

La valeur `ns/op` est le temps médian rapporté par nanobench. Le speedup est :

```text
speedup = temps référence / temps SIMD
```

Un speedup supérieur à `1` est favorable au SIMD. Les lignes marquées
`Unstable` par nanobench doivent être interprétées avec prudence : la durée
totale de mesure est courte par rapport au nombre d'itérations.

## 4. Résultats du benchmark

Mesures obtenues avec :

```powershell
build\benchmark\benchmark.exe all all
```

### 4.1 Vue synthétique

| Traitement | 128 éléments | 4096 éléments | 131072 éléments | Conclusion |
|:--|--:|--:|--:|:--|
| Produit scalaire | 0,30× | 0,31× | 0,32× | SIMD explicite plus lent |
| Normalisation | 1,13× | 1,09× | 1,06× | Gain modéré |
| Transformation de points | 2,46× | 2,52× | 2,50× | Gain net et stable |

Le résultat important est que la vectorisation explicite n'est pas
automatiquement bénéfique. Le produit scalaire utilise ici un stockage AoS :
la reconstruction des lanes par composante coûte suffisamment cher pour
annuler le gain des multiplications SIMD. La transformation de points amortit
mieux ce coût car les coefficients de la matrice sont réutilisés sur tout le
lot.

### 4.2 Produit scalaire

| Taille | Référence ns/op | SIMD SSE2 ns/op | ASM ns/op | Speedup SIMD |
|--:|--:|--:|--:|--:|
| 128 | 150,14 | 505,63 | 257,14 | 0,30× |
| 4096 | 4 988,70 | 16 027,78 | 7 841,20 | 0,31× |
| 131 072 | 165 406,75 | 524 978,94 | 262 986,50 | 0,32× |

La version ASM est plus rapide que le kernel SSE batch actuel, mais reste
plus lente que la référence C++ sur cette campagne. Ce résultat constitue une
régression mesurée et non un échec de validation fonctionnelle.

### 4.3 Normalisation

| Taille | Référence ns/op | SIMD SSE2 ns/op | Speedup |
|--:|--:|--:|--:|
| 128 | 455,34 | 402,19 | 1,13× |
| 4096 | 12 909,73 | 11 806,02 | 1,09× |
| 131 072 | 442 963,51 | 416 318,74 | 1,06× |

Le gain diminue avec la taille du lot, mais reste favorable. La dispersion
est plus élevée sur le petit lot ; les résultats `128` éléments sont donc
moins représentatifs que ceux des deux autres tailles.

### 4.4 Transformation de points

| Taille | Référence ns/op | SIMD SSE2 ns/op | Speedup |
|--:|--:|--:|--:|
| 128 | 917,68 | 373,45 | 2,46× |
| 4096 | 29 027,61 | 11 515,35 | 2,52× |
| 131 072 | 949 270,52 | 380 400,27 | 2,50× |

La réutilisation des coefficients de la matrice et le calcul de quatre points
en parallèle expliquent le gain le plus régulier de la campagne.

### 4.5 Coût des conversions AoS/SoA

| Taille | `ToSOA` ns/op | `ToAOS` ns/op |
|--:|--:|--:|
| 128 | 267,00 | 125,68 |
| 4096 | 8 067,22 | 4 033,99 |
| 131 072 | 270 371,87 | 221 851,91 |

Ces conversions ne sont pas incluses dans les temps des trois kernels. Pour
une utilisation réelle en pipeline AoS, leur coût doit être ajouté au coût
total. La conclusion dépend alors de la réutilisation du format SoA sur
plusieurs traitements successifs.

Les résultats bruts complets sont conservés dans
[`benchmark-results.txt`](benchmark-results.txt).

## 5. Validation fonctionnelle

La suite GoogleTest couvre :

- résultats connus des opérations vectorielles et matricielles ;
- vecteurs nuls et valeurs non finies ;
- identité, translation, rotation et composition ;
- lots vides, petits et non multiples de quatre ;
- comparaison entre référence scalaire et kernel SIMD ;
- round-trip AoS → SoA → AoS ;
- canary contre les écritures hors limites ;
- alignement sur 16 octets.

La commande suivante a été exécutée avec succès :

```powershell
ctest --test-dir build -C Release --output-on-failure
```

Résultat : **264/264 tests passés**.

## 6. Assembleur et analyse

La fonction `Dot_Asm_x64` est implémentée dans
`mathLibSIMD/src/dotProd.asm` et appelée depuis le benchmark. Elle charge les
deux vecteurs avec `movaps`, multiplie avec `mulps`, puis réduit les lanes
avec des additions et des permutations.

### 6.1 Méthode et configuration

Les extraits ci-dessous ont été générés avec MSVC **Release x64** et les
options `/O2 /EHsc /std:c++latest /arch:SSE2`. Le listing assembleur a été
produit à partir de `BatchProcessing.cpp` avec l'option `/FAcs`, puis vérifié
avec les commentaires de code source émis par le compilateur :

```powershell
$env:ComSpec /c "call `"$env:VSINSTALLDIR\Common7\Tools\VsDevCmd.bat`" -arch=x64"
cl /nologo /O2 /EHsc /std:c++latest /arch:SSE2 /FAcs `
   /Fa batchprocessing-release.asm /c mathLibSIMD\src\BatchProcessing.cpp `
   /ImathLibSIMD\include
```

La compilation du projet complet et l'exécution des tests ont également été
validées en Release (`264/264` tests passés). Pour vérifier la campagne de
mesure ciblée :

```powershell
build\benchmark\benchmark.exe dot reference medium
build\benchmark\benchmark.exe dot simd medium
build\benchmark\benchmark.exe dot asm medium
```

Les mesures obtenues sur 4096 éléments sont respectivement d'environ `5,1 µs`
pour la référence, `15,5 µs` pour le SIMD explicite et `7,9 µs` pour l'ASM.

### 6.2 Extrait ASM x64 et commentaire

L'extrait source suivant est vérifiable directement dans
`mathLibSIMD/src/dotProd.asm` :

```asm
movaps  xmm0, [rcx]       ; charge les quatre floats de lhs
mulps   xmm0, [rdx]       ; multiplication composante par composante
movaps  xmm1, xmm0
movhlps xmm1, xmm0        ; déplace les deux lanes hautes
addps   xmm0, xmm1        ; première réduction partielle
movaps  xmm1, xmm0
shufps  xmm1, xmm1, 1     ; duplique la lane contenant le second résultat
addss   xmm0, xmm1        ; somme finale dans xmm0[0]
```

`rcx` et `rdx` sont les deux premiers paramètres de la convention d'appel
x64 Windows. `movaps` suppose que les adresses sont alignées sur 16 octets.
`mulps` produit quatre multiplications en parallèle. Les permutations et les
additions réduisent ensuite les quatre lanes en un seul scalaire retourné
dans `xmm0`.

### 6.3 Désassemblage réel de la référence C++

La référence reste scalaire : chaque itération charge les trois composantes,
utilise `mulss` et `addss`, puis écrit un seul résultat avec `movss`.

```asm
; Fonction source : simd::BatchDotReference
00070  call    Vec3f::getZ
00083  movaps  xmm9,xmm0
0008a  call    Vec3f::getX
0009d  movaps  xmm8,xmm0
000a4  mulss   xmm8,xmm6
000a9  call    Vec3f::getY
000b9  movaps  xmm7,xmm0
000bf  mulss   xmm7,xmm6
000c3  addss   xmm7,xmm8
000c8  call    Vec3f::getZ
000cd  mulss   xmm0,xmm9
000d2  addss   xmm0,xmm7
000d6  movss   [r14+rsi*4],xmm0
```

Ici, `mulss` et `addss` sont des opérations scalaires sur la lane basse.
Cette absence de `mulps`/`addps` est cohérente avec la version de référence.

### 6.4 Désassemblage réel de `BatchDot` SIMD

L'extrait suivant provient du même listing Release et correspond à la boucle
SIMD de `BatchDot` :

```asm
; Fonction source : simd::BatchDot
00374  mulps   xmm6,xmm12
00388  mulps   xmm10,xmm13
00392  mulps   xmm3,xmm11
00396  addps   xmm6,xmm10
0039a  addps   xmm6,xmm3
0039d  movups  [rsi+rdi*4],xmm6
003a1  add     rdi,4
003a9  cmp     rax,rbx
003ac  jb      $LL4@BatchDot
```

Les trois `mulps` calculent en parallèle les termes `x`, `y` et `z` pour
quatre vecteurs. Les deux `addps` additionnent les termes dans les quatre
lanes, puis `movups` écrit les quatre résultats. `add rdi,4` confirme qu'une
itération avance de quatre éléments. Avant ce bloc, le listing montre les
appels à `getX`, `getY` et `getZ` ainsi que les instructions `unpcklps` et
`movlhps` utilisées pour reconstruire les lanes depuis l'AoS. La boucle
résiduelle revient ensuite à `Vec3f::Dot` pour les éléments restants.

Le listing assembleur constitue une preuve reproductible du code généré.

### 6.5 Profilage CPU Visual Studio

Un profilage CPU a été réalisé avec Visual Studio sur l'exécutable `benchmark`
en configuration Release x64. La session visible dans
[`Profilage CPU.png`](Profilage%20CPU.png) dure `2 min 11 s` et représente
`127 508` unités processeur. Le tableau « Top des fonctions » affiche :

| Entrée affichée par Visual Studio | Temps total | Temps UC exclusif |
|:--|--:|--:|
| `benchmark.dll!0x00007ff7793b21d0` | 7,70 % | 7,66 % |
| `benchmark.dll!0x00007ff7793b2f5c` | 4,70 % | 4,66 % |
| `benchmark.dll!0x00007ff7793b2f5f` | 2,85 % | 2,83 % |
| `benchmark.dll!0x00007ff7793b2193` | 2,70 % | 2,68 % |
| `benchmark.dll!0x00007ff7793b2e13` | 2,11 % | 2,10 % |

Le chemin réactif confirme que le processus `benchmark` concentre
`100,00 %` du temps observé. La capture ne contient toutefois pas les symboles
de débogage : les adresses ne peuvent donc pas être attribuées avec certitude
à `BatchDot`, `LoadComponent` ou une autre fonction précise. Elle constitue
une preuve de collecte CPU, mais pas une mesure symbolisée par fonction.

Cette limite doit être prise en compte avec les mesures nanobench et le
désassemblage : `BatchDot` réalise bien trois `mulps` et deux `addps` par
groupe de quatre, mais la capture seule ne permet pas de quantifier
séparément le coût de la reconstruction AoS. Une session complémentaire avec
les symboles PDB chargés permettrait d'obtenir les noms exacts.

## 7. Limites et conclusions

- Le benchmark actuel mesure correctement trois tailles et conserve les
  données brutes, mais les petites opérations peuvent être proches du
  plancher de mesure.
- Le benchmark nanobench signale certaines petites mesures comme instables ;
  elles ne doivent pas être utilisées seules pour conclure.
- Le kernel `BatchDot` doit être considéré comme une expérience mesurée :
  son résultat montre qu'une vectorisation explicite avec extraction AoS peut
  régresser.
- Le stockage SoA ou une conversion amortie sur plusieurs opérations pourrait
  améliorer ce cas, mais relève d'une expérience ultérieure.
- Aucun facteur minimal d'accélération n'est imposé par le cahier des charges ;
  les régressions sont donc acceptables si elles sont expliquées honnêtement.

## 8. Utilisation de l'IA

Une IA a été utilisée comme assistant pour la revue de conformité, la
documentation Doxygen, l'analyse des résultats et l'identification de
corrections minimales.
