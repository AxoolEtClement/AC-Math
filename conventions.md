# Conventions Mathématiques de la Bibliothèque

Cette page définit les conventions fondamentelles utilisées dans toute la bibliothèque `mathLibSIMD`.

## 1. Stockage des Matrices
La bibliothèque utilise un stockage **Row-Major** (ordre ligne par ligne).
Une matrice $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ est stockée en mémoire comme : `[a, b, c, d]`.

## 2. Convention des Vecteurs
Les vecteurs sont traités comme des **vecteurs colonnes**.
L'opération de transformation d'un vecteur $v$ par une matrice $M$ est définie par :
\f[ v' = M \times v \f]

## 3. Composition des Transformations
L'ordre de composition suit la logique mathématique standard. 
Pour deux transformations $A$ et $B$, l'expression \f$ A \times B \times v \f$ signifie que la transformation $B$ est appliquée en premier, puis la transformation $A$.

## 4. Coordonnées Homogènes
Pour la transformation de points 3D via une matrice 4x4 :
- On utilise une coordonnée homogène \f$ w = 1 \f$.
- La bibliothèque implémente des transformations **affines**.
- **Aucune division perspective** n'est effectuée.

## 5. Alignement et SIMD
Tous les types de données SIMD (`Vec3f`, `Vec4`, `Matrix4x4`) sont alignés sur **16 octets** (`alignas(16)`).
L'utilisation d'instructions de chargement alignées (`movaps`) requiert que les pointeurs passés aux fonctions de traitement par lots (`Batch...`) respectent cet alignement.
