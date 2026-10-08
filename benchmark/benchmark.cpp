// =====================================================================================
//  BenchmarkMath.cpp : mathLibCPP (scalaire) vs mathLibSIMD (SSE4.1)
//
//  Principe : chaque benchmark est une lambda GENERIQUE `[](auto& d) { ... }` executee
//  telle quelle sur les deux backends (CppBackend / SimdBackend). Les deux versions
//  executent donc exactement la meme expression, sur exactement les memes valeurs.
//
//  Couverture :
//    - Vec3 / Vec4 : construction, getters/setters, + - * / (vec & scalaire), unaire -,
//      compound ops, ==, !=, Dot, Cross, Magnitude, Normalize, Distance, Angle, Lerp, Min, Max
//    - Matrix3x3   : ctors, operator(), + - * (mat, vec, scalaire), *=, ==, Transpose,
//                    Determinant, Inverse, Scale, RotationX/Y/Z, enchainements
//    - Batch       : memes operations sur des tableaux (cache, boucles, debit reel)
//    - Vec4 AOS vs SOA : simd::Vec4 (AOS : 1 vecteur par registre SSE) contre
//                    simd::Vec4SOA (SOA : 4 vecteurs par bloc, 1 registre par composante).
//                    Meme jeu de donnees, meme nombre de vecteurs traites, temps rapporte
//                    PAR VECTEUR. Resultats dans un second CSV (suffixe "_aos_soa").
//
//  Options :
//    --quick           mesures plus courtes (verification rapide)
//    --quiet           n'affiche que le resume final (pas les tables nanobench)
//    --filter=texte    ne lance que les benchmarks dont "groupe nom" contient le texte
//    --csv=chemin      fichier CSV de sortie (defaut : benchmark_results.csv)
//    --seed=N          graine des donnees aleatoires (defaut : aleatoire, affichee)
//
//  A compiler en Release (-O2/-O3 + -msse4.1, ou /O2), jamais en Debug.
// =====================================================================================

#include <nanobench.h>

#include "mathLibCPP/Vec3.h"
#include "mathLibCPP/Vec4.h"
#include "mathLibCPP/Matrix3x3.h"
#include "mathLibSIMD/include/Vec3f.hpp"
#include "mathLibSIMD/include/Vec4.hpp"
#include "mathLibSIMD/include/Mat3.hpp"
#include "mathLibSIMD/include/Vec4SOA.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace
{
    constexpr std::size_t kVecBatch = 1024; // vecteurs par lot
    constexpr std::size_t kMatBatch = 256;  // matrices par lot

    // Barriere compilateur : force le recalcul a chaque iteration (empeche le compilateur
    // de sortir l'expression de la boucle ou de la reduire a une constante).
    inline void CompilerBarrier()
    {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("" ::: "memory");
#elif defined(_MSC_VER)
        _ReadWriteBarrier();
#endif
    }

    std::string ToLower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    // ---------------------------------------------------------------------------------
    //  Options
    // ---------------------------------------------------------------------------------
    struct Options
    {
        bool quick = false;
        bool quiet = false;
        std::string filter;
        std::string csvPath = "benchmark_results.csv";
        std::uint32_t seed = std::random_device{}();
    };

    Options ParseArgs(int argc, char** argv)
    {
        Options opt;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--quick") opt.quick = true;
            else if (arg == "--quiet") opt.quiet = true;
            else if (arg.rfind("--filter=", 0) == 0) opt.filter = arg.substr(9);
            else if (arg.rfind("--csv=", 0) == 0) opt.csvPath = arg.substr(6);
            else if (arg.rfind("--seed=", 0) == 0) opt.seed = static_cast<std::uint32_t>(std::stoul(arg.substr(7)));
            else if (arg == "--help" || arg == "-h")
            {
                std::cout << "Usage : BenchmarkMath [--quick] [--quiet] [--filter=texte] [--csv=chemin] [--seed=N]\n"
                          << "  --filter=\"aos vs soa\" ne lance que la comparaison Vec4 / Vec4SOA\n";
                std::exit(0);
            }
            else std::cerr << "Option inconnue ignoree : " << arg << "\n";
        }
        return opt;
    }

    // ---------------------------------------------------------------------------------
    //  Donnees d'entree brutes (communes aux deux backends)
    //  Valeurs aleatoires a l'execution => aucune constante pliable par le compilateur.
    //  Composantes dans [0.5, 10] : pas de division par zero, Angle/Normalize valides.
    //  Matrices a diagonale dominante : toujours inversibles.
    // ---------------------------------------------------------------------------------
    struct Inputs
    {
        std::array<float, 4> va{}, vb{};
        std::array<float, 9> ma{}, mb{};
        float scalar{}, t{}, angX{}, angY{}, angZ{};
        std::vector<std::array<float, 4>> batchVa, batchVb;
        std::vector<std::array<float, 9>> batchMa, batchMb;
    };

    Inputs MakeInputs(std::uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> positive(0.5f, 10.0f);
        std::uniform_real_distribution<float> signedUnit(-1.0f, 1.0f);
        std::uniform_real_distribution<float> angle(0.1f, 3.0f);
        std::uniform_real_distribution<float> zeroOne(0.05f, 0.95f);

        auto vec4 = [&] { return std::array<float, 4>{ positive(rng), positive(rng), positive(rng), positive(rng) }; };
        auto mat = [&]
        {
            std::array<float, 9> m{};
            for (auto& e : m) e = signedUnit(rng);
            m[0] += 4.0f; m[4] += 4.0f; m[8] += 4.0f; // diagonale dominante
            return m;
        };

        Inputs in;
        in.va = vec4();
        in.vb = vec4();
        in.ma = mat();
        in.mb = mat();
        in.scalar = positive(rng);
        in.t = zeroOne(rng);
        in.angX = angle(rng);
        in.angY = angle(rng);
        in.angZ = angle(rng);
        for (std::size_t i = 0; i < kVecBatch; ++i) { in.batchVa.push_back(vec4()); in.batchVb.push_back(vec4()); }
        for (std::size_t i = 0; i < kMatBatch; ++i) { in.batchMa.push_back(mat()); in.batchMb.push_back(mat()); }
        return in;
    }

    // ---------------------------------------------------------------------------------
    //  Backends : types + petits adaptateurs pour ce qui differe d'une API a l'autre
    // ---------------------------------------------------------------------------------
    struct CppBackend
    {
        using Vec3 = Maths::Vec3<float>;
        using Vec4 = Maths::Vec4<float>;
        using Mat3 = Maths::Matrix3x3<float>;

        static Vec3 MakeVec3(float x, float y, float z) { return Vec3(x, y, z); }
        static Vec4 MakeVec4(float x, float y, float z, float w) { return Vec4(x, y, z, w); }
        static Mat3 MakeMat3(const std::array<float, 9>& e) { return Mat3(e); }

        static Mat3 Identity() { return Mat3::Identity(); }
        static Mat3 Zero() { return Mat3::Zero(); }
        static Mat3 Scale(const Vec3& s) { return Mat3::Scale(s); }
        static Mat3 RotX(float a) { return Mat3::RotationX(a); }
        static Mat3 RotY(float a) { return Mat3::RotationY(a); }
        static Mat3 RotZ(float a) { return Mat3::RotationZ(a); }

        static Vec3 Lerp3(const Vec3& a, const Vec3& b, float t) { return Vec3::Lerp(a, b, t); }
        static Vec3 Min3(const Vec3& a, const Vec3& b) { return Vec3::Min(a, b); }
        static Vec3 Max3(const Vec3& a, const Vec3& b) { return Vec3::Max(a, b); }
        static Vec4 Lerp4(const Vec4& a, const Vec4& b, float t) { return Vec4::Lerp(a, b, t); }
        static Vec4 Min4(const Vec4& a, const Vec4& b) { return Vec4::Min(a, b); }
        static Vec4 Max4(const Vec4& a, const Vec4& b) { return Vec4::Max(a, b); }

        static float GetX(const Vec3& v) { return v.x; }
        static float GetY(const Vec3& v) { return v.y; }
        static float GetZ(const Vec3& v) { return v.z; }
        static float GetX(const Vec4& v) { return v.x; }
        static float GetY(const Vec4& v) { return v.y; }
        static float GetZ(const Vec4& v) { return v.z; }
        static float GetW(const Vec4& v) { return v.w; }

        static void SetX(Vec3& v, float f) { v.x = f; }
        static void SetY(Vec3& v, float f) { v.y = f; }
        static void SetZ(Vec3& v, float f) { v.z = f; }
        static void SetX(Vec4& v, float f) { v.x = f; }
        static void SetY(Vec4& v, float f) { v.y = f; }
        static void SetZ(Vec4& v, float f) { v.z = f; }
        static void SetW(Vec4& v, float f) { v.w = f; }
    };

    struct SimdBackend
    {
        using Vec3 = simd::Vec3f;
        using Vec4 = simd::Vec4;
        using Mat3 = simd::Matrix3x3;

        static Vec3 MakeVec3(float x, float y, float z) { return Vec3(x, y, z); }
        static Vec4 MakeVec4(float x, float y, float z, float w) { return Vec4(x, y, z, w); }
        static Mat3 MakeMat3(const std::array<float, 9>& e) { return Mat3(e); }

        static Mat3 Identity() { return Mat3::Identity(); }
        static Mat3 Zero() { return Mat3::Zero(); }
        static Mat3 Scale(const Vec3& s) { return Mat3::Scale(s); }
        static Mat3 RotX(float a) { return Mat3::RotationX(a); }
        static Mat3 RotY(float a) { return Mat3::RotationY(a); }
        static Mat3 RotZ(float a) { return Mat3::RotationZ(a); }

        static Vec3 Lerp3(const Vec3& a, const Vec3& b, float t) { return Vec3::Lerp(a, b, t); }
        static Vec3 Min3(const Vec3& a, const Vec3& b) { return Vec3::Min(a, b); }
        static Vec3 Max3(const Vec3& a, const Vec3& b) { return Vec3::Max(a, b); }
        static Vec4 Lerp4(const Vec4& a, const Vec4& b, float t) { return Vec4::Lerp(a, b, t); }
        static Vec4 Min4(const Vec4& a, const Vec4& b) { return Vec4::Min(a, b); }
        static Vec4 Max4(const Vec4& a, const Vec4& b) { return Vec4::Max(a, b); }

        static float GetX(const Vec3& v) { return v.getX(); }
        static float GetY(const Vec3& v) { return v.getY(); }
        static float GetZ(const Vec3& v) { return v.getZ(); }
        static float GetX(const Vec4& v) { return v.getX(); }
        static float GetY(const Vec4& v) { return v.getY(); }
        static float GetZ(const Vec4& v) { return v.getZ(); }
        static float GetW(const Vec4& v) { return v.getW(); }

        static void SetX(Vec3& v, float f) { v.setX(f); }
        static void SetY(Vec3& v, float f) { v.setY(f); }
        static void SetZ(Vec3& v, float f) { v.setZ(f); }
        static void SetX(Vec4& v, float f) { v.setX(f); }
        static void SetY(Vec4& v, float f) { v.setY(f); }
        static void SetZ(Vec4& v, float f) { v.setZ(f); }
        static void SetW(Vec4& v, float f) { v.setW(f); }
    };

    // ---------------------------------------------------------------------------------
    //  Jeu de donnees d'un backend (construit depuis les memes Inputs)
    //  Les lambdas accedent aux adaptateurs statiques via `d.` (Data herite du backend).
    // ---------------------------------------------------------------------------------
    template <typename B>
    struct Data : B
    {
        using Vec3 = typename B::Vec3;
        using Vec4 = typename B::Vec4;
        using Mat3 = typename B::Mat3;

        float f0, f1, f2, f3;
        float scalar, t, angX, angY, angZ;
        std::array<float, 9> elems;

        Vec3 a3, b3, eq3; // eqN : copie de aN, pour tester == dans le pire cas
        Vec4 a4, b4, eq4;
        Mat3 ma, mb, meq;

        std::vector<Vec3> arr3a, arr3b, out3;
        std::vector<Vec4> arr4a, arr4b, out4;
        std::vector<Mat3> arrMa, arrMb, outM;

        explicit Data(const Inputs& in)
            : f0(in.va[0]), f1(in.va[1]), f2(in.va[2]), f3(in.va[3]),
              scalar(in.scalar), t(in.t), angX(in.angX), angY(in.angY), angZ(in.angZ),
              elems(in.ma),
              a3(B::MakeVec3(in.va[0], in.va[1], in.va[2])),
              b3(B::MakeVec3(in.vb[0], in.vb[1], in.vb[2])),
              eq3(a3),
              a4(B::MakeVec4(in.va[0], in.va[1], in.va[2], in.va[3])),
              b4(B::MakeVec4(in.vb[0], in.vb[1], in.vb[2], in.vb[3])),
              eq4(a4),
              ma(B::MakeMat3(in.ma)),
              mb(B::MakeMat3(in.mb)),
              meq(ma)
        {
            for (const auto& e : in.batchVa) { arr3a.push_back(B::MakeVec3(e[0], e[1], e[2])); arr4a.push_back(B::MakeVec4(e[0], e[1], e[2], e[3])); }
            for (const auto& e : in.batchVb) { arr3b.push_back(B::MakeVec3(e[0], e[1], e[2])); arr4b.push_back(B::MakeVec4(e[0], e[1], e[2], e[3])); }
            for (const auto& e : in.batchMa) arrMa.push_back(B::MakeMat3(e));
            for (const auto& e : in.batchMb) arrMb.push_back(B::MakeMat3(e));
            out3.resize(arr3a.size());
            out4.resize(arr4a.size());
            outM.resize(arrMa.size());
        }
    };

    using CppData = Data<CppBackend>;
    using SimdData = Data<SimdBackend>;

    // ---------------------------------------------------------------------------------
    //  Jeu de donnees AOS vs SOA (Vec4)
    //  Les deux layouts contiennent EXACTEMENT les memes kVecBatch vecteurs :
    //  le bloc SOA k contient les vecteurs 4k..4k+3 (un vecteur par "voie" SSE).
    // ---------------------------------------------------------------------------------
    static_assert(kVecBatch % 4 == 0, "kVecBatch doit etre un multiple de 4 pour le layout SOA");

    struct LayoutData
    {
        float scalar, t;

        std::vector<simd::Vec4> aosA, aosB, aosOut;      // 1 element = 1 vecteur
        std::vector<simd::Vec4SOA> soaA, soaB, soaOut;   // 1 element = 4 vecteurs

        explicit LayoutData(const Inputs& in) : scalar(in.scalar), t(in.t)
        {
            for (const auto& e : in.batchVa) aosA.emplace_back(e[0], e[1], e[2], e[3]);
            for (const auto& e : in.batchVb) aosB.emplace_back(e[0], e[1], e[2], e[3]);
            aosOut.resize(aosA.size());

            soaA = Pack(in.batchVa);
            soaB = Pack(in.batchVb);
            soaOut.resize(soaA.size());
        }

        std::size_t Count() const { return aosA.size(); }   // vecteurs
        std::size_t Blocks() const { return soaA.size(); }  // blocs de 4 vecteurs

    private:
        static std::vector<simd::Vec4SOA> Pack(const std::vector<std::array<float, 4>>& src)
        {
            std::vector<simd::Vec4SOA> out;
            out.reserve(src.size() / 4);
            for (std::size_t k = 0; k + 3 < src.size(); k += 4)
            {
                float x[4], y[4], z[4], w[4];
                for (std::size_t j = 0; j < 4; ++j)
                {
                    x[j] = src[k + j][0];
                    y[j] = src[k + j][1];
                    z[j] = src[k + j][2];
                    w[j] = src[k + j][3];
                }
                out.emplace_back(x, y, z, w);
            }
            return out;
        }
    };

    // Somme horizontale des 4 voies d'un registre (pour les reductions cote SOA).
    inline float HSum(__m128 v)
    {
        v = _mm_hadd_ps(v, v);
        v = _mm_hadd_ps(v, v);
        return _mm_cvtss_f32(v);
    }

    // ---------------------------------------------------------------------------------
    //  Suite : lance chaque lambda sur les deux backends et memorise les temps
    // ---------------------------------------------------------------------------------
    struct Row
    {
        std::string group;
        std::string name;
        double cppNs;
        double simdNs;
        double Speedup() const { return cppNs / simdNs; }
    };

    const char* Verdict(double speedup)
    {
        if (speedup >= 1.10) return "SIMD plus rapide";
        if (speedup <= 0.90) return "SIMD plus LENT";
        return "~ egal";
    }

    // Resultat d'une comparaison de layouts (temps en ns PAR VECTEUR)
    struct LayoutRow
    {
        std::string name;
        double aosNs;
        double soaNs;
        double Speedup() const { return aosNs / soaNs; }
    };

    const char* LayoutVerdict(double speedup)
    {
        if (speedup >= 1.10) return "SOA plus rapide";
        if (speedup <= 0.90) return "SOA plus LENT";
        return "~ egal";
    }

    class Suite
    {
    public:
        Suite(const Options& opt, CppData& cpp, SimdData& simdData, LayoutData& layout)
            : _opt(opt), _cpp(cpp), _simd(simdData), _layout(layout) {}

        void Group(const std::string& title)
        {
            _group = title;
            std::cout << "\n## " << title << "\n\n";
        }

        template <typename F>
        void Compare(const std::string& name, F&& fn, std::size_t batch = 1)
        {
            if (!Selected(name)) return;

            ankerl::nanobench::Bench bench;
            // Pas de '|' dans le titre : nanobench le met dans la cellule d'en-tete du tableau
            // markdown, et un '|' y ajoute une colonne fantome (l'en-tete n'a plus le meme
            // nombre de colonnes que la ligne de separation => tableau non reconnu).
            bench.title(_group + " / " + name)
                 .unit("op")
                 .batch(batch)
                 .warmup(1000)
                 .epochs(_opt.quick ? 8 : 25)
                 .minEpochTime(std::chrono::milliseconds(_opt.quick ? 2 : 10))
                 .relative(true)
                 .output(_opt.quiet ? nullptr : &std::cout);

            // Le 1er benchmark sert de reference (100 %) pour la colonne "relative".
            bench.run("Standard C++", [&]
            {
                CompilerBarrier();
                auto r = fn(_cpp);
                ankerl::nanobench::doNotOptimizeAway(r);
            });
            bench.run("SIMD (SSE4.1)", [&]
            {
                CompilerBarrier();
                auto r = fn(_simd);
                ankerl::nanobench::doNotOptimizeAway(r);
            });

            using Measure = ankerl::nanobench::Result::Measure;
            const auto& results = bench.results();
            const double perOp = 1e9 / static_cast<double>(batch);
            _rows.push_back({ _group, name,
                              results[0].median(Measure::elapsed) * perOp,
                              results[1].median(Measure::elapsed) * perOp });

            if (_opt.quiet)
            {
                const Row& r = _rows.back();
                std::cout << std::left << std::setw(70) << (_group + " | " + name)
                          << std::right << std::fixed << std::setprecision(2)
                          << std::setw(8) << r.Speedup() << "x  " << Verdict(r.Speedup()) << "\n";
            }
        }

        // Compare Vec4 (AOS) et Vec4SOA sur le MEME nombre de vecteurs (_layout.Count()).
        // fnAos / fnSoa sont des lambdas `[](LayoutData& d) { ... }` qui bouclent chacune sur
        // leur layout (Count() iterations cote AOS, Blocks() iterations cote SOA).
        template <typename FAos, typename FSoa>
        void CompareLayouts(const std::string& name, FAos&& fnAos, FSoa&& fnSoa)
        {
            if (!Selected(name)) return;

            const std::size_t batch = _layout.Count();

            ankerl::nanobench::Bench bench;
            bench.title(_group + " / " + name)
                 .unit("vec")
                 .batch(batch)
                 .warmup(1000)
                 .epochs(_opt.quick ? 8 : 25)
                 .minEpochTime(std::chrono::milliseconds(_opt.quick ? 2 : 10))
                 .relative(true)
                 .output(_opt.quiet ? nullptr : &std::cout);

            bench.run("Vec4 AOS (1 vec / registre)", [&]
            {
                CompilerBarrier();
                auto r = fnAos(_layout);
                ankerl::nanobench::doNotOptimizeAway(r);
            });
            bench.run("Vec4SOA (4 vec / bloc)", [&]
            {
                CompilerBarrier();
                auto r = fnSoa(_layout);
                ankerl::nanobench::doNotOptimizeAway(r);
            });

            using Measure = ankerl::nanobench::Result::Measure;
            const auto& results = bench.results();
            const double perOp = 1e9 / static_cast<double>(batch);
            _layoutRows.push_back({ name,
                                    results[0].median(Measure::elapsed) * perOp,
                                    results[1].median(Measure::elapsed) * perOp });

            if (_opt.quiet)
            {
                const LayoutRow& r = _layoutRows.back();
                std::cout << std::left << std::setw(70) << (_group + " | " + name)
                          << std::right << std::fixed << std::setprecision(2)
                          << std::setw(8) << r.Speedup() << "x  " << LayoutVerdict(r.Speedup()) << "\n";
            }
        }

        void PrintLayoutSummary() const
        {
            if (_layoutRows.empty()) return;

            std::cout << "\n\n## Resume Vec4 AOS vs Vec4SOA\n\n```text\n"
                      << "ns/vec = nanosecondes par vecteur (et non par bloc de 4)\n"
                      << "speedup = temps AOS / temps SOA  (> 1 : SOA plus rapide)\n\n";

            std::cout << std::left << std::setw(36) << "benchmark"
                      << std::right << std::setw(12) << "AOS ns/vec" << std::setw(12) << "SOA ns/vec"
                      << std::setw(10) << "speedup" << "  verdict\n";
            std::cout << std::string(86, '-') << "\n";

            double sumLog = 0.0;
            for (const LayoutRow& r : _layoutRows)
            {
                std::cout << std::left << std::setw(36) << r.name.substr(0, 35)
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(12) << r.aosNs << std::setw(12) << r.soaNs
                          << std::setprecision(2) << std::setw(9) << r.Speedup() << "x  "
                          << LayoutVerdict(r.Speedup()) << "\n";
                sumLog += std::log(r.Speedup());
            }

            std::cout << "\nmoyenne geometrique du speedup : " << std::fixed << std::setprecision(2)
                      << std::exp(sumLog / static_cast<double>(_layoutRows.size())) << "x\n";
            std::cout << "```\n";
        }

        void WriteLayoutCsv() const
        {
            if (_layoutRows.empty()) return;

            // "benchmark_results.csv" -> "benchmark_results_aos_soa.csv"
            std::string path = _opt.csvPath;
            const std::size_t dot = path.rfind('.');
            const std::size_t slash = path.find_last_of("/\\");
            if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
                path.insert(dot, "_aos_soa");
            else
                path += "_aos_soa";

            std::ofstream out(path);
            if (!out)
            {
                std::cerr << "Impossible d'ecrire " << path << "\n";
                return;
            }
            out << "benchmark,aos_ns_per_vec,soa_ns_per_vec,speedup\n";
            for (const LayoutRow& r : _layoutRows)
                out << '"' << r.name << "\"," << r.aosNs << ',' << r.soaNs << ',' << r.Speedup() << "\n";
            std::cout << "Resultats AOS/SOA ecrits dans " << path << "\n";
        }

        void PrintSummary() const
        {
            std::cout << "\n\n## Resume\n\n```text\n"
                      << "ns/op = nanosecondes par operation (par element pour les lots)\n"
                      << "speedup = temps C++ / temps SIMD  (> 1 : SIMD plus rapide)\n\n";

            std::cout << std::left << std::setw(44) << "benchmark"
                      << std::right << std::setw(12) << "C++ ns/op" << std::setw(12) << "SIMD ns/op"
                      << std::setw(10) << "speedup" << "  verdict\n";
            std::cout << std::string(96, '-') << "\n";

            std::string lastGroup;
            for (const Row& r : _rows)
            {
                if (r.group != lastGroup)
                {
                    std::cout << "--- " << r.group << " ---\n";
                    lastGroup = r.group;
                }
                std::cout << std::left << std::setw(44) << r.name.substr(0, 43)
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(12) << r.cppNs << std::setw(12) << r.simdNs
                          << std::setprecision(2) << std::setw(9) << r.Speedup() << "x  "
                          << Verdict(r.Speedup()) << "\n";
            }

            // Moyenne geometrique par groupe et globale
            std::cout << "\n--- moyenne geometrique du speedup ---\n";
            std::vector<std::string> groups;
            for (const Row& r : _rows)
                if (std::find(groups.begin(), groups.end(), r.group) == groups.end()) groups.push_back(r.group);
            double totalLog = 0.0;
            for (const std::string& g : groups)
            {
                double sumLog = 0.0;
                std::size_t n = 0;
                for (const Row& r : _rows)
                    if (r.group == g) { sumLog += std::log(r.Speedup()); ++n; }
                totalLog += sumLog;
                std::cout << std::left << std::setw(20) << g << std::right << std::fixed << std::setprecision(2)
                          << std::exp(sumLog / static_cast<double>(n)) << "x\n";
            }
            if (!_rows.empty())
                std::cout << std::left << std::setw(20) << "GLOBAL" << std::right
                          << std::exp(totalLog / static_cast<double>(_rows.size())) << "x\n";

            // Pires regressions / meilleurs gains
            std::vector<const Row*> sorted;
            for (const Row& r : _rows) sorted.push_back(&r);
            std::sort(sorted.begin(), sorted.end(), [](const Row* a, const Row* b) { return a->Speedup() < b->Speedup(); });

            std::cout << "\n--- SIMD plus lent que le scalaire (a optimiser en priorite) ---\n";
            int shown = 0;
            for (const Row* r : sorted)
            {
                if (r->Speedup() >= 1.0 || shown >= 10) break;
                std::cout << "  " << std::left << std::setw(56) << (r->group + " | " + r->name)
                          << std::right << std::setw(7) << r->Speedup() << "x\n";
                ++shown;
            }
            if (shown == 0) std::cout << "  (aucun)\n";

            std::cout << "\n--- plus gros gains SIMD ---\n";
            shown = 0;
            for (auto it = sorted.rbegin(); it != sorted.rend() && shown < 10; ++it, ++shown)
                std::cout << "  " << std::left << std::setw(56) << ((*it)->group + " | " + (*it)->name)
                          << std::right << std::setw(7) << (*it)->Speedup() << "x\n";
            std::cout << "```\n";
        }

        void WriteCsv() const
        {
            std::ofstream out(_opt.csvPath);
            if (!out)
            {
                std::cerr << "Impossible d'ecrire " << _opt.csvPath << "\n";
                return;
            }
            out << "group,benchmark,cpp_ns_per_op,simd_ns_per_op,speedup\n";
            for (const Row& r : _rows)
                out << '"' << r.group << "\",\"" << r.name << "\"," << r.cppNs << ',' << r.simdNs << ',' << r.Speedup() << "\n";
            std::cout << "\nResultats ecrits dans " << _opt.csvPath << "\n";
        }

    private:
        bool Selected(const std::string& name) const
        {
            if (_opt.filter.empty()) return true;
            return ToLower(_group + " " + name).find(ToLower(_opt.filter)) != std::string::npos;
        }

        const Options& _opt;
        CppData& _cpp;
        SimdData& _simd;
        LayoutData& _layout;
        std::string _group;
        std::vector<Row> _rows;
        std::vector<LayoutRow> _layoutRows;
    };

    // =================================================================================
    //  Vec3
    // =================================================================================
    void BenchVec3(Suite& s)
    {
        s.Group("Vec3");

        s.Compare("construct (x,y,z)", [](auto& d) { return d.MakeVec3(d.f0, d.f1, d.f2); });
        s.Compare("read x+y+z", [](auto& d) { return d.GetX(d.a3) + d.GetY(d.a3) + d.GetZ(d.a3); });
        s.Compare("write x,y,z", [](auto& d) { auto v = d.a3; d.SetX(v, d.f0); d.SetY(v, d.f1); d.SetZ(v, d.f2); return v; });

        s.Compare("a + b", [](auto& d) { return d.a3 + d.b3; });
        s.Compare("a - b", [](auto& d) { return d.a3 - d.b3; });
        s.Compare("a * b (component)", [](auto& d) { return d.a3 * d.b3; });
        s.Compare("a / b (component)", [](auto& d) { return d.a3 / d.b3; });
        s.Compare("-a", [](auto& d) { return -d.a3; });
        s.Compare("a * scalar", [](auto& d) { return d.a3 * d.scalar; });
        s.Compare("scalar * a", [](auto& d) { return d.scalar * d.a3; });
        s.Compare("a / scalar", [](auto& d) { return d.a3 / d.scalar; });

        s.Compare("a += b", [](auto& d) { auto v = d.a3; v += d.b3; return v; });
        s.Compare("a -= b", [](auto& d) { auto v = d.a3; v -= d.b3; return v; });
        s.Compare("a *= b", [](auto& d) { auto v = d.a3; v *= d.b3; return v; });
        s.Compare("a /= b", [](auto& d) { auto v = d.a3; v /= d.b3; return v; });
        s.Compare("a *= scalar", [](auto& d) { auto v = d.a3; v *= d.scalar; return v; });
        s.Compare("a /= scalar", [](auto& d) { auto v = d.a3; v /= d.scalar; return v; });

        s.Compare("a == a' (equal, worst case)", [](auto& d) { return d.a3 == d.eq3; });
        s.Compare("a == b (different)", [](auto& d) { return d.a3 == d.b3; });
        s.Compare("a != a' (equal)", [](auto& d) { return d.a3 != d.eq3; });

        s.Compare("Dot", [](auto& d) { return d.a3.Dot(d.b3); });
        s.Compare("Cross", [](auto& d) { return d.a3.Cross(d.b3); });
        s.Compare("MagnitudeSquared", [](auto& d) { return d.a3.MagnitudeSquared(); });
        s.Compare("Magnitude", [](auto& d) { return d.a3.Magnitude(); });
        s.Compare("Normalize", [](auto& d) { return d.a3.Normalize(); });
        s.Compare("DistanceSquared", [](auto& d) { return d.a3.DistanceSquared(d.b3); });
        s.Compare("Distance", [](auto& d) { return d.a3.Distance(d.b3); });
        s.Compare("Angle", [](auto& d) { return d.a3.Angle(d.b3); });
        s.Compare("Lerp", [](auto& d) { return d.Lerp3(d.a3, d.b3, d.t); });
        s.Compare("Min", [](auto& d) { return d.Min3(d.a3, d.b3); });
        s.Compare("Max", [](auto& d) { return d.Max3(d.a3, d.b3); });
    }

    // =================================================================================
    //  Vec4
    // =================================================================================
    void BenchVec4(Suite& s)
    {
        s.Group("Vec4");

        s.Compare("construct (x,y,z,w)", [](auto& d) { return d.MakeVec4(d.f0, d.f1, d.f2, d.f3); });
        s.Compare("read x+y+z+w", [](auto& d) { return d.GetX(d.a4) + d.GetY(d.a4) + d.GetZ(d.a4) + d.GetW(d.a4); });
        s.Compare("write x,y,z,w", [](auto& d) { auto v = d.a4; d.SetX(v, d.f0); d.SetY(v, d.f1); d.SetZ(v, d.f2); d.SetW(v, d.f3); return v; });

        s.Compare("a + b", [](auto& d) { return d.a4 + d.b4; });
        s.Compare("a - b", [](auto& d) { return d.a4 - d.b4; });
        s.Compare("a * b (component)", [](auto& d) { return d.a4 * d.b4; });
        s.Compare("a / b (component)", [](auto& d) { return d.a4 / d.b4; });
        s.Compare("-a", [](auto& d) { return -d.a4; });
        s.Compare("a * scalar", [](auto& d) { return d.a4 * d.scalar; });
        s.Compare("scalar * a", [](auto& d) { return d.scalar * d.a4; });
        s.Compare("a / scalar", [](auto& d) { return d.a4 / d.scalar; });

        s.Compare("a += b", [](auto& d) { auto v = d.a4; v += d.b4; return v; });
        s.Compare("a -= b", [](auto& d) { auto v = d.a4; v -= d.b4; return v; });
        s.Compare("a *= b", [](auto& d) { auto v = d.a4; v *= d.b4; return v; });
        s.Compare("a /= b", [](auto& d) { auto v = d.a4; v /= d.b4; return v; });
        s.Compare("a *= scalar", [](auto& d) { auto v = d.a4; v *= d.scalar; return v; });
        s.Compare("a /= scalar", [](auto& d) { auto v = d.a4; v /= d.scalar; return v; });

        s.Compare("a == a' (equal, worst case)", [](auto& d) { return d.a4 == d.eq4; });
        s.Compare("a == b (different)", [](auto& d) { return d.a4 == d.b4; });
        s.Compare("a != a' (equal)", [](auto& d) { return d.a4 != d.eq4; });

        s.Compare("Dot", [](auto& d) { return d.a4.Dot(d.b4); });
        s.Compare("MagnitudeSquared", [](auto& d) { return d.a4.MagnitudeSquared(); });
        s.Compare("Magnitude", [](auto& d) { return d.a4.Magnitude(); });
        s.Compare("Normalize", [](auto& d) { return d.a4.Normalize(); });
        s.Compare("DistanceSquared", [](auto& d) { return d.a4.DistanceSquared(d.b4); });
        s.Compare("Distance", [](auto& d) { return d.a4.Distance(d.b4); });
        s.Compare("Angle", [](auto& d) { return d.a4.Angle(d.b4); });
        s.Compare("Lerp", [](auto& d) { return d.Lerp4(d.a4, d.b4, d.t); });
        s.Compare("Min", [](auto& d) { return d.Min4(d.a4, d.b4); });
        s.Compare("Max", [](auto& d) { return d.Max4(d.a4, d.b4); });
    }

    // =================================================================================
    //  Matrix3x3
    // =================================================================================
    void BenchMatrix3x3(Suite& s)
    {
        s.Group("Matrix3x3");

        s.Compare("default ctor", [](auto& d) { return decltype(d.ma){}; });
        s.Compare("Identity()", [](auto& d) { return d.Identity(); });
        s.Compare("Zero()", [](auto& d) { return d.Zero(); });
        s.Compare("construct from array<9>", [](auto& d) { return d.MakeMat3(d.elems); });

        s.Compare("operator() read x9", [](auto& d)
        {
            float sum = 0.0f;
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                    sum += d.ma(r, c);
            return sum;
        });
        s.Compare("operator() write x9", [](auto& d)
        {
            auto m = d.ma;
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 3; ++c)
                    m(r, c) = d.scalar;
            return m;
        });

        s.Compare("A + B", [](auto& d) { return d.ma + d.mb; });
        s.Compare("A - B", [](auto& d) { return d.ma - d.mb; });
        s.Compare("A * B", [](auto& d) { return d.ma * d.mb; });
        s.Compare("A * vec3", [](auto& d) { return d.ma * d.a3; });
        s.Compare("A * scalar", [](auto& d) { return d.ma * d.scalar; });
        s.Compare("A *= B", [](auto& d) { auto m = d.ma; m *= d.mb; return m; });

        s.Compare("A == A' (equal, worst case)", [](auto& d) { return d.ma == d.meq; });
        s.Compare("A == B (different)", [](auto& d) { return d.ma == d.mb; });
        s.Compare("A != A' (equal)", [](auto& d) { return d.ma != d.meq; });

        s.Compare("Transpose", [](auto& d) { return d.ma.Transpose(); });
        s.Compare("Determinant", [](auto& d) { return d.ma.Determinant(); });
        s.Compare("Inverse", [](auto& d) { return d.ma.Inverse(); });
        s.Compare("Inverse(A) * vec3 (solve)", [](auto& d) { return d.ma.Inverse() * d.a3; });

        s.Compare("Scale(vec3)", [](auto& d) { return d.Scale(d.a3); });
        s.Compare("RotationX", [](auto& d) { return d.RotX(d.angX); });
        s.Compare("RotationY", [](auto& d) { return d.RotY(d.angY); });
        s.Compare("RotationZ", [](auto& d) { return d.RotZ(d.angZ); });

        s.Compare("Rz * Ry * Rx (build + compose)", [](auto& d) { return d.RotZ(d.angZ) * d.RotY(d.angY) * d.RotX(d.angX); });
        s.Compare("(Scale * Rz) * vec3 (pipeline)", [](auto& d) { return (d.Scale(d.a3) * d.RotZ(d.angZ)) * d.b3; });
        s.Compare("A * B * vec3", [](auto& d) { return (d.ma * d.mb) * d.a3; });
    }

    // =================================================================================
    //  Lots : memes operations sur des tableaux (temps rapporte PAR ELEMENT)
    //  Les resultats sont ecrits dans des tableaux de sortie pour eviter l'elimination.
    // =================================================================================
    void BenchBatch(Suite& s)
    {
        s.Group("Batch");

        // ---- Vec3 ----
        s.Compare("[Vec3] out[i] = a[i] + b[i]", [](auto& d)
        {
            const std::size_t n = d.arr3a.size();
            for (std::size_t i = 0; i < n; ++i) d.out3[i] = d.arr3a[i] + d.arr3b[i];
            return d.out3.data();
        }, kVecBatch);

        s.Compare("[Vec3] out[i] = a[i] * scalar", [](auto& d)
        {
            const std::size_t n = d.arr3a.size();
            for (std::size_t i = 0; i < n; ++i) d.out3[i] = d.arr3a[i] * d.scalar;
            return d.out3.data();
        }, kVecBatch);

        s.Compare("[Vec3] sum of Dot(a[i], b[i])", [](auto& d)
        {
            const std::size_t n = d.arr3a.size();
            float acc = 0.0f;
            for (std::size_t i = 0; i < n; ++i) acc += d.arr3a[i].Dot(d.arr3b[i]);
            return acc;
        }, kVecBatch);

        s.Compare("[Vec3] out[i] = Cross(a[i], b[i])", [](auto& d)
        {
            const std::size_t n = d.arr3a.size();
            for (std::size_t i = 0; i < n; ++i) d.out3[i] = d.arr3a[i].Cross(d.arr3b[i]);
            return d.out3.data();
        }, kVecBatch);

        s.Compare("[Vec3] out[i] = Normalize(a[i])", [](auto& d)
        {
            const std::size_t n = d.arr3a.size();
            for (std::size_t i = 0; i < n; ++i) d.out3[i] = d.arr3a[i].Normalize();
            return d.out3.data();
        }, kVecBatch);

        s.Compare("[Vec3] out[i] = Lerp(a[i], b[i], t)", [](auto& d)
        {
            const std::size_t n = d.arr3a.size();
            for (std::size_t i = 0; i < n; ++i) d.out3[i] = d.Lerp3(d.arr3a[i], d.arr3b[i], d.t);
            return d.out3.data();
        }, kVecBatch);

        // ---- Vec4 ----
        s.Compare("[Vec4] out[i] = a[i] + b[i]", [](auto& d)
        {
            const std::size_t n = d.arr4a.size();
            for (std::size_t i = 0; i < n; ++i) d.out4[i] = d.arr4a[i] + d.arr4b[i];
            return d.out4.data();
        }, kVecBatch);

        s.Compare("[Vec4] out[i] = a[i] * scalar", [](auto& d)
        {
            const std::size_t n = d.arr4a.size();
            for (std::size_t i = 0; i < n; ++i) d.out4[i] = d.arr4a[i] * d.scalar;
            return d.out4.data();
        }, kVecBatch);

        s.Compare("[Vec4] sum of Dot(a[i], b[i])", [](auto& d)
        {
            const std::size_t n = d.arr4a.size();
            float acc = 0.0f;
            for (std::size_t i = 0; i < n; ++i) acc += d.arr4a[i].Dot(d.arr4b[i]);
            return acc;
        }, kVecBatch);

        s.Compare("[Vec4] out[i] = Normalize(a[i])", [](auto& d)
        {
            const std::size_t n = d.arr4a.size();
            for (std::size_t i = 0; i < n; ++i) d.out4[i] = d.arr4a[i].Normalize();
            return d.out4.data();
        }, kVecBatch);

        s.Compare("[Vec4] out[i] = Lerp(a[i], b[i], t)", [](auto& d)
        {
            const std::size_t n = d.arr4a.size();
            for (std::size_t i = 0; i < n; ++i) d.out4[i] = d.Lerp4(d.arr4a[i], d.arr4b[i], d.t);
            return d.out4.data();
        }, kVecBatch);

        s.Compare("[Vec4] out[i] = Min(a[i], b[i])", [](auto& d)
        {
            const std::size_t n = d.arr4a.size();
            for (std::size_t i = 0; i < n; ++i) d.out4[i] = d.Min4(d.arr4a[i], d.arr4b[i]);
            return d.out4.data();
        }, kVecBatch);

        // ---- Matrix3x3 ----
        s.Compare("[Mat3] out[i] = A * v[i] (1 matrice, N vec)", [](auto& d)
        {
            const std::size_t n = d.arr3a.size();
            for (std::size_t i = 0; i < n; ++i) d.out3[i] = d.ma * d.arr3a[i];
            return d.out3.data();
        }, kVecBatch);

        s.Compare("[Mat3] out[i] = A[i] * v[i]", [](auto& d)
        {
            const std::size_t n = d.arrMa.size();
            for (std::size_t i = 0; i < n; ++i) d.out3[i] = d.arrMa[i] * d.arr3a[i];
            return d.out3.data();
        }, kMatBatch);

        s.Compare("[Mat3] out[i] = A[i] * B[i]", [](auto& d)
        {
            const std::size_t n = d.arrMa.size();
            for (std::size_t i = 0; i < n; ++i) d.outM[i] = d.arrMa[i] * d.arrMb[i];
            return d.outM.data();
        }, kMatBatch);

        s.Compare("[Mat3] out[i] = Transpose(A[i])", [](auto& d)
        {
            const std::size_t n = d.arrMa.size();
            for (std::size_t i = 0; i < n; ++i) d.outM[i] = d.arrMa[i].Transpose();
            return d.outM.data();
        }, kMatBatch);

        s.Compare("[Mat3] sum of Determinant(A[i])", [](auto& d)
        {
            const std::size_t n = d.arrMa.size();
            float acc = 0.0f;
            for (std::size_t i = 0; i < n; ++i) acc += d.arrMa[i].Determinant();
            return acc;
        }, kMatBatch);

        s.Compare("[Mat3] out[i] = Inverse(A[i])", [](auto& d)
        {
            const std::size_t n = d.arrMa.size();
            for (std::size_t i = 0; i < n; ++i) d.outM[i] = d.arrMa[i].Inverse();
            return d.outM.data();
        }, kMatBatch);
    }

    // =================================================================================
    //  Vec4 : AOS (simd::Vec4) vs SOA (simd::Vec4SOA)
    //  Meme operation sur les memes kVecBatch vecteurs. Cote AOS : kVecBatch iterations ;
    //  cote SOA : kVecBatch / 4 iterations (4 vecteurs par bloc). Temps rapporte PAR VECTEUR.
    //  Les reductions SOA accumulent dans un registre puis font UNE somme horizontale finale.
    // =================================================================================
    void BenchVec4Layout(Suite& s)
    {
        s.Group("Vec4 AOS vs SOA");

        // ---- operateurs arithmetiques ----
        s.CompareLayouts("out = a + b",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = d.aosA[i] + d.aosB[i]; return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = d.soaA[i] + d.soaB[i]; return d.soaOut.data(); });

        s.CompareLayouts("out = a - b",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = d.aosA[i] - d.aosB[i]; return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = d.soaA[i] - d.soaB[i]; return d.soaOut.data(); });

        s.CompareLayouts("out = a * b (component)",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = d.aosA[i] * d.aosB[i]; return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = d.soaA[i] * d.soaB[i]; return d.soaOut.data(); });

        s.CompareLayouts("out = a / b (component)",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = d.aosA[i] / d.aosB[i]; return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = d.soaA[i] / d.soaB[i]; return d.soaOut.data(); });

        s.CompareLayouts("out = -a",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = -d.aosA[i]; return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = -d.soaA[i]; return d.soaOut.data(); });

        s.CompareLayouts("out = a * scalar",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = d.aosA[i] * d.scalar; return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = d.soaA[i] * d.scalar; return d.soaOut.data(); });

        s.CompareLayouts("out = a / scalar",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = d.aosA[i] / d.scalar; return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = d.soaA[i] / d.scalar; return d.soaOut.data(); });

        s.CompareLayouts("a += b (in place)",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] += d.aosB[i]; return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] += d.soaB[i]; return d.soaOut.data(); });

        // ---- produits scalaires / normes (reductions) ----
        s.CompareLayouts("sum of Dot(a, b)",
            [](LayoutData& d) { float acc = 0.0f; for (std::size_t i = 0; i < d.Count(); ++i) acc += d.aosA[i].Dot(d.aosB[i]); return acc; },
            [](LayoutData& d) { __m128 acc = _mm_setzero_ps(); for (std::size_t i = 0; i < d.Blocks(); ++i) acc = _mm_add_ps(acc, d.soaA[i].Dot(d.soaB[i])); return HSum(acc); });

        s.CompareLayouts("sum of MagnitudeSquared(a)",
            [](LayoutData& d) { float acc = 0.0f; for (std::size_t i = 0; i < d.Count(); ++i) acc += d.aosA[i].MagnitudeSquared(); return acc; },
            [](LayoutData& d) { __m128 acc = _mm_setzero_ps(); for (std::size_t i = 0; i < d.Blocks(); ++i) acc = _mm_add_ps(acc, d.soaA[i].MagnitudeSquared()); return HSum(acc); });

        s.CompareLayouts("sum of Magnitude(a)",
            [](LayoutData& d) { float acc = 0.0f; for (std::size_t i = 0; i < d.Count(); ++i) acc += d.aosA[i].Magnitude(); return acc; },
            [](LayoutData& d) { __m128 acc = _mm_setzero_ps(); for (std::size_t i = 0; i < d.Blocks(); ++i) acc = _mm_add_ps(acc, d.soaA[i].Magnitude()); return HSum(acc); });

        s.CompareLayouts("sum of DistanceSquared(a, b)",
            [](LayoutData& d) { float acc = 0.0f; for (std::size_t i = 0; i < d.Count(); ++i) acc += d.aosA[i].DistanceSquared(d.aosB[i]); return acc; },
            [](LayoutData& d) { __m128 acc = _mm_setzero_ps(); for (std::size_t i = 0; i < d.Blocks(); ++i) acc = _mm_add_ps(acc, d.soaA[i].DistanceSquared(d.soaB[i])); return HSum(acc); });

        s.CompareLayouts("sum of Distance(a, b)",
            [](LayoutData& d) { float acc = 0.0f; for (std::size_t i = 0; i < d.Count(); ++i) acc += d.aosA[i].Distance(d.aosB[i]); return acc; },
            [](LayoutData& d) { __m128 acc = _mm_setzero_ps(); for (std::size_t i = 0; i < d.Blocks(); ++i) acc = _mm_add_ps(acc, d.soaA[i].Distance(d.soaB[i])); return HSum(acc); });

        // ---- fonctions vectorielles ----
        s.CompareLayouts("out = Normalize(a)",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = d.aosA[i].Normalize(); return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = d.soaA[i].Normalize(); return d.soaOut.data(); });

        s.CompareLayouts("out = Lerp(a, b, t)",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = simd::Vec4::Lerp(d.aosA[i], d.aosB[i], d.t); return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = simd::Vec4SOA::Lerp(d.soaA[i], d.soaB[i], d.t); return d.soaOut.data(); });

        s.CompareLayouts("out = Min(a, b)",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = simd::Vec4::Min(d.aosA[i], d.aosB[i]); return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = simd::Vec4SOA::Min(d.soaA[i], d.soaB[i]); return d.soaOut.data(); });

        s.CompareLayouts("out = Max(a, b)",
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Count(); ++i) d.aosOut[i] = simd::Vec4::Max(d.aosA[i], d.aosB[i]); return d.aosOut.data(); },
            [](LayoutData& d) { for (std::size_t i = 0; i < d.Blocks(); ++i) d.soaOut[i] = simd::Vec4SOA::Max(d.soaA[i], d.soaB[i]); return d.soaOut.data(); });

        // ---- enchainement (le SOA reste en registres, l'AOS paie ses reductions) ----
        s.CompareLayouts("sum of Dot(Normalize(a), b)",
            [](LayoutData& d) { float acc = 0.0f; for (std::size_t i = 0; i < d.Count(); ++i) acc += d.aosA[i].Normalize().Dot(d.aosB[i]); return acc; },
            [](LayoutData& d) { __m128 acc = _mm_setzero_ps(); for (std::size_t i = 0; i < d.Blocks(); ++i) acc = _mm_add_ps(acc, d.soaA[i].Normalize().Dot(d.soaB[i])); return HSum(acc); });
    }
}

int main(int argc, char** argv)
{
    const Options opt = ParseArgs(argc, argv);

    std::cout << "BenchmarkMath : mathLibCPP (scalaire) vs mathLibSIMD (SSE4.1)\n"
              << "graine = " << opt.seed << (opt.quick ? "  [mode --quick]" : "") << "\n";
#ifndef NDEBUG
    std::cout << "ATTENTION : NDEBUG non defini (build Debug ?). Les resultats seront faux : compile en Release.\n";
#endif

    const Inputs inputs = MakeInputs(opt.seed);
    CppData cppData(inputs);
    SimdData simdData(inputs);

    LayoutData layoutData(inputs);

    Suite suite(opt, cppData, simdData, layoutData);
    BenchVec3(suite);
    BenchVec4(suite);
    BenchMatrix3x3(suite);
    BenchBatch(suite);
    BenchVec4Layout(suite);

    suite.PrintSummary();
    suite.PrintLayoutSummary();
    suite.WriteCsv();
    suite.WriteLayoutCsv();
    return 0;
}