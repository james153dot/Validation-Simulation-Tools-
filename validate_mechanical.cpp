// validate_mech.cpp
// Build (OpenMP preferred):
//   g++ -O3 -std=c++17 -fopenmp validate_mech.cpp -o validate_mech
// Fallback (no OpenMP):
//   g++ -O3 -std=c++17 validate_mech.cpp -o validate_mech
// Run:
//   ./validate_mech --threads 8

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <future>
#include <chrono>
#include <random>
#include <algorithm>

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double elapsed_s;
};

static inline double rel_err(double a, double b) {
    double denom = std::max(1e-12, std::abs(a) + std::abs(b));
    return std::abs(a - b) / denom;
}

// -------------------- Analytic baselines --------------------
static inline double beam_tip_deflection(double F, double L, double E, double I) {
    return F * std::pow(L, 3) / (3.0 * E * I);
}

static inline double sdof_wn(double k, double m) { return std::sqrt(k / m); }
static inline double sdof_zeta(double c, double k, double m) { return c / (2.0 * std::sqrt(k * m)); }
static inline double sdof_step_overshoot(double zeta) {
    if (zeta <= 0.0 || zeta >= 1.0) return 0.0;
    return std::exp(-zeta * M_PI / std::sqrt(1.0 - zeta * zeta));
}
static inline double spring_work(double k, double x) { return 0.5 * k * x * x; }

// -------------------- Tiny test macros ----------------------
#define ASSERT_NEAR(a,b,tol) do { \
    double __ea = (a), __eb = (b); \
    double __re = rel_err(__ea, __eb); \
    if (__re > (tol)) return TestResult{__func__, false, \
        std::string("rel_error=") + std::to_string(__re) + \
        ", a=" + std::to_string(__ea) + ", b=" + std::to_string(__eb) + \
        ", tol=" + std::to_string(tol), 0.0}; \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) return TestResult{__func__, false, (msg), 0.0}; \
} while(0)

// -------------------- Individual tests ----------------------

TestResult test_beam_case(unsigned seed) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uF(-1, 3), uL(-2, 0), uE(9, 12), uI(-12, -6);

    for (int i = 0; i < 50; ++i) {
        double F = std::pow(10.0, uF(rng));
        double L = std::pow(10.0, uL(rng));
        double E = std::pow(10.0, uE(rng));
        double I = std::pow(10.0, uI(rng));
        double model = beam_tip_deflection(F, L, E, I);
        double hw = model; // replace with hardware/solver output
        double tol = 1e-5;
        double re = rel_err(hw, model);
        if (re > tol) {
            return {"beam_case", false, "beam rel_error=" + std::to_string(re), 0.0};
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    return {"beam_case", true, "ok", dt.count()};
}

TestResult test_sdof_case(unsigned seed) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> um(-2, 1), uk(1, 5), uz(0.05, 0.6);

    for (int i = 0; i < 50; ++i) {
        double m = std::pow(10.0, um(rng));
        double k = std::pow(10.0, uk(rng));
        double z = uz(rng);
        double c = 2.0 * z * std::sqrt(k * m);
        double wn = sdof_wn(k, m);
        double Mp = sdof_step_overshoot(sdof_zeta(c, k, m));
        double hw_wn = wn; // replace with measured/solver value
        double hw_Mp = Mp;
        if (rel_err(hw_wn, wn) > 1e-3) {
            return {"sdof_case", false, "wn mismatch", 0.0};
        }
        if (rel_err(hw_Mp, Mp) > 2e-2) {
            return {"sdof_case", false, "Mp mismatch", 0.0};
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    return {"sdof_case", true, "ok", dt.count()};
}

TestResult test_spring_energy_equilibrium(unsigned seed) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uk(0, 5), ux(-4, -1);

    for (int i = 0; i < 50; ++i) {
        double k = std::pow(10.0, uk(rng));
        double x = std::pow(10.0, ux(rng));
        double W = spring_work(k, x);
        // trapezoidal integration of F= kx
        int N = 200;
        double Wnum = 0.0;
        double dx = x / N;
        for (int j = 0; j < N; ++j) {
            double xi = j * dx;
            double xj = (j + 1) * dx;
            double Fi = k * xi, Fj = k * xj;
            Wnum += 0.5 * (Fi + Fj) * (xj - xi);
        }
        if (rel_err(Wnum, W) > 1e-3)
            return {"spring_energy", false, "energy mismatch", 0.0};
        // equilibrium
        double Fext = k * x; // replace with measured
        if (rel_err(Fext, k * x) > 1e-5)
            return {"spring_equil", false, "equilibrium mismatch", 0.0};
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    return {"spring_energy_equilibrium", true, "ok", dt.count()};
}

// -------------------- Runner & parallelism --------------------

struct Args { int threads = 0; };

Args parse_args(int argc, char** argv) {
    Args a; a.threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s == "--threads" && i + 1 < argc) { a.threads = std::stoi(argv[++i]); }
    }
    return a;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    std::vector<TestResult> results;

    // Create a set of per-test seeds so they are independent in parallel
    struct Job { std::string name; unsigned seed; TestResult (*fn)(unsigned); };
    std::vector<Job> jobs = {
        {"beam",   1337u, test_beam_case},
        {"sdof",   2025u, test_sdof_case},
        {"spring", 9001u, test_spring_energy_equilibrium},
    };

    auto t0 = std::chrono::high_resolution_clock::now();

#if defined(_OPENMP)
    int nth = args.threads > 0 ? args.threads : 0; // 0 lets OMP decide
    if (nth > 0) {
        omp_set_num_threads(nth);
    }
    results.resize(jobs.size());
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)jobs.size(); ++i) {
        const auto& jb = jobs[i];
        results[i] = jb.fn(jb.seed);
        results[i].name = jb.name;
    }
#else
    int nth = args.threads > 0 ? args.threads : std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::future<TestResult>> futs;
    futs.reserve(jobs.size());
    for (const auto& jb : jobs) {
        futs.emplace_back(std::async(std::launch::async, [jb]{ return jb.fn(jb.seed); }));
    }
    for (auto& f : futs) {
        results.emplace_back(f.get());
    }
#endif

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    // Summary
    int fails = 0;
    for (const auto& r : results) {
        bool ok = r.passed;
        if (!ok) ++fails;
        std::cout << (ok ? "\x1b[92mPASS\x1b[0m " : "\x1b[91mFAIL\x1b[0m ")
                  << r.name << " | " << r.message << "\n";
    }
    std::cout << "Total: " << results.size() << ", Failures: " << fails
              << ", Time: " << dt.count() << "s\n";
    return fails == 0 ? 0 : 1;
}
