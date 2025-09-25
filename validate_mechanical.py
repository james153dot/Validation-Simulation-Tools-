#!/usr/bin/env python3
"""
Parallel mechanical validation runner.

Usage:
  python validate_mech.py --procs 8 --junit out.xml --seed 42

Tests implemented (each with sweeps):
  1) Cantilever beam tip deflection vs Euler–Bernoulli closed form
  2) SDOF natural frequency & step-response overshoot vs closed form
  3) Linear spring energy consistency (∫F·dx) & static equilibrium

Parallelization:
  - Each parameterized test case is independent and farmed to worker processes
    via multiprocessing.Pool. Results are reduced into a summary + optional JUnit XML.
"""

from __future__ import annotations
import math, argparse, itertools, os, sys, time, random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
from multiprocessing import Pool, cpu_count

# ----------------------------- Utilities -----------------------------

@dataclass
class TestResult:
    name: str
    params: Dict[str, float]
    passed: bool
    message: str
    elapsed_s: float


def relative_error(a: float, b: float) -> float:
    denom = max(1e-12, abs(a) + abs(b))
    return abs(a - b) / denom


# Optional: tiny JUnit writer (no external deps)
def write_junit_xml(results: List[TestResult], path: str) -> None:
    import xml.sax.saxutils as sx
    ts_name = "MechanicalValidation"
    tests = len(results)
    failures = sum(0 if r.passed else 1 for r in results)
    time_sum = sum(r.elapsed_s for r in results)
    lines = [
        f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<testsuite name=\"{ts_name}\" tests=\"{tests}\" failures=\"{failures}\" time=\"{time_sum:.6f}\">",
    ]
    for r in results:
        cname = sx.escape(r.name)
        pstr = ", ".join(f"{k}={v:.6g}" for k, v in r.params.items())
        lines.append(f"  <testcase classname=\"{ts_name}\" name=\"{cname}\" time=\"{r.elapsed_s:.6f}\">")
        if not r.passed:
            msg = sx.escape(r.message)
            lines.append(f"    <failure message=\"{msg}\" />")
        # add properties for params
        if r.params:
            lines.append("    <properties>")
            for k, v in r.params.items():
                lines.append(f"      <property name=\"{sx.escape(k)}\" value=\"{v}\"/>")
            lines.append("    </properties>")
        lines.append("  </testcase>")
    lines.append("</testsuite>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------- Analytic Models ---------------------------

# 1) Euler–Bernoulli cantilever beam, end load F at free tip: tip deflection
#    delta = F L^3 / (3 E I)

def beam_tip_deflection(F: float, L: float, E: float, I: float) -> float:
    return F * L**3 / (3.0 * E * I)


# 2) SDOF vibration properties (linear):
#    w_n = sqrt(k/m), zeta = c/(2*sqrt(k m))
#    Step response peak overshoot: M_p = exp(-zeta*pi/sqrt(1-zeta^2)), 0<zeta<1

def sdof_wn(k: float, m: float) -> float:
    return math.sqrt(k / m)


def sdof_zeta(c: float, k: float, m: float) -> float:
    return c / (2.0 * math.sqrt(k * m))


def sdof_step_overshoot(zeta: float) -> float:
    if zeta <= 0 or zeta >= 1:
        return 0.0
    return math.exp(-zeta * math.pi / math.sqrt(1 - zeta**2))


# 3) Linear spring energy: F = k x; W = ∫ F dx = 0.5 k x^2

def spring_work(k: float, x: float) -> float:
    return 0.5 * k * x**2


# --------------------------- Test Cases ------------------------------

def test_beam_deflection_case(p: Dict[str, float]) -> TestResult:
    t0 = time.perf_counter()
    F, L, E, I = p["F"], p["L"], p["E"], p["I"]
    model_delta = beam_tip_deflection(F, L, E, I)

    # Inject a "hardware model" value for illustration (replace with real model call)
    hardware_delta = model_delta * (1.0 + p.get("bias", 0.0))

    tol = p.get("tol_rel", 1e-4)
    err = relative_error(hardware_delta, model_delta)
    passed = err <= tol
    msg = f"rel_error={err:.3e} (tol {tol:g}), model={model_delta:.6e}, hw={hardware_delta:.6e}"
    return TestResult("beam_tip_deflection", p, passed, msg, time.perf_counter() - t0)


def test_sdof_case(p: Dict[str, float]) -> TestResult:
    t0 = time.perf_counter()
    k, m, c = p["k"], p["m"], p["c"]
    wn = sdof_wn(k, m)
    z = sdof_zeta(c, k, m)
    Mp = sdof_step_overshoot(z)

    # Hypothetical hardware extraction (replace with measured/solver values)
    hw_wn = wn * (1.0 + p.get("bias_wn", 0.0))
    hw_Mp = Mp * (1.0 + p.get("bias_Mp", 0.0))

    tol_wn = p.get("tol_wn", 1e-3)
    tol_Mp = p.get("tol_Mp", 2e-2)
    err_wn = relative_error(hw_wn, wn)
    err_Mp = relative_error(hw_Mp, Mp)
    passed = (err_wn <= tol_wn) and (err_Mp <= tol_Mp)
    msg = f"wn rel_err={err_wn:.2e} (tol {tol_wn}), Mp rel_err={err_Mp:.2e} (tol {tol_Mp}), zeta={z:.3f}"
    return TestResult("sdof_closed_form", p, passed, msg, time.perf_counter() - t0)


def test_spring_energy_equilibrium(p: Dict[str, float]) -> TestResult:
    t0 = time.perf_counter()
    k, x = p["k"], p["x"]
    # Energy check
    W = spring_work(k, x)
    # Numerical quadrature of F dx with simple Riemann sum (emulates sampled hardware trajectory)
    N = max(10, int(p.get("N", 100)))
    xs = [i * x / N for i in range(N + 1)]
    Fxs = [k * xi for xi in xs]
    W_num = 0.0
    for i in range(N):
        W_num += 0.5 * (Fxs[i] + Fxs[i + 1]) * (xs[i + 1] - xs[i])  # trapezoid

    errW = relative_error(W_num, W)

    # Static equilibrium check: external load F_ext balanced by spring force at displacement x
    F_ext = k * x * (1.0 + p.get("bias_F", 0.0))
    eq_err = relative_error(F_ext, k * x)

    tolW = p.get("tol_energy", 1e-3)
    tolEq = p.get("tol_equil", 1e-4)
    passed = (errW <= tolW) and (eq_err <= tolEq)
    msg = f"energy rel_err={errW:.2e} (tol {tolW}), equilibrium rel_err={eq_err:.2e} (tol {tolEq})"
    return TestResult("spring_energy_equilibrium", p, passed, msg, time.perf_counter() - t0)


# ------------------------ Parameter Sweeps ---------------------------

def param_sweep_beam(seed: int) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    cases = []
    for _ in range(50):  # 50 cases per seed chunk
        F = 10**rng.uniform(-1, 3)        # 0.1 .. 1000 N
        L = 10**rng.uniform(-2, 0)        # 0.01 .. 1 m
        E = 10**rng.uniform(9, 12)        # 1e9 .. 1e12 Pa
        I = 10**rng.uniform(-12, -6)      # 1e-12 .. 1e-6 m^4
        cases.append({"F": F, "L": L, "E": E, "I": I, "tol_rel": 1e-5, "bias": 0.0})
    return cases


def param_sweep_sdof(seed: int) -> List[Dict[str, float]]:
    rng = random.Random(1000 + seed)
    cases = []
    for _ in range(50):
        m = 10**rng.uniform(-2, 1)        # 0.01 .. 10 kg
        k = 10**rng.uniform(1, 5)         # 10 .. 1e5 N/m
        z = rng.uniform(0.05, 0.6)
        c = 2.0 * z * math.sqrt(k * m)
        cases.append({"m": m, "k": k, "c": c, "tol_wn": 1e-3, "tol_Mp": 2e-2, "bias_wn": 0.0, "bias_Mp": 0.0})
    return cases


def param_sweep_spring(seed: int) -> List[Dict[str, float]]:
    rng = random.Random(2000 + seed)
    cases = []
    for _ in range(50):
        k = 10**rng.uniform(0, 5)         # 1 .. 1e5 N/m
        x = 10**rng.uniform(-4, -1)       # 0.0001 .. 0.1 m
        cases.append({"k": k, "x": x, "N": 200, "tol_energy": 1e-3, "tol_equil": 1e-5, "bias_F": 0.0})
    return cases


# Registry of test families
TEST_FAMILIES: Dict[str, Tuple[Callable[[Dict[str, float]], TestResult], Callable[[int], List[Dict[str, float]]]]] = {
    "beam": (test_beam_deflection_case, param_sweep_beam),
    "sdof": (test_sdof_case, param_sweep_sdof),
    "spring": (test_spring_energy_equilibrium, param_sweep_spring),
}


def run_family(args) -> List[TestResult]:
    name, func, params = args
    out = []
    for p in params:
        t = func(p)
        # Attach family name for clarity
        t.name = f"{name}::" + t.name
        out.append(t)
    return out


# ----------------------------- Main ---------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parallel mechanical validation runner")
    ap.add_argument("--procs", type=int, default=max(1, cpu_count()//2), help="worker processes")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed for sweeps")
    ap.add_argument("--families", nargs="*", default=["beam", "sdof", "spring"], choices=list(TEST_FAMILIES.keys()), help="which test families to run")
    ap.add_argument("--junit", type=str, default=None, help="optional JUnit XML output path")
    args = ap.parse_args()

    # Build parameter sets per family
    family_args = []
    for fam in args.families:
        func, sweep = TEST_FAMILIES[fam]
        params = sweep(args.seed)
        family_args.append((fam, func, params))

    t0 = time.perf_counter()
    results: List[TestResult] = []

    # Parallel map over families, then within family we loop (keeps payload size moderate)
    with Pool(processes=args.procs) as pool:
        for chunk in pool.imap_unordered(run_family, family_args):
            results.extend(chunk)

    # Summary
    total = len(results)
    failed = [r for r in results if not r.passed]
    elapsed = time.perf_counter() - t0

    # Pretty print
    def green(s):
        return f"\033[92m{s}\033[0m"
    def red(s):
        return f"\033[91m{s}\033[0m"

    print(f"\n=== Mechanical Validation Summary ===")
    for r in results[:10]:  # show a teaser
        tag = green("PASS") if r.passed else red("FAIL")
        print(f"[{tag}] {r.name} | {r.message}")
    if len(results) > 10:
        print(f"... ({len(results)-10} more cases hidden) ...")
    print(f"Total cases: {total}, Failures: {len(failed)}, Time: {elapsed:.2f}s")

    if args.junit:
        write_junit_xml(results, args.junit)
        print(f"JUnit written to: {args.junit}")

    sys.exit(0 if not failed else 1)
