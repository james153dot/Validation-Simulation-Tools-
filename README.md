Mechanical Validation Suite (Python + C++, parallel)

Parallel test harness for mechanical & thermo‑elastic models. Includes closed‑form baselines (beams, springs, SDOF, thermo‑strain), randomized sweeps, CI‑friendly summaries, and JUnit output. Plug in your FEA/rig to auto‑gate merges with physics‑based checks.

Features

Python runner (validate_mech.py) with multiprocessing

C++ runner (validate_mech.cpp) with OpenMP/threads

Deterministic param sweeps via seeds

JUnit XML and non‑zero exit on failure

Thermal–structural checks: free/fixed bars, gradients, lumped thermo‑loads

Quick Start
# Python
python validate_mech.py --procs 8 --seed 1 --families beam sdof spring


# C++ (with OpenMP)
g++ -O3 -std=c++17 -fopenmp validate_mech.cpp -o validate_mech
./validate_mech --threads 8
Wiring your solver/rig

Replace the hardware_* stubs with: FEM API calls, CSV parsers, HTTP/IPC hooks, or DLL/SO bindings.

Normalize units (SI recommended). Set tolerances per spec.

Add mesh/time‑step refinement sweeps; require monotone convergence.

CI

Add both runners to your pipeline; fail the job if any test fails.

Publish JUnit to your CI test tab and archive logs.

Extending

Add material nonlinearity windows (small ΔT to pre‑yield).

Add modal MAC/FRF comparisons for thermal pre‑loads.

Add contact tests under thermal expansion.
