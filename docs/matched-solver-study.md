# Matched solver-control study

The native solver can be audited on identical generated binary-packing instances under **classical controls** (efficacy cuts and most-fractional branching) and **expert controls** (lookahead cuts and strong branching). Supplying an existing trained checkpoint adds the learned presolve/cuts/branching/primal-control pipeline on the same instances.

Example:

    python -m scripts.matched_solver_study --seeds 101 102 103 --vars 10 --constraints 3 --output artifacts/matched-solver-study.json

Optional learned-control comparison:

    python -m scripts.matched_solver_study --checkpoint /path/to/checkpoint --seeds 101 102 103

The script writes JSON with source-instance fingerprints, family and seed, algorithm setting, objective, exact-reference objective, reference feasibility, search-tree node count, LP solves, cut count, and wall time. Two model sizes (base and scale-shift) are compared without pooling their results. The reference uses SciPy HiGHS on the unpresolved MILP; independently recomputing constraints checks that reference solution.

Scope of the evidence:
- This is a **small educational native binary-packing solver**, not a drop-in SCIP/Gurobi benchmark.
- Its SolveResult does not expose a final decision vector, dual bound or time budget. Accordingly only the **reference** solution is independently feasibility-audited; studied solutions are checked for complete search and matching reference objective, not individually feasibility-reconstructed.
- The script compares exact controls on the same seed and fingerprint. Runtime is measured, but no statistical speedup claim follows from the tiny correctness fixtures.
- Scale-shift instances differ by dimension, not by a certified shift in a learned training distribution. Do not call the scale family OOD relative to a given checkpoint without separately documenting its training data.
- Before publishing gains, pin solver, hardware, checkpoint, and environment versions; use larger held-out families, repeat seeds, ensure fair compute budgets, and report failure cases, p90 and paired uncertainty.

Acceptance checks in CI: tiny instances must match the independent reference, data fingerprints must match across policies, and incomplete or inconsistent comparison records must be rejected.
