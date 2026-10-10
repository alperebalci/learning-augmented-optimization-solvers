"""Matched-instance exact-solver comparison for the native binary-packing research sandbox.

This is a reproducible method study, NOT a SCIP/industrial solver-speedup benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from lamip import (
    SolverConfig,
    generate_binary_packing,
    load_components,
    solve,
    solve_reference,
)


def _fingerprint(problem: object) -> str:
    payload = {
        "profits": problem.profits.tolist(),
        "A": problem.A.tolist(),
        "b": problem.b.tolist(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _reference_is_feasible(problem: object, solution: np.ndarray) -> bool:
    return bool(
        solution.shape == (problem.n_vars,)
        and np.all(np.isfinite(solution))
        and np.all(np.abs(solution - np.rint(solution)) <= 1e-7)
        and np.all(solution >= -1e-7)
        and np.all(solution <= 1.0 + 1e-7)
        and np.all(problem.A @ solution <= problem.b + 1e-7)
    )


def validate_matched_rows(rows: list[dict[str, object]], policies: tuple[str, ...]) -> None:
    """Reject missing baselines, unmatched instances, invalid costs and bogus references."""
    if not rows or len(set(policies)) < 2:
        raise ValueError("a nonempty study requires at least two distinct policies")
    groups: dict[tuple[str, int, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["seed"]), str(row["fingerprint"]))
        policy = str(row["policy"])
        group = groups.setdefault(key, {})
        if policy in group:
            raise ValueError(f"duplicate policy for instance: {key} {policy}")
        group[policy] = row
        if row["optimal"] is not True or row["reference_feasible"] is not True:
            raise ValueError(f"invalid exactness/feasibility evidence: {key}")
        for field in ("runtime_seconds", "objective", "reference_objective"):
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"nonfinite {field} in {key}")
        if float(row["runtime_seconds"]) < 0:
            raise ValueError("negative runtime")
        if not math.isclose(
            float(row["objective"]),
            float(row["reference_objective"]),
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(f"objective mismatch for {key}")
        for field in ("nodes_processed", "lp_solves", "cuts_added"):
            if not isinstance(row[field], int) or row[field] < 0:
                raise ValueError(f"invalid {field} for {key}")
    for key, group in groups.items():
        if set(group) != set(policies):
            raise ValueError(f"incomplete matched controls for {key}")
        if len({float(row["reference_objective"]) for row in group.values()}) != 1:
            raise ValueError(f"inconsistent exact reference for {key}")


def run_study(
    seeds: tuple[int, ...],
    *,
    in_distribution: tuple[int, int] = (10, 3),
    scale_shift: tuple[int, int] = (12, 4),
    checkpoint: str | None = None,
) -> dict[str, object]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    families = {
        "in_distribution": in_distribution,
        "scale_shift": scale_shift,
    }
    if any(n < 2 or m < 1 for n, m in families.values()):
        raise ValueError("invalid instance dimensions")
    policies = {
        "classical": SolverConfig(
            presolve="full", cuts="efficacy", branching="most_fractional", primal="greedy"
        ),
        "expert_controls": SolverConfig(
            presolve="full", cuts="oracle", branching="strong", primal="greedy"
        ),
    }
    learned = None
    if checkpoint is not None:
        learned = load_components(checkpoint)
        policies["learned_controls"] = SolverConfig(
            presolve="learned", cuts="learned", branching="learned", primal="learned"
        )
    rows: list[dict[str, object]] = []
    for family, (n_vars, n_constraints) in families.items():
        for seed in seeds:
            problem = generate_binary_packing(n_vars, n_constraints, seed)
            exact = solve_reference(problem, presolve=False)
            if not _reference_is_feasible(problem, exact.solution):
                raise RuntimeError("independent reference solution feasibility audit failed")
            objective_from_vector = float(problem.profits @ exact.solution)
            if not np.isclose(objective_from_vector, exact.objective, atol=1e-7):
                raise RuntimeError("reference objective disagrees with its decision vector")
            fingerprint = _fingerprint(problem)
            for name, config in policies.items():
                started = time.perf_counter()
                solved = solve(problem, config=config, learned=learned)
                runtime = time.perf_counter() - started
                row = {
                    "family": family,
                    "seed": seed,
                    "n_vars": n_vars,
                    "n_constraints": int(problem.n_constraints),
                    "fingerprint": fingerprint,
                    "policy": name,
                    "optimal": solved.optimal,
                    "reference_feasible": True,
                    "objective": float(solved.objective),
                    "reference_objective": float(exact.objective),
                    "nodes_processed": int(solved.nodes_processed),
                    "lp_solves": int(solved.lp_solves),
                    "cuts_added": int(solved.cuts_added),
                    "runtime_seconds": float(runtime),
                }
                rows.append(row)
    validate_matched_rows(rows, tuple(policies))
    return {
        "schema_version": 1,
        "scope": "native educational exact binary-packing solver",
        "claims": "correctness and matched controls only; no production speedup claim",
        "policies": list(policies),
        "seeds": list(seeds),
        "families": {key: list(val) for key, val in families.items()},
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output", type=Path, default=Path("artifacts/matched-solver-study.json"))
    parser.add_argument("--vars", type=int, default=10)
    parser.add_argument("--constraints", type=int, default=3)
    args = parser.parse_args()
    payload = run_study(
        tuple(args.seeds),
        in_distribution=(args.vars, args.constraints),
        scale_shift=(args.vars + 2, args.constraints + 1),
        checkpoint=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(payload['runs'])} matched solver results: {args.output}")


if __name__ == "__main__":
    main()
