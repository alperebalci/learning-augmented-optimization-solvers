"""Regression tests for method-matched solver evidence and reference feasibility."""

from copy import deepcopy

import pytest

from scripts.matched_solver_study import run_study, validate_matched_rows


def test_baseline_controls_match_exact_reference() -> None:
    report = run_study(
        (11,), in_distribution=(7, 2), scale_shift=(8, 3)
    )
    rows = report["runs"]
    assert len(rows) == 4
    assert {row["policy"] for row in rows} == {"classical", "expert_controls"}
    assert all(row["optimal"] and row["reference_feasible"] for row in rows)
    assert all(row["objective"] == pytest.approx(row["reference_objective"]) for row in rows)
    assert all(row["runtime_seconds"] >= 0 for row in rows)


def test_incomplete_policy_controls_are_rejected() -> None:
    report = run_study((13,), in_distribution=(6, 2), scale_shift=(7, 2))
    rows = deepcopy(report["runs"])
    rows.pop()
    with pytest.raises(ValueError, match="incomplete matched controls"):
        validate_matched_rows(rows, ("classical", "expert_controls"))


def test_mismatched_objective_is_rejected() -> None:
    report = run_study((13,), in_distribution=(6, 2), scale_shift=(7, 2))
    rows = deepcopy(report["runs"])
    rows[0]["objective"] += 10.0
    with pytest.raises(ValueError, match="objective mismatch"):
        validate_matched_rows(rows, ("classical", "expert_controls"))


def test_duplicate_or_missing_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="seeds must"):
        run_study(())
    with pytest.raises(ValueError, match="seeds must"):
        run_study((2, 2))
