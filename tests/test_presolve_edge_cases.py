"""Presolve regressions: safe fixings must preserve a valid solver residual."""

import numpy as np
import pytest

from lamip import SolverConfig, apply_presolve, solve, solve_reference
from lamip.problem import BinaryPackingMIP


@pytest.mark.parametrize(
    ("profits", "coefficients", "capacity"),
    [
        ([10.0, -1.0, -3.0], [4.0, 2.0, 3.0], 7.56),
        ([7.0, 6.0, 5.0], [4.0, 6.0, 8.0], 1.0),
        ([7.0, 6.0, 5.0], [0.0, 0.0, 0.0], 0.0),
        ([8.0, 9.0, 3.0, -2.0], [0.0, 5.0, 8.0, 0.0], 7.0),
    ],
)
def test_safe_fixings_preserve_optimum_with_degenerate_residuals(
    profits: list[float], coefficients: list[float], capacity: float
) -> None:
    problem = BinaryPackingMIP(
        profits=np.array(profits),
        A=np.array([coefficients]),
        b=np.array([capacity]),
    )
    reference = solve_reference(problem, presolve=False)
    for config in ("fixing", "full"):
        reduced = apply_presolve(problem, config)
        assert reduced.problem.n_vars >= 2
        assert len(reduced.kept_indices) + len(reduced.fixed_values) == problem.n_vars
        assert set(reduced.kept_indices).isdisjoint(reduced.fixed_values)

        residual_solution = solve_reference(reduced.problem, presolve=False).solution
        original_solution = np.zeros(problem.n_vars)
        original_solution[reduced.kept_indices] = residual_solution
        for j, value in reduced.fixed_values.items():
            original_solution[j] = value
        assert np.all(problem.A @ original_solution <= problem.b + 1e-8)
        assert np.isclose(problem.profits @ original_solution, reference.objective)
        assert np.isclose(
            problem.profits @ original_solution,
            reduced.problem.profits @ residual_solution + reduced.objective_offset,
        )

        native_result = solve(problem, SolverConfig(presolve=config, cuts="none"))
        assert native_result.optimal
        assert np.isclose(native_result.objective, reference.objective)


def test_duplicate_and_dominated_rows_leave_one_representative() -> None:
    problem = BinaryPackingMIP(
        profits=np.array([4.0, 6.0, 7.0]),
        A=np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        ),
        b=np.array([2.0, 2.0, 2.0]),
    )
    reduced = apply_presolve(problem, "redundancy")
    assert reduced.problem.n_constraints == 1
    assert np.array_equal(reduced.problem.A[0], problem.A[2])
    assert np.isclose(
        solve_reference(reduced.problem, presolve=False).objective,
        solve_reference(problem, presolve=False).objective,
    )


def test_identical_rows_choose_stable_lowest_index() -> None:
    problem = BinaryPackingMIP(
        profits=np.array([3.0, 2.0]),
        A=np.array([[1.0, 1.0], [1.0, 1.0]]),
        b=np.array([1.0, 1.0]),
    )
    reduced = apply_presolve(problem, "redundancy")
    assert reduced.problem.n_constraints == 1
    assert np.array_equal(reduced.problem.A[0], problem.A[0])
