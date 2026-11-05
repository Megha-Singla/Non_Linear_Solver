"""Tests for Member 4's experiments and visualization module."""

import math
import os
import sys
from pathlib import Path

import pytest

# Use a non-interactive backend so tests work on computers/CI without a display.
os.environ.setdefault("MPLBACKEND", "Agg")

# The solver modules are stored in the project's Non_linear_solver directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIR = PROJECT_ROOT / "Non_linear_solver"
if not SOLVER_DIR.is_dir():
    # Also allow this test file to be run from inside the tests directory
    # when the project folder has a different working-directory layout.
    SOLVER_DIR = Path(__file__).resolve().parent.parent / "Non_linear_solver"

sys.path.insert(0, str(SOLVER_DIR))

from Non_linear_solver.visualisation import Experiments  # noqa: E402


def test_run_bisection_returns_expected_root(tmp_path):
    """The experiment helper should solve x^2 - 2 on [1, 2]."""
    experiments = Experiments(output_dir=tmp_path)

    result = experiments.run_bisection(
        experiments.sqrt_problem, 1, 2, tol=1e-10
    )

    assert result["converged"] is True
    assert math.isclose(result["root"], math.sqrt(2), abs_tol=1e-8)
    assert result["residual"] < 1e-8
    assert len(result["history"]) == result["iterations"]


def test_run_newton_returns_expected_root(tmp_path):
    """The experiment helper should solve x^2 - 2 from x0 = 1."""
    experiments = Experiments(output_dir=tmp_path)

    result = experiments.run_newton(
        experiments.sqrt_problem, 1.0, tol=1e-10
    )

    assert result["converged"] is True
    assert math.isclose(result["root"], math.sqrt(2), abs_tol=1e-8)
    assert result["residual"] < 1e-8
    assert len(result["history"]) == result["iterations"]


def test_run_all_creates_all_experiment_plots(tmp_path, capsys):
    """Running all experiments should save the four expected PNG figures."""
    experiments = Experiments(output_dir=tmp_path)

    experiments.run_all()

    expected_plots = [
        "experiment_1_comparison.png",
        "experiment_2_initial_guesses.png",
        "experiment_3_challenging_cases.png",
        "experiment_4_tolerance_study.png",
    ]

    for filename in expected_plots:
        plot_path = tmp_path / filename
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 0

    output = capsys.readouterr().out
    assert "All experiments completed." in output
