"""Experiments and plots for Project 4: Nonlinear Equation Solver.

Member 4 contribution: runs the four required experiments using the
Bisection, Newton and Diagnostics classes implemented by the other members.

Run from this directory with: python visualisation.py
The iteration tables are printed in the terminal and PNG figures are saved
in the plots/ directory beside this file.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from bisection import Bisection
from newton import Newton
from diagnostics import Diagnostics


class Experiments:
    """Run, report and visualize the four project experiments."""

    def __init__(self, output_dir=None):
        project_dir = Path(__file__).resolve().parent
        self.output_dir = Path(output_dir) if output_dir else project_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test functions and their analytical derivatives.
        self.sqrt_problem = (lambda x: x**2 - 2, lambda x: 2*x)
        self.cubic_problem = (lambda x: x**3 - x, lambda x: 3*x**2 - 1)
        self.cycle_problem = (lambda x: x**3 - 2*x + 2, lambda x: 3*x**2 - 2)
        self.atan_problem = (lambda x: math.atan(x), lambda x: 1/(1 + x**2))

        # Reuse Member 3's table printer; solver histories remain in each result.
        self.print_table = Diagnostics().print_table
        self.exact_sqrt2 = math.sqrt(2)

    def run_bisection(self, problem, a, b, tol=1e-8, max_iter=100):
        """Run bisection with a fresh solver so histories cannot overlap."""
        solver = Bisection(problem[0], tol=tol, max_iter=max_iter)
        return solver.solve(a, b)

    def run_newton(self, problem, x0, tol=1e-8, max_iter=50):
        """Run Newton's method with a fresh solver for one initial guess."""
        solver = Newton(problem[0], problem[1], tol=tol, max_iter=max_iter)
        return solver.solve(x0)

    @staticmethod
    def print_summary(label, result):
        """Print a compact summary, including failed runs with no residual."""
        residual = result["residual"]
        residual_text = f"{residual:.3e}" if residual is not None else "N/A"
        print(
            f"{label:<20} converged={str(result['converged']):<5} "
            f"iterations={result['iterations']:>3} "
            f"root/last x={result['root']:<20.12g} residual={residual_text}"
        )
        print(f"{'':<20} status: {result['message']}")

    @staticmethod
    def _finish_plot(fig, filename, output_dir):
        """Apply common plot formatting and save the figure."""
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {path}")

    def experiment_1(self):
        """Compare both methods on x^2 - 2 = 0, whose positive root is sqrt(2)."""
        print("\nEXPERIMENT 1: Both methods on f(x) = x^2 - 2")
        print("Bisection interval [1, 2]; Newton initial guess x0 = 1; tolerance = 1e-10")

        bisection_result = self.run_bisection(self.sqrt_problem, 1, 2, tol=1e-10)
        newton_result = self.run_newton(self.sqrt_problem, 1.0, tol=1e-10)

        self.print_table(bisection_result)
        self.print_table(newton_result)
        self.print_summary("Bisection", bisection_result)
        self.print_summary("Newton", newton_result)

        fig, ax = plt.subplots(figsize=(8, 5))
        for result, marker in ((bisection_result, "o"), (newton_result, "s")):
            iterations = [row["k"] for row in result["history"]]
            true_errors = [max(abs(row["x"] - self.exact_sqrt2), 1e-17)
                           for row in result["history"]]
            ax.semilogy(iterations, true_errors, marker + "-",
                        label=f"{result['method']} ({result['iterations']} iterations)")
        ax.set(title=r"Experiment 1: convergence for $x^2 - 2 = 0$",
               xlabel="Iteration", ylabel=r"True error $|x_k - \sqrt{2}|$")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        self._finish_plot(fig, "experiment_1_comparison.png", self.output_dir)

    def experiment_2(self):
        """Show how Newton's iteration path depends on its initial guess."""
        print("\nEXPERIMENT 2: Effect of the initial guess for f(x) = x^3 - x")
        print("The equation has roots -1, 0 and 1; tolerance = 1e-10")
        initial_guesses = [-2.0, -0.7, -0.4, 0.3, 0.5, 0.6, 0.7, 2.0]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for x0 in initial_guesses:
            result = self.run_newton(self.cubic_problem, x0, tol=1e-10)
            self.print_summary(f"Initial guess {x0:+.1f}", result)
            x_values = [x0] + [row["x"] for row in result["history"]]
            iteration_numbers = list(range(len(x_values)))
            ax.plot(iteration_numbers, x_values, marker="o", markersize=3,
                    label=f"x0={x0:+.1f} ({'converged' if result['converged'] else 'failed'})")

        ax.set(title=r"Experiment 2: Newton iterates for $x^3 - x = 0$",
               xlabel="Iteration (0 is the initial guess)", ylabel=r"Approximation $x_k$")
        ax.axhline(-1, linestyle="--", linewidth=0.8, label="Roots: -1, 0, 1")
        ax.axhline(0, linestyle="--", linewidth=0.8)
        ax.axhline(1, linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        self._finish_plot(fig, "experiment_2_initial_guesses.png", self.output_dir)

    def experiment_3(self):
        """Demonstrate a zero derivative, a Newton cycle and poor convergence."""
        print("\nEXPERIMENT 3: Challenging cases for Newton's method")

        zero_derivative = self.run_newton(self.sqrt_problem, 0.0)
        print("\nCase A: f(x)=x^2-2, x0=0 (the derivative at x0 is zero)")
        self.print_summary("Newton", zero_derivative)

        cycle = self.run_newton(self.cycle_problem, 0.0, max_iter=10)
        print("\nCase B: f(x)=x^3-2x+2, x0=0 (iteration cycles)")
        self.print_summary("Newton", cycle)
        print("Recorded iterates:", [row["x"] for row in cycle["history"]])

        poor_start = self.run_newton(self.atan_problem, 2.0)
        print("\nCase C: f(x)=arctan(x), x0=2 (poor starting guess)")
        self.print_summary("Newton", poor_start)
        if poor_start["history"]:
            self.print_table(poor_start)

        # Bisection provides a stable comparison for arctan(x) on [-2, 3].
        bisect_atan = self.run_bisection(self.atan_problem, -2, 3)
        self.print_summary("Bisection arctan", bisect_atan)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        cycle_values = [0.0] + [row["x"] for row in cycle["history"]]
        axes[0].plot(range(len(cycle_values)), cycle_values, "o-")
        axes[0].set(title=r"Newton cycle: $x^3 - 2x + 2$", xlabel="Iteration", ylabel=r"$x_k$")
        axes[0].grid(True, alpha=0.3)

        atan_values = [2.0] + [row["x"] for row in poor_start["history"]]
        axes[1].semilogy(range(len(atan_values)),
                         [max(abs(value), 1e-16) for value in atan_values], "o-")
        axes[1].set(title=r"Newton from $x_0=2$: $f(x)=\arctan(x)$",
                    xlabel="Iteration", ylabel=r"$|x_k|$ (root is 0)")
        axes[1].grid(True, which="both", alpha=0.3)
        self._finish_plot(fig, "experiment_3_challenging_cases.png", self.output_dir)

    def experiment_4(self):
        """Compare iteration counts and accuracy over a range of tolerances."""
        print("\nEXPERIMENT 4: Tolerance study for f(x) = x^2 - 2")
        tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
        methods = ("Bisection", "Newton method")
        iteration_counts = {method: [] for method in methods}

        print(f"{'Method':<16}{'Tolerance':>12}{'Iterations':>12}"
              f"{'Approximation':>20}{'Residual':>14}{'True error':>14}")
        for tol in tolerances:
            results = (
                self.run_bisection(self.sqrt_problem, 1, 2, tol=tol),
                self.run_newton(self.sqrt_problem, 1.0, tol=tol),
            )
            for result in results:
                method = result["method"]
                iteration_counts[method].append(result["iterations"])
                residual = result["residual"]
                residual_text = f"{residual:.3e}" if residual is not None else "N/A"
                true_error = abs(result["root"] - self.exact_sqrt2)
                print(f"{method:<16}{tol:>12.0e}{result['iterations']:>12}"
                      f"{result['root']:>20.12f}{residual_text:>14}"
                      f"{true_error:>14.3e}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for method, marker in (("Bisection", "o"), ("Newton method", "s")):
            ax.semilogx(tolerances, iteration_counts[method], marker + "-", label=method)
        ax.invert_xaxis()
        ax.set(title=r"Experiment 4: iteration count vs tolerance for $x^2-2$",
               xlabel="Tolerance (tighter to the right)", ylabel="Number of iterations")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        self._finish_plot(fig, "experiment_4_tolerance_study.png", self.output_dir)

    def run_all(self):
        """Execute all required experiments in the order given in the brief."""
        self.experiment_1()
        self.experiment_2()
        self.experiment_3()
        self.experiment_4()
        print("\nAll experiments completed.")
        print(f"Plots are available in: {self.output_dir}")


if __name__ == "__main__":
    Experiments().run_all()
