import math


class Diagnostics:
    """Shared stopping criteria, input validation, iteration history and result reporting.

    Stopping rule: the solver stops successfully when the step size (or the
    bisection half-width) is below tol, or when |f(x)| is below tol. It stops
    with a failure when max_iter iterations are used up.
    """

    def __init__(self, tol=1e-8, max_iter=100):
        if not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0:
            raise ValueError(f"Tolerance must be a positive number, got {tol!r}")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
        self.tol = tol
        self.max_iter = max_iter
        self.history = []

    def reset(self):
        """Clear the iteration history before a new solve."""
        self.history = []

    def check_callable(self, func, name="f"):
        """Raise ValueError if func cannot be called."""
        if not callable(func):
            raise ValueError(f"{name} must be a callable function")

    def check_finite(self, value, name="value"):
        """Raise ValueError if value is not a finite real number."""
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")

    def check_interval(self, f, a, b):
        """Validate [a, b] for bisection and return (f(a), f(b))."""
        self.check_finite(a, "a")
        self.check_finite(b, "b")
        if a >= b:
            raise ValueError(f"Interval must satisfy a < b, got a={a}, b={b}")
        fa, fb = f(a), f(b)
        self.check_finite(fa, "f(a)")
        self.check_finite(fb, "f(b)")
        if fa * fb > 0:
            raise ValueError(
                f"No sign change on [{a}, {b}]: f(a)={fa:.4g}, f(b)={fb:.4g}"
            )
        return fa, fb

    def derivative_too_small(self, dfx, eps=1e-12):
        """Return True when |f'(x)| is too close to zero for a Newton step."""
        return abs(dfx) < eps

    def has_converged(self, error, fx):
        """Return True when the error indicator or the residual is below tol."""
        return abs(error) < self.tol or abs(fx) < self.tol

    def record(self, k, x, fx, error):
        """Store one iteration: number, approximation, residual |f(x)| and error."""
        self.history.append({"k": k, "x": x, "fx": fx, "residual": abs(fx), "error": error})

    def result(self, method, root, converged, message):
        """Build the result dictionary returned by every solver."""
        last = self.history[-1] if self.history else None
        return {
            "method": method,
            "root": root,
            "converged": converged,
            "iterations": len(self.history),
            "residual": last["residual"] if last else None,
            "message": message,
            "history": list(self.history),
        }

    def print_table(self, result):
        """Print the iteration history of a result as a table."""
        print(f"\n{result['method']}: {result['message']}")
        print(f"{'k':>4} {'x':>22} {'|f(x)|':>12} {'error':>12}")
        for row in result["history"]:
            print(f"{row['k']:>4} {row['x']:>22.15f} {row['residual']:>12.3e} {abs(row['error']):>12.3e}")
