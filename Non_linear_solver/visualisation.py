"""
Experiments and visualization for Project 4 (Nonlinear Equation Solver) - Member 4.
Uses Bisection (Member 1), Newton (Member 2) and Diagnostics (Member 3).
 
Run (from the Non_linear_solver folder):   python visualisation.py
Output: iteration tables in the terminal and one figure per experiment,
        saved in the "plots" folder next to this file (exp1.png ... exp4.png).
"""
 
import math
import os
import matplotlib.pyplot as plt
 
from bisection import Bisection
from newton import Newton
from diagnostics import Diagnostics
 
 
class Experiments:
    """Runs the four required experiments; each one draws its own figure."""
 
    def __init__(self, out_dir=None):
        # folder for the PNG files (default: "plots" next to this file)
        here = os.path.dirname(os.path.abspath(__file__))
        self.out_dir = out_dir or os.path.join(here, "plots")
        os.makedirs(self.out_dir, exist_ok=True)
 
        self.table = Diagnostics().print_table    # reuse Member 3's iteration table
        self.root2 = math.sqrt(2)                 # exact root of x^2 - 2
        # each test problem is a pair (f, f')
        self.sq = (lambda x: x**2 - 2, lambda x: 2 * x)                    # x^2 - 2
        self.cubic = (lambda x: x**3 - x, lambda x: 3 * x**2 - 1)          # roots -1, 0, 1
        self.cycle = (lambda x: x**3 - 2*x + 2, lambda x: 3 * x**2 - 2)    # Newton cycles at x0=0
        self.atan = (lambda x: math.atan(x), lambda x: 1 / (1 + x**2))     # Newton diverges at x0=2
 
    # ---- helpers: a fresh solver every time, so histories never mix ----
    def bisect(self, p, a, b, tol=1e-8):
        return Bisection(p[0], tol=tol).solve(a, b)
 
    def newton(self, p, x0, tol=1e-8, max_iter=50):
        return Newton(p[0], p[1], tol=tol, max_iter=max_iter).solve(x0)
 
    def line(self, label, r):
        """One-line summary of a result."""
        print(f"{label:<18} converged={r['converged']!s:<5} iters={r['iterations']:>3} "
              f"root={r['root']:<20.14f} |f|={r['residual']:.2e}")
 
    # ---- Experiment 1: both methods work ----
    def exp1(self, ax):
        print("\nEXPERIMENT 1: x^2 - 2 = 0, tol = 1e-10, bisection on [1,2], Newton from x0 = 1")
        b, n = self.bisect(self.sq, 1, 2, 1e-10), self.newton(self.sq, 1.0, 1e-10)
        self.table(b); self.table(n)
        self.line("Bisection", b); self.line("Newton", n)
        for r, style in ((b, "o-"), (n, "s-")):   # true error |x_k - sqrt(2)| per iteration
            ax.semilogy([h["k"] for h in r["history"]],
                        [max(abs(h["x"] - self.root2), 1e-17) for h in r["history"]],
                        style, label=f"{r['method']} ({r['iterations']} iters)")
        ax.set(title="Exp 1: x^2 - 2", xlabel="iteration k", ylabel="true error |x_k - sqrt(2)|")
        ax.legend()
 
    # ---- Experiment 2: effect of the initial guess ----
    def exp2(self, ax):
        print("\nEXPERIMENT 2: Newton on x^3 - x (roots -1, 0, 1), tol = 1e-10")
        for x0 in [-2.0, -0.7, -0.4, 0.3, 0.5, 0.6, 0.7, 2.0]:
            r = self.newton(self.cubic, x0, 1e-10)
            self.line(f"x0 = {x0:+.1f}", r)
            ax.plot([0] + [h["k"] for h in r["history"]],            # k = 0 is the guess itself
                    [x0] + [h["x"] for h in r["history"]], "o-", label=f"x0={x0:+.1f}")
        ax.set(title="Exp 2: x^3 - x, different x0", xlabel="iteration k", ylabel="iterate x_k")
        ax.legend(fontsize=7, ncol=2)
 
    # ---- Experiment 3: challenging cases ----
    def exp3(self, ax):
        print("\nEXPERIMENT 3: challenging cases for Newton")
        a = self.newton(self.sq, 0.0)
        print("A) x^2 - 2, x0 = 0 (f'(0) = 0):", a["message"], f"[{a['iterations']} iterations]")
        b = self.newton(self.cycle, 0.0, max_iter=10)
        print("B) x^3 - 2x + 2, x0 = 0 (cycle):", b["message"],
              "\n   iterates:", [h["x"] for h in b["history"]])
        c = self.newton(self.atan, 2.0)
        print("C) arctan(x), x0 = 2 (far from root):")
        self.table(c)
        self.line("   Bisection [-2,3]", self.bisect(self.atan, -2, 3))
        xs = [2.0] + [h["x"] for h in c["history"]]                  # |x_k| shows the blow-up
        ax.semilogy(range(len(xs)), [abs(x) for x in xs], "o-")
        ax.set(title="Exp 3: arctan(x), x0 = 2 (diverges)", xlabel="iteration k",
               ylabel="|x_k|  (true root is 0)")
 
    # ---- Experiment 4: tolerance study ----
    def exp4(self, ax):
        print("\nEXPERIMENT 4: x^2 - 2 = 0 with different tolerances")
        tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
        iters = {"Bisection": [], "Newton method": []}
        print(f"{'method':<14}{'tol':>7}{'iters':>6}{'approximation':>20}{'|f(x)|':>11}{'true err':>11}")
        for tol in tols:
            for r in (self.bisect(self.sq, 1, 2, tol), self.newton(self.sq, 1.0, tol)):
                iters[r["method"]].append(r["iterations"])
                print(f"{r['method']:<14}{tol:>7.0e}{r['iterations']:>6}{r['root']:>20.14f}"
                      f"{r['residual']:>11.2e}{abs(r['root'] - self.root2):>11.2e}")
        for name, style in (("Bisection", "o-"), ("Newton method", "s-")):
            ax.semilogx(tols, iters[name], style, label=name)
        ax.invert_xaxis()                                            # tighter tolerance to the right
        ax.set(title="Exp 4: iterations vs tolerance", xlabel="tolerance", ylabel="iterations")
        ax.legend()
 
    def run_all(self):
        """Run every experiment and save one PNG per experiment in self.out_dir."""
        for name, exp in (("exp1", self.exp1), ("exp2", self.exp2),
                          ("exp3", self.exp3), ("exp4", self.exp4)):
            fig, ax = plt.subplots(figsize=(7, 5))
            exp(ax)
            ax.grid(True, which="both", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(self.out_dir, name + ".png")
            plt.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[saved figure: {path}]")
 
 
if __name__ == "__main__":
    Experiments().run_all()
 