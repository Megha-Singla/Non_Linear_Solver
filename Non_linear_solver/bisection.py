from diagnostics import Diagnostics


class Bisection:
    """Bisection method for a continuous f on [a, b] with f(a) * f(b) < 0.
    The error indicator is the half-width of the current interval, (b - a) / 2,
    which is an upper bound on the distance from the midpoint to the root."""

    def __init__(self, f, tol=1e-8, max_iter=100):
        self.diag = Diagnostics(tol, max_iter)
        self.diag.check_callable(f, "f")   
        self.f = f
        self.max_iter=max_iter
        self.tol=tol

    def solve(self, a, b):
        fa, fb = self.diag.check_interval(self.f, a, b)    #check whether the interval contains the root or not and a,b, f(a), f(b) are all finite or not.
        if fa == 0:
            return self.diag.result("Bisection", a, True, "f(a) is exactly zero")  #giving result, (which method is used, root, whether converged or not, message) 
        if fb == 0:
            return self.diag.result("Bisection", b, True, "f(b) is exactly zero")

        for k in range(1, self.max_iter+1):
            c = (a + b) / 2
            fc = self.f(c)
            error = (b - a) / 2    #it is the maximum bound of error.
            self.diag.record(k, c, fc, error)    #tracking all the history so far.
            if abs(fc) < self.tol:
                return self.diag.result("Bisection", c, True, f"Converged in {k} iterations")
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        return self.diag.result("Bisection", c, False,
            f"Did not converge within {self.max_iter} iterations")




def f(x):
    return 2*x+3

def g(x):
    return x**2 - 2

f1=Bisection(g)
print(f1.solve(0,3))




