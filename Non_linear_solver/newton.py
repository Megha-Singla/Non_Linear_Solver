import math
from diagnostics import Diagnostics
class Newton:
    """
    This is code for Newton Method
    """
    def __init__(self, f, df, tol=1e-8, max_iter=50, deriv_eps=1e-12, div_limit=1e8):
        self.diag = Diagnostics(tol, max_iter)
        self.diag.check_callable(f, "f")    # checking if the function that is given as input is actually a function or not (like I gave a function "Hello" which is not callable)
        self.diag.check_callable(df, "df")  # similarly for the derivative if the derivative of the function is callable or not
        self.f = f                    # initializing the function
        self.df = df                  # initializing the derivative function
        self.deriv_eps = deriv_eps    # it is the derivative eps that checks if the derivative is closing to zero or not
        self.div_limit = div_limit    # it is about the divergence limit which gives a limit if x is beyond that it means instead of converging, it is diverging

    def solve(self, x0):
        self.diag.reset()    #clearing the iteration, residual details after every new equations
        self.diag.check_finite(x0, "x0")    #it is checking if the value x0 is finite real number (like it is not string or something)
        x = x0

        for k in range(1, self.diag.max_iter + 1):      # creating the iteration to get the root
            fx = self.f(x)
            dfx = self.df(x)
            if not math.isfinite(fx) or not math.isfinite(dfx):     #checking if the value of x is not giving an infinite fx
                return self.diag.result("Newton method", x, False, "f or f' is not finite, change it")
            if self.diag.derivative_too_small(dfx, self.deriv_eps):     #checking if the derivative is closing to zero or zero 
                return self.diag.result(    # so checking if very small eps is taken, so if it is close to that eps, it is near to zero
                    "Newton method", x, False,
                    "Derivative is close to zero and can be zero, it will cause the denominator to be zero, so change it",
                )

            x_new = x - fx / dfx
            #print(x_new)
            if not math.isfinite(x_new) or abs(x_new) > self.div_limit:    #it is checking that the divergence limit as if the x is converging or diverging
                return self.diag.result(
                    "Newton method", x, False,
                    "Iteration is diverging instead of converging, so it is not getting converged",   # here if the value of x is greater than diverging value, it diverges (converging failed)
                )

            f_new = self.f(x_new)
            step = x_new - x   #checking the step size
            self.diag.record(k, x_new, f_new, step)   #stores the one iteration number, approximation and residual
            if self.diag.has_converged(step, f_new):   # if the abs(f_new) value is less than tol, it is getting converged yes:)!!
                return self.diag.result("Newton method", x_new, True, "Converged!!")
            x = x_new    #continuing the iteration

        return self.diag.result(
            "Newton method", x, False,
            "Did not got converged and maximum iteration reached",     #showing that the maximum iteration is reached and it did not got converged oops!!
        )

