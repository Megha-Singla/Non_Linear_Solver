import math
import unittest

from Non_linear_solver.newton import Newton


class TestNewton(unittest.TestCase):

    def test_finds_sqrt2(self):
        r = Newton(lambda x: x ** 2 - 2, lambda x: 2 * x, tol=1e-12).solve(1.0) # function is x^2 -2 with deri: 2*x and x0 as 1
        self.assertTrue(r["converged"])   #should show that the convergence is true
        self.assertAlmostEqual(r["root"], math.sqrt(2), places=12)      # checking if the value of root is almost equalt to the root 2

    def test_zero_derivative(self):            #this function is checking if the derivative is zero or not
        r = Newton(lambda x: x ** 2 - 1, lambda x: 2 * x).solve(0.0)
        self.assertFalse(r["converged"])         # it is showing that convergence should come false
        self.assertIn("Derivative near zero", r["message"])  # it is checking if the message that it is showing is telling that the derivative is zero or not

    def test_max_iterations(self):              #this function is testing the maximum iteration that is if it shows that max iteration is reached
        r = Newton(lambda x: x ** 3 - 2 * x + 2, lambda x: 3 * x ** 2 - 2, max_iter=10).solve(0.0)
        self.assertFalse(r["converged"])     # it is showing that convergence is false (test passed)
        self.assertEqual(r["iterations"], 10)   #checks if the iterations has reached 10.

    def test_divergence_reported(self):           # this function is testing the divergence with the example that whether the function is diverging
        r = Newton(math.atan, lambda x: 1 / (1 + x ** 2), max_iter=30).solve(2.0)
        self.assertFalse(r["converged"])   # will show false when it says conveged

    def test_non_callable_function(self):     # this function is not callable as the function is 5 which is not actual function
        with self.assertRaises(ValueError):      #need to show value error for this example
            Newton(5, lambda x: 1)

    def test_non_finite_guess(self):    # this function tests that this x0 value is invalid and will give valueerror for this
        with self.assertRaises(ValueError):    #need to show the value error 
            Newton(lambda x: x, lambda x: 1).solve(float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)