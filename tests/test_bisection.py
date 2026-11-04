import math
import unittest

from Non_linear_solver.bisection import Bisection


class TestBisection(unittest.TestCase):

    def test_finds_sqrt2(self):
        r = Bisection(lambda x: x ** 2 - 2, tol=1e-10).solve(1, 2)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["root"], math.sqrt(2), places=9)

    def test_no_sign_change(self):
        with self.assertRaises(ValueError):
            Bisection(lambda x: x ** 2 + 1).solve(-1, 1)

    def test_bad_interval_order(self):
        with self.assertRaises(ValueError):
            Bisection(lambda x: x).solve(2, 1)

    def test_exact_endpoint_root(self):
        r = Bisection(lambda x: x - 1).solve(1, 3)
        self.assertTrue(r["converged"])
        self.assertEqual(r["root"], 1)



if __name__ == "__main__":
    unittest.main(verbosity=2)