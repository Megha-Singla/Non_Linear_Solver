import unittest

from Non_linear_solver.diagnostics import Diagnostics


class TestDiagnostics(unittest.TestCase):

    def setUp(self):
        # runs before every test: makes a fresh Diagnostics object
        self.d = Diagnostics(tol=1e-6, max_iter=50)
        self.f = lambda x: x ** 2 - 2

    # ---- input validation ----
    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            Diagnostics(tol=-1)

    def test_bad_max_iter(self):
        with self.assertRaises(ValueError):
            Diagnostics(max_iter=0)

    def test_valid_interval(self):
        fa, fb = self.d.check_interval(self.f, 0, 2)
        self.assertEqual((fa, fb), (-2, 2))

    def test_no_sign_change(self):
        with self.assertRaises(ValueError):
            self.d.check_interval(self.f, 2, 3)

    def test_a_not_less_than_b(self):
        with self.assertRaises(ValueError):
            self.d.check_interval(self.f, 3, 1)

    # ---- failure and stopping checks ----
    def test_zero_derivative(self):
        self.assertTrue(self.d.derivative_too_small(0.0))
        self.assertFalse(self.d.derivative_too_small(2.0))

    def test_converged_small_error(self):
        self.assertTrue(self.d.has_converged(error=1e-9, fx=0.5))

    def test_converged_small_residual(self):
        self.assertTrue(self.d.has_converged(error=0.5, fx=1e-9))

    def test_not_converged(self):
        self.assertFalse(self.d.has_converged(error=0.5, fx=0.5))

    # ---- history and result ----
    def test_record_and_result(self):
        self.d.record(1, 1.5, -0.25, 0.5)
        r = self.d.result("Bisection", 1.5, True, "ok")
        self.assertEqual(r["iterations"], 1)
        self.assertEqual(r["residual"], 0.25)
        self.assertTrue(r["converged"])


if __name__ == "__main__":
    unittest.main()