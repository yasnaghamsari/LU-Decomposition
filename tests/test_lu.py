"""
Unit tests for the LU decomposition project.

Run from the project folder with:
    python -m unittest discover tests
"""

import os
import sys
import unittest

import numpy as np

# Make the project folder importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lu_decomposition import (  # noqa: E402
    MatrixError,
    SingularMatrixError,
    ZeroPivotError,
    crout_lu,
    doolittle_lu,
)


class TestDoolittle(unittest.TestCase):
    def test_known_factors(self):
        A = [[2, -1, -2], [-4, 6, 3], [-4, -2, 8]]
        L, U = doolittle_lu(A)
        np.testing.assert_allclose(L, [[1, 0, 0], [-2, 1, 0], [-2, -1, 1]])
        np.testing.assert_allclose(U, [[2, -1, -2], [0, 4, -1], [0, 0, 3]])

    def test_structure(self):
        A = np.random.default_rng(0).random((5, 5)) + 5 * np.eye(5)
        L, U = doolittle_lu(A)
        np.testing.assert_allclose(np.diag(L), np.ones(5))  # unit diagonal
        np.testing.assert_allclose(L, np.tril(L))          # lower triangular
        np.testing.assert_allclose(U, np.triu(U))          # upper triangular
        np.testing.assert_allclose(L @ U, A)


class TestCrout(unittest.TestCase):
    def test_structure(self):
        A = np.random.default_rng(1).random((5, 5)) + 5 * np.eye(5)
        L, U = crout_lu(A)
        np.testing.assert_allclose(np.diag(U), np.ones(5))  # unit diagonal
        np.testing.assert_allclose(L, np.tril(L))
        np.testing.assert_allclose(U, np.triu(U))
        np.testing.assert_allclose(L @ U, A)

    def test_relation_to_doolittle(self):
        A = [[4, 3], [6, 3]]
        L_d, U_d = doolittle_lu(A)
        L_c, U_c = crout_lu(A)
        D = np.diag(np.diag(U_d))
        np.testing.assert_allclose(L_c, L_d @ D)
        np.testing.assert_allclose(U_c, np.linalg.inv(D) @ U_d)


class TestErrors(unittest.TestCase):
    def test_singular(self):
        for method in (doolittle_lu, crout_lu):
            with self.assertRaises(SingularMatrixError):
                method([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_needs_pivoting(self):
        for method in (doolittle_lu, crout_lu):
            with self.assertRaises(ZeroPivotError):
                method([[0, 1], [1, 0]])

    def test_invalid_inputs(self):
        bad_inputs = [
            [[1, 2, 3], [4, 5, 6]],        # not square
            [],                            # empty
            [[1, 2], [3]],                 # ragged rows
            [[1, "a"], [3, 4]],            # non-numeric
            [[1, float("nan")], [3, 4]],   # NaN
            [1, 2, 3],                     # 1-D
        ]
        for data in bad_inputs:
            with self.subTest(data=data):
                with self.assertRaises(MatrixError):
                    doolittle_lu(data)


if __name__ == "__main__":
    unittest.main()
