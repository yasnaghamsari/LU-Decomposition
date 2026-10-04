"""
doolittle.py
------------
Doolittle LU decomposition:   A = L U

    * L is LOWER triangular with 1's on its diagonal  (unit lower triangular)
    * U is UPPER triangular

Formulas (for row i = 0, 1, ..., n-1):

    U[i][k] = A[i][k] - sum_{j<i} L[i][j] * U[j][k]               for k >= i
    L[k][i] = ( A[k][i] - sum_{j<i} L[k][j] * U[j][i] ) / U[i][i]  for k >  i

So we compute ROW i of U first, then COLUMN i of L.
The pivots are the diagonal entries U[i][i]; they must not be zero.

No library LU routine is used - only plain Python loops.
"""

import numpy as np

from .validation import DEFAULT_TOLERANCE, ZeroPivotError, check_matrix


def doolittle_lu(data, tol=DEFAULT_TOLERANCE):
    """
    Compute the Doolittle LU decomposition of a square matrix.

    Parameters
    ----------
    data : list of lists or NumPy array
        The square matrix A.
    tol : float
        Pivots with absolute value <= tol are treated as zero.

    Returns
    -------
    L, U : NumPy arrays
        L is unit lower triangular, U is upper triangular, and A = L @ U.

    Raises
    ------
    MatrixError (or a subclass) if the matrix is invalid, singular,
    or needs pivoting.
    """
    A = check_matrix(data, tol)
    n = A.shape[0]

    # Start with zero matrices; we fill them in step by step.
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # ---- Row i of U ------------------------------------------------
        for k in range(i, n):
            total = 0.0
            for j in range(i):
                total += L[i][j] * U[j][k]
            U[i][k] = A[i][k] - total

        # The pivot U[i][i] is used as a divisor below, so it must be non-zero.
        if abs(U[i][i]) <= tol:
            raise ZeroPivotError(
                f"Doolittle: zero pivot U[{i + 1}][{i + 1}] = {U[i][i]:.3e}. "
                f"Row exchanges (pivoting) would be required."
            )

        # ---- Column i of L ---------------------------------------------
        L[i][i] = 1.0  # Doolittle convention: unit diagonal in L
        for k in range(i + 1, n):
            total = 0.0
            for j in range(i):
                total += L[k][j] * U[j][i]
            L[k][i] = (A[k][i] - total) / U[i][i]

    return L, U
