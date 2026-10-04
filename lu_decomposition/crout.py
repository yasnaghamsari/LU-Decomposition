"""
crout.py
--------
Crout LU decomposition:   A = L U

    * L is LOWER triangular
    * U is UPPER triangular with 1's on its diagonal  (unit upper triangular)

Formulas (for column j = 0, 1, ..., n-1):

    L[i][j] = A[i][j] - sum_{k<j} L[i][k] * U[k][j]               for i >= j
    U[j][i] = ( A[j][i] - sum_{k<j} L[j][k] * U[k][i] ) / L[j][j]  for i >  j

So we compute COLUMN j of L first, then ROW j of U.
The pivots are the diagonal entries L[j][j]; they must not be zero.

No library LU routine is used - only plain Python loops.
"""

import numpy as np

from .validation import DEFAULT_TOLERANCE, ZeroPivotError, check_matrix


def crout_lu(data, tol=DEFAULT_TOLERANCE):
    """
    Compute the Crout LU decomposition of a square matrix.

    Parameters
    ----------
    data : list of lists or NumPy array
        The square matrix A.
    tol : float
        Pivots with absolute value <= tol are treated as zero.

    Returns
    -------
    L, U : NumPy arrays
        L is lower triangular, U is unit upper triangular, and A = L @ U.

    Raises
    ------
    MatrixError (or a subclass) if the matrix is invalid, singular,
    or needs pivoting.
    """
    A = check_matrix(data, tol)
    n = A.shape[0]

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        # ---- Column j of L ---------------------------------------------
        for i in range(j, n):
            total = 0.0
            for k in range(j):
                total += L[i][k] * U[k][j]
            L[i][j] = A[i][j] - total

        # The pivot L[j][j] is used as a divisor below, so it must be non-zero.
        if abs(L[j][j]) <= tol:
            raise ZeroPivotError(
                f"Crout: zero pivot L[{j + 1}][{j + 1}] = {L[j][j]:.3e}. "
                f"Row exchanges (pivoting) would be required."
            )

        # ---- Row j of U ------------------------------------------------
        U[j][j] = 1.0  # Crout convention: unit diagonal in U
        for i in range(j + 1, n):
            total = 0.0
            for k in range(j):
                total += L[j][k] * U[k][i]
            U[j][i] = (A[j][i] - total) / L[j][j]

    return L, U
