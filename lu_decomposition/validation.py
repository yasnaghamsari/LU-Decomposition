"""
validation.py
-------------
Checks that a matrix is suitable for LU decomposition *before* we try to
decompose it.

Two different questions are asked here:

1. Is the matrix invertible (non-singular)?
   A square matrix A is invertible  <=>  det(A) != 0  <=>  rank(A) = n.

2. Does an LU decomposition WITHOUT row exchanges (pivoting) exist?
   Theorem: an invertible matrix A has an LU factorization without
   pivoting if and only if every leading principal minor is non-zero:

        det(A[0:k, 0:k]) != 0      for k = 1, 2, ..., n

   Example: A = [[0, 1], [1, 0]] is invertible (det = -1), but its first
   leading minor is a11 = 0, so plain Doolittle/Crout cannot start.

NumPy is used here only for basic operations (determinant, rank) to
*check* the matrix - never to compute the LU factors themselves.
"""

import numpy as np

# Numbers smaller than this (in absolute value) are treated as zero.
DEFAULT_TOLERANCE = 1e-12


class MatrixError(ValueError):
    """Raised when a matrix is invalid or unsuitable for LU decomposition."""


class SingularMatrixError(MatrixError):
    """Raised when the matrix is singular (not invertible)."""


class ZeroPivotError(MatrixError):
    """Raised when a zero pivot appears, i.e. LU without pivoting fails."""


def to_square_matrix(data):
    """
    Convert the input (list of lists, NumPy array, ...) into a float NumPy
    array and make sure it is a valid, non-empty, square, finite matrix.

    Returns the matrix as a 2-D NumPy array of floats.
    Raises MatrixError if anything is wrong.
    """
    try:
        A = np.array(data, dtype=float)
    except (ValueError, TypeError):
        raise MatrixError(
            "The matrix contains non-numeric entries or rows of different lengths."
        )

    if A.ndim != 2:
        raise MatrixError(
            f"Expected a 2-D matrix, but got an object with {A.ndim} dimension(s)."
        )

    rows, cols = A.shape
    if rows == 0 or cols == 0:
        raise MatrixError("The matrix is empty.")

    if rows != cols:
        raise MatrixError(
            f"LU decomposition needs a square matrix, but this one is {rows} x {cols}."
        )

    if not np.all(np.isfinite(A)):
        raise MatrixError("The matrix contains NaN or infinite values.")

    return A


def leading_principal_minors(A):
    """
    Return the list [det(A_1), det(A_2), ..., det(A_n)] where A_k is the
    top-left k x k sub-matrix of A.
    """
    n = A.shape[0]
    return [np.linalg.det(A[:k, :k]) for k in range(1, n + 1)]


def check_matrix(data, tol=DEFAULT_TOLERANCE):
    """
    Full suitability check for LU decomposition without pivoting.

    Steps:
        1. Valid square numeric matrix?            (MatrixError)
        2. Invertible?  rank(A) == n               (SingularMatrixError)
        3. All leading principal minors non-zero?  (ZeroPivotError)

    Returns the matrix as a float NumPy array if every check passes.
    """
    A = to_square_matrix(data)
    n = A.shape[0]

    # --- Step 2: invertibility -------------------------------------------
    # matrix_rank is more reliable than "det == 0" for floating point
    # numbers, because a determinant can be tiny even for a good matrix
    # (e.g. 0.1 * identity of size 20 has det = 1e-20).
    rank = np.linalg.matrix_rank(A)
    if rank < n:
        raise SingularMatrixError(
            f"The matrix is singular (rank {rank} < {n}), so it is not invertible "
            f"and has no unique LU decomposition."
        )

    # --- Step 3: leading principal minors --------------------------------
    # We compare each minor relative to the size of the entries involved,
    # so the test does not depend on how the matrix is scaled.
    for k, minor in enumerate(leading_principal_minors(A), start=1):
        scale = max(1.0, np.max(np.abs(A[:k, :k]))) ** k
        if abs(minor) <= tol * scale:
            raise ZeroPivotError(
                f"Leading principal minor of order {k} is zero, so LU decomposition "
                f"without row exchanges does not exist. "
                f"(The matrix is invertible, but it would need pivoting: PA = LU.)"
            )

    return A
