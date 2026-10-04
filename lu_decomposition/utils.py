"""
utils.py
--------
Helper functions for printing matrices and verifying results.
NumPy is used here only for basic operations (matrix product, norms).
"""

import numpy as np


def format_matrix(M, decimals=4):
    """Return a matrix as a nicely aligned multi-line string."""
    M = np.asarray(M, dtype=float)
    width = decimals + 7  # room for sign, integer part and decimal point
    lines = []
    for row in M:
        # "+ 0.0" turns -0.0 into 0.0 so we never print "-0.0000"
        cells = [f"{value + 0.0:{width}.{decimals}f}" for value in row]
        lines.append("  [" + " ".join(cells) + " ]")
    return "\n".join(lines)


def print_matrix(name, M, decimals=4):
    """Print a matrix with a title above it."""
    print(f"{name} =")
    print(format_matrix(M, decimals))


def verify_decomposition(A, L, U, tol=1e-9):
    """
    Check that A ~= L U.

    Returns a dictionary with:
        max_error      : largest absolute entry of (A - L U)
        relative_error : ||A - L U||_F / ||A||_F   (Frobenius norm)
        ok             : True if the relative error is below `tol`
    """
    A = np.asarray(A, dtype=float)
    residual = A - L @ U
    max_error = float(np.max(np.abs(residual)))
    relative_error = float(np.linalg.norm(residual) / np.linalg.norm(A))
    return {
        "max_error": max_error,
        "relative_error": relative_error,
        "ok": relative_error < tol,
    }


def determinant_from_lu(L, U):
    """
    det(A) = det(L) * det(U), and the determinant of a triangular matrix is
    the product of its diagonal entries.  This is a nice by-product of LU.
    """
    return float(np.prod(np.diag(L)) * np.prod(np.diag(U)))
