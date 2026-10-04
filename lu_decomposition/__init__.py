"""
lu_decomposition
================
LU decomposition from scratch (Doolittle and Crout) for a
Numerical Linear Algebra course project.
"""

from .crout import crout_lu
from .doolittle import doolittle_lu
from .utils import determinant_from_lu, format_matrix, print_matrix, verify_decomposition
from .validation import (
    MatrixError,
    SingularMatrixError,
    ZeroPivotError,
    check_matrix,
    leading_principal_minors,
)

__all__ = [
    "doolittle_lu",
    "crout_lu",
    "check_matrix",
    "leading_principal_minors",
    "verify_decomposition",
    "determinant_from_lu",
    "format_matrix",
    "print_matrix",
    "MatrixError",
    "SingularMatrixError",
    "ZeroPivotError",
]
