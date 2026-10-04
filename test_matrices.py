"""
test_matrices.py
----------------
A collection of example matrices used by main.py to demonstrate and
compare the Doolittle and Crout methods.

Each entry is (name, description, matrix).  The last few are chosen on
purpose to FAIL, so we can show the error handling.
"""

# Hilbert matrix H[i][j] = 1 / (i + j + 1): famous for being ill-conditioned.
HILBERT_4 = [[1.0 / (i + j + 1) for j in range(4)] for i in range(4)]

TEST_MATRICES = [
    (
        "2x2 simple matrix",
        "Small example that is easy to check by hand.",
        [[4, 3],
         [6, 3]],
    ),
    (
        "3x3 textbook matrix",
        "Classic example with integer L and U factors (Doolittle).",
        [[ 2, -1, -2],
         [-4,  6,  3],
         [-4, -2,  8]],
    ),
    (
        "4x4 diagonally dominant matrix",
        "Strictly diagonally dominant => LU without pivoting always exists.",
        [[10, -1,  2,  0],
         [-1, 11, -1,  3],
         [ 2, -1, 10, -1],
         [ 0,  3, -1,  8]],
    ),
    (
        "4x4 Hilbert matrix",
        "Ill-conditioned matrix: decomposition works, but rounding errors grow.",
        HILBERT_4,
    ),
    (
        "3x3 singular matrix",
        "Rows are linearly dependent (row3 = 2*row2 - row1) -> det = 0.",
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
    ),
    (
        "2x2 matrix that needs pivoting",
        "Invertible (det = -1) but a11 = 0, so LU without pivoting fails.",
        [[0, 1],
         [1, 0]],
    ),
    (
        "2x3 non-square matrix",
        "LU decomposition is only defined here for square matrices.",
        [[1, 2, 3],
         [4, 5, 6]],
    ),
]
