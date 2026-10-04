"""
main.py
-------
LU Decomposition - Numerical Linear Algebra Course Project

Run:
    python main.py           interactive menu
    python main.py --demo    run all built-in test matrices and exit
"""

import sys

import numpy as np

from lu_decomposition import (
    MatrixError,
    check_matrix,
    crout_lu,
    determinant_from_lu,
    doolittle_lu,
    leading_principal_minors,
    print_matrix,
    verify_decomposition,
)
from test_matrices import TEST_MATRICES

LINE = "=" * 70
THIN_LINE = "-" * 70


# ----------------------------------------------------------------------
# Running one method
# ----------------------------------------------------------------------
def run_method(name, method, A):
    """
    Decompose A with the given method, print L and U, and verify A ~= L U.
    Returns (L, U), or None if the decomposition failed.
    """
    print(f"\n>>> {name} method")
    print(THIN_LINE)
    try:
        L, U = method(A)
    except MatrixError as error:
        print(f"[ERROR] {error}")
        return None

    print_matrix("L", L)
    print_matrix("U", U)
    print_matrix("L x U", L @ U)

    check = verify_decomposition(A, L, U)
    status = "PASSED" if check["ok"] else "FAILED"
    print(f"Verification A = L x U : {status}")
    print(f"  max |A - LU|            = {check['max_error']:.3e}")
    print(f"  ||A - LU||_F / ||A||_F  = {check['relative_error']:.3e}")
    return L, U


# ----------------------------------------------------------------------
# Comparing Doolittle and Crout
# ----------------------------------------------------------------------
def compare_methods(A, doolittle_result, crout_result):
    """
    Compare the two factorizations.

    Theory: if A = L_d U_d (Doolittle) and D = diag(U_d), then the Crout
    factors are   L_c = L_d D   and   U_c = D^(-1) U_d.
    Both methods also give the same determinant.
    """
    L_d, U_d = doolittle_result
    L_c, U_c = crout_result

    D = np.diag(np.diag(U_d))  # diagonal matrix of the Doolittle pivots
    D_inv = np.diag(1.0 / np.diag(U_d))

    same_L = np.allclose(L_c, L_d @ D)
    same_U = np.allclose(U_c, D_inv @ U_d)

    det_d = determinant_from_lu(L_d, U_d)
    det_c = determinant_from_lu(L_c, U_c)
    det_np = np.linalg.det(A)

    print("\n>>> Comparison: Doolittle vs. Crout")
    print(THIN_LINE)
    print(f"Pivots (Doolittle diag(U)) : {np.round(np.diag(U_d), 6)}")
    print(f"Pivots (Crout     diag(L)) : {np.round(np.diag(L_c), 6)}")
    print(f"L_crout = L_doolittle * D        : {'YES' if same_L else 'NO'}")
    print(f"U_crout = D^-1 * U_doolittle     : {'YES' if same_U else 'NO'}")
    print(f"det(A) from Doolittle (prod diag U) = {det_d: .6g}")
    print(f"det(A) from Crout     (prod diag L) = {det_c: .6g}")
    print(f"det(A) from numpy (for reference)   = {det_np: .6g}")


# ----------------------------------------------------------------------
# Full analysis of a matrix
# ----------------------------------------------------------------------
def analyze_matrix(A, title="Input matrix"):
    """Check, decompose with both methods, verify, and compare."""
    print(f"\n{LINE}\n{title}\n{LINE}")

    # Step 1: suitability check (gives a clear message before we start).
    try:
        A = check_matrix(A)
    except MatrixError as error:
        try:
            print_matrix("A", np.array(A, dtype=float))
        except (ValueError, TypeError):
            print(f"A = {A}")
        print(f"\n[ERROR] Matrix is not suitable for LU decomposition:\n        {error}")
        return

    print_matrix("A", A)
    minors = leading_principal_minors(A)
    print("Leading principal minors:", ", ".join(f"{m:.4g}" for m in minors))
    print(f"Condition number cond(A) = {np.linalg.cond(A):.4g}")
    print("Matrix check: OK (square, invertible, all leading minors non-zero)")

    # Step 2: both decompositions.
    doolittle_result = run_method("Doolittle", doolittle_lu, A)
    crout_result = run_method("Crout", crout_lu, A)

    # Step 3: comparison.
    if doolittle_result and crout_result:
        compare_methods(A, doolittle_result, crout_result)


def run_all_tests():
    """Run the analysis on every matrix in test_matrices.py."""
    for number, (name, description, matrix) in enumerate(TEST_MATRICES, start=1):
        analyze_matrix(matrix, title=f"Test {number}: {name}\n{description}")


# ----------------------------------------------------------------------
# Reading a matrix from the keyboard
# ----------------------------------------------------------------------
def read_matrix_from_user():
    """
    Ask the user for the size n and then n rows of numbers.
    Returns a list of lists, or None if the input was invalid.
    """
    try:
        n = int(input("Size of the square matrix n: "))
    except ValueError:
        print("[ERROR] n must be a whole number.")
        return None
    if n <= 0:
        print("[ERROR] n must be positive.")
        return None

    print(f"Enter {n} rows, each with {n} numbers separated by spaces:")
    matrix = []
    for i in range(n):
        text = input(f"  row {i + 1}: ").replace(",", " ")
        try:
            row = [float(value) for value in text.split()]
        except ValueError:
            print("[ERROR] Rows may only contain numbers.")
            return None
        if len(row) != n:
            print(f"[ERROR] Row {i + 1} has {len(row)} numbers, expected {n}.")
            return None
        matrix.append(row)
    return matrix


def interactive_menu():
    """Simple text menu."""
    while True:
        print(f"\n{LINE}")
        print("LU Decomposition - Numerical Linear Algebra Course Project")
        print(LINE)
        print("1) Run all built-in test matrices")
        print("2) Enter my own matrix")
        print("3) Exit")
        choice = input("Choose an option (1-3): ").strip()

        if choice == "1":
            run_all_tests()
        elif choice == "2":
            matrix = read_matrix_from_user()
            if matrix is not None:
                analyze_matrix(matrix, title="Your matrix")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Please type 1, 2 or 3.")


def main():
    if "--demo" in sys.argv:
        run_all_tests()
    else:
        try:
            interactive_menu()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")


if __name__ == "__main__":
    main()
