# LU Decomposition – Numerical Linear Algebra Course Project

This project implements **LU decomposition from scratch** in Python using two classical
methods, **Doolittle** and **Crout**. It checks whether a matrix is suitable before
decomposing it, prints the factors `L` and `U`, verifies that `A ≈ L × U`, and compares the
two methods on several test matrices.

NumPy is used **only** for storing matrices and for basic operations such as the matrix
product, norms, rank and determinant used for checks and verification. The LU factors
themselves are computed with plain Python loops. No library LU routine
(`scipy.linalg.lu` or similar) is used.

---

## 1. Project structure

```
LU Decomposition/
├── main.py                  # Entry point: menu, demo, printing and comparison
├── test_matrices.py         # Example matrices (good ones and ones that should fail)
├── lu_decomposition/        # The core package
│   ├── __init__.py
│   ├── validation.py        # Input checks: square, invertible, leading minors
│   ├── doolittle.py         # Doolittle algorithm (unit diagonal in L)
│   ├── crout.py             # Crout algorithm (unit diagonal in U)
│   └── utils.py             # Printing, verification A ≈ LU, determinant from LU
├── tests/
│   └── test_lu.py           # Unit tests (unittest)
├── requirements.txt
└── README.md
```

---

## 2. The mathematical idea

LU decomposition writes a square matrix `A` as the product

$$
A = L\,U
$$

where `L` is **lower triangular** and `U` is **upper triangular**. It is basically Gaussian
elimination written down as a matrix factorization: `U` is the matrix you get after
elimination, and `L` stores the multipliers you used.

### Why is it useful?

* **Solving linear systems.** To solve `Ax = b`, factor once and then do two cheap triangular
  solves:
  1. Forward substitution: `L y = b`
  2. Back substitution: `U x = y`

  Factoring costs about `(2/3) n³` operations. After that, each new right-hand side `b`
  costs only about `2 n²`.
* **Determinants.** `det(A) = det(L) · det(U)`, which is the product of the diagonal
  entries.
* **Inverses.** You get `A⁻¹` by solving `A x = eᵢ` for each column of the identity.

### When does it exist (without row exchanges)?

The factorization is not unique as written: for any invertible diagonal `D`,
`A = (L D)(D⁻¹ U)` is another valid LU pair. To make it unique, we fix the diagonal of one
factor to ones. That choice is exactly what separates Doolittle from Crout.

**Theorem.** An invertible matrix `A` has a unique LU factorization without pivoting
**if and only if** all its leading principal minors are non-zero:

$$
\det(A_k) \neq 0 \quad \text{for } k = 1, \dots, n,
$$

where `A_k` is the top-left `k × k` block of `A`.

The program therefore runs three checks before decomposing:

| Check | Why it matters | Error raised |
|-------|----------------|--------------|
| Square, non-empty, numeric, finite | LU is defined here for `n × n` matrices only | `MatrixError` |
| `rank(A) = n` | A singular matrix is not invertible | `SingularMatrixError` |
| All leading principal minors ≠ 0 | Otherwise a zero pivot appears and we would divide by 0 | `ZeroPivotError` |

As a second safety net, both algorithms also check every pivot while they run.

> **Note.** A matrix such as `[[0, 1], [1, 0]]` is invertible, but its first pivot is 0.
> Such matrices need **partial pivoting** (`PA = LU`, with `P` a permutation matrix). Pivoting
> is beyond the scope of this project and is listed under possible extensions.

---

## 3. Doolittle vs. Crout

Both methods compute the same kind of factorization. They differ only in **which factor gets
ones on its diagonal**, and therefore in the order in which entries are computed.

### Doolittle: `L` has a unit diagonal

$$
L = \begin{bmatrix} 1 & 0 & 0 \\ \ell_{21} & 1 & 0 \\ \ell_{31} & \ell_{32} & 1 \end{bmatrix},
\qquad
U = \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}
$$

For `i = 1, …, n`, first compute **row i of U**, then **column i of L**:

$$
u_{ik} = a_{ik} - \sum_{j=1}^{i-1} \ell_{ij}\,u_{jk} \qquad (k = i, \dots, n)
$$

$$
\ell_{ki} = \frac{1}{u_{ii}}\Big(a_{ki} - \sum_{j=1}^{i-1} \ell_{kj}\,u_{ji}\Big) \qquad (k = i+1, \dots, n)
$$

The pivots are `u_ii`. Doolittle gives exactly the `L` and `U` of ordinary Gaussian
elimination.

### Crout: `U` has a unit diagonal

$$
L = \begin{bmatrix} \ell_{11} & 0 & 0 \\ \ell_{21} & \ell_{22} & 0 \\ \ell_{31} & \ell_{32} & \ell_{33} \end{bmatrix},
\qquad
U = \begin{bmatrix} 1 & u_{12} & u_{13} \\ 0 & 1 & u_{23} \\ 0 & 0 & 1 \end{bmatrix}
$$

For `j = 1, …, n`, first compute **column j of L**, then **row j of U**:

$$
\ell_{ij} = a_{ij} - \sum_{k=1}^{j-1} \ell_{ik}\,u_{kj} \qquad (i = j, \dots, n)
$$

$$
u_{ji} = \frac{1}{\ell_{jj}}\Big(a_{ji} - \sum_{k=1}^{j-1} \ell_{jk}\,u_{ki}\Big) \qquad (i = j+1, \dots, n)
$$

The pivots are `ℓ_jj`.

### How the two results are related

Let `D = diag(U_Doolittle)`. Then

$$
L_{\text{Crout}} = L_{\text{Doolittle}}\, D, \qquad U_{\text{Crout}} = D^{-1}\, U_{\text{Doolittle}}.
$$

Both methods therefore have **the same pivots** and give **the same determinant**. The
program checks these relations numerically for every test matrix.

| | Doolittle | Crout |
|---|---|---|
| Unit diagonal in | `L` | `U` |
| Pivots stored in | diagonal of `U` | diagonal of `L` |
| Computation order | row of `U`, then column of `L` | column of `L`, then row of `U` |
| Cost | ≈ `(2/3) n³` flops | ≈ `(2/3) n³` flops |
| `det(A)` | `∏ uᵢᵢ` | `∏ ℓᵢᵢ` |
| Relation to Gaussian elimination | `L` = multipliers, `U` = eliminated matrix | same factors rescaled by `D` |

---

## 4. How to run

**Requirements:** Python 3.8 or newer, and NumPy.

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3a. run the interactive menu
python main.py

# 3b. or run all built-in test matrices at once
python main.py --demo

# 4. run the unit tests
python -m unittest discover tests
```

In the interactive menu, choose **2** to type in your own matrix, one row per line with
numbers separated by spaces or commas:

```
Size of the square matrix n: 2
Enter 2 rows, each with 2 numbers separated by spaces:
  row 1: 1 2
  row 2: 3 4
```

You can also use the functions directly in your own code:

```python
from lu_decomposition import doolittle_lu, crout_lu, verify_decomposition

A = [[2, -1, -2], [-4, 6, 3], [-4, -2, 8]]
L, U = doolittle_lu(A)
print(verify_decomposition(A, L, U))   # {'max_error': 0.0, 'relative_error': 0.0, 'ok': True}
```

---

## 5. Test matrices

| # | Matrix | Purpose | Expected result |
|---|--------|---------|-----------------|
| 1 | 2×2 `[[4,3],[6,3]]` | Easy to check by hand | Success |
| 2 | 3×3 textbook matrix | Integer factors | Success |
| 3 | 4×4 strictly diagonally dominant | LU always exists for this class | Success |
| 4 | 4×4 Hilbert matrix | Ill-conditioned (`cond ≈ 1.5·10⁴`) | Success, small residual |
| 5 | `[[1,2,3],[4,5,6],[7,8,9]]` | Singular (`rank = 2`) | `SingularMatrixError` |
| 6 | `[[0,1],[1,0]]` | Invertible but needs pivoting | `ZeroPivotError` |
| 7 | 2×3 matrix | Not square | `MatrixError` |

---

## 6. Example output

Output of `python main.py --demo` for test 2 (abridged):

```
======================================================================
Test 2: 3x3 textbook matrix
Classic example with integer L and U factors (Doolittle).
======================================================================
A =
  [     2.0000     -1.0000     -2.0000 ]
  [    -4.0000      6.0000      3.0000 ]
  [    -4.0000     -2.0000      8.0000 ]
Leading principal minors: 2, 8, 24
Condition number cond(A) = 30.26
Matrix check: OK (square, invertible, all leading minors non-zero)

>>> Doolittle method
----------------------------------------------------------------------
L =
  [     1.0000      0.0000      0.0000 ]
  [    -2.0000      1.0000      0.0000 ]
  [    -2.0000     -1.0000      1.0000 ]
U =
  [     2.0000     -1.0000     -2.0000 ]
  [     0.0000      4.0000     -1.0000 ]
  [     0.0000      0.0000      3.0000 ]
Verification A = L x U : PASSED
  max |A - LU|            = 0.000e+00
  ||A - LU||_F / ||A||_F  = 0.000e+00

>>> Crout method
----------------------------------------------------------------------
L =
  [     2.0000      0.0000      0.0000 ]
  [    -4.0000      4.0000      0.0000 ]
  [    -4.0000     -4.0000      3.0000 ]
U =
  [     1.0000     -0.5000     -1.0000 ]
  [     0.0000      1.0000     -0.2500 ]
  [     0.0000      0.0000      1.0000 ]
Verification A = L x U : PASSED
  max |A - LU|            = 0.000e+00
  ||A - LU||_F / ||A||_F  = 0.000e+00

>>> Comparison: Doolittle vs. Crout
----------------------------------------------------------------------
Pivots (Doolittle diag(U)) : [2. 4. 3.]
Pivots (Crout     diag(L)) : [2. 4. 3.]
L_crout = L_doolittle * D        : YES
U_crout = D^-1 * U_doolittle     : YES
det(A) from Doolittle (prod diag U) =  24
det(A) from Crout     (prod diag L) =  24
det(A) from numpy (for reference)   =  24
```

Error handling (tests 5 and 6):

```
Test 5: 3x3 singular matrix
[ERROR] Matrix is not suitable for LU decomposition:
        The matrix is singular (rank 2 < 3), so it is not invertible and has no unique LU decomposition.

Test 6: 2x2 matrix that needs pivoting
[ERROR] Matrix is not suitable for LU decomposition:
        Leading principal minor of order 1 is zero, so LU decomposition without row exchanges
        does not exist. (The matrix is invertible, but it would need pivoting: PA = LU.)
```

---

## 7. Observations

* On every valid test matrix, Doolittle and Crout reproduce `A` up to machine precision
  (relative error ≈ `10⁻¹⁷`). Both have the same pivots and the same determinant, as the
  theory predicts.
* The **Hilbert matrix** still factors with a tiny residual. Its pivots, however, shrink
  quickly (`1, 0.083, 0.0056, 0.00036`). A **small residual `‖A − LU‖` does not mean
  that solutions of `Ax = b` will be accurate**: that accuracy is governed by the
  condition number.
* Being invertible is **not enough** for LU without pivoting (test 6). This is why real
  libraries always use partial pivoting.

## 8. Possible extensions

* Partial pivoting (`PA = LU`) so that every invertible matrix can be handled.
* Forward and back substitution to solve `Ax = b` with the computed factors.
* Cholesky decomposition (`A = L Lᵀ`) for symmetric positive definite matrices.
* Timing experiments to observe the `O(n³)` cost.
