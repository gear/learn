"""coding=utf-8
python=3.5.2
"""

import numpy as np


def solve(mat, y):
    """Solve a system of linear equations given by `mat`
    and y.  The rows represent coefficients of each
    equation."""
    reduced = gaussian_elim(mat)
    sol = np.zeros(shape=(mat.shape[0]))
    S = 0
    for i in reversed(range(len(sol))):
        sol[i] = (y[i]-S) / reduced[i][i]
        S += y[i] - S
    return sol

def gaussian_elim(mat):
    """Perform gaussian elimination on matrix `mat`.
    The result is a new upper triangle matrix.
    `mat` must be a n-by-n 2d numpy array."""
    up_mat = np.array(mat, dtype=float)
    n = up_mat.shape[0]
    for r in range(0,n-1):
        for rr in range(r+1, n):
            try:
                ratio = up_mat[rr][r] / up_mat[r][r]
            except ZeroDivisionError:
                print("zero")
                continue
            for i in range(r,n):
                up_mat[rr][i] -= up_mat[r][i] * ratio
    return up_mat

def test():
    print("Testing gaussian elimination...")
    matrix = np.array([[1,2,3,4], [7,5,3,3], [6,3,7,3], [8,1,9,2]])
    print(gaussian_elim(matrix))
    print("Testing system of linear equations...")
    sol = solve(matrix, [1,2,3,4])
    print(sol)
    print("Correct result:")
    print(np.matmul(matrix, sol))

if __name__ == '__main__':
    test()
