import numpy as np

from sgp import decodeMatrix

neutral_index = 1
mult_index = 2  # multiplication index
add_index = 1  # addition index
neg_index = 3  # negative index


def get_control(mat_set, xCount, qCount, colCount, q, xi, xf, uMin, uMax, uCount):
    u = np.full(uCount, np.nan, dtype=float)
    for k in range(uCount):
        u[k] = decodeMatrix(mat_set[k], xi, xf, q, xCount, qCount, colCount, uMin, uMax)
    return u


def encode_u1(free_space, pm_col_count, xCount, qCount):
    b_space = 8
    mat = np.zeros((6, b_space + free_space), dtype=int)

    mat[0][0] = add_index
    mat[1][0] = neutral_index
    mat[2][0] = 4
    mat[3][0] = neg_index
    mat[4][0] = 1
    mat[5][0] = 1

    mat[0][1] = add_index
    mat[1][1] = neutral_index
    mat[2][1] = 5
    mat[3][1] = neg_index
    mat[4][1] = 2
    mat[5][1] = 1

    mat[0][2] = add_index
    mat[1][2] = neutral_index
    mat[2][2] = 6
    mat[3][2] = neg_index
    mat[4][2] = 3
    mat[5][2] = 1

    mat[0][3] = mult_index
    mat[1][3] = neutral_index
    mat[2][3] = 7
    mat[3][3] = neutral_index
    mat[4][3] = 13
    mat[5][3] = 1

    mat[0][4] = mult_index
    mat[1][4] = neutral_index
    mat[2][4] = 8
    mat[3][4] = neutral_index
    mat[4][4] = 14
    mat[5][4] = 1

    mat[0][5] = mult_index
    mat[1][5] = neutral_index
    mat[2][5] = 9
    mat[3][5] = neutral_index
    mat[4][5] = 15
    mat[5][5] = 1

    mat[0][6] = add_index
    mat[1][6] = neutral_index
    mat[2][6] = 16
    mat[3][6] = neutral_index
    mat[4][6] = 17
    mat[5][6] = 1

    mat[0][7] = add_index
    mat[1][7] = neutral_index
    mat[2][7] = 19
    mat[3][7] = neutral_index
    mat[4][7] = 18
    mat[5][7] = 1

    for k in range(free_space):
        mat[0][b_space + k] = add_index
        mat[1][b_space + k] = neutral_index
        mat[2][b_space + k] = 4 * xCount + qCount + 2 + k
        mat[3][b_space + k] = 0
        mat[4][b_space + k] = 0
        mat[5][b_space + k] = 1

    return mat


def encode_u2(free_space, pm_col_count, xCount, qCount):
    b_space = 8
    mat = np.zeros((6, b_space + free_space), dtype=int)

    mat[0][0] = add_index
    mat[1][0] = neutral_index
    mat[2][0] = 4
    mat[3][0] = neg_index
    mat[4][0] = 1
    mat[5][0] = 1

    mat[0][1] = add_index
    mat[1][1] = neutral_index
    mat[2][1] = 5
    mat[3][1] = neg_index
    mat[4][1] = 2
    mat[5][1] = 1

    mat[0][2] = add_index
    mat[1][2] = neutral_index
    mat[2][2] = 6
    mat[3][2] = neg_index
    mat[4][2] = 3
    mat[5][2] = 1

    mat[0][3] = mult_index
    mat[1][3] = neutral_index
    mat[2][3] = 10
    mat[3][3] = neutral_index
    mat[4][3] = 13
    mat[5][3] = 1

    mat[0][4] = mult_index
    mat[1][4] = neutral_index
    mat[2][4] = 11
    mat[3][4] = neutral_index
    mat[4][4] = 14
    mat[5][4] = 1

    mat[0][5] = mult_index
    mat[1][5] = neutral_index
    mat[2][5] = 12
    mat[3][5] = neutral_index
    mat[4][5] = 15
    mat[5][5] = 1

    mat[0][6] = add_index
    mat[1][6] = neutral_index
    mat[2][6] = 16
    mat[3][6] = neutral_index
    mat[4][6] = 17
    mat[5][6] = 1

    mat[0][7] = add_index
    mat[1][7] = neutral_index
    mat[2][7] = 19
    mat[3][7] = neutral_index
    mat[4][7] = 18
    mat[5][7] = 1

    for k in range(free_space):
        mat[0][b_space + k] = add_index
        mat[1][b_space + k] = neutral_index
        mat[2][b_space + k] = 4 * xCount + qCount + 2 + k
        mat[3][b_space + k] = 0
        mat[4][b_space + k] = 0
        mat[5][b_space + k] = 1

    return mat


def encode_u_simple(free_space, pm_col_count, x_count, q_count):
    mat = np.array(
        [
            encode_u1(free_space, pm_col_count, x_count, q_count),
            encode_u2(free_space, pm_col_count, x_count, q_count),
        ]
    )
    return mat
