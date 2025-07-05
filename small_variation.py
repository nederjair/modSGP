import numpy as np
from random import randrange
from random import choice
from my_functions import funcs1arg, funcs2arg


def encodeSVRow(permitted_rows, permitted_cols,
                xCount, u_count, qCount):
    svRow = np.zeros(4, dtype=int)
    rowIndex = choice(permitted_rows)
    colIndex = choice(permitted_cols)
    u_index = randrange(1, u_count + 1)

    svRow[0] = u_index
    svRow[1] = rowIndex
    svRow[2] = colIndex

    if rowIndex == 1:
        svRow[3] = randrange(0, len(funcs2arg) + 1)
    elif rowIndex == 2 or rowIndex == 4:
        svRow[3] = randrange(0, len(funcs1arg) + 1)
    elif rowIndex == 3 or rowIndex == 5:
        svRow[3] = randrange(0, xCount + qCount + colIndex)
    elif rowIndex == 6:
        svRow[3] = randrange(1, 3)
    return svRow


def encodeSVMat(permitted_cols, permitted_rows, xCount,
                u_count, qCount, svRowCount):
    sv_mat = np.zeros((svRowCount, 4), dtype=int)
    for i in range(svRowCount):
        sv_mat[i] = encodeSVRow(permitted_cols, permitted_rows,
                                xCount, u_count, qCount)
    return sv_mat


def applySVRow(mat, svRow):
    uIndex = svRow[0] - 1
    rowIndex = svRow[1] - 1
    colIndex = svRow[2] - 1
    b = mat[uIndex, :, colIndex]
    if rowIndex == 1 and b[5] == 1 and svRow[3] == 0:
        pass
    elif rowIndex == 3 and b[5] == 2 and svRow[3] == 0:
        pass
    elif rowIndex == 2 and b[5] == 1 and svRow[3] == 0:
        pass
    elif rowIndex == 4 and b[5] == 2 and svRow[3] == 0:
        pass
    elif rowIndex == 5 and b[1]*b[2]*b[3]*b[4] == 0:
        pass
    else:
        mat[uIndex][rowIndex][colIndex] = svRow[3]


def applySVMat(basic_solution, svMat, svRowCount):
    basic_solution_copy = basic_solution.copy()
    for i in range(svRowCount):
        applySVRow(basic_solution_copy, svMat[i])
    return basic_solution_copy
