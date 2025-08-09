import numpy as np

from basic_functions import clamp
from my_functions import funcs1arg, funcs2arg


def decodeCol(b, f0):
    indxF2 = b[0]
    indxF11 = b[1]
    indxF12 = b[3]
    indxA1 = b[2]
    indxA2 = b[4]

    if (indxF2 == 0 and b[5] == 2) or (indxF11 == 0 or indxA1 == 0):
        return funcs1arg[indxF12 - 1].method(f0[indxA2 - 1])
    if (indxF2 == 0 and b[5] == 1) or (indxF12 == 0 or indxA2 == 0):
        return funcs1arg[indxF11 - 1].method(f0[indxA1 - 1])
    return funcs2arg[indxF2 - 1].method(
        funcs1arg[indxF11 - 1].method(f0[indxA1 - 1]),
        funcs1arg[indxF12 - 1].method(f0[indxA2 - 1]),
    )


def decodeMatrix(mat, xi, xf, q, xCount, qCount, colCount, uMin, uMax):
    f0 = np.full(xCount + qCount + colCount, np.nan, dtype=float)
    xq = np.concatenate((xi - xf, q), axis=0)
    f0[0 : qCount + xCount] = xq
    for k in range(colCount):
        f0[xCount + qCount + k] = decodeCol(mat[:, k], f0)
    return clamp(f0[-1], min_value=uMin, max_value=uMax)


def symDecodeCol(b, f0):
    indxF2 = b[0]
    indxF11 = b[1]
    indxF12 = b[3]
    indxA1 = b[2]
    indxA2 = b[4]

    if (indxF2 == 0 and b[5] == 2) or (indxF11 == 0 or indxA1 == 0):
        return funcs1arg[indxF12 - 1].s_method(f0[indxA2 - 1])
    if (indxF2 == 0 and b[5] == 1) or (indxF12 == 0 or indxA2 == 0):
        return funcs1arg[indxF11 - 1].s_method(f0[indxA1 - 1])
    return funcs2arg[indxF2 - 1].s_method(
        funcs1arg[indxF11 - 1].s_method(f0[indxA1 - 1]),
        funcs1arg[indxF12 - 1].s_method(f0[indxA2 - 1]),
    )


def symDecodeMatrix(mat, xCount, qCount, colCount):
    x_ = ["(x" + str(i + 1) + "- xf" + str(i + 1) + ')' for i in range(xCount)]
    q = ["q" + str(i + 1) for i in range(qCount)]
    f0 = x_ + q
    for k in range(colCount):
        f0.append(symDecodeCol(mat[:, k], f0))
    return f0[-1]
