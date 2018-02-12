import numpy as np
import numpy.linalg as la

# positive infinity
POS_INF = 999999999999999999999999999999


def simplex(c, P, b):
    """
        m equalities
        n variable (m basis variables and n-m non-basis variables)
    :param C:
    :param P:
    :return:
    """
    m, n = P.shape
    B_index = list(range(n - m, n))  # Basis Variables index
    while True:
        CB = c[B_index]  # C_B
        B = P[:, B_index]  # B matrix
        B_inv = la.inv(B)  # B^{-1}
        XB = B_inv * b  # X_B
        z = CB.T * XB  # z

        # optimality computations
        r = [(i, CB.T * B_inv * P[:, i] - c[i]) for i in range(n) if i not in B_index]
        enter_index, enter_check = r[0]
        for i in range(1, len(r)):
            if enter_check > r[i][1]:
                enter_index, enter_check = r[i]

        # Feasibility computations
        theta = XB / (B_inv * P[:, enter_index])
        out_index = np.argmin(np.where(theta > 0, theta, POS_INF))
        out_theta = theta[out_index]
        if enter_check > 0:
            break
        B_index = [i if i != B_index[out_index] else enter_index for i in B_index]

    solution = np.mat(np.zeros((n, 1)))
    for i in range(m):
        solution[B_index[i], 0] = XB[i]
    return solution, z


if __name__ == '__main__':
    c = np.mat([5, 4, 0, 0, 0, 0]).reshape(6, 1)
    P = np.mat(
        [
            [6, 4, 1, 0, 0, 0],
            [1, 2, 0, 1, 0, 0],
            [-1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1]
        ]
    )
    b = np.mat([24, 6, 1, 2]).reshape(4, 1)
    solu, z = simplex(c, P, b)
    print("hello world")
