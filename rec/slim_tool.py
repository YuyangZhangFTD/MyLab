from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
from sklearn import linear_model
import numpy as np


class SparseMatrix(lil_matrix):

    def lil_get_col_to_csc(self, j):
        row_num, col_num = self.shape
        new_mat = lil_matrix((row_num, col_num - 1), dtype=self.dtype)
        new_col = self.getcol(j)
        new_mat[:, :j] = self[:, :j]
        new_mat[:, j:] = self[:, j + 1:]
        return csc_matrix(new_col), csc_matrix(new_mat)


def train(
        A,
        l1_ratio=0.5,
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        positive=True,
        max_iter=1000):
    sample_num, feature_num = A.shape
    W = lil_matrix((feature_num, feature_num))
    A = SparseMatrix(A)

    for j in range(feature_num):
        aj, Aj = A.lil_get_col_to_csc(j)

        if aj.nnz == 0:
            continue

        aj = aj.toarray().ravel()

        alphas, coefs, __ = linear_model.enet_path(
            Aj, aj, l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, alphas=alphas, positive=positive, max_iter=max_iter)

        W[:j, j] = coefs[:j, -1].reshape(j, 1)
        W[j + 1:, j] = coefs[j:, -1].reshape(feature_num - j - 1, 1)

    return W


if __name__ == "__main__":
    a = SparseMatrix((10, 10))
    a[1, 1] = 1
    b, c = a.lil_get_col_to_csc(1)
    print(b.shape)
    print(c.shape)

    b = lil_matrix(np.random.random([1000, 20]))
    print(train(b))
