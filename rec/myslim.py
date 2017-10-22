from scipy.sparse import lil_matrix, csr_matrix, csc_matrix


class SparseMatrix(lil_matrix):

    def lil_get_col_to_csc(self, j):
        row_num, col_num = self.shape
        new_mat = lil_matrix((row_num, col_num - 1), dtype=self.dtype)
        new_col = self.getcol(j)
        new_mat[:, :j] = self[:, :j]
        new_mat[:, j:] = self[:, j + 1:]
        return csc_matrix(new_col), csc_matrix(new_mat)

    def lil_get_row_to_csc(self, i):
        row_num, col_num = self.shape
        new_mat = lil_matrix((row_num - 1, col_num), dtype=self.dtype)
        new_row = self.getrow(i)
        new_mat[:i, :] = self[:i, :]
        new_mat[i:, :] = self[i + 1:, :]
        return csc_matrix(new_row), csc_matrix(new_mat)

    def get_row_other(self, i):
        new_row = lil_matrix((1, self.shape[1]), dtype=self.dtype)
        new_row.rows[0] = self.rows[i][:]
        new_row.data[0] = self.data[i][:]

        new_matrix = lil_matrix(
            (self.shape[0] - 1, self.shape[1]), dtype=self.dtype)
        new_matrix.rows[:i][:] = self.rows[:i][:]
        new_matrix.data[:i][:] = self.data[:i][:]
        new_matrix.rows[i:][:] = self.rows[i + 1:][:]
        new_matrix.data[i:][:] = self.data[i + 1:][:]
        return csr_matrix(new_row), csr_matrix(new_matrix)

    def get_col_other(self, j):

        return None


a = SparseMatrix((10, 10))
a[1, 1] = 1
b, c = a.lil_get_col_to_csc(1)
d, e = a.lil_get_row_to_csc(1)
print(b.shape)
print(c.shape)
print(d.shape)
print(e.shape)
