from scipy.sparse import lil_matrix


class sparse_matrix(lil_matrix):

	def __init__(self, arg1, shape=None, dtype=None, copy=False):
		lil_matrix.__init__(arg1, shape=shape, dtype=dtype, copy=copy)

	def get_row_other(self,i):
		i = sp.lil_matrix._check_row_bounds(i)
		
		new_row = lil_matrix((1, self.shape[1]), dtype=self.dtype)
		new_row.rows[0] = self.rows[i][:]
		new_row.data[0] = self.data[i][:]
		
		new_matrix = lil_matrix((self.shape[0]-1, self.shape[1]), dtype=self.dtype)
		new_matrix.rows[:i] = self.rows[:i][:]
		new_matrix.data[:i] = self.data[:i][:]
		new_matrix.row[i:] = self.rows[i+1:][:]
		new_matrix.data[i:] = self.data[i+1:][:]
