# necessary imports
import numpy as np
import random as rand


class Matrix:

    matrix_rows, matrix_cols = 0, 0

    def __init__(self, matrix: np.ndarray):
        """
        Desired matrix specifications:
        [ [0, 1, 2, 3, ...] ,
          [ ... ] ,
          [ ... ] , 
          ...]
        """

        # check if matrix is a numpy array
        if not isinstance(matrix, np.ndarray):
            # if input is a 2d-list, convert to a numpy array
            if isinstance(matrix, list):
                matrix = np.array(matrix)
            # otherwise raise an error
            else:
                raise ValueError("Matrix must be a numpy array or a 2d-list")
        
        # check if matrix has > 0 columns & rows
        if len(matrix) == 0 or len(matrix[0]) == 0:
            raise ValueError("Matrix must not be empty")

        # set matrix
        self.matrix = matrix

        # define matrix size outside of constructor for easier use in later methods
        self.matrix_rows, self.matrix_cols = matrix.shape

    def get_matrix(self) -> np.ndarray:
        return self.matrix

    def get_size(self) -> tuple:
        return (self.matrix_rows, self.matrix_cols)

    def get_transpose(self) -> np.ndarray:
        """
        return: matrix transpose
        """

        # no checks required because any valid matrix can be transposed

        # initialize empty matrix
        ret = [[0 for _ in range(self.matrix_rows)] for _ in range(self.matrix_cols)]

        # fill in the transpose
        for i in range(self.matrix_rows):
            for j in range(self.matrix_cols):
                ret[j][i] = self.matrix[i][j]
        
        # return as numpy array
        return np.array(ret)

    @staticmethod
    def combine_rows(self, row1: int, row2: int, scalar: int, matrix = None) -> None:
        """
        return: None
        combines row 1 into row 2 with scalar multiple
        i.e. row2 = row2 + scalar * row1
        alters input matrix in-place
        """

        if matrix is None:
            matrix = self.matrix

        # check to make sure the rows are valid
        if row1 < 0 or row1 >= self.matrix_rows or row2 < 0 or row2 >= self.matrix_rows:
            raise ValueError("Invalid row index")

        # combine rows
        for i in range(self.matrix_cols):
            matrix[row2][i] += scalar * matrix[row1][i]

    @staticmethod
    def scale_row(self, row: int, scalar: int, matrix = None) -> None:
        """
        return: None
        scales row by scalar
        modifies input matrix in-place
        """

        if matrix is None:
            matrix = self.matrix
            
        # check to make sure the row is valid
        if row < 0 or row >= self.matrix_rows:
            raise ValueError("Invalid row index")
    
        # scale row
        for i in range(self.matrix_cols):
            matrix[row][i] *= scalar

    @staticmethod
    def swap_rows(self, row1: int, row2: int, matrix = None) -> None:
        """
        return: None
        swaps row1 and row2
        modifies input matrix in-place
        """

        if matrix is None:
            matrix = self.matrix

        # check to make sure the rows are valid
        if row1 < 0 or row1 >= self.matrix_rows or row2 < 0 or row2 >= self.matrix_rows:
            raise ValueError("Invalid row index")

        # create temporary storage value for row1
        temp = matrix[row1].copy()

        # swap rows
        matrix[row1] = matrix[row2]
        matrix[row2] = temp

    def rref(self, reduced_form = True, augment = None):
        """
        reduced_form: bool, whether to return the matrix in reduced row echelon form
        augment: the augment matrix
        return: augment if specified
        modifies self.matrix in-place
        """

        if augment is None:
            augment = [[0] for _ in range(self.matrix_rows)]

        # initialize augment matrix
        augmented_matrix = self.matrix.copy().tolist()

        # augment the matrix
        for idx in range(self.matrix_rows):
            augmented_matrix[idx].extend(augment[idx])

        # recover as numpy array
        augmented_matrix = np.array(augmented_matrix, dtype=float)

        # get the augmented matrix dimensions
        augment_rows, augment_cols = augmented_matrix.shape
        
        # rref code magic (I do not know how this works)
        row = 0
        
        for col in range(augment_cols):

            if row >= augment_rows:
                break
            
            pivot = np.argmax(np.abs(augmented_matrix [ row:augment_rows, col ] )) + row

            if augmented_matrix [ pivot, col ] == 0:
                continue

            augmented_matrix[[row, pivot]] = augmented_matrix[[pivot, row]]

            augmented_matrix[row] = augmented_matrix[row] / augmented_matrix[row, col]

            for row_idx in range(augment_rows):
                if row_idx != row:
                    augmented_matrix[row_idx] = augmented_matrix[row_idx] - \
                                                augmented_matrix[row_idx, col] * augmented_matrix[row]

            row += 1

        # set the reduced matrix
        self.reduced_matrix = augmented_matrix[:, 0:self.matrix_cols]
        
        # return augment
        return augmented_matrix[:, self.matrix_cols:augment_cols]
    
    def get_reduced_form(self):
        return self.reduced_matrix

    def get_rank(self):
        """
        return: matrix rank (# of pivots)
        """

        # get the reduced matrix
        reduced_matrix = self.get_reduced_form()

        if reduced_matrix is None:
            self.rref()
            reduced_matrix = self.get_reduced_form()

        # get the reduced matrix dimensions
        reduced_rows, reduced_cols = reduced_matrix.shape

        # find pivot rows
        pivot_rows = []

        for row in range(reduced_rows):

            for col in range(reduced_cols):
                
                if reduced_matrix[row][col] == 1:

                    pivot_rows.append(row)
                    continue
        
        return len(pivot_rows)

    def invert(self):
        """
        return: matrix inverse
        """

        # check to make sure the matrix is square
        if self.matrix_rows != self.matrix_cols:
            raise ValueError("Matrix must be square to have an inverse")
        
        # check to make sure the matrix is invertible
        if self.determinant() == 0:
            raise ValueError("Matrix is not invertible")

        def identity_matrix(n: int) -> np.ndarray:
            """
            return: n x n identity matrix
            """

            # initialize empty matrix
            ret = [[0 for _ in range(n)] for _ in range(n)]

            # fill in the identity matrix
            for i in range(n):
                ret[i][i] = 1
            
            # return as numpy array
            return np.array(ret)
        
        print(f'augmenting {self.matrix} with {identity_matrix(self.matrix_rows)}')
        return self.rref(reduced_form = True, augment = identity_matrix(self.matrix_rows))

    def determinant(self, matrix = None) -> int:
        """
        return: matrix determinant
        """

        # get the matrix in usable form
        m = self.matrix if matrix is None else matrix

        # check to make sure the matrix is square
        if len(m) != len(m[0]):
            raise ValueError("Matrix must be square to have a determinant")

        # base case
        if len(m) == 1:
            return m[0][0]
        
        # alternate base case
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        @staticmethod
        def MINOR(matrix: np.ndarray, row: int, col: int) -> np.ndarray:
            """
            return: matrix minor (matrix without the row and column of the element)
            """

            # initialize empty matrix
            ret = [[0 for _ in range(len(matrix) - 1)] for _ in range(len(matrix) - 1)]

            # fill in the minor
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if i != row and j != col:
                        ret[i if i < row else i - 1][j if j < col else j - 1] = matrix[i][j]
            
            # return as numpy array
            return np.array(ret)


        # recursive case
        # returns sum of the determinants of the minors of the first row
        return sum( [ (-1)**i * m[0][i] * self.determinant(MINOR(m, 0, i)) for i in range(len(m)) ] )

    def null_space(self) -> np.ndarray:
        """
        calculates the null space (kernel) of the matrix
        return: null space as a list of vectors
        """

        # make sure Matrix is not empty
        if self.matrix_rows == 0 or self.matrix_cols == 0:
            raise ValueError("Matrix must not be empty to have a null space")
    
        # generate reduced matrix
        self.rref()

        # get the reduced matrix
        reduced_matrix = self.get_reduced_form()

        # get the reduced matrix dimensions
        reduced_rows, reduced_cols = reduced_matrix.shape

        # find pivot rows
        pivot_rows, pivot_cols, free_cols = [], [], [n for n in range(reduced_cols)]

        for row in range(reduced_rows):

            for col in range(reduced_cols):
                
                if reduced_matrix[row][col] == 1:

                    pivot_rows.append(row)
                    pivot_cols.append(col)
                    free_cols.remove(col)
                    continue

        # initialize null space
        null_space = []

        # case where there are free variables
        if len(free_cols) > 0:

            # populate null space
            null_space = [[0 for _ in range(len(free_cols))] for _ in range(reduced_cols)]

            # get free columns
            free_matrix = reduced_matrix[:, free_cols]

            # fill in null space
            for idx, row in enumerate(free_matrix):

                for col_idx, value in enumerate(row):

                    null_space[idx][col_idx] = (-float(value))
            
            # fill in free variable "1"s
            for idx, free_col in enumerate(free_cols):

                null_space[free_col][idx] = 1
        
        return np.array(null_space)

    def col_space(self) -> np.ndarray:
        """
        calculates the column space of the matrix
        return: column space as a list of vectors
        """

        # make sure Matrix is not empty
        if self.matrix_rows == 0 or self.matrix_cols == 0:
            raise ValueError("Matrix must not be empty to have a column space")
    
        # generate reduced matrix
        self.rref()

        # get the reduced matrix
        reduced_matrix = self.get_reduced_form()

        # get the reduced matrix dimensions
        reduced_rows, reduced_cols = reduced_matrix.shape

        # find pivot rows
        pivot_cols = []

        for row in range(reduced_rows):

            for col in range(reduced_cols):
                
                if reduced_matrix[row][col] == 1:

                    pivot_cols.append(col)
                    continue
        
        return self.matrix[:, pivot_cols]

    @staticmethod
    def MMULT(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """
        static method to be used only when multiplying two matrices directly related to self.matrix
        return: product of matrix1 & matrix2
        """
        pass

    def eigenvalues(self):
        """
        return: list of eigenvalues
        """

        from numpy import linalg as LA

        return LA.eig(self.matrix)[0]

    def eigenvectors(self):
        """
        return: list of eigenvectors
        """

        from numpy import linalg as LA

        return LA.eig(self.matrix)[1]

    def diagonalization(self):
        pass

    def factorization(self):
        pass

    def nonnegative_factorization(self):
        pass
    

def msum(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    return: sum of matrix1 & matrix2
    """

    # check to make sure they are the same size
    if matrix1.get_size() != matrix2.get_size():
        raise ValueError("Matrices must be the same size to be added")

    # get the matrices in usable form
    m1, m2 = matrix1.get_matrix(), matrix2.get_matrix()
    
    # return summed matrix
    return Matrix(np.array([ [m1[row][col] + m2[row][col] for col in range(matrix1.get_size()[1])] \
                    for row in range(matrix1.get_size()[0]) ]))


def mscale(matrix: Matrix, scalar: int) -> Matrix:
    """
    return: matrix scaled by scalar
    """

    # no need for checks because any matrix can be scaled

    # get the matrix in usable form
    m = matrix.get_matrix()

    # return scaled matrix
    return Matrix(np.array([ [m[row][col] * scalar for col in range(matrix.get_size()[1])] \
                    for row in range(matrix.get_size()[0]) ]))


def mmult(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """
    return: product of matrix1 & matrix2
    """

    # check to make sure the two are compatible sizes
    if matrix1.get_size()[1] != matrix2.get_size()[0]:
        raise ValueError("Matrices must have compatible sizes to be multiplied")
    
    # get the matrices in usable form
    m1, m2 = matrix1.get_matrix(), matrix2.get_matrix()

    # initialize empty matrix
    ret = [[0 for _ in range(matrix2.get_size()[1])] for _ in range(matrix1.get_size()[0])]

    # fill in the product
    for i in range(matrix1.get_size()[0]):
        for j in range(matrix2.get_size()[1]):
            for k in range(matrix1.get_size()[1]):
                ret[i][j] += m1[i][k] * m2[k][j]
    
    # return as Matrix
    return Matrix(np.array(ret))
