"""
Checklist:

[n] Matrix class:
- [y] constructor
- [y] get_matrix
- [y] get_transpose
- [y] combine_rows
- [y] scale_row
- [y] swap_rows
- [n] rref
- [n] invert
- [y] determinant
- [n] null_space
- [n] col_space
- [n] eigenvalues
- [n] eigenvectors
- [n] diagonalization
- [n] factorization
- [n] nonnegative_factorization

[n] sys_eq_calc()
[y] msum()
[y] mscale()
[y] mmult() 
"""

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
        self.matrix_rows = len(matrix)
        self.matrix_cols = len(matrix[0])

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
    def combine_rows(self, row1: int, row2: int, scalar: int) -> None:
        """
        return: None
        combines row 1 into row 2 with scalar multiple
        i.e. row2 = row2 + scalar * row1
        alters self.matrix in-place
        """

        # check to make sure the rows are valid
        if row1 < 0 or row1 >= self.matrix_rows or row2 < 0 or row2 >= self.matrix_rows:
            raise ValueError("Invalid row index")

        # combine rows
        for i in range(self.matrix_cols):
            self.matrix[row2][i] += scalar * self.matrix[row1][i]

    @staticmethod
    def scale_row(self, row: int, scalar: int) -> None:
        """
        return: None
        scales row by scalar
        modifies self.matrix in-place
        """
            
        # check to make sure the row is valid
        if row < 0 or row >= self.matrix_rows:
            raise ValueError("Invalid row index")
    
        # scale row
        for i in range(self.matrix_cols):
            self.matrix[row][i] *= scalar

    @staticmethod
    def swap_rows(self, row1: int, row2: int) -> None:
        """
        return: None
        swaps row1 and row2
        modifies self.matrix in-place
        """

        # check to make sure the rows are valid
        if row1 < 0 or row1 >= self.matrix_rows or row2 < 0 or row2 >= self.matrix_rows:
            raise ValueError("Invalid row index")

        # create temporary storage value for row1
        temp = self.matrix[row1].copy()

        # swap rows
        self.matrix[row1] = self.matrix[row2]
        self.matrix[row2] = temp

    def rref(self, reduced_form = True, augment = [0 for _ in range(matrix_rows)]):
        pass

    def invert(self):
        pass

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

    def null_space(self):
        pass

    def col_space(self):
        pass

    def eigenvalues(self):
        pass

    def eigenvectors(self):
        pass

    def diagonalization(self):
        pass

    def factorization(self):
        pass

    def nonnegative_factorization(self):
        pass
    

def sys_eq_calc(matrix: np.ndarray) -> list:
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

    print(ret)

    # fill in the product
    for i in range(matrix1.get_size()[0]):
        for j in range(matrix2.get_size()[1]):
            for k in range(matrix1.get_size()[1]):
                ret[i][j] += m1[i][k] * m2[k][j]
    
    # return as Matrix
    return Matrix(np.array(ret))


def mseed(rows: int, cols: int, max_val: int, min_val: int, decimals = False) -> Matrix:
    """
    return: random matrix with specified rows and columns
    """

    # check to make sure the dimensions are valid
    if rows <= 0 or cols <= 0:
        raise ValueError("Matrix must have positive dimensions")

    # return random matrix
    if decimals:
        return Matrix(np.array([[rand.randint(min_val, max_val) * rand.random() for _ in range(cols)] for _ in range(rows)]))
    else:
        return Matrix(np.array([[rand.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]))