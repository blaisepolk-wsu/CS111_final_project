"""
Checklist:

[n] Matrix class:
- [y] constructor
- [y] get_matrix
- [y] get_transpose
- [n] combine_rows
- [n] scale_row
- [n] swap_rows
- [n] rref
- [n] invert
- [n] determinant
- [n] null_space
- [n] col_space
- [n] eigenvalues
- [n] eigenvectors
- [n] diagonalization
- [n] factorization
- [n] nonnegative_factorization

[n] sys_eq_calc()
[n] msum()
[n] mscale()
[n] mmult() 
"""

import numpy as np


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
    def combine_rows(self, row1: int, row2: int, scalar: int):
        pass

    @staticmethod
    def scale_row(self, row: int, scalar: int):
        pass

    @staticmethod
    def swap_rows(self, row1: int, row2: int):
        pass

    def rref(self, reduced_form: True, augment = [0 for _ in range(matrix_rows)]):
        pass

    def invert(self):
        pass

    def determinant(self):
        pass

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
    pass


def mscale(matrix1: Matrix, scalar: int) -> Matrix:
    pass


def mmult(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    pass