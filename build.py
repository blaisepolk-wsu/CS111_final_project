"""
Authors: Ryan Ludwig, Blaise Polk
Completion Date: 12/7/24

Project Description:
This project is a Linear Algebra calculator that can perform a variety of operations on matrices.
The calculator works in a custom GUI using 2D arrays to represent matrices.
The calculator can perform the following operations:
add and multiply matrices
find the transpose, inverse, determinant, rank, null space, column space, eigenvalues, and eigenvectors of a matrix
row reduce the matrix
The calculator also displays the outputs in a copyable format for easy repeated use.
"""

import gui


def main():
    """
    The main function of the Linear Algebra calculator build.
    This function exclusively creates the GUI for the Linear Algebra calculator.
    See gui.py for the implementation of the GUI.
    See Matrix.py for the background calculations involving the custom matrices.
    """
    
    gui.create_gui()

if __name__ == "__main__":
    main()