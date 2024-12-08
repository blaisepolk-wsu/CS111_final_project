import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
import Matrix


class MatrixGUI:
    """
    This block of code initializes the main application window and starts the Tkinter event loop.
    If this script is run as the main program, it creates an instance of the Tkinter Tk class,
    which represents the main window of the application. It then creates an instance of the
    MatrixGUI class, passing the main window as an argument. Finally, it starts the Tkinter
    event loop by calling the mainloop() method on the root window, which waits for user
    interaction and updates the GUI accordingly.
    MatrixGUI is a class that creates a graphical user interface (GUI) for performing various matrix operations.

    Attributes:
    root (tk.Tk): The root window of the GUI.
    matrix1 (None): Placeholder for the first matrix.
    matrix2 (None): Placeholder for the second matrix.
    matrix1_label (tk.Label): Label for the first matrix entry.
    matrix1_entry (tk.Entry): Entry widget for the first matrix.
    matrix2_label (tk.Label): Label for the second matrix entry.
    matrix2_entry (tk.Entry): Entry widget for the second matrix.
    operation_label (tk.Label): Label for the operation selection.
    operation_var (tk.StringVar): Variable to store the selected operation.
    operation_menu (tk.OptionMenu): Dropdown menu for selecting the matrix operation.
    execute_button (tk.Button): Button to execute the selected operation.
    result_label (tk.Label): Label for the result display.
    result_text (tk.Text): Text widget to display the result of the operation.

    Methods:
    __init__(self, root):
        Initializes the MatrixGUI with the given root window and sets up the initial state.
    create_widgets(self):
        Creates and places the widgets (labels, entries, buttons, etc.) in the GUI.
    execute_operation(self):
        Executes the selected matrix operation based on the user input and displays the result.
    """    
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Operations")

        self.matrix1 = None
        self.matrix2 = None

        self.create_widgets()

    def create_widgets(self):
        self.matrix_explanation1 = tk.Label(self.root, text="Enter matrices as 2D lists (e.g., [[1, 2], [3, 4]]).")
        self.matrix_explanation1.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.matrix_explanation2 = tk.Label(self.root, text="All operations not involving 2 matrices will act on Matrix 1.")
        self.matrix_explanation2.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

        self.matrix1_label = tk.Label(self.root, text="Matrix 1:")
        self.matrix1_label.grid(row=2, column=0, padx=10, pady=10)
        self.matrix1_entry = tk.Entry(self.root, width=50)
        self.matrix1_entry.grid(row=2, column=1, padx=10, pady=10)

        self.matrix2_label = tk.Label(self.root, text="Matrix 2:")
        self.matrix2_label.grid(row=3, column=0, padx=10, pady=10)
        self.matrix2_entry = tk.Entry(self.root, width=50)
        self.matrix2_entry.grid(row=3, column=1, padx=10, pady=10)

        self.operation_label = tk.Label(self.root, text="Operation:")
        self.operation_label.grid(row=4, column=0, padx=10, pady=10)
        self.operation_var = tk.StringVar(self.root)
        self.operation_var.set("Add")
        self.operation_menu = tk.OptionMenu(self.root, self.operation_var, "Add", "Multiply", "Transpose", "Inverse", "Determinant", "Rank", "Null Space", "Column Space", "Eigenvalues", "Eigenvectors", "Reduced Row Echelon Form")
        self.operation_menu.grid(row=4, column=1, padx=10, pady=10)

        self.execute_button = tk.Button(self.root, text="Execute", command=self.execute_operation)
        self.execute_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        self.result_label = tk.Label(self.root, text="Result:")
        self.result_label.grid(row=6, column=0, padx=10, pady=10)
        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.grid(row=6, column=1, padx=10, pady=10)

    def execute_operation(self):
        try:
            matrix1 = np.array(eval(self.matrix1_entry.get()))
            matrix2 = np.array(eval(self.matrix2_entry.get())) if self.matrix2_entry.get() else None
            mat1 = Matrix(matrix1)
            mat2 = Matrix(matrix2) if matrix2 is not None else None

            operation = self.operation_var.get()
            result = None
            """
            Executes the selected matrix operation based on user input.
            This method retrieves matrices from the user input, performs the selected
            operation, and displays the result. Supported operations include addition,
            multiplication, transpose, inverse, determinant, rank, null space, column
            space, eigenvalues, and eigenvectors.
            Raises:
                Exception: If an error occurs during matrix operations or input retrieval.
            Operations:
                - Add: Adds two matrices.
                - Multiply: Multiplies two matrices.
                - Transpose: Computes the transpose of a matrix.
                - Inverse: Computes the inverse of a matrix.
                - Determinant: Computes the determinant of a matrix.
                - Rank: Computes the rank of a matrix.
                - Null Space: Computes the null space of a matrix.
                - Column Space: Computes the column space of a matrix.
                - Eigenvalues: Computes the eigenvalues of a matrix.
                - Eigenvectors: Computes the eigenvectors of a matrix.
                - Reduced Row Echelon Form: Computes the reduced row echelon form of a matrix.
            """
            if operation == "Add":
                result = mat1.msum(mat1, mat2).get_matrix()
            elif operation == "Multiply":
                result = mat1.mmult(mat1, mat2).get_matrix()
            elif operation == "Transpose":
                result = mat1.get_transpose()
            elif operation == "Inverse":
                result = mat1.invert()[::, 2::]
            elif operation == "Determinant":
                result = mat1.determinant()
            elif operation == "Rank":
                result = mat1.get_rank()
            elif operation == "Null Space":
                result = mat1.null_space()
                if result.tolist() == []:
                    result = "Matrix has no null space"
            elif operation == "Column Space":
                result = mat1.col_space()
            elif operation == "Eigenvalues":
                result = mat1.eigenvalues()
            elif operation == "Eigenvectors":
                result = mat1.eigenvectors()
            elif operation == "Reduced Row Echelon Form":
                result = mat1.rref()
    
            self.result_text.delete(1.0, tk.END)
            if type(result) is not int and type(result) is not str:
                self.result_text.insert(tk.END, ( str(result) + '\n\ncopyable version:\n\n' + str(result.tolist()) ) )
            else:
                self.result_text.insert(tk.END, str(result) )
    
        except Exception as e:
            messagebox.showerror("Error", str(e))


def create_gui():
    root = tk.Tk()
    app = MatrixGUI(root)
    root.mainloop()
