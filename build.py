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

Data files are stored in the user's local app data folder.
All data (themes & saved matrices) are stored in JSON format in the file linalg_data.json.
Personalized data (default themes) are stored in the file personal.txt.
"""

import gui
import os
import json


def main():
    """
    The main function of the Linear Algebra calculator build.
    This function imports theme data, saved matrix data, & initializes the GUI for the Linear Algebra calculator.
    See gui.py for the implementation of the GUI.
    See Matrix.py for the background calculations involving the custom matrices.
    """

    # Get the file path of the data file
    folder_path = os.getenv('LOCALAPPDATA') + '\Linear Algebra Calculator'
    file_path = folder_path + '\\linalg_data.json'

    # check if the directory exists
    if not os.path.exists(folder_path):

        os.makedirs(folder_path)

    # check if the file exists
    if not os.path.exists(file_path):

        with open(file_path, 'w') as file:

            # write default data to the file
            file.write('{\
                       \n\t"themes": [\
                       \n\t\t{\
                       \n\t\t\t"name": "light",\
                       \n\t\t\t"background": "white",\
                       \n\t\t\t"text": "black",\
                       \n\t\t\t"button": "#D3D3D3",\
                       \n\t\t\t"button_pressed": "#808080",\
                       \n\t\t\t"entrybox_inactive": "#F6F6F6",\
                       \n\t\t\t"entrybox_active": "#F6F6F6",\
                       \n\t\t\t"border": "#A9A9A9"\
                       \n\t\t},\
                       \n\t\t{\
                       \n\t\t\t"name": "dark",\
                       \n\t\t\t"background": "black",\
                       \n\t\t\t"text": "white",\
                       \n\t\t\t"button": "#A9A9A9",\
                       \n\t\t\t"button_pressed": "#D3D3D3",\
                       \n\t\t\t"entrybox_inactive": "#5A5A5A",\
                       \n\t\t\t"entrybox_active": "#5A5A5A",\
                       \n\t\t\t"border": "#D3D3D3"\
                       \n\t\t}\
                       \n\t],\
                       \n\t"matrices": [\
                       \n\t\t{\
                       \n\t\t\t"name": "ID2",\
                       \n\t\t\t"matrix": [[1, 0], [0, 1]]\
                       \n\t\t}\
                       \n\t]\
                       \n}')
    
    # Import the theme data
    theme_data, matrix_data = {}, {}
    with open(file_path, 'r') as file_data:
        raw_data = json.load(file_data)
        print(f'raw data: {raw_data}')

        theme_data_raw = raw_data['themes']
        for theme in theme_data_raw:
            theme_name = theme['name']
            theme.pop('name')
            theme_data[theme_name] = theme
        
        matrix_data_raw = raw_data['matrices']
        for matrix in matrix_data_raw:
            matrix_name = matrix['name']
            matrix.pop('name')
            matrix_data[matrix_name] = matrix

    print(f'theme data: {theme_data}')
    print(f'matrix data: {matrix_data}')

    # Initialize the GUI
    gui.create_gui(theme_data, matrix_data)

if __name__ == "__main__":
    main()