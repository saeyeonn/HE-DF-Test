import numpy as np

def dot_product(vector1, vector2):
    result = 0
    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]
    return result

def transpose(matrix):
    rows, cols = len(matrix), len(matrix[0])
    transposed = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed

def matrix_multiplication(matrix1, matrix2):
    result = []
    transposed = transpose(matrix2)
    for row in matrix1:
        new_row = []
        for col in transposed:
            new_row.append(dot_product(row, col))
        result.append(new_row)
    return result

def linear_regression(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    
    # calculate parameter
    X_transpose = transpose(X)
    X_transpose_X = matrix_multiplication(X_transpose, X)
    X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
    X_transpose_y = matrix_multiplication(X_transpose, [[val] for val in y])
    theta = matrix_multiplication(X_transpose_X_inverse, X_transpose_y)
    
    return [val[0] for val in theta]

# test
X = [[1], [2], [3], [4], [5]]
y = [2, 3, 4, 5, 6]
theta = linear_regression(X, y)
print("Linear Regression Parameter:", theta)
