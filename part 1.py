import numpy as np

def eigenvectors_eigenvalues(original_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(original_matrix)
    for i in range(len(eigenvalues)):
        if np.dot(original_matrix, eigenvectors[:, i]).all() == (eigenvalues[i] * eigenvectors[:, i]).all():
            print("For eigenvalue: ", eigenvalues[i], " and eigenvector: ", eigenvectors[:, i],
                  " the equation A * v = λ * v is correct.")
        else:
            print("For eigenvalue: ", eigenvalues[i], " and eigenvector: ", eigenvectors[:, i],
                  " the equation A * v = λ * v is incorrect.")
    return eigenvalues, eigenvectors


matrix = np.array([[1, 0, 5],
                   [3, 2, 10],
                   [7, 1, 4]])

values, vectors = eigenvectors_eigenvalues(matrix)
for i in range(len(values)):
    print("Value: ", values[i], ";  Vector: ", vectors[:, i])
