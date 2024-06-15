import numpy as np


def decrypt_message(encrypted_vector, key_matrix):
    vector = np.array([ord(char) for char in encrypted_vector])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonal_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    decrypted_vector = np.dot(diagonal_matrix, vector)
    return decrypted_vector


message = input("Message: ")
matrix = np.random.randint(0, 256, (len(message), len(message)))
print(decrypt_message(message, matrix))