# Linear Algebra and Image Processing with Python

This project includes three major tasks, focusing on eigenvalues and eigenvectors, image compression using Principal Component Analysis (PCA), and cryptography using matrix diagonalization.

## Part 1: Eigenvalues and Eigenvectors Calculation

### Description

The task involves writing a function that accepts a square matrix and returns its eigenvalues and eigenvectors using the NumPy library. It also verifies the equality A * v = λ * v for each eigenvalue and its corresponding eigenvector.

```python
import numpy as np

def eigenvectors_eigenvalues(original_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(original_matrix)
    for i in range(len(eigenvalues)):
        if np.allclose(np.dot(original_matrix, eigenvectors[:, i]), eigenvalues[i] * eigenvectors[:, i]):
            print(f"For eigenvalue: {eigenvalues[i]}, the equation A * v = λ * v is correct.")
        else:
            print(f"For eigenvalue: {eigenvalues[i]}, the equation A * v = λ * v is incorrect.")
    return eigenvalues, eigenvectors

matrix = np.array([[1, 0, 5], [3, 2, 10], [7, 1, 4]])
values, vectors = eigenvectors_eigenvalues(matrix)
for i in range(len(values)):
    print(f"Value: {values[i]}, Vector: {vectors[:, i]}")
```

## Part 2: Image Compression using PCA (4 pts)

### Description

This task involves implementing PCA to reduce the dimensionality of images, thus achieving compression. The steps include loading an image, converting it to grayscale, applying PCA, and reconstructing the image using a limited number of components.

## Part 3: Diagonalization and Cryptography (1.5 pts)

#### Description
This task demonstrates the use of diagonalization for decrypting messages. The function decrypt_message uses the inverse operation of diagonalization with a key matrix to decode an encrypted vector.
Example Code
```python
Копіювати код
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
```

## Conclusion

This project showcases the practical applications of linear algebra concepts such as eigenvalues, eigenvectors, and diagonalization in diverse fields like image processing and cryptography. By implementing these tasks, we explored:

1. **Eigenvalues and Eigenvectors**: The foundation of many advanced mathematical techniques, crucial for understanding linear transformations and stability in systems.

2. **Principal Component Analysis (PCA)**: A powerful tool in data science and machine learning for dimensionality reduction, data compression, and noise reduction, demonstrated here through image processing.

3. **Cryptography**: The use of matrix diagonalization in cryptographic algorithms, illustrating the intersection of abstract mathematics and practical security applications.

These exercises not only reinforce theoretical knowledge but also provide hands-on experience in applying mathematical concepts to real-world problems. Through this project, we hope to inspire further exploration and understanding of how mathematics can be utilized to solve complex challenges in technology and science.
