import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA


def vector_image(path):
    img = imread(path)
    print("Image shape: ", img.shape)
    sum_img = img.sum(axis=2)
    print("Black and white image shape: ", sum_img.shape)
    bw_img = sum_img / sum_img.max()
    print("Black and white image shape after normalization: ", bw_img.shape)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title("Black and White Image")
    plt.imshow(bw_img, cmap="gray")

    plt.show()

    return bw_img


def pca_image(img, variance=0.95):
    height, width = img.shape
    image_2d = img.reshape(height, width)

    pca = PCA()
    pca.fit(image_2d)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    components = np.argmax(cumulative_variance >= variance) + 1

    print("Number of components to cover 95% variance: ", components)

    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=variance, color='r', linestyle='-')
    plt.axvline(x=components, color='r', linestyle='-')
    plt.title('Cumulative variance explained by the components')
    plt.xlabel('Components')
    plt.ylabel('Cumulative variance explained')
    plt.grid()
    plt.show()

    return components


def reconstruct_image(img, n_components):
    height, width = img.shape
    image_2d = img.reshape(height, width)

    pca = PCA(n_components=n_components)
    img_pca = pca.fit_transform(image_2d)

    print("Original image shape: ", image_2d.shape)
    print("Image shape after PCA: ", img_pca.shape)

    img_restored = pca.inverse_transform(img_pca)
    img_restored = img_restored.reshape(img.shape)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title(f"Restored Image with {n_components} Components")
    plt.imshow(img_restored, cmap="gray")

    plt.show()

    return img_pca


path = "image_lab2.png"
bw = vector_image(path)

components = pca_image(bw, variance=0.95)

reconstruct_image(bw, components)

for component in [1, 10, 30, 50, 70, 90, components]:
    print(f"Reconstructing with {component} components")
    reconstruct_image(bw, component)
