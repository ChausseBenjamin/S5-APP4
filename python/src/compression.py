import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import is_orthogonal
from brs_debug import Debug
import matplotlib.pyplot as plt
import numpy as np
import img

def get_covariance_matrix(image: np.ndarray):
    """
    Calculates the covariance matrix of a given image.
    Returns 3 values, the matrix itself, its mean matrix
    and finally the centered matrix.
    """
    Debug.Start("get_covariance_matrix")
    
    # Centrage
    Debug.Log("Getting mean of image")
    mean = np.mean(image, axis=0)
    Debug.Log("Getting centered image from its mean")
    centered = image - mean
    
    # Matrice de covariance
    Debug.Log("Obtaining covariance matrix")
    covariance = np.cov(centered, rowvar=False)
    Debug.End()
    return covariance, mean, centered


def get_eigen(covariance: np.ndarray):
    """
    This function takes a covariance array and computes
    its clean vectors. Afterwards, it sorts them from
    the biggest to smallest value, so an importance can
    easily be attributed to each of them.
    """
    Debug.Start("get_eigen")
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    Debug.Log(f"{len(eigenvalues)} eigenvalues")
    Debug.Log(f"{len(eigenvectors)} eigenvectors")
    
    # Tri décroissant
    Debug.Log("Sorting from most important to least important")
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    Debug.End()
    return eigenvalues, eigenvectors

def get_change_of_basis_matrix(eigenvectors: np.ndarray):
    """
    Obtain a change of basis matrix from clean vectors,
    which were in terms computed from a covariance matrix.
    
    The change of basis matrix is simply built by placing
    each clean vector as a column of the matrix
    """
    Debug.Start("get_change_of_basis_matrix")
    Debug.Log(f"it is {is_orthogonal(eigenvectors)} that the vector matrix is orthogonal")
    change_of_basis = eigenvectors.T
    Debug.End()
    return change_of_basis

def compute_change_of_basis(input: np.ndarray, change_of_basis_matrix: np.ndarray):
    """
    Apply a change of basis matrix, to an input. That's it.
    This is used to project the centered matrix with the 
    calculated change of basis matrix.
    """
    Debug.Start("compute_change_of_basis")
    Debug.Log(f"it is {is_orthogonal(change_of_basis_matrix)} that the change_of_basis_matrix matrix is orthogonal")
    projected = input @ change_of_basis_matrix.T
    Debug.End()
    return projected

def compress_projected(projected: np.ndarray, keep_ratio: float):
    """
    This sets a certain amount of columns in a projected image to 0.
    The columns that are set to 0 are the least important ones.
    You must give an image that has been computed through a change
    of basis.
    """
    Debug.Start("compress_projected")
    compressed = projected.copy()
    n_components = projected.shape[1]
    
    k = int(n_components * keep_ratio)
    Debug.Log(f"Keeping {keep_ratio*100}% of columns")
    Debug.Log(f"Going from {n_components} to {k} non 0 columns")
    # On met à zéro les composantes faibles
    compressed[:, k:] = 0
    Debug.End()
    return compressed

def reconstruct_image(compressed, change_of_basis_matrix, mean):
    """
    Rebuilds a coherent image through the inverse of the P matrix
    (change of basis) that was used to compress it in the first
    place.
    """
    Debug.Start("reconstruct_image")
    reconstructed = compressed @ change_of_basis_matrix
    reconstructed += mean
    Debug.Log(f"The compression is {compressed.size}")
    Debug.Log(f"The reconstruction is {reconstructed.size}")
    Debug.End()
    return reconstructed

def compress_image(image: np.ndarray, keep_ratio=0.5):
    """
    Compresses an image with the Eigen method... Or PCA
    or Karhumen transform... We got told it's all the same thing.
    The steps are:
    - Covariance matrix calculated from the image
    - Clean values and vectors are obtained from the covarience matrix.
    - Change of basis matrix calculated from clean vectors
    - Centered matrix is projected with the change of basis matrix.
    - A ratio of low importance columns are removed
    - The image is reconstructed with the inverse of its change of basis matrix.
    """
    Debug.Start("compress_image")
    cov, mean, centered = get_covariance_matrix(image)
    eigenvalues, eigenvectors = get_eigen(cov)
    
    P = get_change_of_basis_matrix(eigenvectors)
    projected = compute_change_of_basis(centered, P)
    
    compressed = compress_projected(projected, keep_ratio)
    reconstructed = reconstruct_image(compressed, P, mean)

    analyze_worth_of_compression(image, projected, eigenvectors, mean, keep_ratio, reconstructed)
    Debug.End()
    return reconstructed

def mse(img1: np.ndarray, img2: np.ndarray):
    """
    Uses the mean square root average to check how different 2 images are.
    Essentially telling if it's been compressed or not.
    """
    Debug.Start("mse")
    if img1.shape != img2.shape:
        Debug.Error(f"Images are of different sizes! {img1.shape} != {img2.shape}")
        raise ValueError("Images must have the same shape")
    result = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    Debug.Log(f"RMS difference is: {result}")
    Debug.End()
    return result

def compare_images(img1: np.ndarray, img2: np.ndarray, max_pixel_value=255.0) -> float:
    Debug.Start("compare_images")
    mse_value = mse(img1, img2)
    if mse_value == 0:
        Debug.Warn("mse is 0, difference is infinite")
        Debug.End()
        return float("inf")
    result = 10 * np.log10(max_pixel_value / np.sqrt(mse_value))
    Debug.Log(f"There's {result} dB difference between the 2 images. ({max_pixel_value / np.sqrt(mse_value)})")
    Debug.End()
    return result

def analyze_worth_of_compression(original: np.ndarray,
                                 projected: np.ndarray,
                                 eigenvectors: np.ndarray,
                                 mean: np.ndarray,
                                 keep_ratio,
                                 compressed):
    """
    Checks whether PCA compression reduces the amount of data
    required to reconstruct the image.
    """

    Debug.Start("analyze_worth_of_compression")

    n_components = projected.shape[1]
    k = int(n_components * keep_ratio)

    # Original storage
    original_storage = original.size

    Debug.Log("Removing columns outside of kept ones that stayed for simplicity")
    projected = projected[:, :k]
    eigenvectors = eigenvectors[:, :k]

    # Determine how many components are actually kept
    # Count non-zero columns in projected (since you zeroed weak components)
    non_zero_cols = np.any(projected != 0, axis=0)
    k = np.sum(non_zero_cols)

    n_samples = projected.shape[0]
    n_features = eigenvectors.shape[0]

    # Storage needed for reconstruction
    projected_storage = n_samples * k
    eigenvectors_storage = n_features * k
    mean_storage = mean.size

    total_compressed_storage = (
        projected_storage +
        eigenvectors_storage +
        mean_storage
    )

    Debug.Log(f"Kept components (k): {k}")
    Debug.Log(f"Original:            {original_storage} values")
    Debug.Log(f"Projected:           {projected_storage} values")
    Debug.Log(f"Eigenvectors:        {eigenvectors_storage} values")
    Debug.Log(f"Mean:                {mean_storage} values")
    Debug.Log(f"Total compressed:    {total_compressed_storage} values")

    compression_ratio = total_compressed_storage / original_storage
    Debug.Log(f"Compression ratio: {compression_ratio:.3f}")

    db = compare_images(original, compressed)
    if db < 45:
        Debug.Error("Image is noticably compressed. <45")
    else:
        Debug.Log("Compression dB is judged adequat")

    if total_compressed_storage < original_storage:
        Debug.Log("Compression is worth it.")
    else:
        Debug.Warn("Compression is NOT worth it.")

    Debug.End()

    return compression_ratio

def main():
    Debug.Start("compression.py")
    image = img.original()
    compressed = compress_image(image=image, keep_ratio=0.5)
    img.save("compressed.png", compressed)
    Debug.End()

if __name__ == "__main__":
    main()
