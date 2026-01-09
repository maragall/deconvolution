"""
Evaluation metrics for comparing deconvolved images to ground truth.
"""

import numpy as np
from typing import Dict


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min > 0:
        return (image - img_min) / (img_max - img_min)
    return np.zeros_like(image)


def compute_mse(image: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((image - ground_truth) ** 2))


def compute_psnr(image: np.ndarray, ground_truth: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio in dB."""
    mse = compute_mse(image, ground_truth)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(max_val**2 / mse))


def compute_ssim(
    image: np.ndarray,
    ground_truth: np.ndarray,
    win_size: int = 7,
    data_range: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Simplified 3D SSIM implementation without skimage dependency.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Compute means
    mu1 = np.mean(image)
    mu2 = np.mean(ground_truth)

    # Compute variances and covariance
    sigma1_sq = np.var(image)
    sigma2_sq = np.var(ground_truth)
    sigma12 = np.mean((image - mu1) * (ground_truth - mu2))

    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


def compute_pearson(image: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    img_flat = image.flatten()
    gt_flat = ground_truth.flatten()

    # Handle constant arrays
    if np.std(img_flat) == 0 or np.std(gt_flat) == 0:
        return 0.0

    corr = np.corrcoef(img_flat, gt_flat)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_nrmse(image: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Normalized Root Mean Squared Error."""
    rmse = np.sqrt(compute_mse(image, ground_truth))
    gt_range = ground_truth.max() - ground_truth.min()
    if gt_range == 0:
        return float("inf")
    return float(rmse / gt_range)


def unify_shape(image: np.ndarray, ground_truth: np.ndarray) -> tuple:
    """
    Pad or crop images to the same shape.

    Centers the smaller image within the larger one's dimensions.
    """
    shape1 = np.array(image.shape)
    shape2 = np.array(ground_truth.shape)

    # Target shape is the maximum along each dimension
    target_shape = np.maximum(shape1, shape2)

    def pad_to_shape(img, target):
        pad_width = []
        for s, t in zip(img.shape, target):
            diff = t - s
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
        return np.pad(img, pad_width, mode='constant', constant_values=0)

    img1_padded = pad_to_shape(image, target_shape)
    img2_padded = pad_to_shape(ground_truth, target_shape)

    return img1_padded, img2_padded


def rescale_to_match(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Rescale image intensities to optimally match reference using least squares.

    Finds optimal a, b such that: a * image + b ≈ reference
    This is standard practice in deconvolution evaluation.
    """
    img_flat = image.flatten()
    ref_flat = reference.flatten()

    # Solve least squares: [img, 1] @ [a, b]^T = ref
    A = np.column_stack([img_flat, np.ones_like(img_flat)])
    result = np.linalg.lstsq(A, ref_flat, rcond=None)
    a, b = result[0]

    return a * image + b


def evaluate(deconvolved: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics comparing deconvolved image to ground truth.

    Uses optimal intensity rescaling to ensure fair comparison
    (deconvolution may change absolute intensity scale).

    Args:
        deconvolved: Deconvolved image array
        ground_truth: Ground truth (phantom) array

    Returns:
        Dictionary with metrics: mse, psnr, ssim, pearson, nrmse
    """
    # Unify shapes if needed
    if deconvolved.shape != ground_truth.shape:
        deconvolved, ground_truth = unify_shape(deconvolved, ground_truth)

    # Rescale deconvolved to optimally match ground truth intensity
    dec_rescaled = rescale_to_match(deconvolved, ground_truth)

    # Normalize both to [0, 1] for consistent metrics
    gt_norm = normalize(ground_truth)
    dec_norm = normalize(dec_rescaled)

    return {
        "mse": compute_mse(dec_norm, gt_norm),
        "psnr": compute_psnr(dec_norm, gt_norm),
        "ssim": compute_ssim(dec_norm, gt_norm),
        "pearson": compute_pearson(dec_norm, gt_norm),
        "nrmse": compute_nrmse(dec_norm, gt_norm),
    }
