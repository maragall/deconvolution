"""
Utility functions for deconvolution.
"""

import numpy as np
import torch


def get_device():
    """Get the compute device. Raises error if CUDA not available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this deconvolution package. "
            "Please install PyTorch with CUDA support."
        )
    return torch.device('cuda')


# Lazy device initialization
device = None


def _get_device():
    """Internal device getter with lazy initialization."""
    global device
    if device is None:
        device = get_device()
    return device


def get_gpu_memory_info():
    """Get current GPU memory usage in MB."""
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**2,
        'reserved': torch.cuda.memory_reserved() / 1024**2,
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,
    }


def reset_peak_memory():
    """Reset peak memory tracking."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def to_tensor(x, dtype=torch.float32):
    """Convert numpy array to torch tensor on GPU."""
    dev = _get_device()
    if isinstance(x, torch.Tensor):
        return x.to(device=dev, dtype=dtype)
    return torch.from_numpy(np.asarray(x)).to(device=dev, dtype=dtype)


def to_numpy(x):
    """Convert torch tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def clip(x, min_val=1e-12):
    """Clip values to avoid numerical issues."""
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, min=min_val)
    return np.clip(x, a_min=min_val, a_max=None)


def poisson_noise(average_counts):
    """Add Poisson noise to expected counts."""
    rng = np.random.default_rng()
    if isinstance(average_counts, torch.Tensor):
        avg_cpu = average_counts.cpu().numpy()
        result = rng.poisson(avg_cpu).astype(avg_cpu.dtype)
        return torch.from_numpy(result).to(_get_device())
    return rng.poisson(average_counts).astype(average_counts.dtype)


def coinflip(n, probability=0.5):
    """
    GPU-accelerated binomial sampling for gradient consensus.
    Uses normal approximation which is accurate for large counts (n*p > 5).

    For microscopy data with photon counts typically > 1000, this is very accurate.
    """
    # Normal approximation to binomial: N(np, sqrt(np(1-p)))
    mean = n * probability
    std = torch.sqrt(n * probability * (1 - probability))

    # Sample from normal and round to integers
    normal_samples = torch.randn_like(n)
    result = mean + std * normal_samples

    # Clamp to valid range [0, n]
    result = torch.clamp(result, min=0)
    result = torch.minimum(result, n)

    return result


def poisson_log_likelihood(expected_counts, measured_counts):
    """Compute Poisson log-likelihood."""
    if isinstance(expected_counts, torch.Tensor):
        e, m = clip(expected_counts), measured_counts
        return torch.sum(m * torch.log(e) - e - torch.lgamma(m + 1))
    else:
        from scipy.special import gammaln
        e, m = clip(expected_counts), measured_counts
        return np.sum(m * np.log(e) - e - gammaln(m + 1))
