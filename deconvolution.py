"""
GPU-accelerated deconvolution with Richardson-Lucy and Gradient Consensus.
"""

import numpy as np
import torch
import tifffile as tf

device = torch.device('cuda')


def clip(x, min_val=1e-12):
    return torch.clamp(x, min=min_val)


def coinflip(n, p=0.5):
    """GPU binomial sampling via normal approximation."""
    mean = n * p
    std = torch.sqrt(n * p * (1 - p))
    return torch.clamp(mean + std * torch.randn_like(n), min=0)


class PSFConvolution:
    """FFT-based convolution with a PSF."""

    def __init__(self, psf):
        self.psf = psf
        self._otf_cache = {}

    def _get_otf(self, shape):
        if shape not in self._otf_cache:
            psf_padded = torch.zeros(shape, device=device, dtype=self.psf.dtype)
            slices = tuple(slice(0, s) for s in self.psf.shape)
            psf_padded[slices] = self.psf

            # Center PSF at origin
            for dim in range(len(shape)):
                psf_padded = torch.roll(psf_padded, -self.psf.shape[dim] // 2, dims=dim)

            self._otf_cache[shape] = torch.fft.rfftn(psf_padded)
        return self._otf_cache[shape]

    def forward(self, x):
        otf = self._get_otf(x.shape)
        return torch.fft.irfftn(otf * torch.fft.rfftn(x), s=x.shape)

    def transpose(self, x):
        otf = self._get_otf(x.shape)
        return torch.fft.irfftn(torch.conj(otf) * torch.fft.rfftn(x), s=x.shape)


def deconvolve(measured, psf, iterations=50, method='richardson_lucy', background=0.0, callback=None):
    """
    Deconvolve measured data using Richardson-Lucy or Gradient Consensus.

    Args:
        measured: Measured image (numpy array or torch tensor)
        psf: Point spread function (numpy array or torch tensor)
        iterations: Number of iterations
        method: 'richardson_lucy' or 'gradient_consensus'
        background: Background offset
        callback: Optional function(iteration, estimate)

    Returns:
        Deconvolved estimate (torch tensor on GPU)
    """
    assert method in ('richardson_lucy', 'gradient_consensus')

    # Move to GPU
    if not isinstance(measured, torch.Tensor):
        measured = torch.from_numpy(np.asarray(measured, dtype=np.float32)).to(device)
    if not isinstance(psf, torch.Tensor):
        psf = torch.from_numpy(np.asarray(psf, dtype=np.float32)).to(device)

    # Normalize PSF
    psf = psf / psf.sum()

    H = PSFConvolution(psf)
    H_T_ones = clip(H.transpose(torch.ones_like(measured)))
    estimate = torch.ones_like(measured)

    for i in range(iterations):
        estimate = clip(_gradient_step(estimate, measured, H, H_T_ones, method, background))
        if callback:
            callback(i, estimate)

    return estimate


def _gradient_step(estimate, measured, H, H_T_ones, method, background):
    """Single gradient step."""
    est = estimate.clone().requires_grad_(True)
    predicted = clip(H.forward(est) + background)

    loss = torch.sum(measured * torch.log(predicted) - predicted)
    gradient = torch.autograd.grad(loss, est)[0]
    step_size = estimate / H_T_ones

    if method == 'gradient_consensus':
        heads = coinflip(measured, 0.5)

        est_h = estimate.clone().requires_grad_(True)
        pred_h = clip(H.forward(est_h) + background)
        loss_h = torch.sum(heads * torch.log(pred_h) - 0.5 * pred_h)
        grad_h = torch.autograd.grad(loss_h, est_h)[0]
        grad_t = gradient - grad_h

        # Crosstalk check
        h_prod = H.forward(grad_h * grad_t)
        dummy = torch.zeros_like(estimate).requires_grad_(True)
        dummy_pred = H.forward(dummy)
        crosstalk = torch.autograd.grad(torch.sum(h_prod * dummy_pred), dummy)[0]

        step_size = torch.where(crosstalk <= 0, torch.zeros_like(step_size), step_size)

    return estimate + gradient * step_size


def load_tiff(path):
    """Load TIFF as float32 tensor on GPU."""
    data = tf.imread(str(path)).astype(np.float32)
    return torch.from_numpy(data).to(device)


def save_tiff(path, data):
    """Save tensor as TIFF."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    tf.imwrite(str(path), data.astype(np.float32))


def gaussian_psf(shape, sigma):
    """Generate Gaussian PSF on GPU."""
    coords = [torch.arange(s, device=device) - s // 2 for s in shape]
    grids = torch.meshgrid(*coords, indexing='ij')

    if isinstance(sigma, (int, float)):
        sigma = [sigma] * len(shape)

    r2 = sum((g / s) ** 2 for g, s in zip(grids, sigma))
    psf = torch.exp(-0.5 * r2)
    return psf / psf.sum()
