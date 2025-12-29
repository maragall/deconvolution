"""
Linear operators for deconvolution.
"""

import torch
import torch.nn.functional as F
from .utils import get_device, clip


class LinearOperator:
    """Base class for linear operators."""

    def forward(self, x):
        """Apply the forward operator."""
        raise NotImplementedError

    def transpose(self, x):
        """Apply the transpose (adjoint) operator."""
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class Crop3D(LinearOperator):
    """3D cropping operator to handle PSF boundary effects."""

    def __init__(self, crop_pix):
        """
        Args:
            crop_pix: (nz, ny, nx) pixels to crop from each side
        """
        if len(crop_pix) == 2:
            crop_pix = (0, crop_pix[0], crop_pix[1])
        self.crop_pix = crop_pix

    def forward(self, x):
        nz, ny, nx = self.crop_pix
        if x.ndim == 2:
            return x[ny:x.shape[-2]-ny if ny > 0 else x.shape[-2],
                     nx:x.shape[-1]-nx if nx > 0 else x.shape[-1]]
        elif x.ndim == 3:
            z_slice = slice(nz, x.shape[0]-nz if nz > 0 else x.shape[0])
            y_slice = slice(ny, x.shape[1]-ny if ny > 0 else x.shape[1])
            x_slice = slice(nx, x.shape[2]-nx if nx > 0 else x.shape[2])
            return x[z_slice, y_slice, x_slice]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

    def transpose(self, x):
        nz, ny, nx = self.crop_pix
        if x.ndim == 2:
            return F.pad(x, (nx, nx, ny, ny))
        elif x.ndim == 3:
            return F.pad(x, (nx, nx, ny, ny, nz, nz))
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")


class Bucket3D(LinearOperator):
    """3D binning/bucketing operator."""

    def __init__(self, shape):
        """
        Args:
            shape: (bz, by, bx) binning factors
        """
        if len(shape) == 2:
            shape = (1, shape[0], shape[1])
        self.shape = shape

    def forward(self, x):
        bz, by, bx = self.shape
        if x.ndim == 2:
            return x.reshape(x.shape[0]//by, by, x.shape[1]//bx, bx).sum(dim=(1, 3))
        elif x.ndim == 3:
            return x.reshape(
                x.shape[0]//bz, bz,
                x.shape[1]//by, by,
                x.shape[2]//bx, bx
            ).sum(dim=(1, 3, 5))
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

    def transpose(self, x):
        device = get_device()
        if x.ndim == 2:
            by, bx = self.shape[1], self.shape[2]
            return torch.kron(x, torch.ones((by, bx), device=device))
        elif x.ndim == 3:
            bz, by, bx = self.shape
            ones = torch.ones((bz, by, bx), device=device)
            # 3D Kronecker product
            result = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * ones
            return result.reshape(
                x.shape[0] * bz,
                x.shape[1] * by,
                x.shape[2] * bx
            )
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")


class PSFConvolution(LinearOperator):
    """Convolution with a Point Spread Function using FFT."""

    def __init__(self, psf, image_shape=None):
        """
        Args:
            psf: Point Spread Function tensor
            image_shape: Shape of images to convolve (for pre-computing OTF)
        """
        self.psf = psf
        self._otf_cache = {}
        self._image_shape = image_shape

        if image_shape is not None:
            self._precompute_otf(image_shape)

    def _precompute_otf(self, shape):
        """Pre-compute the OTF for a given image shape."""
        if shape in self._otf_cache:
            return self._otf_cache[shape]

        device = get_device()
        psf = self.psf

        # Pad PSF to image size
        pad_shape = []
        for psf_size, img_size in zip(psf.shape, shape):
            pad_before = (img_size - psf_size) // 2
            pad_after = img_size - psf_size - pad_before
            pad_shape.extend([pad_before, pad_after])

        # Reverse for F.pad (it expects [..., x, y, z] order)
        pad_shape = pad_shape[::-1]
        psf_padded = F.pad(psf, pad_shape)

        # Shift PSF so center is at origin
        for dim in range(psf_padded.ndim):
            psf_padded = torch.roll(psf_padded, -psf_padded.shape[dim]//2, dims=dim)

        # Compute OTF
        otf = torch.fft.rfftn(psf_padded)
        self._otf_cache[shape] = otf

        return otf

    def forward(self, x):
        """Apply PSF convolution."""
        shape = tuple(x.shape)
        otf = self._precompute_otf(shape)

        x_fft = torch.fft.rfftn(x)
        result = torch.fft.irfftn(otf * x_fft, s=shape)

        return result.real if torch.is_complex(result) else result

    def transpose(self, x):
        """Apply transpose of PSF convolution (correlation)."""
        shape = tuple(x.shape)
        otf = self._precompute_otf(shape)

        x_fft = torch.fft.rfftn(x)
        # Conjugate for correlation
        result = torch.fft.irfftn(torch.conj(otf) * x_fft, s=shape)

        return result.real if torch.is_complex(result) else result


class CompositeOperator(LinearOperator):
    """Compose multiple linear operators."""

    def __init__(self, operators):
        """
        Args:
            operators: List of operators to compose (applied in order)
        """
        self.operators = operators

    def forward(self, x):
        for op in self.operators:
            x = op.forward(x)
        return x

    def transpose(self, x):
        for op in reversed(self.operators):
            x = op.transpose(x)
        return x


class Identity(LinearOperator):
    """Identity operator (pass-through)."""

    def forward(self, x):
        return x

    def transpose(self, x):
        return x
