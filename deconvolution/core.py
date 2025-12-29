"""
Core deconvolution algorithms.
"""

import torch
from .utils import clip, coinflip, get_device, to_tensor
from .operators import CompositeOperator, Identity


class DeconvolutionEngine:
    """
    GPU-accelerated deconvolution using Richardson-Lucy or Gradient Consensus.
    """

    def __init__(self, forward_operator, background=0.0):
        """
        Args:
            forward_operator: Linear operator H that models the imaging system
            background: Background signal to add to the model
        """
        if isinstance(forward_operator, list):
            self.H = CompositeOperator(forward_operator)
        else:
            self.H = forward_operator

        self.background = background
        self._H_T_ones = None

    def _compute_H_T_ones(self, shape):
        """Compute H^T(1) for step size scaling."""
        ones = torch.ones(shape, device=get_device(), dtype=torch.float32)
        return clip(self.H.transpose(ones))

    def _scaled_gradient_step(self, estimate, measured, H_T_ones, method='richardson_lucy'):
        """
        Compute one scaled gradient step.

        Args:
            estimate: Current fluorophore estimate
            measured: Measured photon counts
            H_T_ones: Precomputed H^T(1) for step size
            method: 'richardson_lucy' or 'gradient_consensus'

        Returns:
            Updated estimate
        """
        # Forward pass with gradient tracking
        estimate_grad = estimate.clone().requires_grad_(True)
        predicted = clip(self.H.forward(estimate_grad) + self.background)

        # Compute gradient of log-likelihood
        loss = torch.sum(measured * torch.log(predicted) - predicted)
        gradient = torch.autograd.grad(loss, estimate_grad, create_graph=False)[0]

        # Step size from RL derivation
        step_size = estimate / H_T_ones

        if method == 'gradient_consensus':
            # Split measurements randomly
            heads_photons = coinflip(measured, 0.5)

            # Compute gradient for heads subset
            estimate_heads = estimate.clone().requires_grad_(True)
            predicted_heads = clip(self.H.forward(estimate_heads) + self.background)
            heads_loss = torch.sum(
                heads_photons * torch.log(predicted_heads) - 0.5 * predicted_heads
            )
            heads_gradient = torch.autograd.grad(heads_loss, estimate_heads, create_graph=False)[0]

            tails_gradient = gradient - heads_gradient

            # Compute local crosstalk: H^T(H(heads_gradient * tails_gradient))
            product = heads_gradient * tails_gradient
            h_product = self.H.forward(product)

            # Use autograd to compute H^T
            dummy = torch.zeros_like(estimate).requires_grad_(True)
            dummy_pred = self.H.forward(dummy)
            crosstalk_loss = torch.sum(h_product * dummy_pred)
            local_dot_product = torch.autograd.grad(crosstalk_loss, dummy, create_graph=False)[0]

            # Zero step size where gradients disagree
            step_size = torch.where(
                local_dot_product <= 0,
                torch.tensor(0.0, device=step_size.device, dtype=step_size.dtype),
                step_size
            )

        updated = estimate + gradient * step_size
        return updated

    def deconvolve(self, measured, iterations=100, method='richardson_lucy',
                   initial_estimate=None, callback=None):
        """
        Perform deconvolution.

        Args:
            measured: Measured image/volume (numpy array or torch tensor)
            iterations: Number of iterations
            method: 'richardson_lucy' or 'gradient_consensus'
            initial_estimate: Initial estimate (defaults to uniform)
            callback: Optional function called each iteration with (iteration, estimate)

        Returns:
            Deconvolved estimate as torch tensor
        """
        assert method in ('richardson_lucy', 'gradient_consensus'), \
            f"Unknown method: {method}"

        # Convert to tensor
        measured = to_tensor(measured)
        measured_shape = tuple(measured.shape)

        # Compute H^T(1) for this measurement shape
        # We need the shape of the estimate, which depends on the operator
        if initial_estimate is not None:
            initial_estimate = to_tensor(initial_estimate)
            estimate_shape = tuple(initial_estimate.shape)
        else:
            # For most cases, estimate has same shape as measured
            # But if there's cropping/binning, we need to infer the shape
            estimate_shape = self._infer_estimate_shape(measured_shape)

        H_T_ones = self._compute_H_T_ones(estimate_shape)

        # Initialize estimate
        if initial_estimate is not None:
            estimate = initial_estimate.clone()
        else:
            estimate = torch.ones(estimate_shape, device=get_device(), dtype=torch.float32)

        # Iterate
        for i in range(iterations):
            estimate = clip(self._scaled_gradient_step(
                estimate, measured, H_T_ones, method=method
            ))

            if callback is not None:
                callback(i, estimate)

        return estimate

    def _infer_estimate_shape(self, measured_shape):
        """
        Infer the shape of the estimate from the measured shape.
        For now, assumes same shape (no binning/cropping changes size).
        """
        # This is a simplification - in practice, you'd need to track
        # how each operator changes the shape
        return measured_shape

    def compute_residual(self, estimate, measured):
        """Compute the residual between predicted and measured."""
        predicted = self.H.forward(to_tensor(estimate)) + self.background
        return to_tensor(measured) - predicted


class DeconvolutionResult:
    """Container for deconvolution results with history tracking."""

    def __init__(self):
        self.estimate = None
        self.history = []
        self.method = None
        self.iterations = 0

    def add_iteration(self, iteration, estimate):
        """Store estimate from an iteration."""
        self.history.append(estimate.clone())
        self.estimate = estimate
        self.iterations = iteration + 1


class ChunkedDeconvolver:
    """
    Processes large datasets in chunks to avoid memory issues.

    For 3D data, processes in Z-chunks with overlap to handle PSF extent.
    Results are written to disk incrementally.
    """

    def __init__(self, psf, chunk_size=20, background=0.0):
        """
        Args:
            psf: Point spread function tensor
            chunk_size: Number of Z-slices per chunk
            background: Background signal level
        """
        from .operators import PSFConvolution

        self.psf = psf
        self.chunk_size = chunk_size
        self.background = background

        # Overlap needed based on PSF Z extent
        self.overlap = psf.shape[0] // 2 + 1 if psf.ndim == 3 else 0

    def process_stack(self, input_stack, output_path, iterations=50,
                      method='richardson_lucy', progress_callback=None):
        """
        Process a lazy-loaded stack and write results incrementally.

        Args:
            input_stack: LazyTiffStack or similar lazy-loading object
            output_path: Path to write output TIFF
            iterations: Deconvolution iterations per chunk
            method: 'richardson_lucy' or 'gradient_consensus'
            progress_callback: Optional callback(chunk_idx, total_chunks, z_start, z_end)

        Returns:
            Path to output file
        """
        from .io import TiffStackWriter
        from .operators import PSFConvolution

        n_slices = len(input_stack)
        total_chunks = (n_slices + self.chunk_size - 1) // self.chunk_size

        with TiffStackWriter(output_path) as writer:
            z = 0
            chunk_idx = 0

            while z < n_slices:
                # Determine chunk boundaries with overlap
                z_start = max(0, z - self.overlap)
                z_end = min(n_slices, z + self.chunk_size + self.overlap)

                # Load chunk
                chunk = input_stack.load_chunk(z_start, z_end)

                if progress_callback:
                    progress_callback(chunk_idx, total_chunks, z_start, z_end)

                # Create operator for this chunk shape
                psf_op = PSFConvolution(self.psf, image_shape=tuple(chunk.shape))
                engine = DeconvolutionEngine(psf_op, background=self.background)

                # Deconvolve
                result = engine.deconvolve(chunk, iterations=iterations, method=method)

                # Extract non-overlapping region
                out_start = z - z_start  # Offset into result
                out_end = out_start + min(self.chunk_size, n_slices - z)

                # Write valid slices
                for i in range(out_start, out_end):
                    writer.write_slice(result[i])

                z += self.chunk_size
                chunk_idx += 1

                # Free memory
                del chunk, result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return output_path

    def process_2d_slices(self, input_stack, output_path, iterations=50,
                          method='richardson_lucy', progress_callback=None):
        """
        Process each Z-slice independently with 2D deconvolution.

        Faster than 3D but ignores axial blur. Good for thin samples.

        Args:
            input_stack: LazyTiffStack or similar
            output_path: Path to write output TIFF
            iterations: Deconvolution iterations per slice
            method: 'richardson_lucy' or 'gradient_consensus'
            progress_callback: Optional callback(slice_idx, total_slices)

        Returns:
            Path to output file
        """
        from .io import TiffStackWriter
        from .operators import PSFConvolution

        # Use center slice of PSF for 2D
        if self.psf.ndim == 3:
            psf_2d = self.psf[self.psf.shape[0] // 2]
        else:
            psf_2d = self.psf

        n_slices = len(input_stack)

        with TiffStackWriter(output_path) as writer:
            for i in range(n_slices):
                if progress_callback:
                    progress_callback(i, n_slices)

                # Load single slice
                slice_data = to_tensor(input_stack[i])

                # Create operator for this slice
                psf_op = PSFConvolution(psf_2d, image_shape=tuple(slice_data.shape))
                engine = DeconvolutionEngine(psf_op, background=self.background)

                # Deconvolve
                result = engine.deconvolve(slice_data, iterations=iterations, method=method)

                # Write result
                writer.write_slice(result)

                # Free memory
                del slice_data, result

        return output_path
