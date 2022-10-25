import math
from typing import Optional
from . import ArrayOrTensor, ArrayOrTensorDispatch, mutually_exclusive_kwargs, OptionalArrayOrTensor


dispatch = ArrayOrTensorDispatch()
sqrt2 = 1.4142135623730951
log2 = 0.6931471805599453
log2pi = 1.8378770664093453


def _get_rfft_scale(cov: OptionalArrayOrTensor, rfft_scale: OptionalArrayOrTensor) -> ArrayOrTensor:
    if (cov is None) == (rfft_scale is None):  # pragma: no cover
        raise ValueError("exactly one of `cov` and `rfft_scale` must be given")
    if rfft_scale is not None:
        return rfft_scale
    return evaluate_rfft_scale(cov=cov)


def log_prob_stdnorm(y: ArrayOrTensor) -> ArrayOrTensor:
    """
    Evaluate the log probability of a standard normal random variable.
    """
    return - (log2pi + y * y) / 2


@mutually_exclusive_kwargs("cov", ("rfft", "size"))
def evaluate_rfft_scale(*, cov: OptionalArrayOrTensor = None, rfft: OptionalArrayOrTensor = None,
                        size: Optional[int] = None) -> ArrayOrTensor:
    """
    Evaluate the scale of Fourier coefficients.

    Args:
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., n)`.

    Returns:
        scale: Scale of Fourier coefficients with shape `(..., n // 2 + 1)`.
    """
    if rfft is None:
        *_, size = cov.shape
        rfft = dispatch[cov].fft.rfft(cov).real
    scale: ArrayOrTensor = dispatch.sqrt(size * rfft / 2)
    # Rescale for the real-only zero frequency term.
    scale[0] *= sqrt2
    if size % 2 == 0:
        # Rescale for the real-only Nyqvist frequency term.
        scale[..., -1] *= sqrt2
    return scale


def expand_rfft(rfft: ArrayOrTensor, n: int) -> ArrayOrTensor:
    """
    Convert truncated real Fourier coefficients to full Fourier coefficients.

    Args:
        rfft: Truncated real Fourier coefficients with shape `(n // 2 + 1,)`.
        n: Number of samples.

    Returns:
        fft: Full Fourier coefficients with shape `(n,)`.
    """
    nrfft = n // 2 + 1
    ncomplex = (n - 1) // 2
    fft = dispatch[rfft].empty(n, dtype=rfft.dtype)
    fft[:nrfft] = rfft
    fft[nrfft:] = dispatch.flip(rfft[1:1 + ncomplex]).conj()
    return fft


def unpack_rfft(z: ArrayOrTensor, size: int) -> ArrayOrTensor:
    """
    Unpack the Fourier coefficients of a real Fourier transform with `size // 2 + 1` elements to a
    vector of `size` elements.

    Args:
        z: Real Fourier transform coefficients.
        size: Size of the real signal. Necessary because the size cannot be inferred from `rfft`.

    Returns:
        z: Unpacked vector of `size` elements comprising the `size // 2 + 1` real parts of the zero
            frequency term, complex terms, and Nyqvist frequency term (for even `size`). The
            subsequent `(size - 1) // 2` elements are the imaginary parts of complex coefficients.
    """
    ncomplex = (size - 1) // 2
    parts = [z.real, z.imag[..., 1: ncomplex + 1]]
    return dispatch.concatenate(parts, axis=-1)


def pack_rfft(z: ArrayOrTensor, full_fft: bool = False) -> ArrayOrTensor:
    """
    Transform a real vector with `size` elements to a vector of complex Fourier coefficients with
    `size // 2 + 1` elements ready for inverse real fast Fourier transformation.

    Args:
        z: Unpacked vector of `size` elements. See :func:`unpack_rfft` for details.
        full_fft: Whether to return the full set of Fourier coefficients rather than just the
            reduced representation for the real fast Fourier transform. The full representation is
            required for :func:`pack_rfft2`.

    Returns:
        rfft: Real Fourier transform coefficients.
    """
    *_, size = z.shape
    fftsize = size // 2 + 1
    ncomplex = (size - 1) // 2
    # Zero frequency term, real parts of complex coefficients and possible Nyqvist frequency.
    rfft = z[..., :fftsize] * (1 + 0j)
    # Imaginary parts of complex coefficients.
    rfft[..., 1:ncomplex + 1] += 1j * z[..., fftsize:]
    if not full_fft:
        return rfft
    # Add the redundant complex coefficients (use `flip` because torch does not support negative
    # strides).
    return dispatch.concatenate([rfft, dispatch.flip(rfft[..., 1:ncomplex + 1].conj(), (-1,))], -1)


def transform_irfft(z: ArrayOrTensor, loc: ArrayOrTensor, *, cov: OptionalArrayOrTensor = None,
                    rfft_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Transform white noise in the Fourier domain to a Gaussian process realization.

    Args:
        z: Fourier-domain white noise with shape `(..., size)`. See :func:`unpack_rfft` for details.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov: First row of the covariance matrix with shape `(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape `(..., size // 2 + 1)`.

    Returns:
        y: Realization of the Gaussian process with shape `(..., size)`.
    """
    rfft_scale = _get_rfft_scale(cov, rfft_scale)
    rfft = pack_rfft(z) * rfft_scale
    return dispatch[rfft].fft.irfft(rfft, z.shape[-1]) + loc


def transform_rfft(y: ArrayOrTensor, loc: ArrayOrTensor, *, cov: OptionalArrayOrTensor = None,
                   rfft_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        y: Realization of the Gaussian process with shape `(..., size)`.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov: First row of the covariance matrix with shape `(..., size)`.

    Returns:
        z: Fourier-domain white noise with shape `(..., size)`.. See :func:`transform_irrft` for
            details.
    """
    rfft_scale = _get_rfft_scale(cov, rfft_scale)
    return unpack_rfft(dispatch[y].fft.rfft(y - loc) / rfft_scale, y.shape[-1])


def evaluate_log_prob_rfft(y: ArrayOrTensor, loc: ArrayOrTensor, *,
                           cov: OptionalArrayOrTensor = None,
                           rfft_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the log probability of a one-dimensional Gaussian process realization in Fourier space.

    Args:
        y: Realization of a Gaussian process with shape `(..., size)`, where `...` is the batch
            shape and `size` is the number of grid points.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., size)`.

    Returns:
        log_prob: Log probability of the Gaussian process realization with shape `(...)`.
    """
    rfft_scale = _get_rfft_scale(cov, rfft_scale)
    rfft = transform_rfft(y, loc, rfft_scale=rfft_scale)
    return log_prob_stdnorm(rfft).sum(axis=-1) \
        + evaluate_rfft_log_abs_det_jacobian(y.shape[-1], rfft_scale=rfft_scale)


def evaluate_rfft_log_abs_det_jacobian(size: int, *, cov: OptionalArrayOrTensor = None,
                                       rfft_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the log absolute determinant of the Jacobian associated with :func:`transform_rfft`.

    Args:
        cov: First row of the covariance matrix with shape `(..., size)`.
        rfft_scale: Optional precomputed scale of Fourier coefficients with shape
            `(..., size // 2 + 1)`.

    Returns:
        log_abs_det_jacobian
    """
    imagidx = (size + 1) // 2
    rfft_scale = _get_rfft_scale(cov, rfft_scale)
    assert rfft_scale.shape[-1] == size // 2 + 1
    return - dispatch.log(rfft_scale).sum(axis=-1) \
        - dispatch.log(rfft_scale[1:imagidx]).sum(axis=-1) - log2 * ((size - 1) // 2) \
        + size * math.log(size) / 2


def _get_rfft2_scale(cov: OptionalArrayOrTensor, rfft2_scale: OptionalArrayOrTensor) \
        -> ArrayOrTensor:
    if (cov is None) == (rfft2_scale is None):  # pragma: no cover
        raise ValueError("exactly one of `cov` and `rfft2_scale` must be given")
    if rfft2_scale is not None:
        return rfft2_scale
    return evaluate_rfft2_scale(cov)


def evaluate_rfft2_scale(cov: ArrayOrTensor) -> ArrayOrTensor:
    """
    Evaluate the scale of Fourier coefficients.

    Args:
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.

    Returns:
        scale: Scale of Fourier coefficients with shape `(..., height, width // 2 + 1)`.
    """
    *_, height, width = cov.shape
    size = width * height
    rfft2_scale = size * dispatch[cov].fft.rfft2(cov).real / 2

    # Recall how the two-dimensional RFFT is computed. We first take an RFFT of rows of the matrix.
    # This leaves us with a real first column (zero frequency term) and a real last column if the
    # number of columns is even (Nyqvist frequency term). Second, we take a *full* FFT of the
    # columns. The first column will have a real coefficient in the first row (zero frequency in the
    # "row-dimension"). All elements in rows beyond n // 2 + 1 are irrelevant because the column was
    # real. The same applies to the last column if there is a Nyqvist frequency term. Finally, we
    # will also have a real-only Nyqvist frequency term in the first and last column if the number
    # of rows is even.

    # The zero-frequency term in both dimensions which must always be real.
    rfft2_scale[..., 0, 0] *= 2

    # If the number of colums is even, the last row will be real after the row-wise RFFT.
    # Consequently, the first element is real after the column-wise FFT.
    if width % 2 == 0:
        rfft2_scale[..., 0, width // 2] *= 2

    # If the number of rows is even, we have a Nyqvist frequency term in the first column.
    if height % 2 == 0:
        rfft2_scale[..., height // 2, 0] *= 2

    # If the number of rows and columns is even, we also have a Nyqvist frequency term in the last
    # column.
    if height % 2 == 0 and width % 2 == 0:
        rfft2_scale[..., height // 2, width // 2] *= 2

    return dispatch.sqrt(rfft2_scale)


def unpack_rfft2(z: ArrayOrTensor, shape: tuple[int]) -> ArrayOrTensor:
    """
    Unpack the Fourier coefficients of a two-dimensional real Fourier transform with shape
    `(..., height, width // 2 + 1)` to a batch of matrices with shape `(..., height, width)`.

    TODO: add details on packing structure.

    Args:
        z: Two-dimensional real Fourier transform coefficients.
        shape: Shape of the real signal. Necessary because the number of columns cannot be inferred
            from `rfft2`.

    Returns:
        z: Unpacked matrices with shape `(..., height, width)`.
    """
    *_, height, width = shape
    ncomplex = (width - 1) // 2
    parts = [
        # First column is always real.
        unpack_rfft(z[..., :height // 2 + 1, 0], height)[..., None],
        # Real and imaginary parts of complex coefficients.
        z[..., 1:ncomplex + 1].real,
        z[..., 1:ncomplex + 1].imag,
    ]
    if width % 2 == 0:  # Nyqvist frequency terms if the number of columns is even.
        parts.append(unpack_rfft(z[..., :height // 2 + 1, width // 2], height)[..., None])
    return dispatch.concatenate(parts, axis=-1)


def pack_rfft2(z: ArrayOrTensor) -> ArrayOrTensor:
    """
    Transform a batch of real matrices with shape `(..., height, width)` to a batch of complex
    Fourier coefficients with shape `(..., height, width // 2 + 1)` ready for inverse real fast
    Fourier transformation in two dimensions.

    Args:
        z: Unpacked matrices with shape `(..., height, width)`. See :func:`unpack_rfft2` for
            details.

    Returns:
        rfft: Two-dimensional real Fourier transform coefficients.
    """
    *batch_shape, height, width = z.shape
    ncomplex = (width - 1) // 2
    rfft2 = dispatch[z].empty((*batch_shape, height, width // 2 + 1),
                              dtype=dispatch.get_complex_dtype(z))
    # Real FFT in the first column due to zero-frequency terms for the row-wise Fourier transform.
    rfft2[..., 0] = pack_rfft(z[..., 0], full_fft=True)
    # Complex Fourier coefficients.
    rfft2[..., 1:ncomplex + 1] = z[..., 1:ncomplex + 1] + 1j * z[..., ncomplex + 1:2 * ncomplex + 1]
    # Real FFT in the last column due to the Nyqvist frequency terms for the row-wise Fourier
    # transform if the number of columns is even.
    if width % 2 == 0:
        rfft2[..., width // 2] = pack_rfft(z[..., width - 1], full_fft=True)
    return rfft2


def transform_irfft2(z: ArrayOrTensor, loc: ArrayOrTensor, *, cov: OptionalArrayOrTensor = None,
                     rfft2_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Transform white noise in the Fourier domain to a Gaussian process realization.

    Args:
        z: Unpacked matrices with shape `(..., height, width)`. See :func:`unpack_rfft2` for
            details.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., n, m)`.

    Returns:
        y: Realization of the Gaussian process.
    """
    rfft2 = pack_rfft2(z) * _get_rfft2_scale(cov, rfft2_scale)
    return dispatch[rfft2].fft.irfft2(rfft2, z.shape[-2:]) + loc


def transform_rfft2(y: ArrayOrTensor, loc: ArrayOrTensor, *, cov: OptionalArrayOrTensor = None,
                    rfft2_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        y: Realization of the Gaussian process.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., n, m)`.

    Returns:
        z: Unpacked matrices with shape `(..., height, width)`. See :func:`unpack_rfft2` for
            details.
    """
    rfft2_scale = _get_rfft2_scale(cov, rfft2_scale)
    return unpack_rfft2(dispatch[y].fft.rfft2(y - loc) / rfft2_scale, y.shape)


def evaluate_rfft2_log_abs_det_jacobian(width: int, *, cov: OptionalArrayOrTensor = None,
                                        rfft2_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the log absolute determinant of the Jacobian associated with :func:`transform_rfft`.

    Args:
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.
        rfft_scale: Optional precomputed scale of Fourier coefficients with shape
            `(..., height, width // 2 + 1)`.

    Returns:
        log_abs_det_jacobian
    """
    rfft2_scale = _get_rfft2_scale(cov, rfft2_scale)
    height = rfft2_scale.shape[-2]
    assert rfft2_scale.shape[-1] == width // 2 + 1
    ncomplex_horizontal = (width - 1) // 2
    ncomplex_vertical = (height - 1) // 2
    parts = [
        # Real part of the first-column RFFT.
        - dispatch.log(rfft2_scale[..., :height // 2 + 1, 0]).sum(axis=-1),
        # Imaginary part of the first-column RFFT.
        - dispatch.log(rfft2_scale[..., 1:ncomplex_vertical + 1, 0]).sum(axis=-1),
        # Complex coefficients in subsequent columns.
        - 2 * dispatch.log(rfft2_scale[..., 1:1 + ncomplex_horizontal]).sum(axis=(-2, -1))
    ]
    # Account for Nyqvist frequencies in the last column if the number of columns is even.
    if width % 2 == 0:
        parts.extend([
            # Real part of the last-column RFFT.
            - dispatch.log(rfft2_scale[..., :height // 2 + 1, width // 2]).sum(axis=-1),
            # Imaginary part of the last-column RFFT.
            - dispatch.log(rfft2_scale[..., 1:ncomplex_vertical + 1, width // 2]).sum(axis=-1),
        ])
    size = width * height
    nterms = (size - 1) // 2
    if height % 2 == 0 and width % 2 == 0:
        nterms -= 1
    return sum(parts) - log2 * nterms + size * math.log(size) / 2


def evaluate_log_prob_rfft2(y: ArrayOrTensor, loc: ArrayOrTensor, *,
                            cov: OptionalArrayOrTensor = None,
                            rfft2_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the log probability of a two-dimensional Gaussian process realization in Fourier space.

    Args:
        y: Realization of a Gaussian process with shape `(..., n, m)`, where `...` is the batch
            shape, `n` is the number of rows, and `m` is the number of columns.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., n, m)`.

    Returns:
        log_prob: Log probability of the Gaussian process realization with shape `(...)`.
    """
    rfft2_scale = _get_rfft2_scale(cov, rfft2_scale)
    rfft2 = transform_rfft2(y, loc, rfft2_scale=rfft2_scale)
    return log_prob_stdnorm(rfft2).sum(axis=(-2, -1)) \
        + evaluate_rfft2_log_abs_det_jacobian(y.shape[-1], rfft2_scale=rfft2_scale)
