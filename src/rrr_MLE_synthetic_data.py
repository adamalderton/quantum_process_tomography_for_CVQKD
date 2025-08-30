import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from scipy.special import gammaln

matplotlib.use("Agg")

################### NUMERICAL HELPER FUNCTIONS ###################

def factorial_array(n: int, exact: bool = True, return_log: bool = False) -> np.ndarray:
    """
    Factorials 0!, 1!, ..., n! with safe options.

    Args:
        n (int): maximum n (inclusive).
        exact (bool): if True, return arbitrary-precision integers (dtype=object).
                      if False, return floating-point approximations.
        return_log (bool): if True, return log(k!) as float (using math.lgamma).

    Returns:
        np.ndarray: shape (n+1,). If return_log=True -> float array of log(k!).
                    If exact=True -> object array of Python ints (arbitrary precision).
                    Else -> float array (may overflow to inf for very large n).
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    if return_log:
        # log(k!) via gammaln(k+1)
        ks = np.arange(n + 1, dtype=int)
        return gammaln(ks + 1).astype(float)

    if exact:
        # Arbitrary-precision integers, no overflow
        if n == 0:
            return np.array([1], dtype=object)
        vals = np.arange(1, n + 1, dtype=object)
        acc = np.multiply.accumulate(vals, dtype=object)
        return np.concatenate(([1], acc))

    # Float approximations (fast, but can overflow for large n). Use gammaln for stability
    ks = np.arange(n + 1, dtype=int)
    return np.exp(gammaln(ks + 1)).astype(float)

def build_grid(grid_half_side: float, grid_side_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2D grid of points.

    Args:
        grid_half_side (float): The half side length of the grid, such that it spans [-grid_half_side, grid_half_side]. grid_half_side is a coordinate halfwidth in (y_1, y_2) space.
        grid_side_points (int): The number of discrete points along an axis of the grid.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - alphas_grid (np.ndarray): A 2D array of complex numbers representing the coherent state alpha at each grid point, in canonical units.
            - y1s_grid (np.ndarray): A 2D array of y1 coordinates of the grid points.
            - y2s_grid (np.ndarray): A 2D array of y2 coordinates of the grid points.
    """
    y1s = np.linspace(-grid_half_side, grid_half_side, grid_side_points)
    y2s = np.linspace(-grid_half_side, grid_half_side, grid_side_points)
    y1s_grid, y2s_grid = np.meshgrid(y1s, y2s, indexing = "xy")
    
    # Determine the coherent state alpha from the y1 and y2 grid points, in canonical units for now. TODO: Convert to SNU.
    alphas_grid = (y1s_grid + 1j * y2s_grid) / np.sqrt(2.0)

    return alphas_grid, y1s_grid, y2s_grid

def analytic_Q_number_state(alpha: np.ndarray, k: int) -> np.ndarray:
    """Compute the Q-function of a number state |k> at given coherent state points.

    Args:
        alpha (np.ndarray): A 2D array of complex numbers representing the coherent state alpha at which to evaluate the Q-function.
        k (int): The photon number of the number state |k>.
    
    Returns:
        np.ndarray: A 2D array of the Q-function values at each alpha.
    """

    if k < 0:
        raise ValueError("k must be >= 0")
    abs_sq = np.abs(alpha) ** 2
    if k == 0:
        # Avoid 0*log(0) -> NaN at origin
        return (1.0 / np.pi) * np.exp(-abs_sq)
    # For k>0: safe log with where; values where abs_sq==0 contribute 0 anyway.
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.where(abs_sq > 0, k * np.log(abs_sq), 0.0)
    out = (1.0 / np.pi) * np.exp(log_term - abs_sq - gammaln(k + 1))
    # Explicitly zero out any spurious NaNs where abs_sq==0
    if np.isscalar(abs_sq):
        return out
    out = np.where(abs_sq == 0, 0.0, out)
    return out

def analytic_Q_coherent_state(alpha: np.ndarray, alpha_0: complex) -> np.ndarray:
    """Compute the Q-function of a coherent state |alpha_0> at given coherent state points.

    Args:
        alpha (np.ndarray): A 2D array of complex numbers representing the coherent state alpha at which to evaluate the Q-function.
        alpha_0 (complex): The coherent state parameter of the state |alpha_0>.
    
    Returns:
        np.ndarray: A 2D array of the Q-function values at each alpha.
    """
    Q_values = (1 / np.pi) * np.exp(-np.abs(alpha - alpha_0)**2)
    return Q_values

def sample_pure_coherent_state(alpha_0: complex, num_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample dual-homodyne measurement outcomes from a pure coherent state |alpha_0>.
    
        Args:
            alpha_0 (complex): The coherent state parameter of the state |alpha_0>.
            num_samples (int): The number of measurement samples to draw.
            rng (Optional[np.random.Generator]): An optional random number generator for reproducibility. If None, a new generator will be created.

        Returns:
            np.ndarray: A 1D array of complex numbers representing the sampled coherent state parameters from the dual-homodyne measurements.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # The dual-homodyne measurement outcomes (y1, y2) are drawn from a Gaussian distribution centered at (Re(alpha_0), Im(alpha_0)) with variance 1/2 in each quadrature.
    y1s = rng.normal(loc=np.sqrt(2) * alpha_0.real, scale=1.0 / np.sqrt(2), size=num_samples)
    y2s = rng.normal(loc=np.sqrt(2) * alpha_0.imag, scale=1.0 / np.sqrt(2), size=num_samples)
    
    alpha_from_y = (y1s + 1j * y2s) / np.sqrt(2.0)

    return alpha_from_y

def sample_number_state(k: int, num_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:

    """Sample heterodyne outcomes α for the number (Fock) state |k> in canonical units.
    The function maps to canonical Cartesian quadratures y1 = sqrt(2) Re(α), y2 = sqrt(2) Im(α)
    and reconstructs α = (y1 + i y2) / sqrt(2) (numerically equivalent up to floating point roundoff).

    Distribution: Q_k(α) = (1/π) e^{-|α|^2} |α|^{2k} / k!.
    Implementation: sample r^2 ~ Gamma(k+1, 1), θ ~ Uniform[0, 2π), α = r e^{iθ}.

    Args:
        k (int): Fock state index (k >= 0).
        num_samples (int): Number of complex samples to draw.
        rng (Optional[np.random.Generator]): Optional NumPy Generator instance to control randomness.
            If None, a new default_rng() is created.
        np.ndarray[complex]: Complex heterodyne outcomes α of shape (num_samples,).

    Returns:
        np.ndarray[complex] of shape (num_samples,).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if rng is None:
        rng = np.random.default_rng()

    # Sample radius in alpha-space: |α|^2 ~ Gamma(k+1, 1)
    r2_alpha = rng.gamma(shape=k + 1, scale=1.0, size=num_samples)

    # Convert to y-space radius (y1^2 + y2^2 = 2 * |α|^2)
    r_y = np.sqrt(2.0 * r2_alpha)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=num_samples)

    # Sample directly in (y1, y2) polar coordinates
    y1s = r_y * np.cos(theta)
    y2s = r_y * np.sin(theta)

    # Cast to complex α space: α = (y1 + i y2) / sqrt(2)
    alpha_from_y = (y1s + 1j * y2s) / np.sqrt(2.0)

    return alpha_from_y

def adaptive_grid_half_size(alpha_samples: np.ndarray, padding: float = 2.0, percentile: float = 95.0) -> float:
    """Adaptive grid half-size via high-percentile |α| radius.

    Rule: half_size = sqrt(2) * r_p + padding, with r_p the specified percentile of |α|.
    This better captures ring-shaped number state Q distributions (e.g. k=3) than mean-based rules.

    Args:
        alpha_samples: complex samples α.
        padding: additive margin (y units) after mapping r_p -> sqrt(2) r_p.
        percentile: percentile of |α| (default 95).

    Returns:
        Half-side length in y-coordinates.
    """
    if alpha_samples.size == 0:
        return float(padding)
    r_p = np.percentile(np.abs(alpha_samples), percentile)
    return float(np.sqrt(2.0) * r_p + padding)

def histogram_on_grid(alpha_samples: np.ndarray, grid_half_size: Optional[float], grid_side_points: int,
                      *, padding: float = 3.0) -> Tuple[np.ndarray, np.ndarray, float, int, float]:
    """Form a simple point-mass (nearest-grid) histogram estimate Q̂ on the design grid.

    Adaptive behaviour:
        If grid_half_size is None, it is set to |mean(α_i)| + padding. This targets the typical
        displacement magnitude; it may exclude outlier samples if they are far from the mean.
        Increase padding if you require broader coverage.

    Args:
        alpha_samples: 1D (or flattenable) array of complex samples α.
        grid_half_size: Half side length of square in (y1,y2). If None -> adaptive = max|α| + padding.
        grid_side_points: Number of discrete grid points along each axis.
        padding: Extra margin added when grid_half_size is None.

    Returns:
        (Q_hat, counts_2d, coverage_hist, counts_in, area_alpha)
    """

    if alpha_samples.ndim != 1:
        alpha_samples = alpha_samples.ravel()

    if grid_half_size is None:
        grid_half_size = adaptive_grid_half_size(alpha_samples, padding=padding)

    # Convert complex α back to (x,y) coordinates used for grid generation: x=√2 Re α, y=√2 Im α
    xs = np.linspace(-grid_half_size, grid_half_size, grid_side_points)
    ys = np.linspace(-grid_half_size, grid_half_size, grid_side_points)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    area_alpha = (dx * dy) / 2.0
    sx = np.sqrt(2.0) * alpha_samples.real
    sy = np.sqrt(2.0) * alpha_samples.imag

    # Mask samples within window
    in_mask = (sx >= -grid_half_size) & (sx <= grid_half_size) & (sy >= -grid_half_size) & (sy <= grid_half_size)
    sx_in = sx[in_mask]
    sy_in = sy[in_mask]
    counts_in = sx_in.size

    # Map to nearest index
    fx = (sx_in + grid_half_size) / (2 * grid_half_size) * (grid_side_points - 1)
    fy = (sy_in + grid_half_size) / (2 * grid_half_size) * (grid_side_points - 1)
    ix = np.rint(fx).astype(int)
    iy = np.rint(fy).astype(int)
    # Safety clamp
    ix = np.clip(ix, 0, grid_side_points - 1)
    iy = np.clip(iy, 0, grid_side_points - 1)
    counts_2d = np.zeros((grid_side_points, grid_side_points), dtype=int)
    np.add.at(counts_2d, (iy, ix), 1)  # note: Y first (row), X second (col)

    # Empirical coverage (mass inside window)
    coverage_hist = counts_in / alpha_samples.size if alpha_samples.size > 0 else 0.0

    # Convert to Q_hat via dividing by total samples and area element
    Q_hat = counts_2d.astype(np.float64) / (alpha_samples.size * area_alpha)

    return Q_hat, counts_2d, coverage_hist, counts_in, area_alpha

def true_rho_coherent(alpha_0: complex, fock_cutoff: int) -> Tuple[np.ndarray, float]:
    """
    Build the truncated coherent state |alpha_0> in the Fock basis {|0>,...,|fock_cutoff>}
    using a stable recurrence (no factorials), return:
      - rho: the normalized density matrix inside the truncated subspace (trace=1 there),
      - tail: the probability mass discarded by truncation (Poisson survival beyond cutoff).

    Args:
        alpha_0: coherent-state parameter α0 (complex).
        fock_cutoff: largest Fock number N to keep (inclusive). Must be >= 0.

    Returns:
        rho: (N+1, N+1) complex ndarray, rho = |c><c| with ||c||=1 inside the cutoff.
        tail: float, 1 - sum_{n=0}^N |<n|alpha_0>|^2.
    """
    if fock_cutoff < 0:
        raise ValueError("fock_cutoff must be >= 0")

    N = fock_cutoff
    mu = float(abs(alpha_0) ** 2)

    # Coherent amplitudes c_n = e^{-mu/2} * alpha_0^n / sqrt(n!) via stable recurrence:
    c = np.zeros(N + 1, dtype=np.complex128)
    c[0] = np.exp(-mu / 2.0)
    for n in range(N):
        # c_{n+1} = c_n * alpha_0 / sqrt(n+1)
        c[n + 1] = c[n] * alpha_0 / np.sqrt(n + 1.0)

    # Pre-normalization norm^2 equals Poisson CDF up to N
    norm2_trunc = float(np.vdot(c, c).real)
    # Probability mass above cutoff (Poisson tail)
    tail = max(0.0, 1.0 - norm2_trunc)

    # Renormalize inside the truncated subspace
    if norm2_trunc > 0.0:
        c /= np.sqrt(norm2_trunc)

    # Pure-state projector in the truncated space
    rho = np.outer(c, np.conjugate(c))  # shape (N+1, N+1), complex128

    return rho, tail

def numeric_coverage(alpha: np.ndarray, Q_vals: np.ndarray, grid_half_size: float, grid_size_points: int) -> Tuple[float, float]:
	"""Approximate integral of Q over current square grid via Riemann sum.

	Coordinates x,y are in [-grid_half_size, grid_half_size]; spacing Δ = 2grid_half_size/(grid_size_points-1). Relation d^2α = (1/2) dx dy.
	Therefore coverage ≈ Σ_j Q(α_j) * (Δx Δy)/2.
	Returns (coverage, tail_mass=1-coverage).
	"""
	delta = (2 * grid_half_size) / (grid_size_points - 1)
	area_element = (delta * delta) / 2.0
	coverage = float(Q_vals.sum() * area_element)
	return coverage, max(0.0, 1.0 - coverage)

def Q_from_rho(alpha: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute Q(α) = (1/π) e^{-|α|^2} Σ_{m,n<=N} ρ_{mn} (α*)^m α^n / √(m! n!)
    using a truncated forward model and stable recurrences.

    Works for alpha of any shape; returns an array of the same shape.
    """
    alpha = np.asarray(alpha, dtype=np.complex128)
    shape = alpha.shape
    a = alpha.reshape(-1)  # flatten to (M,)

    Np1 = rho.shape[0]
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be square (N+1)x(N+1).")
    if Np1 == 0:
        return np.zeros_like(alpha, dtype=float)

    # Build d[n, k] = α_k^n / √(n!), via recurrence (no factorials)
    M = a.size
    d = np.empty((Np1, M), dtype=np.complex128)
    d[0] = 1.0
    for n in range(Np1 - 1):
        d[n + 1] = d[n] * a / np.sqrt(n + 1.0)

    # c[m, k] = (α_k*)^m / √(m!) is just conj(d[m, k])
    c = np.conjugate(d)

    # Contract: Q_k = Σ_{m,n} ρ_{mn} c[m,k] d[n,k]
    # einsum does it efficiently over all k
    Q = np.einsum('mn,mk,nk->k', rho, c, d)

    # Multiply envelope and reshape back
    Q = (np.exp(-np.abs(a) ** 2) / np.pi) * Q
    Q = Q.real  # true Q is real; discard tiny imag from round-off

    return Q.reshape(shape)

def build_coherent_state_matrix(alpha: np.ndarray, fock_cutoff: int) -> np.ndarray:
    """
    Construct the (fock_cutoff+1) x M matrix C with entries
        C[n, j] = <n | α_j> = exp(-|α_j|^2 / 2) * α_j^n / sqrt(n!)
    where α_j are the input phase-space points.

    Notes:
    - Accepts alpha of any shape; flattens to length M internally.
    - Uses a stable recurrence: C[n+1, :] = C[n, :] * α / sqrt(n+1).

    Args:
        alpha: array-like of complex α points (any shape).
        fock_cutoff: largest Fock number N to include (>= 0).

    Returns:
        C: complex ndarray of shape (N+1, M), with M = alpha.size.
    """
    if fock_cutoff < 0:
        raise ValueError("fock_cutoff must be >= 0")

    a = np.asarray(alpha, dtype=np.complex128).reshape(-1)  # flatten to (M,)
    M = a.size
    N = fock_cutoff

    C = np.empty((N + 1, M), dtype=np.complex128)
    C[0] = np.exp(-0.5 * np.abs(a) ** 2)  # n = 0 term

    for n in range(N):
        C[n + 1] = C[n] * a / np.sqrt(n + 1.0)

    return C

def initialise_rho(fock_cutoff: int, method: str = 'alpha_bar', alpha_bar: Optional[complex] = None) -> np.ndarray:
    """Initialise a density matrix ρ in the truncated Fock basis { |0>,...,|N> }.

    Methods:
        'alpha_bar'       : ρ = |ᾱ><ᾱ| where ᾱ is a user-specified coherent amplitude (should be the expected state given the samples).
                             If alpha_bar is None, defaults to ᾱ = 0 (vacuum projector).
                             The coherent state is truncated and renormalised within the
                             subspace using `true_rho_coherent` (tail mass discarded).
        'maximally_mixed' : ρ = I / (N+1) (trace = 1, full rank).

    Args:
        fock_cutoff: N (largest Fock number retained, N >= 0).
        method: 'alpha_bar' or 'maximally_mixed' (case-insensitive; hyphens allowed).
        alpha_bar: complex coherent amplitude for initial state when method='alpha_bar'.

    Returns:
        ndarray (N+1, N+1) complex128 density matrix (trace=1 for both methods).
    """
    if fock_cutoff < 0:
        raise ValueError("fock_cutoff must be >= 0")
    Np1 = fock_cutoff + 1

    key = method.strip().lower().replace('-', '_')
    if key == 'alpha_bar':
        if alpha_bar is None:
            alpha_bar = 0.0 + 0.0j
        rho, _tail = true_rho_coherent(alpha_bar, fock_cutoff)
        # true_rho_coherent already renormalises within truncation, so trace=1.
        return rho
    elif key == 'maximally_mixed':
        return np.eye(Np1, dtype=np.complex128) / Np1
    else:
        raise ValueError("Unknown initialisation method. Use 'alpha_bar' or 'maximally_mixed'.")

def _hermitize(X: np.ndarray) -> np.ndarray:
    return 0.5 * (X + X.conjugate().T)

def _project_psd_trace_one(X: np.ndarray) -> np.ndarray:
    # Hermitize
    Xh = _hermitize(X)
    # Eigendecompose and clamp negatives
    vals, vecs = np.linalg.eigh(Xh)
    vals = np.maximum(vals, 0.0)
    s = float(vals.sum())
    if s <= 0.0:
        # Fallback: maximally mixed
        d = X.shape[0]
        return np.eye(d, dtype=np.complex128) / d
    vals /= s
    return (vecs * vals) @ vecs.conjugate().T

#################### MLE ROUTINES ###################

def compute_model_probabilities(rho: np.ndarray, C: np.ndarray, bin_area: float) -> np.ndarray:
    """Compute model probabilities for each histogram/bin center.

    Given:
        rho: (N+1, N+1) density matrix in Fock basis.
        C:   (N+1, M) matrix with columns C[:, j] = <n|alpha_j>.
        bin_area: scalar ΔA representing the phase–space area element associated
                  with each grid point in α-space (already includes the 1/2 from
                  d^2α = (1/2) dx dy if constructed that way).

    We form per-bin probabilities
        p_j = (ΔA/π) <α_j|ρ|α_j>
            = (ΔA/π) Σ_{m,n} C[m,j]^* ρ_{m n} C[n,j].

    Notes:
        * The set { (ΔA/π)|α_j><α_j| } approximates a POVM over the truncated
          domain; hence Σ_j p_j ≈ Tr(ρ * Σ_j (ΔA/π)|α_j><α_j|). It need not equal 1
          exactly (finite window + discretisation error).
        * Small negative numerical round‑off is clipped to 0.
        * A tiny floor (eps) is applied to avoid division by zero in EM steps.

    Returns:
        p (M,) float64 array of per-bin model probabilities.
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2D array")
    if C.ndim != 2:
        raise ValueError("C must be a 2D array with shape (N+1, M)")
    if rho.shape[0] != C.shape[0]:
        raise ValueError("Dimension mismatch: rho (N+1, N+1) and C (N+1, M) must share N+1")
    if bin_area <= 0:
        raise ValueError("bin_area must be positive")

    # Efficient contraction: overlaps_j = <α_j|ρ|α_j>.
    # Compute R = rho @ C -> (N+1, M), then elementwise multiply by C.conj() and sum over rows.
    RC = rho @ C  # (N+1, M)
    overlaps = np.sum(np.conjugate(C) * RC, axis=0)  # (M,), complex but should be real.
    overlaps = overlaps.real  # Discard tiny imaginary parts (Hermiticity guarantees reality up to round-off).

    p = (bin_area / np.pi) * overlaps

    # Numerical guards: clip tiny negatives and enforce a small floor for stability.
    p = np.maximum(p, 0.0)
    eps = 1e-30  # protects later divisions
    p = np.clip(p, eps, None)

    return p

def compute_R_operator(counts_2d: np.ndarray, bin_probabilities: np.ndarray, C: np.ndarray, bin_area: float) -> np.ndarray:
    """Construct the EM reweighting operator R for the current iteration.

    POVM elements (discretised heterodyne bins):
        E_j = (ΔA/π) |α_j><α_j|,   with |α_j> amplitudes encoded by columns of C.

    Given observed counts n_j and model probabilities p_j, the EM "R" operator is
        R(ρ) = Σ_j (n_j / p_j) E_j  (independent of ρ once n_j, p_j fixed).

    Using C (N+1, M) with C[:, j] = <n|α_j>, we have
        |α_j><α_j|  ↦  (C[:, j]) (C[:, j])^†  (rank-1 outer product in Fock basis).
    Therefore
        R = Σ_j w_j C[:, j] C[:, j]^†,  where w_j = (n_j / p_j) (ΔA/π).

    Vectorised form:
        R = (C * w) @ C^†  with broadcasting w over rows.

    Args:
        counts_2d: 2D integer array of counts on the α-grid (shape GxG).
        probabilities: 1D float array p_j of model bin probabilities (length M = G*G).
        C: (N+1, M) coherent state coefficient matrix.
        bin_area: ΔA (float), positive.

    Returns:
        R: (N+1, N+1) complex Hermitian positive semidefinite matrix.
    """
    if C.ndim != 2:
        raise ValueError("C must be 2D (N+1, M)")
    if counts_2d.ndim != 2:
        raise ValueError("counts_2d must be a 2D grid array")
    if bin_area <= 0:
        raise ValueError("bin_area must be positive")

    n_vec = counts_2d.ravel(order='C').astype(np.float64)
    M = C.shape[1]
    if n_vec.size != M:
        raise ValueError(f"Counts size {n_vec.size} does not match number of columns in C ({M})")
    bin_probabilities = np.asarray(bin_probabilities, dtype=np.float64)
    if bin_probabilities.ndim != 1 or bin_probabilities.size != M:
        raise ValueError("probabilities must be 1D of length matching columns of C")

    # Guard against zeros (should already be floored in probability computation)
    eps = 1e-30
    p_safe = np.clip(bin_probabilities, eps, None)

    weights = (n_vec / p_safe) * (bin_area / np.pi)  # (M,)
    # Broadcast weights over columns: scale each column of C
    CW = C * weights  # (N+1, M)
    R = CW @ C.conjugate().T  # (N+1, N+1)

    # Hermitise to suppress numerical asymmetry
    R = 0.5 * (R + R.conjugate().T)

    return R

def neg_log_likelihood_poisson(
    counts_2d: np.ndarray,
    bin_masses: np.ndarray,   # p_j = Tr(ρ E_j) including (ΔA/π)
    N_total: Optional[float] = None,
    include_constant: bool = False,
) -> float:
    """
    Poisson-factorized NLL for binned heterodyne data.

    Model: n_j ~ Poisson(μ_j), with μ_j = N_total * p_j, where
           p_j = Tr(ρ E_j) are the (unnormalized) bin masses that already
           include the POVM weight (ΔA/π).

    NLL(ρ) = Σ_j [ μ_j - n_j log μ_j ] + Σ_j log n_j!
            = N_total * Σ_j p_j - Σ_j n_j log p_j - (Σ_j n_j) log N_total + const.

    Args:
        counts_2d: integer counts on the grid (G×G).
        bin_masses: length-M vector p_j = Tr(ρ E_j) (M = G*G).
        N_total: total number of shots; if None, uses N_total = counts.sum().
        include_constant: if True, adds Σ_j log n_j! to the return value.

    Returns:
        float NLL value.
    """
    n = counts_2d.ravel(order="C").astype(np.float64)
    p = np.asarray(bin_masses, dtype=np.float64)
    if n.size != p.size:
        raise ValueError("counts and probability/mass vector length mismatch")

    # Total shots
    if N_total is None:
        N_total = float(n.sum())

    # Safety floor (must match your p_j flooring elsewhere)
    eps = 1e-18
    p = np.clip(p, eps, None)

    S = p.sum()                     # coverage/mass inside the window
    core = N_total * S - np.sum(n * np.log(p)) - n.sum() * np.log(max(N_total, 1.0))

    if not include_constant:
        return float(core)

    # Add combinatorial constant Σ log n_j!
    # Prefer SciPy if available; otherwise fall back to math.lgamma in a loop.
    try:
        from scipy.special import gammaln
        const = np.sum(gammaln(n + 1.0))
    except Exception:
        import math
        const = float(np.sum([math.lgamma(x + 1.0) for x in n]))

    return float(core + const)

def mle_rrr(
    counts_2d: np.ndarray,
    C: np.ndarray,
    bin_area: float,
    init_rho: np.ndarray,
    *,
    max_iters: int = 300,
    tol: float = 1e-6,
    entropy_tau: float = 0.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    RρR (EM-style) MLE loop for binned heterodyne data.

    Model:
        - Discretised POVM bins: E_j = (ΔA/π) |α_j⟩⟨α_j|, columns of C encode |α_j⟩ in Fock basis.
        - Bin "masses": p_j(ρ) = Tr(ρ E_j) = (ΔA/π) ⟨α_j|ρ|α_j⟩ (not renormalised to 1 in general).
        - Counts n_j are modeled via Poisson factorisation: n_j ~ Poisson(N_total * p_j).

    Update (no outside-bin term):
        R_t = Σ_j (n_j / p_j(ρ_t)) E_j
        ρ_{t+1} ←  R_t ρ_t R_t
        then project to PSD and trace one.

    Entropy regularisation (optional, small τ):
        approximate proximal step by modifying the reweighting operator:
            R_t ← R_t - τ (log ρ_t + I)
        before the sandwich; this may sacrifice strict EM monotonicity but works well in practice.

    Args:
        counts_2d: (G, G) integer array of counts.
        C: (N+1, M) coherent-state matrix, columns C[:, j] = ⟨n|α_j⟩.
        bin_area: ΔA (scalar) for each grid point in α-space.
        init_rho: (N+1, N+1) initial density matrix (will be projected to PSD, trace 1).
        max_iters: maximum number of RρR iterations.
        tol: relative NLL improvement tolerance for early stopping.
        entropy_tau: weight τ ≥ 0 for +τ Tr(ρ log ρ) penalty.

    Returns:
        rho: final (N+1, N+1) density matrix.
        nll_history: list of Poisson NLL values per iteration (evaluated at ρ_{t}).
    """

    # --- basic shape checks ---
    if init_rho.ndim != 2 or init_rho.shape[0] != init_rho.shape[1]:
        raise ValueError("init_rho must be square (N+1)x(N+1).")
    if C.ndim != 2 or init_rho.shape[0] != C.shape[0]:
        raise ValueError("C must be (N+1, M) and share N+1 with init_rho.")
    if not (bin_area > 0):
        raise ValueError("bin_area must be positive.")
    if counts_2d.ndim != 2:
        raise ValueError("counts_2d must be a 2D array (G, G).")

    # --- initial state: ensure PSD, trace 1 ---
    rho = _project_psd_trace_one(init_rho)

    # Precompute flattened counts and total shots
    n_vec = counts_2d.ravel(order="C").astype(np.float64)
    N_total = float(n_vec.sum())

    nll_history: List[float] = []

    # Evaluate initial NLL
    p0 = compute_model_probabilities(rho, C, bin_area)  # (M,)
    nll0 = neg_log_likelihood_poisson(counts_2d, p0, N_total=N_total, include_constant=False)
    nll_history.append(float(nll0))

    # Main loop
    for _ in range(1, max_iters + 1):
        # E-step like: responsibilities via R operator
        p = compute_model_probabilities(rho, C, bin_area)  # p_j = Tr(ρ E_j) (masses), floored internally
        R = compute_R_operator(counts_2d, p, C, bin_area)  # Hermitian PSD

        # Entropy-regularised tweak, if requested
        if entropy_tau > 0.0:
            # Compute matrix logarithm of Hermitian part with eigen floor.
            Xh = _hermitize(rho)
            vals, vecs = np.linalg.eigh(Xh)
            vals = np.clip(vals, 1e-18, None)
            log_rho = (vecs * np.log(vals)) @ vecs.conjugate().T
            R = R - entropy_tau * (log_rho + np.eye(rho.shape[0], dtype=np.complex128))

        # M-step like: sandwich update and projection
        rho_new = R @ rho @ R
        rho_new = _project_psd_trace_one(rho_new)

        # Evaluate NLL at the new iterate
        p_new = compute_model_probabilities(rho_new, C, bin_area)
        nll_new = neg_log_likelihood_poisson(counts_2d, p_new, N_total=N_total, include_constant=False)
        nll_history.append(float(nll_new))

        # Check convergence: relative decrease in NLL
        prev = nll_history[-2]
        curr = nll_history[-1]
        denom = abs(prev) + 1e-18
        if prev - curr >= 0 and (prev - curr) / denom < tol:
            rho = rho_new
            break

        rho = rho_new

    return rho, nll_history

#################### MAIN SCRIPT ###################

def run_mle_workflow(
    *,
    state_type: str,
    num_samples: int,
    grid_side_points: int,
    fock_cutoff: int,
    grid_half_size: float | None = None,
    padding: float = 3.0,
    coherent_alpha: complex = 0 + 0j,
    number_k: int = 0,
    init_method: str = "alpha_bar",
    entropy_tau: float = 0.0,
    max_iters: int = 500,
    tol: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    plot: bool = False,
) -> Dict[str, object]:
    """End-to-end MLE driver following the requested workflow.

    Workflow:
        1. Sample heterodyne (dual-homodyne) data from a chosen test state.
        2. (Optionally adaptive) choose grid_half_size.
        3. build_grid -> α grid points.
        4. histogram_on_grid -> counts on grid (POVM discretisation).
        5. build_coherent_state_matrix -> C.
        6. initialise_rho -> initial density matrix guess.
        7. mle_rrr loop.

    Args:
        state_type: 'coherent' or 'number'. (Generic states can be added later.)
        num_samples: number of heterodyne samples to draw.
        grid_side_points: number of points per axis (G). Total bins = G^2.
        fock_cutoff: largest retained Fock number N.
        grid_half_size: half-side of square window in (y1,y2); if None adaptive.
        padding: extra margin used only when grid_half_size is None.
        coherent_alpha: α0 for coherent test state.
        number_k: k for number state test.
        init_method: 'alpha_bar' or 'maximally_mixed'.
        entropy_tau: entropy regularisation weight τ ≥ 0.
        max_iters: maximum EM (RρR) iterations.
        tol: relative NLL improvement tolerance.
        rng: optional numpy Generator.
        verbose: print progress if True.

    Returns:
        dict with keys: rho, nll_history, counts_2d, bin_area, C, grid_half_size, alpha_grid,
                        state_type, true_rho (if available), fidelity (if true_rho known),
                        coverage_hist, counts_in, Q_hat.
    """
    if rng is None:
        rng = np.random.default_rng()

    state_key = state_type.strip().lower()
    if state_key not in {"coherent", "number"}:
        raise ValueError("state_type must be 'coherent' or 'number'")

    # 1. Sample data
    if state_key == "coherent":
        samples_alpha = sample_pure_coherent_state(coherent_alpha, num_samples, rng=rng)
    else:
        samples_alpha = sample_number_state(number_k, num_samples, rng=rng)

    # 2. Determine window size (if adaptive). We compute it *outside* histogram_on_grid
    #    so the same value is reused for building grid & histogram.
    if grid_half_size is None:
        grid_half_size = adaptive_grid_half_size(samples_alpha, padding=padding)

    # 3. Build grid (α points and coordinate grids)
    alphas_grid, _, _ = build_grid(grid_half_size, grid_side_points)

    # 4. Histogram counts on grid. We pass the explicit grid_half_size to avoid mismatch.
    Q_hat, counts_2d, coverage_hist, counts_in, bin_area = histogram_on_grid(
        samples_alpha, grid_half_size, grid_side_points, padding=padding
    )

    # 5. Coherent state matrix C (flatten α grid in C-order to match counts flatten ordering)
    C = build_coherent_state_matrix(alphas_grid.ravel(order="C"), fock_cutoff)

    # 6. Initial density matrix
    if init_method.strip().lower().replace('-', '_') == "alpha_bar":
        alpha_bar = samples_alpha.mean()
        init_rho = initialise_rho(fock_cutoff, method="alpha_bar", alpha_bar=alpha_bar)
    else:
        init_rho = initialise_rho(fock_cutoff, method="maximally_mixed")

    # 7. MLE loop
    rho_est, nll_history = mle_rrr(
        counts_2d=counts_2d,
        C=C,
        bin_area=bin_area,
        init_rho=init_rho,
        max_iters=max_iters,
        tol=tol,
        entropy_tau=entropy_tau,
    )

    # Build (optional) true rho to assess fidelity (within truncation)
    true_rho = None
    fidelity = None
    if state_key == "coherent":
        true_rho, tail = true_rho_coherent(coherent_alpha, fock_cutoff)
        # For pure true state: F = <ψ|ρ|ψ>
        # Extract coherent amplitudes |ψ> as principal eigenvector of true_rho (rank-1)
        vals, vecs = np.linalg.eigh(true_rho)
        idx = np.argmax(vals)
        psi = vecs[:, idx]
        fidelity = float(np.real(np.vdot(psi, rho_est @ psi)))
        # Adjust for truncation tail not represented (optional note)
    elif state_key == "number":
        if number_k > fock_cutoff:
            # Entire true state outside truncation -> fidelity undefined / 0
            fidelity = 0.0
        else:
            # True pure number state |k>
            basis_vec = np.zeros(fock_cutoff + 1, dtype=np.complex128)
            basis_vec[number_k] = 1.0
            fidelity = float(np.real(np.vdot(basis_vec, rho_est @ basis_vec)))
            true_rho = np.outer(basis_vec, basis_vec.conjugate())

    if verbose:
        print("--- MLE Summary ---")
        print(f"State type          : {state_key}")
        if state_key == 'coherent':
            print(f"True alpha          : {coherent_alpha:.5g}")
        else:
            print(f"Number state k      : {number_k}")
        print(f"Samples (total)     : {num_samples}")
        print(f"Grid side points    : {grid_side_points} (bins={grid_side_points**2})")
        print(f"Grid half-size (y)  : {grid_half_size:.5g}")
        print(f"Heterodyne coverage : {coverage_hist*100:.2f}% of samples inside window")
        print(f"Counts inside       : {counts_in}")
        print(f"Bin area (d^2α)     : {bin_area:.5g}")
        print(f"Fock cutoff N       : {fock_cutoff}")
        print(f"Init method         : {init_method}")
        print(f"Entropy τ           : {entropy_tau}")
        print(f"Iterations run      : {len(nll_history)-1}")
        print(f"Final NLL           : {nll_history[-1]:.6f}")
        if fidelity is not None:
            print(f"Truncated fidelity  : {fidelity:.6f}")
        print("--------------------")

    result: Dict[str, object] = {
        "rho": rho_est,
        "nll_history": nll_history,
        "counts_2d": counts_2d,
        "bin_area": bin_area,
        "C": C,
        "grid_half_size": grid_half_size,
        "alpha_grid": alphas_grid,
        "state_type": state_key,
        "true_rho": true_rho,
        "fidelity": fidelity,
        "coverage_hist": coverage_hist,
        "counts_in": counts_in,
        "Q_hat": Q_hat,
        "init_rho": init_rho,
    }

    # 8. Plotting (3x3 panel) if requested and matplotlib available
    if plot:
        try:
            import os
            import matplotlib.pyplot as plt
            from matplotlib.colors import TwoSlopeNorm

            G = grid_side_points
            # Reconstructed Q on grid from rho_est
            Q_recon = Q_from_rho(alphas_grid, rho_est)

            # True Q
            if state_key == "coherent":
                Q_true = analytic_Q_coherent_state(alphas_grid, coherent_alpha)
                identifier = f"a={coherent_alpha.real:.2f}{'+' if coherent_alpha.imag>=0 else ''}{coherent_alpha.imag:.2f}i"
            else:
                Q_true = analytic_Q_number_state(alphas_grid, number_k)
                identifier = f"n={number_k}"

            # Figure & axes
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            extent = (-grid_half_size, grid_half_size, -grid_half_size, grid_half_size)

            def im(ax, data, title, cmap="viridis", vmin=None, vmax=None):
                im_ = ax.imshow(data, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                ax.set_title(title)
                ax.set_xlabel("y1")
                ax.set_ylabel("y2")
                fig.colorbar(im_, ax=ax, fraction=0.046, pad=0.04)

            # Row 1: True Q, Histogram Q_hat, Reconstructed Q
            if true_rho is not None:
                im(axes[0,0], Q_true, "True Q(α)")
            else:
                axes[0,0].set_title("True Q(α) unavailable")
                axes[0,0].axis('off')
            im(axes[0,1], Q_hat, "Histogram Q̂(α)")
            im(axes[0,2], Q_recon, "Reconstructed Q(α)")

            # Row 2: True rho, Reconstructed rho, Diagonal comparison
            if true_rho is not None:
                v_rho = np.max(np.abs(true_rho))
                rtrue = np.abs(true_rho)
                im2 = axes[1,0].imshow(rtrue, origin='lower', cmap='viridis', vmin=0, vmax=v_rho)
                axes[1,0].set_title("|ρ_true|")
                axes[1,0].set_xlabel('n')
                axes[1,0].set_ylabel('m')
                fig.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)
            else:
                axes[1,0].set_title("ρ_true unavailable")
                axes[1,0].axis('off')

            v_est = np.max(np.abs(rho_est))
            im3 = axes[1,1].imshow(np.abs(rho_est), origin='lower', cmap='viridis', vmin=0, vmax=v_est)
            axes[1,1].set_title("|ρ_est|")
            axes[1,1].set_xlabel('n')
            axes[1,1].set_ylabel('m')
            fig.colorbar(im3, ax=axes[1,1], fraction=0.046, pad=0.04)

            ax_diag = axes[1,2]
            diag_est = np.real(np.diag(rho_est))
            if true_rho is not None:
                diag_true = np.real(np.diag(true_rho))
                idx = np.arange(len(diag_est))
                width = 0.4
                ax_diag.bar(idx - width/2, diag_true, width=width, label='true')
                ax_diag.bar(idx + width/2, diag_est, width=width, label='est')
                ax_diag.legend()
            else:
                idx = np.arange(len(diag_est))
                ax_diag.bar(idx, diag_est, width=0.6, label='est')
                ax_diag.legend()
            ax_diag.set_title("Diag(ρ) comparison")
            ax_diag.set_xlabel('n')
            ax_diag.set_ylabel('Population')

            # Row 3 (updated): NLL history, difference, initial rho
            axes[2,0].plot(range(len(nll_history)), nll_history, marker='o', ms=3)
            axes[2,0].set_title("NLL vs iteration")
            axes[2,0].set_xlabel('Iteration')
            axes[2,0].set_ylabel('Poisson NLL')
            axes[2,0].grid(alpha=0.3)

            ax_diff = axes[2,1]
            if true_rho is not None:
                diff = (rho_est - true_rho).real
                vmax = np.max(np.abs(diff)) if diff.size else 1.0
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
                im_diff = ax_diff.imshow(diff, origin='lower', cmap='RdBu_r', norm=norm)
                ax_diff.set_title("Re(ρ_est - ρ_true)")
                ax_diff.set_xlabel('n')
                ax_diff.set_ylabel('m')
                fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
            else:
                ax_diff.set_title("No true ρ for diff")
                ax_diff.axis('off')

            im_init = axes[2,2].imshow(np.abs(init_rho), origin='lower', cmap='viridis')
            axes[2,2].set_title("|ρ_0|")
            axes[2,2].set_xlabel('n')
            axes[2,2].set_ylabel('m')
            fig.colorbar(im_init, ax=axes[2,2], fraction=0.046, pad=0.04)

            fig.suptitle(f"MLE Reconstruction ({identifier})  samples={num_samples}", fontsize=14)
            fig.tight_layout(rect=[0,0,1,0.97])

            results_dir = os.path.join(os.path.dirname(__file__), 'results')
            os.makedirs(results_dir, exist_ok=True)
            filename = f"{identifier}.png".replace(' ', '')
            filepath = os.path.join(results_dir, filename)
            fig.savefig(filepath, dpi=160)
            plt.close(fig)
            result['figure_path'] = filepath
            if verbose:
                print(f"Saved figure -> {filepath}")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"Plotting failed: {e}")

    return result


def _add_cli_arguments(parser):  # helper to keep __main__ concise
    import argparse
    parser.add_argument("--state", choices=["coherent", "number"], default="coherent", help="Test state type")
    parser.add_argument("-N", "--fock-cutoff", type=int, default=5, help="Fock cutoff N")
    parser.add_argument("-S", "--num-samples", type=int, default=500, help="Number of heterodyne samples")
    parser.add_argument("-G", "--grid-side-points", type=int, default=41, help="Grid side points (G)")
    parser.add_argument("--grid-half-size", type=float, default=None, help="Override grid half-size in y-quadratures")
    parser.add_argument("--padding", type=float, default=3.0, help="Padding when adaptive grid used")
    parser.add_argument("--alpha-real", type=float, default=0.5, help="Re(alpha0) for coherent state")
    parser.add_argument("--alpha-imag", type=float, default=0.0, help="Im(alpha0) for coherent state")
    parser.add_argument("--k", type=int, default=1, help="Photon number k for number state")
    parser.add_argument("--init", choices=["alpha_bar", "maximally_mixed"], default="alpha_bar", help="Initialisation method")
    parser.add_argument("--entropy-tau", type=float, default=0.0, help="Entropy regularisation weight τ")
    parser.add_argument("--max-iters", type=int, default=500, help="Maximum MLE iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Relative NLL improvement tolerance")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress printed summary")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Generate 3x3 diagnostic figure and save to results/")
    return parser


def main_cli():  # entry point
    import argparse
    parser = _add_cli_arguments(argparse.ArgumentParser(description="Heterodyne MLE (RρR) from scratch"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed) if args.seed is not None else None

    coherent_alpha = complex(args.alpha_real, args.alpha_imag)
    result = run_mle_workflow(
        state_type=args.state,
        num_samples=args.num_samples,
        grid_side_points=args.grid_side_points,
        fock_cutoff=args.fock_cutoff,
        grid_half_size=args.grid_half_size,
        padding=args.padding,
        coherent_alpha=coherent_alpha,
        number_k=args.k,
        init_method=args.init,
        entropy_tau=args.entropy_tau,
        max_iters=args.max_iters,
        tol=args.tol,
        rng=rng,
        verbose=not args.no_verbose,
    plot=args.plot,
    )

    # Optional: save outputs (kept minimal; user can extend)
    # Example: np.savez('mle_result.npz', rho=result['rho'], nll_history=result['nll_history'])
    return result


#################### DEFAULT TEST BATTERY ###################

def run_default_tests(
    *,
    num_samples: int = 500,
    fock_cutoff: int = 8,
    grid_side_points: int = 31,
    coherent_range: float = 1.0,
    iters: int = 300,
    tol: float = 1e-6,
    entropy_tau: float = 0.0,
    seed: int | None = 1234,
    verbose: bool = True,
    make_panel_figs: bool = False,
    summary_filename: str = "test_summary.png",
    init_method: str = "alpha_bar",
) -> Dict[str, object]:
    """Run a batch of MLE reconstructions:

    * Number (Fock) states k=0..4.
    * 5x5 coherent state grid over Re/Im in linspace(-R, R, 5) (includes origin).

    Produces a 1x2 summary figure: left panel summarises Fock states (fidelity + iterations),
    right panel shows a fidelity heatmap for coherent states annotated with iteration counts.

    Args:
        num_samples: shots per state.
        fock_cutoff: truncation N (>= 4 recommended for number states up to 4; using 8 default for headroom).
        grid_side_points: per reconstruction grid; adaptively sized half-width.
        coherent_range: R so that Re,Im ∈ linspace(-R,R,5).
        iters: max iterations per MLE run.
        tol: relative NLL tolerance.
        entropy_tau: entropy regularisation weight.
        seed: RNG seed.
        verbose: print progress.
        make_panel_figs: if True, produce individual 3x3 diagnostic panels for each state.
        summary_filename: file name for summary figure in results/.

    Returns:
        Dict with collected metrics.
    """
    rng = np.random.default_rng(seed)
    results: Dict[str, object] = {}

    # Storage for metrics
    fock_fidelities = []
    fock_iterations = []
    fock_final_nll = []

    # --- Fock states ---
    for k in range(5):
        if verbose:
            print(f"[Batch] Fock state k={k}")
        out = run_mle_workflow(
            state_type="number",
            num_samples=num_samples,
            grid_side_points=grid_side_points,
            fock_cutoff=fock_cutoff,
            grid_half_size=None,
            padding=3.0,
            coherent_alpha=0+0j,
            number_k=k,
            init_method=init_method,
            entropy_tau=entropy_tau,
            max_iters=iters,
            tol=tol,
            rng=rng,
            verbose=False,
            plot=make_panel_figs,
        )
        fock_fidelities.append(out.get("fidelity", np.nan))
        fock_iterations.append(len(out.get("nll_history", [])) - 1)
        fock_final_nll.append(out.get("nll_history", [np.nan])[-1])

    # --- Coherent states grid ---
    re_vals = np.linspace(-coherent_range, coherent_range, 5)
    im_vals = np.linspace(-coherent_range, coherent_range, 5)
    coh_fidelity = np.zeros((5,5), dtype=float)
    coh_iters = np.zeros((5,5), dtype=float)
    coh_final_nll = np.zeros((5,5), dtype=float)

    for i, re in enumerate(re_vals):
        for j, im in enumerate(im_vals):
            alpha0 = complex(re, im)
            if verbose:
                print(f"[Batch] Coherent α={alpha0:.3g}")
            out = run_mle_workflow(
                state_type="coherent",
                num_samples=num_samples,
                grid_side_points=grid_side_points,
                fock_cutoff=fock_cutoff,
                grid_half_size=None,
                padding=3.0,
                coherent_alpha=alpha0,
                number_k=0,
                init_method=init_method,
                entropy_tau=entropy_tau,
                max_iters=iters,
                tol=tol,
                rng=rng,
                verbose=False,
                plot=make_panel_figs,
            )
            coh_fidelity[i,j] = out.get("fidelity", np.nan)
            coh_iters[i,j] = len(out.get("nll_history", [])) - 1
            coh_final_nll[i,j] = out.get("nll_history", [np.nan])[-1]

    results.update({
        "fock_fidelity": np.array(fock_fidelities),
        "fock_iterations": np.array(fock_iterations),
        "fock_final_nll": np.array(fock_final_nll),
        "coherent_fidelity": coh_fidelity,
        "coherent_iterations": coh_iters,
        "coherent_final_nll": coh_final_nll,
        "re_vals": re_vals,
        "im_vals": im_vals,
    })

    # --- Summary figure (only if plotting enabled) ---
    if make_panel_figs:
        try:
            import os
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12,5))

            # Left: Fock states fidelity + iterations
            ks = np.arange(5)
            bar = ax_left.bar(ks, fock_fidelities, color='tab:blue', alpha=0.7, label='Fidelity')
            ax_left.set_ylim(0, 1.05)
            ax_left.set_xlabel('Fock k')
            ax_left.set_ylabel('Fidelity')
            ax_left.set_title('Fock states reconstruction')
            # iterations as line
            ax2 = ax_left.twinx()
            ax2.plot(ks, fock_iterations, color='tab:orange', marker='o', label='Iterations')
            ax2.set_ylabel('Iterations')
            # Combine legends
            lines, labels = [], []
            for h in [bar]:
                lines.append(h)
                labels.append('Fidelity')
            l2, = ax2.plot([], [], color='tab:orange', marker='o')
            lines.append(l2)
            labels.append('Iterations')
            ax_left.legend(lines, labels, loc='lower center')

            # Right: coherent fidelity heatmap annotated with iterations
            im_show = ax_right.imshow(coh_fidelity, origin='lower', cmap='viridis',
                                      extent=(re_vals[0], re_vals[-1], im_vals[0], im_vals[-1]),
                                      aspect='equal', vmin=0, vmax=1)
            ax_right.set_xlabel('Re(α)')
            ax_right.set_ylabel('Im(α)')
            ax_right.set_title('Coherent grid fidelity (iterations)')
            fig.colorbar(im_show, ax=ax_right, fraction=0.046, pad=0.04, label='Fidelity')

            # Annotate iterations
            for ii, re in enumerate(re_vals):
                for jj, imv in enumerate(im_vals):
                    ax_right.text(re, imv, f"{int(coh_iters[ii,jj])}", ha='center', va='center', color='white' if coh_fidelity[ii,jj] < 0.5 else 'black', fontsize=8)

            fig.suptitle(f'MLE Batch Summary (Fock 0-4 & 5x5 Coherent Grid)  samples/state={num_samples}  init={init_method}', fontsize=14)
            fig.tight_layout(rect=[0,0,1,0.95])
            results_dir = os.path.join(os.path.dirname(__file__), 'results')
            os.makedirs(results_dir, exist_ok=True)
            out_path = os.path.join(results_dir, summary_filename)
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            results['summary_figure'] = out_path
            if verbose:
                print(f"Saved summary figure -> {out_path}")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"Failed to create summary figure: {e}")

    if verbose:
        print("Batch test complete.")
    return results


if __name__ == "__main__":
    import sys
    # After all definitions so run_default_tests is available.
    if len(sys.argv) == 1:
        run_default_tests(make_panel_figs=True, verbose=True)
    else:
        main_cli()

