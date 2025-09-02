import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.special import gammaln
from scipy.linalg import eigh
from scipy.linalg.blas import zher, zgemm

# ---------------------------------------------------------------------------
# Lightweight logging utility (JavaScript-style console.log interface)
# ---------------------------------------------------------------------------
class console:  # lower-case to mirror user request for console.log
    enabled: bool = True  # master switch
    level: int = 1        # 0 = silent, 1 = minimal (iterations), 2 = verbose (previous default)

    @staticmethod
    def log(message: str, lvl: int = 1, end: str = "\n", flush: bool = False):
        """Minimal/info logging (use lvl=1 for high-level iteration progress).

        Args:
            message: text to print (auto prepended with [LOG]).
            lvl: required verbosity level.
            end: line termination (can pass '\r' for inline updates).
            flush: force stream flush (useful for inline progress).
        """
        if console.enabled and console.level >= lvl:
            # Use print directly so that 'end' works; keep tag consistent.
            print(f"[LOG] {message}", end=end, flush=flush)

    @staticmethod
    def debug(message: str):
        """Detailed helper-function logging (set console.level >=2 to see)."""
        console.log(message, lvl=2)

# Optional alias with capital if ever used elsewhere
Console = console

# Allow runtime override (e.g., set console.enabled = False externally if noisy)


################ CHANNEL RECONSTRUCTION MACHINERY ################

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
    console.debug(f"factorial_array: start (n={n}, exact={exact}, return_log={return_log})")
    if n < 0:
        raise ValueError("n must be >= 0")

    if return_log:
        ks = np.arange(n + 1, dtype=int)
        out = gammaln(ks + 1).astype(float)
        console.debug(f"factorial_array: produced log factorial array shape={out.shape}")
        return out

    if exact:
        if n == 0:
            console.debug("factorial_array: n==0 -> [1]")
            return np.array([1], dtype=object)
        vals = np.arange(1, n + 1, dtype=object)
        acc = np.multiply.accumulate(vals, dtype=object)
        out = np.concatenate(([1], acc))
        console.debug(f"factorial_array: produced exact factorials length={out.size}")
        return out

    ks = np.arange(n + 1, dtype=int)
    out = np.exp(gammaln(ks + 1)).astype(float)
    console.debug(f"factorial_array: produced float factorials length={out.size}")
    return out

def adaptive_grid_half_size(
    samples: List[np.ndarray],
    percentile: float = 0.975,
    padding: float = 1.3,
    min_half_size: float = 3.5,
    max_half_size: Optional[float] = None,
) -> float:
    """
    Choose a square grid half-size L for the shared output grid by pooling all outputs.

    Inputs can be either:
      - complex alpha samples (shape (N,) or (N,1), complex), where alpha = (y1 + i y2)/√2, OR
      - real y-plane samples (shape (N, 2)) giving (y1, y2) directly.

    We compute the percentile of the y-plane radius r = sqrt(y1^2 + y2^2) over all samples,
    then inflate by `padding`. A small `min_half_size` protects against too-tight grids when
    data are sparse; `max_half_size` can cap extremes.

    Returns:
        L (float): grid half-size in y-units so that the grid is [-L, L] × [-L, L].
    """
    console.debug(f"adaptive_grid_half_size: start with {len(samples)} sample arrays, percentile={percentile}, padding={padding}")
    if not samples:
        raise ValueError("adaptive_grid_half_size: `samples` must be a non-empty list of arrays.")

    y1_all = []
    y2_all = []
    for S in samples:
        S = np.asarray(S)
        # Case A: complex alpha samples
        if np.iscomplexobj(S):
            a = S.reshape(-1).astype(np.complex128)
            y1_all.append(np.sqrt(2.0) * a.real)
            y2_all.append(np.sqrt(2.0) * a.imag)
        else:
            # Case B: real y-plane samples (N,2)
            if S.ndim != 2 or S.shape[1] != 2:
                raise ValueError("Each real sample array must have shape (N, 2) for (y1, y2).")
            y1_all.append(S[:, 0].astype(float))
            y2_all.append(S[:, 1].astype(float))

    y1 = np.concatenate(y1_all, axis=0)
    y2 = np.concatenate(y2_all, axis=0)
    r = np.sqrt(y1**2 + y2**2)

    # Robust size from percentile of radius
    r_q = np.quantile(r, percentile)
    L = max(r_q * padding, float(min_half_size))
    if max_half_size is not None:
        L = min(L, float(max_half_size))
    console.debug(f"adaptive_grid_half_size: computed half-size L={L:.4f}")
    return float(L)

def grid_bin_width_and_area(grid_half_size: float, grid_side_points: int) -> Tuple[float, float]:
    """
    Convenience: compute the Cartesian bin width Δy and the heterodyne bin area ΔA.

    Notes:
        alpha = (y1 + i y2)/√2  ⇒  d^2alpha = (1/2) d y1 d y2
        With square bins of side Δy, the area in alpha-units is ΔA = (Δy^2)/2.

    Returns:
        (delta_y, bin_area_alpha) where bin_area_alpha = ΔA for use in E_j = (ΔA/π)|alpha_j><alpha_j|.
    """
    console.debug(f"grid_bin_width_and_area: start (L={grid_half_size}, G={grid_side_points})")
    if grid_side_points < 2:
        raise ValueError("grid_side_points must be at least 2.")
    L = float(grid_half_size)
    G = int(grid_side_points)
    delta_y = 2.0 * L / (G - 1)
    bin_area_alpha = (delta_y ** 2) / 2.0
    console.debug(f"grid_bin_width_and_area: delta_y={delta_y:.4f}, bin_area_alpha={bin_area_alpha:.6e}")
    return delta_y, bin_area_alpha

def build_grid(grid_half_size: float, grid_side_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a square Cartesian grid in y-space and its corresponding coherent points alpha.

    Grid:
        y_axis = linspace(-L, L, G)
        (Y1, Y2) = meshgrid(y_axis, y_axis, indexing='xy')
        alpha_grid = (Y1 + i Y2)/√2   (canonical dual-homodyne mapping)

    Args:
        grid_half_size: L > 0, the half-side of the square grid in y-units.
        grid_side_points: G >= 2, number of points per side (total bins M = G^2).

    Returns:
        alpha_grid_flat: complex array of shape (G*G,), alpha at each grid bin center (row-major).
        Y1: float array of shape (G, G), x-quadrature mesh.
        Y2: float array of shape (G, G), p-quadrature mesh.
    """
    console.debug(f"build_grid: start (L={grid_half_size}, G={grid_side_points})")
    if grid_half_size <= 0:
        raise ValueError("grid_half_size must be positive.")
    if grid_side_points < 2:
        raise ValueError("grid_side_points must be at least 2.")

    L = float(grid_half_size)
    G = int(grid_side_points)

    y_axis = np.linspace(-L, L, G, dtype=float)
    Y1, Y2 = np.meshgrid(y_axis, y_axis, indexing="xy")

    alpha_grid = (Y1 + 1j * Y2) / np.sqrt(2.0)  # canonical mapping
    alpha_grid_flat = alpha_grid.reshape(-1)    # row-major flatten

    console.debug(f"build_grid: built grid with {alpha_grid_flat.size} points")
    return alpha_grid_flat, Y1, Y2

def compute_channel_probabilites(
    J: np.ndarray,
    C_inputs: np.ndarray,
    C_outputs: np.ndarray,
    bin_area: float,
    eps: float = 1e-30
) -> np.ndarray:
    """
    Forward model for a *channel*: p_{i,j}(J) = Tr[ J * (E_j ⊗ |alpha_i><alpha_i|^T) ],
    where E_j = (bin_area/π) |alpha_j><alpha_j| (heterodyne POVM per output bin).
    
    Notes:
      - S is the number of input states, M = G^2 is the number of output bins.
      - Uses the identity p_{ij} = (ΔA/π) Σ_{m,n,p,q} J[(mn),(pq)] A^{(i)}_{nm} B^{(j)}_{pq},
        with A^{(i)}_{nm} = a_n a_m* and B^{(j)}_{pq} = b_p* b_q.
      - Floors probabilities at `eps` to avoid log(0) in the NLL.
    
    Args:
        J: Choi matrix of the channel, shape (d*d, d*d)
        C_inputs: Coherent state overlaps for input states, shape (d, S). Columns are a^(i).
        C_outputs: Coherent state overlaps for output states, shape (d, M). Columns are b^(j).
        bin_area: Area of each output bin in phase space (dual-homodyne grid).
        eps: Small constant to avoid numerical issues.

    Returns:
        p: Probability matrix, shape (S, M), where p[i][j] = p_{i,j}(J).
    """
    # Basic shape checks
    console.debug("compute_channel_probabilites: start")
    d_in = C_inputs.shape[0]
    d_out = C_outputs.shape[0]
    if d_in != d_out:
        raise ValueError(f"C_inputs and C_outputs must have same first dim (d). Got {d_in} vs {d_out}.")
    d = d_in
    S = C_inputs.shape[1]
    M = C_outputs.shape[1]
    if J.shape != (d*d, d*d):
        raise ValueError(f"J must have shape ({d*d}, {d*d}); got {J.shape}.")

    # Ensure contiguous complex arrays
    J = np.ascontiguousarray(J, dtype=np.complex128)
    C_in = np.ascontiguousarray(C_inputs, dtype=np.complex128)
    C_out = np.ascontiguousarray(C_outputs, dtype=np.complex128)

    # Rank-1 factors for inputs: A^{(i)}_{nm} = a_n a_m*
    # Build A_nm with shape (S, d, d) indexed as [i, n, m]
    a = C_in.T  # (S, d)
    A_nm = a[:, :, None] * np.conj(a)[:, None, :]  # (S, d, d) : [i, n, m]
    # We need A indexed as [i, m, n] to match J[m,n,*,*]
    A_mn = np.swapaxes(A_nm, -2, -1)  # (S, d, d) : [i, m, n]

    # Rank-1 factors for outputs: B^{(j)}_{pq} = b_p* b_q
    b = C_out.T  # (M, d)
    B_pq = np.conj(b)[:, :, None] * b[:, None, :]  # (M, d, d) : [j, p, q]

    # Reshape J to 4-tensor J4[m,n,p,q] consistent with (mn),(pq) -> row-major mapping
    J4 = J.reshape(d, d, d, d)

    # Contract over (m,n): T[i,p,q] = Σ_{m,n} J[m,n,p,q] * A_mn[i,m,n]
    # -> einsum indices: 'mnpq,imn->ipq'
    T = np.einsum('mnpq,imn->ipq', J4, A_mn, optimize=True)  # (S, d, d)

    # Now contract T with B over (p,q): p[i,j] = (bin_area/π) * Σ_{p,q} T[i,p,q] * B[j,p,q]
    # -> einsum indices: 'ipq,jpq->ij'
    scale = float(bin_area) / np.pi
    p = scale * np.einsum('ipq,jpq->ij', T, B_pq, optimize=True)  # (S, M)

    # Numerical guards: kill tiny imaginary parts, clamp to eps
    if np.iscomplexobj(p):
        p = p.real
    # Occasionally round-off may produce tiny negatives; clamp then floor
    p = np.maximum(p, eps)

    console.debug(f"compute_channel_probabilites: done (S={S}, M={M})")
    return p

def neg_log_likelihood_poisson_multiinput(
    counts_ij: np.ndarray,
    p_ij: np.ndarray
) -> float:
    """
    Poisson-factorised negative log-likelihood over inputs i and bins j.

    Model:
        n_{ij} ~ Poisson(N_i * p_{ij}),
    where N_i = sum_j n_{ij}. If rows of p_ij are normalised (sum_j p_{ij} = 1),
    the term sum_j N_i * p_{ij} reduces to sum_i N_i (a constant w.r.t. the parameters).

    Args:
        counts_ij: Measured counts, shape (S, M).
        p_ij: Probability matrix from the channel model, shape (S, M). \sum_j p_ij = 1 for each i.

    Returns:
        nll: Negative log-likelihood value (without the constant term).
    """
    console.debug("neg_log_likelihood_poisson_multiinput: start")
    counts = np.asarray(counts_ij, dtype=np.float64) # Cast from integer to floats for numerics
    p = np.asarray(p_ij, dtype=np.float64)

    if counts.shape != p.shape:
        raise ValueError(f"counts_ij and p_ij must have the same shape; got {counts.shape} vs {p.shape}.")

    if np.any(counts < 0):
        raise ValueError("counts_ij must be nonnegative.")

    # Total shots per input (row)
    N_i = counts.sum(axis=1, keepdims=True)  # shape (S, 1)

    # Numerical guard to avoid log(0); keep extremely small floor
    # Use machine tiny to avoid underflow to -inf in log.
    eps = np.finfo(np.float64).tiny
    p_safe = np.maximum(p, eps)

    # NLL (dropping Σ log n_{ij}! constant)
    #   L = Σ_{i,j} [ N_i * p_{ij} - n_{ij} * log p_{ij} ]
    nll = np.sum(N_i * p_safe) - np.sum(counts * np.log(p_safe))

    # Return as plain Python float
    val = float(nll)
    console.debug(f"neg_log_likelihood_poisson_multiinput: nll={val:.6f}")
    return val

def project_CPTP(
    J: np.ndarray,
    d_in: Optional[int] = None,
    d_out: Optional[int] = None,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    eig_clip: float = 1e-12,
    verbose: bool = False,
    log_every: int = 5,
    inline: bool = True,
    return_diagnostics: bool = False,
) -> np.ndarray:
    """
    Project an arbitrary Choi matrix J onto the CPTP set under the Frobenius norm
    using Dykstra's alternating projections.

    Convention: UNNORMALIZED Choi, i.e. Tr_out X = I_{d_in}.

    Args:
        J:        Choi matrix, shape (N, N), with N = d_in * d_out.
        d_in:     Input dimension. If None, inferred assuming d_in == d_out == sqrt(N).
        d_out:    Output dimension. If None and d_in is given, inferred as N // d_in.
        max_iter: Maximum Dykstra iterations.
        tol:      Stopping tolerance (TP residual, min eigenvalue, relative change).
        eig_clip: After PSD projection, zero out eigenvalues below this (numerical dust).
        verbose:  If True, prints diagnostics every few iterations.

    Returns:
        If return_diagnostics is False (default):
            X: The CPTP projection of J, same shape and dtype complex128.
        If return_diagnostics is True:
            (X, diagnostics) where diagnostics is a dict containing:
                - 'iterations': number of Dykstra iterations executed.
                - 'converged': bool indicating stopping by tolerance (vs max_iter).
                - 'tp_residuals': np.ndarray of trace-preserving residual norms per iter.
                - 'min_eigs': np.ndarray of minimum eigenvalue each iter.
                - 'rel_changes': np.ndarray of relative Frobenius changes each iter.
                - 'stopping_reason': textual description.
                - 'tolerance': the tolerance used.
    """

    # ---------- dimension handling ----------
    console.debug("project_CPTP: start")
    J = np.asarray(J, dtype=np.complex128, order="C")
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be square; got shape {J.shape}.")
    N = J.shape[0]

    if d_in is None and d_out is None:
        # Assume square channel (common in this file): d_in == d_out == sqrt(N)
        rt = int(round(np.sqrt(N)))
        if rt * rt != N:
            raise ValueError(
                "Cannot infer (d_in, d_out): N is not a perfect square. "
                "Provide d_in (and optionally d_out)."
            )
        d_in = rt
        d_out = rt
    elif d_in is not None and d_out is None:
        if N % d_in != 0:
            raise ValueError(f"N={N} is not divisible by d_in={d_in}.")
        d_out = N // d_in
    elif d_in is None and d_out is not None:
        if N % d_out != 0:
            raise ValueError(f"N={N} is not divisible by d_out={d_out}.")
        d_in = N // d_out
    else:
        if d_in * d_out != N:
            raise ValueError(f"d_in * d_out = {d_in * d_out} != N={N}.")

    d_in = int(d_in)
    d_out = int(d_out)

    # ---------- helpers (closed-form projections) ----------
    def partial_trace_out(X: np.ndarray) -> np.ndarray:
        """
        Tr_out X: view X as (d_in, d_out, d_in, d_out) and trace over the 2nd/4th axes.
        Returns (d_in, d_in).
        """
        X4 = X.reshape(d_in, d_out, d_in, d_out)
        # einsum index mapping: sum over output index k => "ikjk->ij"
        return np.einsum("ikjk->ij", X4, optimize=True)

    def project_TP(Y: np.ndarray) -> np.ndarray:
        """
        Orthogonal projection onto { X : Tr_out X = I_{d_in} }.
        P(Y) = Y - (I_out ⊗ Δ), with Δ = (Tr_out Y - I_{d_in}) / d_out.
        Implemented without forming an explicit Kronecker product.
        """
        Delta = (partial_trace_out(Y) - np.eye(d_in, dtype=Y.dtype)) / float(d_out)
        Y4 = Y.reshape(d_in, d_out, d_in, d_out).copy()
        I_out = np.eye(d_out, dtype=Y.dtype)
        # Subtract Δ[i,j] * I_out from each (i,j) block
        for i in range(d_in):
            for j in range(d_in):
                Y4[i, :, j, :] -= Delta[i, j] * I_out
        return Y4.reshape(N, N)

    def project_CP(Y: np.ndarray) -> np.ndarray:
        """
        Orthogonal projection onto the PSD cone:
        - Hermitize, eigen-decompose, clip negatives to 0, drop tiny positive dust (< eig_clip).
        """
        H = 0.5 * (Y + Y.conj().T)
        w, U = np.linalg.eigh(H)
        w = np.clip(w, 0.0, None)
        if eig_clip > 0.0:
            w[w < eig_clip] = 0.0
        Z = (U * w) @ U.conj().T
        return 0.5 * (Z + Z.conj().T)

    # ---------- Dykstra's algorithm ----------
    X = J.copy()
    R = np.zeros_like(X)
    S = np.zeros_like(X)
    X_prev = X.copy()

    # simple diagnostic helper
    def diagnostics(Xk: np.ndarray) -> tuple[float, float, float]:
        Tr_out = partial_trace_out(Xk)
        tp_res = float(np.linalg.norm(Tr_out - np.eye(d_in)))
        # Hermitize before eigenvalues for safety
        evals = np.linalg.eigvalsh(0.5 * (Xk + Xk.conj().T))
        min_eig = float(evals[0])
        rel_change = float(
            np.linalg.norm(Xk - X_prev) / max(1.0, np.linalg.norm(X_prev))
        )
        return tp_res, min_eig, rel_change

    tp_res_hist: List[float] = []
    min_eig_hist: List[float] = []
    rel_change_hist: List[float] = []
    stopping_reason = "max_iter_reached"
    converged = False

    for k in range(1, max_iter + 1):
        # TP step
        Yk = project_TP(X + R)
        R = X + R - Yk

        # CP step
        Zk = project_CP(Yk + S)
        S = Yk + S - Zk

        X = Zk

        # Always compute diagnostics (cheap vs projection cost)
        tp_res, min_eig, rel_change = diagnostics(X)
        tp_res_hist.append(tp_res)
        min_eig_hist.append(min_eig)
        rel_change_hist.append(rel_change)

        if verbose and (k == 1 or (log_every > 0 and k % log_every == 0)):
            msg = f"CPTP k={k:03d} tp={tp_res:.2e} minEig={min_eig:.2e} dRel={rel_change:.2e}"
            if inline:
                console.log(msg, end="\r", flush=True)
            else:
                console.log(msg)

        if tp_res <= tol and min_eig >= -tol and rel_change <= tol:
            converged = True
            stopping_reason = "tolerance_met"
            break

        X_prev = X.copy()

    # Ensure exact Hermiticity on exit
    X = 0.5 * (X + X.conj().T)

    if not return_diagnostics:
        if verbose and inline:
            console.log("", end="\n")
        console.debug(f"project_CPTP: end iterations={k} converged={converged}")
        return X

    diag = {
        "iterations": k,
        "converged": converged,
        "tp_residuals": np.asarray(tp_res_hist, dtype=float),
        "min_eigs": np.asarray(min_eig_hist, dtype=float),
        "rel_changes": np.asarray(rel_change_hist, dtype=float),
        "stopping_reason": stopping_reason,
        "tolerance": float(tol),
    }
    if verbose and inline:
        console.log("", end="\n")
    console.debug(f"project_CPTP: end (returning diagnostics) iterations={k} converged={converged}")
    return X, diag

def _coherent_matrix(alphas: np.ndarray, d: int) -> np.ndarray:
    """
    Return C of shape (d, K) with columns c_n = exp(-|alpha|^2/2) * alpha^n / sqrt(n!)
    using the canonical number basis truncation to d.
    """
    console.debug(f"_coherent_matrix: start (K={alphas.size if isinstance(alphas, np.ndarray) else 'unk'}, d={d})")
    alphas = np.asarray(alphas, dtype=np.complex128).reshape(-1)
    K = alphas.size
    n = np.arange(d, dtype=np.int64)  # 0..d-1
    # Precompute 1/sqrt(n!)
    fact = factorial_array(d - 1, exact=False)  # shape (d,)
    inv_sqrt_fact = 1.0 / np.sqrt(fact.astype(np.float64))
    # Powers alpha^n (broadcast: K x d)
    A = np.power(alphas[:, None], n[None, :])  # (K, d)
    # Prefactors
    pref = np.exp(-0.5 * np.abs(alphas) ** 2)  # (K,)
    C = (pref[:, None] * A) * inv_sqrt_fact[None, :]
    out = C.T.astype(np.complex128, copy=False)
    console.debug(f"_coherent_matrix: done shape={out.shape}")
    return out  # (d, K)

################ OPTIMISATION MACHINERY ################

def mle_projected_gradient_descent(
    counts_ij: np.ndarray,
    C_inputs: np.ndarray,
    C_outputs: np.ndarray,
    bin_area: float,
    J_initialisation_style: str = 'identity',
    transmissivity: Optional[float] = None,
    excess_noise: Optional[float] = None,
    max_iters: int = 100,
    step_init: float = 1e-3,
    min_step: float = 1e-8,
    armijo_c: float = 1e-4,
    armijo_tau: float = 0.5,
    verbose: bool = False,
    seed: Optional[int] = None,
    track_cptp_diagnostics: bool = False,
    projection_verbose: bool = False,
    projection_log_every: int = 25,
    projection_inline: bool = True,
    # --- SPG (Spectral Projected Gradient) parameters ---
    use_spg: bool = True,
    bb_alpha_min: float = 1e-6,
    bb_alpha_max: float = 1e3,
    nonmonotone_M: int = 5,
    nonmonotone_c: float = 1e-4,
    max_shrinks: int = 2,
    shrink_factor: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Perform MLE of a quantum channel using projected gradient descent.

        Projected gradient descent loop (SPG variant by default):
        - Initialise J (CPTP).
        - Iterate:
                * Compute gradient G_k.
                * Set BB spectral step α_k = <s_{k-1}, s_{k-1}> / <s_{k-1}, y_{k-1}> (clipped) with s = J_k - J_{k-1}, y = G_k - G_{k-1}.
                * Single projected gradient trial J' = Π_CPTP(J_k - α_k G_k).
                * Non-monotone acceptance against max of last M NLLs with a simple Armijo-style decrease term.
                    If it fails, shrink α (×shrink_factor) up to max_shrinks times (each requires a new projection) then accept last.
        This caps projections to ~1–(max_shrinks+1) per outer iter and usually 1.

    Args:
        counts_ij: Measured counts, shape (S, M).
        C_inputs: Coherent state overlaps for input states, shape (d, S).
        C_outputs: Coherent state overlaps for output states, shape (d, M).
        bin_area: Area of each output bin in phase space (dual-homodyne grid).
        J_initialisation_style: Style for initial Choi matrix. Options:
            - 'identity': J = |Φ+><Φ+|, where |Φ+> = (1/√d) ∑_k |k,k>.
            - 'maximally_mixed': J = I / d^2
            - 'known_Gaussian': Gaussian CPTP channel with transmissivity and excess noise (see below).
        transmissivity: Required if J_initialisation_style is 'known_Gaussian'.
        excess_noise: Required if J_initialisation_style is 'known_Gaussian'.
        max_iters: Maximum number of iterations.
        step_init: Initial step size for gradient descent.
        min_step: Minimum step size before stopping.
    (Deprecated) armijo_c / armijo_tau: retained for signature compatibility; ignored when use_spg=True.
    use_spg: If True (default) use Spectral Projected Gradient with non-monotone check; if False, legacy Armijo backtracking (removed logic).
    bb_alpha_min / bb_alpha_max: Clip bounds for BB step.
    nonmonotone_M: Window size for non-monotone reference (use max of last M NLL values).
    nonmonotone_c: Sufficient decrease coefficient in acceptance test.
    max_shrinks: Maximum additional projection attempts if acceptance fails.
    shrink_factor: Multiplicative factor for α when shrinking.
    verbose: Whether to print progress.
        seed: Random seed for reproducibility.
    projection_verbose: If True, print internal CPTP projection progress each line-search attempt.
    projection_log_every: Emit a CPTP progress line every this many Dykstra iterations.
    projection_inline: If True, progress lines overwrite (carriage return) instead of stacking.

    Returns:
        results: Dictionary containing:
            - 'J': Estimated Choi matrix of the channel, shape (d*d, d*d).
            - 'nll_history': Array of NLL values per accepted iteration (length >=1 incl. initial).
            - 'step_history': Step sizes for accepted iterations.
            - 'bt_history': Backtracking counts per accepted iteration.
            - (if track_cptp_diagnostics):
                * 'init_projection_diag': diagnostics dict from initial CPTP projection.
                * 'projection_history': list (length = #accepted iterations) of dicts with keys:
                       'outer_iter', 'attempt_iters', 'accepted_iters', 'backtracks'.
    """
    console.debug("mle_projected_gradient_descent: start")
    rng = np.random.default_rng(seed)

    # ---------- shapes & guards ----------
    counts = np.asarray(counts_ij, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts_ij must be a 2D array (S, M).")
    if bin_area <= 0:
        raise ValueError("bin_area must be positive.")

    d_in = int(C_inputs.shape[0])
    d_out = int(C_outputs.shape[0])
    if d_in != d_out:
        raise ValueError("C_inputs and C_outputs must have the same first dimension (d).")
    d = d_in
    S = int(C_inputs.shape[1])   # number of inputs
    M = int(C_outputs.shape[1])  # number of output bins

    # ---------- initial J (CPTP) ----------
    D = d * d
    def _init_J(style: str) -> np.ndarray:
        key = style.strip().lower().replace('-', '_')
        if key == 'identity':
            # Unnormalised |Ω⟩ = Σ_k |k⟩⊗|k⟩  ⇒  J = |Ω⟩⟨Ω| has Tr_out J = I  (identity channel).
            Omega = np.zeros((D,), dtype=np.complex128)
            # basis ordering: (out,in) in row-major ⇒ index = out*d + in
            for k in range(d):
                Omega[k * d + k] = 1.0
            J0 = np.outer(Omega, Omega.conjugate())
        elif key == 'maximally_mixed':
            # Completely depolarising Choi: J = I_{d^2} / d so that Tr_out J = I_in.  (Positive & TP.)
            J0 = np.eye(D, dtype=np.complex128) / float(d)
        elif key == 'known_gaussian':
            # Placeholder: could seed with a fitted Gaussian Choi; for now fall back to maximally mixed.
            if transmissivity is None or excess_noise is None:
                raise ValueError("known_Gaussian initialisation requires transmissivity and excess_noise.")
            J0 = np.eye(D, dtype=np.complex128) / float(d)
        else:
            # Small random PSD + TP projection
            X = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2*D)
            X = 0.5 * (X + X.conjugate().T)
            w, U = np.linalg.eigh(X)
            w = np.clip(w, 0.0, None)
            J0 = (U * w) @ U.conjugate().T
            J0 = project_CPTP(J0, d_in=d, d_out=d)
        # Ensure exact CPTP
        return project_CPTP(J0, d_in=d, d_out=d)

    if track_cptp_diagnostics:
        J_init_result = _init_J(J_initialisation_style)
        if isinstance(J_init_result, tuple):  # safety
            J, init_diag = J_init_result
        else:
            # Force a diagnostics projection (cheap) to have a baseline
            J, init_diag = project_CPTP(J_init_result, d_in=d, d_out=d, return_diagnostics=True)
    else:
        J = _init_J(J_initialisation_style)
        if isinstance(J, tuple):  # shouldn't happen unless user passes already tuple
            J = J[0]

    # ---------- precompute coherent rank-1 tensors ----------
    # Inputs: A^{(i)}_{nm} = a_n a_m* . We'll need A^T_{mn} = a_n a_m* as an array with axes (i,m,n).
    a = np.ascontiguousarray(C_inputs.T, dtype=np.complex128)   # (S, d)
    A_nm = a[:, :, None] * np.conjugate(a)[:, None, :]          # (S, n, m) = a_n a_m*
    A_mn = np.swapaxes(A_nm, -2, -1)                            # (S, m, n) = a_m a_n*
    A_T_mn = np.conjugate(A_mn)                                 # (S, m, n) = a_n a_m*  (i.e. A^T)

    # Outputs: B^{(j)}_{pq} = b_p* b_q
    b = np.ascontiguousarray(C_outputs.T, dtype=np.complex128)  # (M, d)
    B_pq = np.conjugate(b)[:, :, None] * b[:, None, :]          # (M, p, q)

    scale = float(bin_area) / np.pi                              # E_j scale
    N_i = counts.sum(axis=1, keepdims=True)                      # (S, 1)

    # ---------- objective helper ----------
    def _forward_p(J_mat: np.ndarray) -> np.ndarray:
        return compute_channel_probabilites(J_mat, C_inputs, C_outputs, bin_area)

    def _nll(J_mat: np.ndarray) -> float:
        p = _forward_p(J_mat)
        return neg_log_likelihood_poisson_multiinput(counts, p)

    # ---------- gradient builder (vectorised; never forms F_ij explicitly) ----------
    # Using ∇L = Σ_{i,j}(N_i - n_ij/p_ij) (E_j ⊗ A_i^T).
    def _grad(J_mat: np.ndarray, p_ij: np.ndarray) -> np.ndarray:
        eps = np.finfo(np.float64).tiny
        p_safe = np.maximum(p_ij, eps)                 # (S, M)
        W = (N_i - counts / p_safe)                    # (S, M)
        # First accumulate over j into S_ipq = Σ_j W_ij * B^{(j)}_{pq}
        S_ipq = np.einsum('jpq,ij->ipq', B_pq, W, optimize=True)     # (S, d, d)
        # Then combine with A^T: G_{mnpq} = scale * Σ_i A^T_{i,mn} * S_{i,pq}
        G4 = scale * np.einsum('imn,ipq->mnpq', A_T_mn, S_ipq, optimize=True)  # (d,d,d,d)
        G = G4.reshape(d*d, d*d)
        # Hermitize to kill any tiny asymmetry
        return 0.5 * (G + G.conjugate().T)

    # ---------- run Projected Gradient (SPG by default) ----------
    nll_history: List[float] = []
    step_history: List[float] = []
    bt_history: List[int] = []

    # initial NLL
    cur_p = _forward_p(J)
    cur_nll = neg_log_likelihood_poisson_multiinput(counts, cur_p)
    nll_history.append(float(cur_nll))
    if verbose:
        console.log(f"PGD init NLL={cur_nll:.6f}")

    projection_history: List[Dict[str, object]] = [] if track_cptp_diagnostics else []  # empty list either way

    if use_spg:
        # Spectral Projected Gradient implementation
        J_prev: Optional[np.ndarray] = None
        G_prev: Optional[np.ndarray] = None
        alpha = float(step_init)

        for it in range(1, max_iters + 1):
            G = _grad(J, cur_p)
            grad_norm_sq = float(np.vdot(G, G).real)
            if grad_norm_sq <= 1e-30:
                if verbose:
                    console.log(f"PGD iter {it:4d} tiny gradient -> stop")
                break

            # BB step (type-1) using previous iterate information
            if J_prev is not None and G_prev is not None:
                s = J - J_prev
                y = G - G_prev
                num = float(np.vdot(s, s).real)
                den = float(np.vdot(s, y).real)
                if den <= 1e-18 or num <= 0:
                    # Fallback: keep previous alpha (or re-use initial)
                    pass
                else:
                    alpha = np.clip(num / den, bb_alpha_min, bb_alpha_max)

            proj_attempt_iters: List[int] = []
            accepted = False
            trial_alpha = float(alpha)
            nll_ref = max(nll_history[-nonmonotone_M:])  # non-monotone reference
            shrinks = 0

            for attempt in range(max_shrinks + 1):  # initial + possible shrinks
                J_trial = J - trial_alpha * G
                if track_cptp_diagnostics:
                    J_trial, proj_diag = project_CPTP(
                        J_trial, d_in=d, d_out=d,
                        verbose=projection_verbose,
                        log_every=projection_log_every,
                        inline=projection_inline,
                        return_diagnostics=True
                    )
                    proj_attempt_iters.append(int(proj_diag["iterations"]))
                else:
                    J_trial = project_CPTP(
                        J_trial, d_in=d, d_out=d,
                        verbose=projection_verbose,
                        log_every=projection_log_every,
                        inline=projection_inline,
                        return_diagnostics=False
                    )

                p_trial = _forward_p(J_trial)
                nll_trial = neg_log_likelihood_poisson_multiinput(counts, p_trial)

                # Acceptance: non-monotone Armijo-style decrease vs window max
                if nll_trial <= nll_ref - nonmonotone_c * trial_alpha * grad_norm_sq:
                    accepted = True
                    break

                # Not accepted and attempts remain -> shrink & retry
                if attempt < max_shrinks:
                    trial_alpha *= shrink_factor
                    shrinks += 1
                else:
                    # Force accept last trial even if criterion not met
                    break

            # Accept (either satisfied criterion or exhausted shrinks)
            J_prev = J
            G_prev = G
            J = J_trial
            cur_p = p_trial
            cur_nll = nll_trial
            nll_history.append(float(cur_nll))
            step_history.append(trial_alpha)
            bt_history.append(shrinks)  # reuse field name; counts shrinks

            if track_cptp_diagnostics:
                projection_history.append(
                    {
                        "outer_iter": it,
                        "attempt_iters": proj_attempt_iters.copy(),
                        "accepted_iters": proj_attempt_iters[-1] if proj_attempt_iters else None,
                        "backtracks": shrinks,
                        "accepted": bool(accepted),
                    }
                )

            grad_norm = np.sqrt(grad_norm_sq)
            rel_improve = (nll_history[-2] - cur_nll) / (abs(nll_history[-2]) + 1e-18)
            inline_extra = ""
            if track_cptp_diagnostics and proj_attempt_iters:
                inline_extra = f" pIt={proj_attempt_iters[-1]}"
            status = (f"SPG i={it:03d} NLL={cur_nll:.4e} dNLL={(nll_history[-2]-cur_nll):.2e} "
                      f"relImp={rel_improve:.2e} |grad|={grad_norm:.2e} step={trial_alpha:.2e} shr={shrinks}{inline_extra}")
            end_char = "\n" if (it % 10 == 0) else "\r"
            console.log(status, end=end_char, flush=True)

            # Convergence: relative improvement tiny
            if rel_improve < 1e-9:
                if verbose:
                    console.log(f"SPG iter {it:4d} rel_improve={rel_improve:.2e} < 1e-9 stop")
                break

        # end SPG loop
    else:
        raise NotImplementedError("Legacy Armijo backtracking removed; set use_spg=True (default).")

    results = {
        "J": J,
        "nll_history": np.asarray(nll_history, dtype=float),
        "step_history": np.asarray(step_history, dtype=float),
        "bt_history": np.asarray(bt_history, dtype=int),
    }
    if track_cptp_diagnostics:
        results["init_projection_diag"] = init_diag
        results["projection_history"] = projection_history
    if verbose:
        console.log(f"PGD done iters={len(step_history)} final NLL={cur_nll:.6f}")
    console.debug("mle_projected_gradient_descent: end")
    return results

def generate_synthetic_channel_data(
    probe_states: np.ndarray, # shape (S,), complex alpha_in
    shots_per_probe: np.ndarray, # shape (S,), int
    transmissivity: float,
    excess_noise: float,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate synthetic dual-homodyne (heterodyne) data from a known Gaussian channel.

    For each input coherent state ∣alpha_in⟩, we transmit it through a phase-insensitive Gaussian channel
    with transmissivity T and excess noise ξ, then perform dual-homodyne (heterodyne)
    detection at the output. Conditional on a fixed ∣alpha_in⟩, the heterodyne outcome β is distributed as a
    circular complex Gaussian centered at \sqrt{T} alpha_in, with per-quadrature variance σ² = 1 + ξ/2 in
    shot-noise units (SNU); equivalently, p(β ∣ ∣alpha_in⟩) ∝ exp(−|β − T alpha_in|² / σ²). The unit "1" here is
    the extra vacuum noise from heterodyne (dual-homodyne) detection. If ∣alpha_in⟩ itself is drawn from a
    zero-mean (discretised, see paper) Gaussian modulation with per-quadrature variance Vmod, the unconditional variance observed
    by Bob becomes VB = T² Vmod + 1 + ξ/2 per quadrature (SNU). In "natural units" (where vacuum
    quadrature variance is 1/2), divide all SNU variances by 2 and use alpha = (y1 + i y2) / 2; under that
    convention the conditional output cloud remains centered at \sqrt{T} alpha_in with per-quadrature variance
    1/2 + ξ/4.

    Args:
        probe_states: Array of input coherent state amplitudes, shape (S,), complex128.
        shots_per_probe: Array of number of shots per probe state, shape (S,), int.
        transmissivity: Transmissivity of the Gaussian channel (0 < T <= 1).
        excess_noise: Excess noise of the Gaussian channel (ξ >= 0).
        seed: Random seed for reproducibility.
    
    Returns:
        synthetic_data : List of length S, each entry is an array of shape (N_i, 2) with dual-homodyne outcomesfor the i-th input probe state, where N_i = shots_per_probe[i].
    """
    console.debug("generate_synthetic_channel_data: start")
    a_in = np.asarray(probe_states, dtype=np.complex128).reshape(-1)
    N_shots = np.asarray(shots_per_probe, dtype=np.int64).reshape(-1)

    if a_in.shape[0] != N_shots.shape[0]:
        raise ValueError(f"probe_states and shots_per_probe must have same length; "
                         f"got {a_in.shape[0]} vs {N_shots.shape[0]}.")
    if transmissivity <= 0.0 or transmissivity > 1.0:
        raise ValueError("transmissivity must be in (0, 1].")
    if excess_noise < 0.0:
        raise ValueError("excess_noise must be ≥ 0.")
    if np.any(N_shots < 0):
        raise ValueError("shots_per_probe must be nonnegative integers.")

    rng = np.random.default_rng(seed)
    sqrtT = float(np.sqrt(transmissivity))
    # Per-quadrature variance in y-space under these conventions
    sigma2 = float(1.0 + 0.5 * excess_noise)
    sigma = float(np.sqrt(sigma2))

    # ---------- sample per probe ----------
    data: List[np.ndarray] = []
    for alpha_in, shots in zip(a_in, N_shots):
        mu_alpha = sqrtT * complex(alpha_in)             # mean in α-plane
        mu_y1 = np.sqrt(2.0) * mu_alpha.real            # map to y-space mean
        mu_y2 = np.sqrt(2.0) * mu_alpha.imag

        if shots == 0:
            samples = np.empty((0, 2), dtype=np.float64)
        else:
            samples = rng.normal(loc=[mu_y1, mu_y2], scale=sigma, size=(int(shots), 2))
            samples = samples.astype(np.float64, copy=False)

        data.append(samples)
    console.debug(f"generate_synthetic_channel_data: produced {len(data)} arrays")

    return data

def run_mle_workflow(
    # Data config
    probe_states: np.ndarray, # shape (S,), complex alpha_in
    shots_per_probe: np.ndarray, # shape (S,), int
    fock_cutoff: int,
    grid_size_points: int,
    grid_half_size: Optional[float] = None,
    adaptive_grid: bool = True,
    # Optimisation config
    max_iters: int = 100,
    step_init: float = 1e-3,
    min_step: float = 1e-8,
    armijo_c: float = 1e-4,
    armijo_tau: float = 0.5,
    J_initialisation_style: str = 'identity',
    transmissivity: Optional[float] = None,
    excess_noise: Optional[float] = None,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Complete workflow to run MLE channel reconstruction from probe states and shots.

    End-to-end pipeline:
    - Build output grid (adaptive or fixed).
    - Histogram output (dual-homodyne) data for each input probe state.
    - Build coherent-state ovelap matrices for the input probe states and the output grid.
    - Run projected gradient descent MLE.
    - Return results (J, history, diagnostics).

    Args:
        probe_states: Array of input coherent state amplitudes, shape (S,), complex128.
        shots_per_probe: Array of number of shots per probe state, shape (S,), int.
        fock_cutoff: Fock space cutoff dimension d.
        grid_size_points: Number of points along one axis for the output grid (G x G total bins).
        grid_half_size: Half-size of the output grid in phase space. If None and adaptive_grid=True,
            it will be set to 3 * sqrt(V_bob) where V_bob is the variance of all output data.
        adaptive_grid: Whether to use an adaptive grid based on output data or a fixed grid.
        max_iters: Maximum number of iterations for MLE.
        step_init: Initial step size for gradient descent.
        min_step: Minimum step size before stopping.
        armijo_c: Armijo condition constant.
        armijo_tau: Step size reduction factor for backtracking line search.
        J_initialisation_style: Style for initial Choi matrix. Options (default 'identity'):
            - 'identity': J = |Φ+><Φ+| (Tr_out J = I, ideal identity channel).
            - 'maximally_mixed': J = I / d^2 (completely depolarising warm start).
            - 'known_Gaussian': Placeholder (currently same as maximally mixed; requires params).
        transmissivity: Required if J_initialisation_style is 'known_Gaussian'.
        excess_noise: Required if J_initialisation_style is 'known_Gaussian'.
        verbose: Whether to print progress.
        seed: Random seed for reproducibility.
    
    Returns:
        results: Dictionary containing:
            - 'J': Estimated Choi matrix of the channel, shape (d*d, d*d).
            - 'history': NLL values, step size, back tracks, etc.
            - 'C_inputs': Coherent state overlaps for input states, shape (d, S).
            - 'C_outputs': Coherent state overlaps for output states, shape (d, M).
            - 'counts_ij': Measured counts histogram, shape (S, M).
            - 'bin_area': Area of each output bin in phase space.
            - 'grid_x': X-coordinates of output grid points, shape (G,).
            - 'grid_p': P-coordinates of output grid points, shape (G,).
    """
    console.debug("run_mle_workflow: start")
    rng = np.random.default_rng(seed)

    # ----------------------------- sanity & shapes -----------------------------
    S = int(np.asarray(probe_states).size)
    if S != int(np.asarray(shots_per_probe).size):
        raise ValueError("probe_states and shots_per_probe must have the same length.")
    if fock_cutoff <= 0:
        raise ValueError("fock_cutoff must be positive.")
    if grid_size_points < 2:
        raise ValueError("grid_size_points must be >= 2.")

    a_in = np.asarray(probe_states, dtype=np.complex128).reshape(-1)
    N_shots = np.asarray(shots_per_probe, dtype=np.int64).reshape(-1)

    # ------------------------- 1) synthetic data generation --------------------
    # If channel params not given, default to identity (T=1, xi=0) per original notes.
    T_true = 1.0 if transmissivity is None else float(transmissivity)
    xi_true = 0.0 if excess_noise   is None else float(excess_noise)

    samples_per_probe: List[np.ndarray] = generate_synthetic_channel_data(
        probe_states=a_in,
        shots_per_probe=N_shots,
        transmissivity=T_true,
        excess_noise=xi_true,
        seed=None if seed is None else int(rng.integers(0, 2**31 - 1))
    )
    console.debug("run_mle_workflow: synthetic data generated")

    # ----------------------- 2) grid selection & histogram ---------------------
    # Pick half-size L (adaptive by default) and build grid & bin area.
    if adaptive_grid or grid_half_size is None:
        L = adaptive_grid_half_size(samples_per_probe, percentile=0.975, padding=1.3, min_half_size=3.5)
    else:
        L = float(grid_half_size)

    alpha_grid_flat, Y1, Y2 = build_grid(L, grid_size_points)  # alpha grid (flattened, row-major)
    G = grid_size_points
    M = G * G
    console.log(f"Grid G={G} M={M} L={L:.3f}")

    # Bin geometry
    delta_y, bin_area = grid_bin_width_and_area(L, G)
    # Histogram edges centered on grid points
    edge_lo = -L - 0.5 * delta_y
    edge_hi =  L + 0.5 * delta_y
    edges = np.linspace(edge_lo, edge_hi, G + 1)

    # Histogram each probe's (y1,y2); out-of-range samples are ignored by histogram2d.
    counts_ij = np.zeros((S, M), dtype=np.int64)
    out_of_bounds = np.zeros(S, dtype=np.int64)

    for i, S_i in enumerate(samples_per_probe):
        if S_i.size == 0:
            continue
        y1 = S_i[:, 0]
        y2 = S_i[:, 1]
        # Count how many fell outside the histogram range (for diagnostics)
        mask_in = (y1 >= edge_lo) & (y1 < edge_hi) & (y2 >= edge_lo) & (y2 < edge_hi)
        out_of_bounds[i] = int((~mask_in).sum())

        H, _, _ = np.histogram2d(y1[mask_in], y2[mask_in], bins=[edges, edges])
        # histogram2d returns shape (G, G) with first index for y1/x, second for y2/p.
        counts_ij[i, :] = H.astype(np.int64, copy=False).ravel(order="C")
    console.log(f"Histogram built total_counts={int(counts_ij.sum())}")

    # ---------------- 3) coherent-state overlap matrices (Fock basis) ----------
    d = int(fock_cutoff)

    # Inputs: columns are |alpha_in>
    C_inputs = _coherent_matrix(a_in, d)                     # (d, S)
    # Outputs: columns are |alpha_j> at grid bin centers (flattened row-major)
    C_outputs = _coherent_matrix(alpha_grid_flat, d)         # (d, M)

    # Quick column-norm diagnostic (truncation error; <= 1.0)
    col_norms_inputs = np.sum(np.abs(C_inputs) ** 2, axis=0).real
    col_norms_outputs = np.sum(np.abs(C_outputs) ** 2, axis=0).real

    # --------------------- 4) run projected-gradient MLE -----------------------
    console.log("MLE start")
    # Enable detailed CPTP projection logging by default (can toggle via verbose flag)
    mle_res = mle_projected_gradient_descent(
        counts_ij=counts_ij,
        C_inputs=C_inputs,
        C_outputs=C_outputs,
        bin_area=bin_area,
        J_initialisation_style=J_initialisation_style,
        transmissivity=transmissivity,
        excess_noise=excess_noise,
        max_iters=max_iters,
        step_init=step_init,
        min_step=min_step,
        armijo_c=armijo_c,
        armijo_tau=armijo_tau,
        verbose=verbose,
        seed=None if seed is None else int(rng.integers(0, 2**31 - 1)),
        track_cptp_diagnostics=True,
        projection_verbose=True if verbose else False,
        projection_log_every=20,
        projection_inline=True,
    )
    console.log("MLE end")

    J_hat = mle_res["J"]
    nll_hist = mle_res["nll_history"]
    step_hist = mle_res["step_history"]
    bt_hist = mle_res["bt_history"]

    # -------------------- 5a) Aggregated projection statistics -----------------
    proj_hist = mle_res.get("projection_history", [])
    if proj_hist:
        accepted_iters = [ph.get("accepted_iters") for ph in proj_hist if ph.get("accepted_iters") is not None]
        attempt_counts = [len(ph.get("attempt_iters", [])) for ph in proj_hist]
        backtracks_list = [ph.get("backtracks", 0) for ph in proj_hist]
        if accepted_iters:
            acc_arr = np.asarray(accepted_iters, dtype=float)
            console.log(
                "Projection stats: accIters mean={:.1f} med={:.0f} p90={:.0f} max={:.0f} total={}".format(
                    acc_arr.mean(), np.median(acc_arr), np.quantile(acc_arr, 0.9), acc_arr.max(), int(acc_arr.sum())
                )
            )
        if attempt_counts:
            at_arr = np.asarray(attempt_counts, dtype=float)
            console.log(
                "Projection attempts/outer: mean={:.2f} med={:.0f} max={:.0f}".format(
                    at_arr.mean(), np.median(at_arr), at_arr.max()
                )
            )
        if backtracks_list:
            bt_arr = np.asarray(backtracks_list, dtype=float)
            console.log(
                "Line search backtracks: mean={:.2f} med={:.0f} max={:.0f} total={}".format(
                    bt_arr.mean(), np.median(bt_arr), bt_arr.max(), int(bt_arr.sum())
                )
            )

    # -------------------- 5) helpful post-run diagnostics ----------------------
    # Model probabilities from initial & final J for overlays.
    # Initial probabilities weren’t stored; recompute from the first iterate’s NLL point by
    # calling forward on the initial J we used inside mle (not returned). Instead, compute
    # only the final model; that’s most useful for overlays.
    p_final = compute_channel_probabilites(J_hat, C_inputs, C_outputs, bin_area)  # (S, M)

    # Row-normalised versions (for heatmap comparison to histograms)
    with np.errstate(invalid='ignore', divide='ignore'):
        counts_row_sums = counts_ij.sum(axis=1, keepdims=True).astype(float)
        hist_row = np.divide(counts_ij, np.maximum(counts_row_sums, 1.0))
        p_row = np.divide(p_final, np.maximum(p_final.sum(axis=1, keepdims=True), 1e-300))

    # Expected counts under the final model
    lambda_final = counts_row_sums * p_final

    # Fraction dropped by histogram bounds
    dropped_frac = (out_of_bounds.astype(float) / np.maximum(N_shots, 1)).astype(float)

    # Grid axes for plotting (x ≡ y1, p ≡ y2)
    y_axis = np.linspace(-L, L, G, dtype=float)

    result = {
        # Estimate
        "J": J_hat,
        # Histories
        "nll_history": nll_hist,
        "step_history": step_hist,
        "bt_history": bt_hist,
        "projection_history": mle_res.get("projection_history", []),
        "init_projection_diag": mle_res.get("init_projection_diag", {}),
        # Data & grid
        "probe_states": a_in,
        "shots_per_probe": N_shots,
        "synthetic_samples": samples_per_probe,        # list of (N_i,2) arrays
        "out_of_bounds_per_probe": out_of_bounds,      # samples outside histogram range
        "dropped_fraction_per_probe": dropped_frac,
        "counts_ij": counts_ij,                        # (S, M) row-major ↔ alpha_grid_flat
        "grid_x": y_axis,                              # for visuals: x-quadrature axis
        "grid_p": y_axis,                              # for visuals: p-quadrature axis
        "Y1": Y1, "Y2": Y2,                            # full meshes if you prefer pcolormesh
        "alpha_grid": alpha_grid_flat,                 # (M,)
        "bin_area": float(bin_area),
        "delta_y": float(delta_y),
        "grid_half_size": float(L),
        # Coherent overlaps (for reproducing forward model)
        "C_inputs": C_inputs,
        "C_outputs": C_outputs,
        "col_norms_inputs": col_norms_inputs,          # truncation diagnostics
        "col_norms_outputs": col_norms_outputs,
        # Model overlays
        "p_final": p_final,                            # unnormalised
        "p_final_row_norm": p_row,                     # row-normalised
        "hist_row_norm": hist_row,                     # row-normalised histogram
        "lambda_final": lambda_final,                  # expected counts under final model
        # True channel (for later comparison in synthetic studies)
        "true_transmissivity": T_true,
        "true_excess_noise": xi_true,
        "fock_cutoff": d,
    }
    console.debug("run_mle_workflow: end")
    return result

################# POST-ESTIMATION ANALYSIS ################

def uncertainty_quantification_Fisher():
    pass

def distance_to_Gaussian_channel():
    # TODO: Construct Gaussian channel
    # TODO: Use distance measure from paper
    pass

################# TESTS AND MAIN ################

def run_default_tests():
    # TODO: identity channel, lossy channel, lossy_noisey channel, phase rotation channel (just rotate data).
    """Lightweight default test harness.

    Current scope (incremental):
      1. Generate a set of 100 probe coherent states distributed radially up to r_max=3.
         We distribute them along a spiral so that both radius and angle are covered:
             r_k = r_max * sqrt(k/(N-1)),  theta_k = 2π k / N.
         (This gives roughly uniform coverage in area.)
      2. Assign a fixed number of shots per probe (assumption: 500) and simulate an
         identity channel (T=1, xi=0) via existing synthetic machinery.
      3. Run the MLE workflow attempting to reconstruct the (approximate) identity channel.
      4. Produce a simple matplotlib scatter plot of the probe state set in the α-plane.
      5. Print a few diagnostics comparing the estimated Choi matrix to the ideal identity Choi.

    Notes / Assumptions:
      - Chosen fock_cutoff=8 (can be increased later if truncation errors matter).
      - Grid: 41×41 adaptive grid (covers the sampled support with padding).
      - Shots per probe kept moderate (500) to keep runtime reasonable (< few seconds typically).
      - If matplotlib is not available, plotting is skipped gracefully.
    """

    # ---------------- Configuration ----------------
    num_probes = 100
    r_max = 1.5  # reduced max |alpha| per request
    shots_per_probe_value = 1000  # Increased from 500 per user request.
    fock_cutoff = 10  # increased truncation dimension per request
    grid_side_points = 41
    seed = 12345

    # ---------------- Probe state construction ----------------
    k = np.arange(num_probes, dtype=float)
    # Spiral-style radial distribution for approximate uniform area coverage
    r = r_max * np.sqrt(k / (num_probes - 1))
    theta = 2.0 * np.pi * k / num_probes
    probe_states = r * np.exp(1j * theta)

    shots_per_probe = np.full(num_probes, shots_per_probe_value, dtype=int)

    # ---------------- Run MLE workflow (identity channel synthetic data) ----------------
    mle_res = run_mle_workflow(
        probe_states=probe_states,
        shots_per_probe=shots_per_probe,
        fock_cutoff=fock_cutoff,
        grid_size_points=grid_side_points,
        grid_half_size=None,          # let adaptive take over
        adaptive_grid=True,
        max_iters=75,                 # modest iterations for speed
        step_init=1e-3,
        min_step=1e-8,
        armijo_c=1e-4,
        armijo_tau=0.5,
        J_initialisation_style='maximally_mixed',
        transmissivity=1.0,
        excess_noise=0.0,
        verbose=True,
        seed=seed,
    )

    # ---------------- Identity Choi reference ----------------
    d = fock_cutoff
    D = d * d
    Omega = np.zeros((D,), dtype=np.complex128)
    for j in range(d):
        Omega[j * d + j] = 1.0
    J_identity = np.outer(Omega, Omega.conjugate())
    # Ensure exact CPTP form already (Tr_out = I). No additional scaling needed since our convention
    # uses unnormalised |Ω> = Σ_i |i,i>.

    J_hat = mle_res['J']

    # Diagnostics
    frob_diff = np.linalg.norm(J_hat - J_identity)
    frob_id = np.linalg.norm(J_identity)
    rel_frob = frob_diff / max(frob_id, 1e-12)
    nll_final = float(mle_res['nll_history'][-1]) if mle_res['nll_history'].size else float('nan')
    proj_iters = [p['accepted_iters'] for p in mle_res.get('projection_history', []) if p.get('accepted_iters') is not None]
    avg_proj_iters = float(np.mean(proj_iters)) if proj_iters else 0.0

    console.log("=== Identity Channel Reconstruction ===")
    console.log(f"# Probes: {num_probes}, shots/probe: {shots_per_probe_value}, total shots ≈ {shots_per_probe_value * num_probes}")
    console.log(f"Fock cutoff d = {d}, grid = {grid_side_points}x{grid_side_points}, adaptive L = {mle_res['grid_half_size']:.3f}")
    console.log(f"Final NLL: {nll_final:.4f}")
    console.log(f"Frobenius ||J_hat - J_id|| = {frob_diff:.3e}  (relative {rel_frob:.3e})")
    console.log(f"Average projection iterations (accepted steps): {avg_proj_iters:.2f}")
    console.log(f"Mean input truncation norm deficit: {(1 - mle_res['col_norms_inputs']).mean():.3e}")
    console.log(f"Mean output truncation norm deficit: {(1 - mle_res['col_norms_outputs']).mean():.3e}")
    console.log("==========================================================")

    # ---------------- Plot probe states ----------------
    try:
        import matplotlib.pyplot as plt  # local import to keep module lightweight
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(probe_states.real, probe_states.imag, c=np.linspace(0, 1, num_probes), s=25, cmap='viridis')
        circ = plt.Circle((0, 0), r_max, color='black', linestyle='--', linewidth=1, fill=False, alpha=0.5)
        ax.add_artist(circ)
        ax.set_xlabel('Re(alpha)')
        ax.set_ylabel('Im(alpha)')
        ax.set_title('Probe Coherent States (Spiral Coverage)')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
    except Exception as e:  # pragma: no cover - plotting fallback
        console.log(f"[run_default_tests] Plot skipped (matplotlib not available or error: {e})")

    # Return results for potential further inspection by caller
    return {
        'probe_states': probe_states,
        'shots_per_probe': shots_per_probe,
        'mle_results': mle_res,
        'J_identity': J_identity,
        'frob_diff': frob_diff,
        'relative_frob_diff': rel_frob,
    }

def main():
    # TODO: Parse command line args, call run_default_tests or run_mle_workflow.
    run_default_tests()

if __name__ == "__main__":
    main()