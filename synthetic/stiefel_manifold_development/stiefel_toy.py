# toy_stiefel_kraus_demo.py
import numpy as np
import os, sys, time
from datetime import datetime
import matplotlib.pyplot as plt

# --- Ensure the parent synthetic directory is on sys.path so we can import PG_MLE_synthetic_channel.py ---
_THIS_DIR = os.path.dirname(__file__)
_SYNTHETIC_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))  # path to synthetic/
if _SYNTHETIC_DIR not in sys.path:
    sys.path.insert(0, _SYNTHETIC_DIR)

# --- Reuse your machinery (grid, histogram, coherent overlaps, Poisson NLL, logging) ---
try:
    from PG_MLE_synthetic_channel import (                   # :contentReference[oaicite:1]{index=1}
        console,
        adaptive_grid_half_size,
        build_grid,
        grid_bin_width_and_area,
        generate_synthetic_channel_data,
        _coherent_matrix,
        neg_log_likelihood_poisson_multiinput,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Could not import PG_MLE_synthetic_channel. Ensure this script is inside synthetic/ subdirectory."
    ) from e

# ------------------------------------------------------------
# Helpers: Kraus <-> stacked Stiefel variable, QR retraction
# ------------------------------------------------------------
def stack_kraus(K_list):
    """Stack r Kraus operators K_k (d x d) into V of shape ((r*d) x d)."""
    return np.vstack(K_list).astype(np.complex128, copy=False)

def unstack_kraus(V, d):
    """Split V ((r*d) x d) into r Kraus operators of shape (d x d)."""
    r = V.shape[0] // d
    return [V[k*d:(k+1)*d, :] for k in range(r)]

def riemannian_grad_on_stiefel(V, G):
    """Project Euclidean grad G onto tangent at V: grad = G - V sym(V^† G)."""
    H = V.conj().T @ G
    symH = 0.5 * (H + H.conj().T)
    return G - V @ symH

def qr_retraction(X):
    """Retract to Stiefel via thin QR with positive diag (column-orthonormal)."""
    Q, R = np.linalg.qr(X)
    # fix signs for stability
    sign = np.sign(np.diag(R).real)
    sign[sign == 0] = 1.0
    return Q @ np.diag(sign)

# ------------------------------------------------------------
# Forward model & gradient in Kraus/Stiefel parametrization
# ------------------------------------------------------------
def forward_probs_from_Kraus(K_list, C_in, C_out, bin_area):
    """
    p_{ij} = (ΔA/π) * sum_k | b_j^† K_k a_i |^2
    Vectorized: for each k, Y_k = (C_out^† @ K_k @ C_in) has shape (M x S).
    """
    d, S = C_in.shape
    M = C_out.shape[1]
    scale = float(bin_area) / np.pi

    p_SM = np.zeros((S, M), dtype=np.float64)
    C_out_H = C_out.conj().T      # (M, d)

    for K in K_list:
        Y = C_out_H @ (K @ C_in)  # (M, S)
        p_SM += (np.abs(Y) ** 2).T

    return scale * p_SM  # (S, M)

def kraus_gradient(K_list, C_in, C_out, counts_ij, p_ij, bin_area, eps: float = 1e-12):
    """
    ∇_{K_k} L = 2c * sum_i S_i K_k A_i,
    where  W_ij = N_i - n_ij / p_ij,
           S_i  = sum_j W_ij B_j,
           A_i  = a_i a_i^†,  B_j = b_j b_j^†,
           c    = bin_area / π.
    Shapes: A_i (d,d), B_j (d,d), K_k (d,d).
    """
    d, S = C_in.shape
    M = C_out.shape[1]
    scale = float(bin_area) / np.pi

    # Totals per input
    N_i = counts_ij.sum(axis=1, keepdims=True)  # (S,1)
    p_safe = np.maximum(p_ij, eps)
    W = (N_i - counts_ij / p_safe)             # (S, M)

    # Precompute rank-1s
    A = [np.outer(C_in[:, i], C_in[:, i].conj()) for i in range(S)]   # list of (d,d)
    B = [np.outer(C_out[:, j], C_out[:, j].conj()) for j in range(M)] # list of (d,d)

    # S_i = sum_j W_ij B_j  -> vectorized accumulation
    # To keep skeleton simple (and tiny sizes fast), do a loop; swap to einsum later if needed.
    S_acc = []
    for i in range(S):
        Si = np.zeros((d, d), dtype=np.complex128)
        Wi = W[i, :]
        # Only accumulate non-zero weights for speed in a toy
        nz = np.nonzero(np.abs(Wi) > 0)[0]
        for j in nz:
            Si += Wi[j] * B[j]
        S_acc.append(Si)

    grads = []
    for K in K_list:
        Gk = np.zeros_like(K)
        for i in range(S):
            Gk += S_acc[i] @ K @ A[i]
        grads.append(2.0 * scale * Gk)

    return grads  # list of (d,d)

# ------------------------------------------------------------
# Choi construction with explicit 4-index tensor (m,n,p,q)
# ------------------------------------------------------------
def choi_from_kraus(K_list):
        """Row-major (NumPy) convention.

        Returns (J, J4) with:
            J4[m,n,p,q] = sum_k K_k[m,p] * K_k[n,q].conj()
            J[(m,p),(n,q)] = J4[m,n,p,q]  (row-major vec mapping)

        This J is the one that satisfies  Tr[J (B ⊗ A^T)] = sum_k |b^† K_k a|^2
        under the rule vec_C(B X A) = (B ⊗ A^T) vec_C(X), where vec_C uses
        NumPy's row-major (order='C') flattening: (m,p) -> m*d + p.

        Note: trace-preserving check continues to use J4 directly via
        Tr_out J = sum_m J4[m,m,:,:] = I.
        """
        d = K_list[0].shape[0]
        J4 = sum(np.einsum('mp,nq->mnpq', K, K.conj()) for K in K_list)  # (d,d,d,d)
        # Row-major vec groups (m,p) for rows and (n,q) for cols
        J = np.transpose(J4, (0, 2, 1, 3)).reshape(d*d, d*d)
        return J, J4

# ------------------------------------------------------------
# Canonical Kraus extraction (for diagnostics only)
# ------------------------------------------------------------
def canonical_kraus_from_choi(J, d, tol=1e-12):
    """Return a canonical Kraus list from Choi matrix J via eigen-decomposition.

    J = sum_k lambda_k |v_k><v_k| with lambda_k >= 0.
    Each eigenvector v_k (length d^2) reshaped to (d,d) in *row-major* (order='C')
    with prefactor sqrt(lambda_k). Sorted by descending eigenvalue, discarding
    eigenvalues <= tol.
    """
    evals, evecs = np.linalg.eigh(J)
    K_list = []
    for lam, vec in sorted(zip(evals, evecs.T), key=lambda x: -x[0]):
        if lam <= tol:
            continue
        # Explicit order='C' to document row-major unvec convention
        K = np.sqrt(max(lam, 0.0)) * vec.reshape(d, d, order='C')
        K_list.append(K)
    return K_list, evals

# ------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------
def save_channel_dat(filepath, K_list, J):
    """Save Kraus operators and Choi matrix (complex) to a human-readable .dat file.

    Format:
      # d=<d> r=<r>
      # Kraus <k>
      <row entries as a+bi j separated by space>
      ... (d rows)
      # Choi (d^2 x d^2)
      <row entries>
    Complex numbers written as 'Re+Imj' with 16-digit exponential precision.
    """
    d = K_list[0].shape[0]
    r = len(K_list)
    def fmt(z: complex):
        return f"{z.real:.16e}+{z.imag:.16e}j"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# d={d} r={r}\n")
        for k, K in enumerate(K_list):
            f.write(f"# Kraus {k}\n")
            for row in K:
                f.write(' '.join(fmt(z) for z in row) + '\n')
        f.write(f"# Choi {J.shape[0]}x{J.shape[1]}\n")
        f.write("# Choi uses row-major vec: J[(m,p),(n,q)] = sum_k K[m,p] K[n,q]^*\n")
        for row in J:
            f.write(' '.join(fmt(z) for z in row) + '\n')
    return filepath

# ------------------------------------------------------------
# A minimal Stiefel-optimiser loop (BB step + simple backtracking)
# ------------------------------------------------------------
def stiefel_mle_toy(counts_ij, C_in, C_out, bin_area, r=1,
                    max_iters=100, step_init=1e-2, bb_alpha_min=1e-6, bb_alpha_max=1e+2,
                    backtracks=5, shrink=0.5, seed=0, verbose=True,
                    init_mode: str = "identity",
                    show_iter_choi: bool = False,
                    tol_grad: float = 1e-6,
                    prob_floor: float = 1e-12,
                    auto_rescale_expected: bool = True,
                    rescale_tol_rel_std: float = 1e-3,
                    capture_first_n_choi: int = 0):
    """
    Returns: dict with Kraus operators, NLL history, and (optionally) the Choi at the end.
    """
    rng = np.random.default_rng(seed)
    d = C_in.shape[0]

    # init Kraus list
    if init_mode == "identity":
        K_list = [np.eye(d, dtype=np.complex128)] + [np.zeros((d, d), np.complex128) for _ in range(r-1)]
    elif init_mode == "mixed":
        # Completely depolarising channel: Choi = (I ⊗ I)/d, Kraus set E_{ij} = 1/sqrt(d) |i><j|
        # Requires r = d^2. If provided r mismatches, override.
        r = d * d
        K_list = []
        scale = 1.0 / np.sqrt(d)
        for i in range(d):
            for j in range(d):
                K = np.zeros((d, d), dtype=np.complex128)
                K[i, j] = scale
                K_list.append(K)
    else:
        raise ValueError(f"Unknown init_mode '{init_mode}' (use 'identity' or 'mixed')")
    V = stack_kraus(K_list)  # ((r*d) x d)
    # orthonormalise columns just in case
    V = qr_retraction(V)

    # helpers
    scale_factor_hist = []
    def unpack_and_eval(Vmat):
        Ks = unstack_kraus(Vmat, d)
        p_raw = forward_probs_from_Kraus(Ks, C_in, C_out, bin_area)
        p = np.maximum(p_raw, prob_floor)
        scale_factor = 1.0
        if auto_rescale_expected:
            # Exposure per input
            N_i = counts_ij.sum(axis=1)
            row_sums = p.sum(axis=1)
            tiny = 1e-18
            ratios = N_i / (row_sums + tiny)
            mean_ratio = np.mean(ratios)
            # relative std of ratios
            rel_std = (np.std(ratios) / (abs(mean_ratio) + tiny)) if mean_ratio != 0 else np.inf
            # If near-constant factor but not ~1, rescale p
            if rel_std < rescale_tol_rel_std and not np.isclose(mean_ratio, 1.0, rtol=1e-3, atol=1e-6):
                p *= mean_ratio
                scale_factor = mean_ratio
                if verbose and not scale_factor_hist:  # log only first time we apply
                    console.log(f"[scale-fix] Rescaled model probabilities by factor {mean_ratio:.6g} (rel std {rel_std:.2e}) to match exposures")
        scale_factor_hist.append(scale_factor)
        nll = neg_log_likelihood_poisson_multiinput(counts_ij, p)
        return Ks, p, float(nll)

    Ks, p, cur_nll = unpack_and_eval(V)
    Ks_initial = [K.copy() for K in Ks]
    # Precompute truth (identity) and initial Choi for iterative plotting if requested
    if show_iter_choi:
        # Precompute truth & initial Choi with consistent ordering
        K_true = [np.eye(d, dtype=np.complex128)]
        J_truth, _ = choi_from_kraus(K_true)
        J_init, _ = choi_from_kraus(Ks_initial)
    if verbose:
        total_shots = int(counts_ij.sum())
        S, M = counts_ij.shape
        console.log(
            "init NLL={:.6f} d={} r={} S={} M={} total_shots={} bin_area={:.3e}".format(
                cur_nll, d, r, S, M, total_shots, bin_area
            )
        )

    # histories
    nll_hist = [cur_nll]
    step_hist = []
    backtrack_hist = []
    iter_time_hist = []
    grad_norm_hist = []
    first_choi = []  # store first N reconstructed Choi matrices after updates

    # previous iterate for BB
    V_prev = None
    Rgrad_prev = None
    alpha = float(step_init)

    t0 = time.perf_counter()
    for it in range(1, max_iters + 1):
        iter_start = time.perf_counter()
        # Euclidean gradient blocks, then stack
        G_blocks = kraus_gradient(Ks, C_in, C_out, counts_ij, p, bin_area, eps=prob_floor)
        G = stack_kraus(G_blocks)

        # Riemannian gradient on Stiefel
        Rgrad = riemannian_grad_on_stiefel(V, G)
        grad_norm_sq = float(np.vdot(Rgrad, Rgrad).real)
        g_norm = np.sqrt(grad_norm_sq)
        grad_norm_hist.append(g_norm)
        if grad_norm_sq <= 1e-30:
            if verbose: console.log(f"iter {it:03d} tiny grad -> stop")
            break
        # Stationarity-based stopping test (scaled by ||V|| as suggested)
        V_norm = np.linalg.norm(V)
        if g_norm <= tol_grad * (1.0 + V_norm):
            if verbose: console.log(f"iter {it:03d} grad-norm {g_norm:.3e} <= tol {tol_grad*(1.0+V_norm):.3e} -> stop")
            break

        # BB1 step from Riemannian differences
        if V_prev is not None and Rgrad_prev is not None:
            s = V - V_prev
            y = Rgrad - Rgrad_prev
            num = float(np.vdot(s, s).real)
            den = float(np.vdot(s, y).real)
            if den > 1e-18:
                alpha = np.clip(num / den, bb_alpha_min, bb_alpha_max)

        # backtracking on retracted trial
        accepted = False
        trial_alpha = float(alpha)
        attempts = 0
        for _ in range(backtracks + 1):
            attempts += 1
            V_trial = qr_retraction(V - trial_alpha * Rgrad)
            Ks_trial, p_trial, nll_trial = unpack_and_eval(V_trial)
            # Armijo-like sufficient decrease
            if nll_trial <= cur_nll - 1e-4 * trial_alpha * grad_norm_sq:
                accepted = True
                break
            trial_alpha *= shrink  # shrink step

        # accept last trial even if not meeting strict decrease
        V_prev, Rgrad_prev = V, Rgrad
        prev_nll = cur_nll
        V, Ks, p, cur_nll = V_trial, Ks_trial, p_trial, nll_trial
        nll_hist.append(cur_nll)
        step_hist.append(trial_alpha)
        backtrack_hist.append(attempts - 1)
        iter_dt = time.perf_counter() - iter_start
        iter_time_hist.append(iter_dt)
        dNLL = prev_nll - cur_nll
        rel_imp = dNLL / (abs(prev_nll) + 1e-18)
        elapsed = time.perf_counter() - t0
        if capture_first_n_choi > 0 and len(first_choi) < capture_first_n_choi:
            J_iter, _ = choi_from_kraus(Ks)
            first_choi.append(J_iter.copy())
        if verbose:
            console.log(
                ("iter {it:03d} NLL={nll:.6f} dNLL={dNLL:.2e} rel={rel:.2e} |grad|={g:.2e} "
                 "step={st:.2e} attempts={att:d} acc={acc} t={t:.2f}s dt={dt:.3f}s").format(
                    it=it, nll=cur_nll, dNLL=dNLL, rel=rel_imp, g=np.sqrt(grad_norm_sq),
                    st=trial_alpha, att=attempts, acc=accepted, t=elapsed, dt=iter_dt
                )
            )
        if show_iter_choi:
            # Build current Choi and display alongside truth and initial (consistent ordering)
            J_cur, _ = choi_from_kraus(Ks)
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
            axes[0].imshow(J_truth.real, cmap='viridis', origin='lower')
            axes[0].set_title('Truth (Re)')
            axes[1].imshow(J_init.real, cmap='viridis', origin='lower')
            axes[1].set_title('Init (Re)')
            im2 = axes[2].imshow(J_cur.real, cmap='viridis', origin='lower')
            axes[2].set_title(f'Iter {it} (Re)')
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8)
            fig.suptitle(f'Choi Convergence d={d} iter={it} NLL={cur_nll:.3f}')
            plt.show()
            plt.close(fig)
    # NOTE: Removed premature stopping on tiny relative NLL improvement; rely on gradient norm / max_iters.

    total_elapsed = time.perf_counter() - t0
    # final gradient norm
    G_blocks = kraus_gradient(Ks, C_in, C_out, counts_ij, p, bin_area, eps=prob_floor)
    G = stack_kraus(G_blocks)
    Rg_final = riemannian_grad_on_stiefel(V, G)
    final_grad_norm = float(np.sqrt(np.vdot(Rg_final, Rg_final).real))
    if verbose:
        console.log(
            "summary iters={} final_NLL={:.6f} total_dNLL={:.2e} avg_dt={:.3f}s total_t={:.2f}s final|grad|={:.2e}".format(
                len(nll_hist)-1, nll_hist[-1], nll_hist[0]-nll_hist[-1],
                (np.mean(iter_time_hist) if iter_time_hist else 0.0), total_elapsed, final_grad_norm
            )
        )
        if backtrack_hist:
            console.log(
                "backtracks per iter: mean={:.2f} max={} total={}".format(
                    float(np.mean(backtrack_hist)), int(np.max(backtrack_hist)), int(np.sum(backtrack_hist))
                )
            )
    return {
        "Kraus": Ks,
        "V": V,
        "Kraus_initial": Ks_initial,
        "nll_history": np.array(nll_hist, float),
        "step_history": np.array(step_hist, float),
        "backtrack_history": np.array(backtrack_hist, int),
        "iter_time_history": np.array(iter_time_hist, float),
        "grad_norm_history": np.array(grad_norm_hist, float),
        "final_grad_norm": final_grad_norm,
        "total_time": total_elapsed,
    "prob_scale_factors": np.array(scale_factor_hist, float),
    "first_choi": np.array(first_choi) if first_choi else np.empty((0, d*d, d*d), dtype=np.complex128),
    }

# ------------------------------------------------------------
# Toy dataset builder reusing your grid/hist machinery
# ------------------------------------------------------------
def build_toy_problem(S=6, shots_per_probe=500, d=6, G=25,
                      T_true=1.0, xi_true=0.0, seed=1234,
                      ring_radii=None):
    """Construct a synthetic tomography toy problem.

    Steps:
      1) Generate S probe coherent states (either single ring or concentric rings if ``ring_radii`` given).
      2) Simulate dual-homodyne (heterodyne) data for each probe.
      3) Build an adaptive square grid and histogram counts per probe.
      4) Build coherent-state overlap matrices for inputs/outputs.

    Parameters
    ----------
    S : int
        Total number of probe coherent states desired.
    shots_per_probe : int
        Number of synthetic measurement shots per probe (uniform across probes).
    d : int
        Fock cutoff (dimension of truncated Hilbert space).
    G : int
        Grid side length (G x G bins).
    T_true, xi_true : float
        Underlying (transmissivity, excess noise) parameters for synthetic channel.
    seed : int
        RNG seed.
    ring_radii : sequence[float] | None
        If provided, arrange probes on concentric rings with these radii (around 0+0j).
        Probes are distributed as evenly as possible over the rings; each ring's
        states are uniformly spaced in phase. If None, a single ring of radius 1.2 is used.
    """
    rng = np.random.default_rng(seed)

    def _concentric_probes(num_total: int, radii):
        R = len(radii)
        # Distribute counts as evenly as possible (first 'remainder' rings get +1)
        base = num_total // R
        rem = num_total % R
        counts = [base + (1 if i < rem else 0) for i in range(R)]
        alphas = []
        for rad, cnt in zip(radii, counts):
            if cnt == 1:
                thetas = np.array([0.0])
            else:
                thetas = 2 * np.pi * np.arange(cnt) / cnt
            alphas.append(rad * np.exp(1j * thetas))
        return np.concatenate(alphas).astype(np.complex128)

    if ring_radii is not None and len(ring_radii) > 0:
        probe_states = _concentric_probes(S, list(ring_radii))
        # In case rounding produced mismatch (shouldn't), truncate or pad with center state.
        if probe_states.size > S:
            probe_states = probe_states[:S]
        elif probe_states.size < S:
            probe_states = np.concatenate([probe_states, np.zeros(S - probe_states.size, dtype=np.complex128)])
    else:
        # Legacy behaviour: single ring (small amplitude for low d)
        r = 1.2
        theta = 2 * np.pi * np.arange(S) / S
        probe_states = (r * np.exp(1j * theta)).astype(np.complex128)

    N_per = np.full(S, shots_per_probe, dtype=int)

    # synthetic dual-homodyne samples via your function
    data = generate_synthetic_channel_data(           # :contentReference[oaicite:2]{index=2}
        probe_states=probe_states,
        shots_per_probe=N_per,
        transmissivity=T_true,
        excess_noise=xi_true,
        seed=int(rng.integers(0, 2**31-1))
    )

    # adaptive grid + ΔA
    L = adaptive_grid_half_size(data, percentile=0.975, padding=1.3, min_half_size=3.5)   # :contentReference[oaicite:3]{index=3}
    alpha_grid_flat, Y1, Y2 = build_grid(L, G)                                            # :contentReference[oaicite:4]{index=4}
    delta_y, bin_area = grid_bin_width_and_area(L, G)                                     # :contentReference[oaicite:5]{index=5}

    # histogram per probe (mirror your MLE preamble)
    edge_lo, edge_hi = -L - 0.5 * delta_y, L + 0.5 * delta_y
    edges = np.linspace(edge_lo, edge_hi, G + 1)
    M = G * G
    counts_ij = np.zeros((S, M), dtype=np.int64)
    for i, S_i in enumerate(data):
        if S_i.size == 0:
            continue
        y1, y2 = S_i[:, 0], S_i[:, 1]
        mask = (y1 >= edge_lo) & (y1 < edge_hi) & (y2 >= edge_lo) & (y2 < edge_hi)
        H, _, _ = np.histogram2d(y1[mask], y2[mask], bins=[edges, edges])
        counts_ij[i, :] = H.astype(np.int64).ravel(order="C")

    # coherent overlaps (Fock basis)
    C_in = _coherent_matrix(probe_states, d)               # (d, S)  :contentReference[oaicite:6]{index=6}
    C_out = _coherent_matrix(alpha_grid_flat, d)           # (d, M)  :contentReference[oaicite:7]{index=7}

    return {
        "probe_states": probe_states,
        "data_samples": data,      # list/array of arrays (Ni,2)
        "counts_ij": counts_ij,
        "C_inputs": C_in,
        "C_outputs": C_out,
        "bin_area": bin_area,
        "grid_side": G,
        "grid_half_size": L,
        "grid_edges": edges,       # 1D edges used for both axes
        "fock_cutoff": d,
    }


def plot_data_diagnostics(toy_dict, max_points_all=6000, seed=0):
    """Plot probe constellation, one probe's sample cloud, all samples, and grid coverage."""
    rng = np.random.default_rng(seed)
    probes = toy_dict["probe_states"]
    data_list = toy_dict["data_samples"]
    L = toy_dict["grid_half_size"]
    edges = toy_dict["grid_edges"]
    G = toy_dict["grid_side"]
    # Flatten all samples
    all_samples = np.concatenate([d for d in data_list if d.size > 0], axis=0) if data_list else np.empty((0,2))
    inside_mask = (np.abs(all_samples[:,0]) <= L) & (np.abs(all_samples[:,1]) <= L)
    coverage_frac = inside_mask.mean() if all_samples.size else 0.0
    # Choose an example probe with data
    valid_indices = [i for i,d in enumerate(data_list) if d.size>0]
    ex_idx = rng.choice(valid_indices) if valid_indices else 0
    ex_samples = data_list[ex_idx]
    # Subsample all samples for scatter if large
    if all_samples.shape[0] > max_points_all:
        sel = rng.choice(all_samples.shape[0], max_points_all, replace=False)
        all_plot = all_samples[sel]
    else:
        all_plot = all_samples
    # Color map per probe for all data: build concatenated arrays
    colors = None
    if all_samples.size:
        probe_labels = np.concatenate([
            np.full(d.shape[0], i) for i,d in enumerate(data_list) if d.size>0
        ])
        if all_samples.shape[0] > max_points_all:
            colors = probe_labels[sel]
        else:
            colors = probe_labels
    # Use constrained_layout to avoid tight_layout warnings with colorbars
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()
    # Panel 1: probe constellation
    ax0.scatter(probes.real, probes.imag, c='C0')
    for i,a in enumerate(probes):
        ax0.text(a.real, a.imag, str(i), fontsize=8, ha='center', va='center')
    ax0.set_title('Probe states (alpha)')
    ax0.set_xlabel('Re')
    ax0.set_ylabel('Im')
    ax0.set_aspect('equal')
    # Panel 2: example probe data
    if ex_samples.size:
        ax1.scatter(ex_samples[:,0], ex_samples[:,1], s=5, alpha=0.6)
    ax1.set_title(f'Example probe {ex_idx} samples (N={ex_samples.shape[0]})')
    ax1.set_xlabel('y1')
    ax1.set_ylabel('y2')
    ax1.set_aspect('equal')
    # Panel 3: all data cloud (color by probe)
    if all_plot.size:
        sc = ax2.scatter(all_plot[:,0], all_plot[:,1], s=3, c=colors, cmap='tab20', alpha=0.5)
        cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cb.set_label('Probe index')
    ax2.set_title(f'All samples (subsampled) total={all_samples.shape[0]}')
    ax2.set_xlabel('y1')
    ax2.set_ylabel('y2')
    ax2.set_aspect('equal')
    # Panel 4: grid coverage heatmap
    if all_samples.size:
        H, _, _ = np.histogram2d(all_samples[:,0], all_samples[:,1], bins=[edges, edges])
        im = ax3.imshow(H.T, origin='lower', extent=[edges[0], edges[-1], edges[0], edges[-1]],
                        aspect='equal', cmap='magma')
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Counts')
    ax3.set_title(f'Coverage heatmap (inside frac={coverage_frac:.3f})')
    ax3.set_xlabel('y1')
    ax3.set_ylabel('y2')
    # Draw square boundary
    ax3.plot([-L, L, L, -L, -L], [-L, -L, L, L, -L], 'w--', lw=1)
    # Removed plt.tight_layout(); constrained_layout handles spacing
    plt.show()

# ------------------------------------------------------------
# Main: run the toy Stiefel MLE and print minimal diagnostics
# ------------------------------------------------------------
def main():
    console.level = 1  # 0 silent, 1 minimal, 2 verbose  :contentReference[oaicite:8]{index=8}
    # Use 100 probe states on 5 concentric rings within |alpha| <= 1
    toy = build_toy_problem(S=100, shots_per_probe=1600, d=4, G=31,
                            T_true=1.0, xi_true=0.0, seed=2025,
                            ring_radii=[0.2, 0.4, 0.6, 0.8, 1.0])

    # Plot data diagnostics before optimisation
    plot_data_diagnostics(toy, max_points_all=8000, seed=2025)

    counts = toy["counts_ij"]
    C_in = toy["C_inputs"]
    C_out = toy["C_outputs"]
    bin_area = toy["bin_area"]
    d = toy["fock_cutoff"]

    # Start from maximally mixed (completely depolarising) channel initialisation
    res = stiefel_mle_toy(
        counts_ij=counts,
        C_in=C_in,
        C_out=C_out,
        bin_area=bin_area,
        r=16,                  # d^2 Kraus operators for mixed initialisation (d=4)
        max_iters=500,
        step_init=5e-3,
        verbose=True,
        seed=7,
    init_mode="mixed",
    show_iter_choi=False,  # suppress interactive iteration plots for batch capture
    capture_first_n_choi=20,
    )

    # simple check: final NLL and TP/CP diagnostics
    K = res["Kraus"]
    # Build Choi with consistent ordering (m,n,p,q)
    J, J4 = choi_from_kraus(K)
    # TP check: Tr_out J = sum_m J4[m,m,:,:] ≈ I_d
    Tr_out = np.einsum('mmpq->pq', J4)
    tp_err = np.linalg.norm(Tr_out - np.eye(d))

    console.log(f"Final NLL: {res['nll_history'][-1]:.6f}")
    console.log(f"TP residual ||Tr_out(J)-I||_F: {tp_err:.2e}")
    console.log(f"Iterations: {len(res['nll_history'])-1}, last step={res['step_history'][-1]:.2e}, total time={res['total_time']:.2f}s, final |grad|={res['final_grad_norm']:.2e}")

    # Persist Kraus & Choi to .dat file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(_THIS_DIR, f"channel_reconstruction_{timestamp}.dat")
    save_channel_dat(out_path, K, J)
    console.log(f"Saved Kraus & Choi to {out_path}")

    # --- Plot truth, initial, reconstructed Choi matrices ---
    # Truth channel (identity for current synthetic data generator parameters T_true=1, xi_true=0)
    K_true = [np.eye(d, dtype=np.complex128)]
    J_true, _ = choi_from_kraus(K_true)
    # Initial Choi (before optimisation)
    K_init = res.get("Kraus_initial", K_true)
    J_init, _ = choi_from_kraus(K_init)

    # --- Fresh 3-panel Choi plot (truth, init, reconstructed) ---
    c_abs = max(np.abs(J_true).max(), np.abs(J_init).max(), np.abs(J).max())
    fro_err = np.linalg.norm(J - J_true) / np.linalg.norm(J_true)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    titles = ["Truth Choi Re", "Init Choi Re", f"Final Choi Re\nrel Fro err={fro_err:.2e}"]
    mats = [J_true.real, J_init.real, J.real]
    ims = []
    for ax, M, title in zip(axes, mats, titles):
        im = ax.imshow(M, cmap='viridis', vmin=-c_abs, vmax=c_abs, origin='lower')
        ims.append(im)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    # Single shared colorbar
    fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.75, location='right', pad=0.02, label='Value (Re)')
    fig.suptitle(f'Stiefel MLE Channel Reconstruction (d={d})')
    plt.show()

    # --- Born-rule sanity check: forward probs vs Choi + Kronecker ---
    rng_local = np.random.default_rng(0)
    for _ in range(3):
        i = int(rng_local.integers(0, C_in.shape[1]))
        j = int(rng_local.integers(0, C_out.shape[1]))
        a = C_in[:, i]; A = np.outer(a, a.conj())
        b = C_out[:, j]; B = np.outer(b, b.conj())
        lhs = sum(abs(b.conj() @ Kk @ a)**2 for Kk in K) * (bin_area/np.pi)
        rhs = (J @ np.kron(B, A.T)).trace().real * (bin_area/np.pi)
        assert np.allclose(lhs, rhs, atol=1e-10), (lhs, rhs)
    console.log("Born-rule sanity check passed (forward vs Choi).")

    # --- Save first 20 reconstructed Choi matrices (real parts) into one PNG ---
    first_choi = res.get("first_choi", np.empty((0,)))
    if first_choi.size:
        n = min(first_choi.shape[0], 20)
        cols = 5 if n >= 5 else n
        rows = int(np.ceil(n / cols))
        c_abs_iter = np.max(np.abs(first_choi[:n].real))
        fig_iter, axes_iter = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), constrained_layout=True)
        axes_iter = np.atleast_1d(axes_iter).ravel()
        for idx in range(n):
            ax = axes_iter[idx]
            ax.imshow(first_choi[idx].real, cmap='viridis', vmin=-c_abs_iter, vmax=c_abs_iter, origin='lower')
            ax.set_title(f'Iter {idx+1}')
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes_iter[n:]:
            ax.axis('off')
        fig_iter.suptitle(f'First {n} Reconstructed Choi Matrices (Real part)')
        out_png = os.path.join(_THIS_DIR, 'first_20_iterations.png')
        fig_iter.savefig(out_png, dpi=150)
        console.log(f'Saved first {n} iteration Choi montage to {out_png}')
        plt.close(fig_iter)

if __name__ == "__main__":
    main()
