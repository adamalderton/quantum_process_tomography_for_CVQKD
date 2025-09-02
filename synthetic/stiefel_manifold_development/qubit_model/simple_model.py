"""Simple qubit channel tomography toy model (Stiefel-parametrised Kraus set).

This is a stripped‑down analogue of the fuller `stiefel_toy.py` machinery, tuned
to *just* exercise the physics for a qubit (d=2) with a small number of Kraus
operators and a simple informationally complete POVM (qubit SIC / tetra POVM).

Key ideas:
  * A trace‑preserving CP map with r Kraus operators {K_k} on C^2 is enforced by
	stacking the r (2x2) Kraus matrices into V in C^{2r x 2} and constraining
	V^† V = I_2 (complex Stiefel manifold St(2, 2r)). Then automatically
	Σ_k K_k^† K_k = I_2.
  * Data generation: choose S probe pure states {|ψ_i>}, push each through the
	channel, measure with SIC POVM {E_m}_{m=1..4}. Sample counts from the
	multinomial distribution with probabilities p_{i,m} = Tr[E_m Φ(|ψ_i><ψ_i|)].
  * Likelihood: product over independent multinomials (or equivalently sum of
	probe NLLs). Gradient uses the same weight structure as the heterodyne
	model in the larger code base: W_{i,m} = N_i - n_{i,m} / p_{i,m}. Then
	  ∇_{K_k} L = 2 Σ_i ( Σ_m W_{i,m} E_m ) K_k ρ_i
	where ρ_i = |ψ_i><ψ_i|. (Derivation mirrors the coherent‑state version.)
  * Riemannian gradient: project Euclidean grad onto tangent space at V and use
	a simple backtracking line search with a QR retraction.

This file deliberately avoids any heavy dependencies or plotting. It is meant
only for quick physics sanity checks (TP, CP, basic MLE convergence).

Assumptions (since `notes.md` was empty):
  1. Focus on qubit (d=2) only.
  2. Use SIC POVM for measurement (tetrahedron on Bloch sphere) – informationally complete.
  3. Pure probe states uniformly (approximately) distributed on Bloch sphere via random Haar vectors.
  4. No adaptive grids / continuous variable structure here.

If later you want to plug in different POVMs or higher dimensions, extend:
  * `build_sic_povm_qubit` for general POVMs;
  * generalise d, adapt tetra vectors to appropriate IC POVM / SIC if available;
  * reuse gradient formula with rank‑1 projectors replaced by POVM elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence, Tuple
import numpy as np
import time
import math

# Plotting (matplotlib kept optional; functions degrade gracefully if unavailable)
try:  # pragma: no cover - optional dependency
	import matplotlib.pyplot as plt
	_HAVE_MPL = True
except Exception:  # pragma: no cover
	plt = None  # type: ignore
	_HAVE_MPL = False


# ---------------------------------------------------------------------------
# Linear algebra helpers (Kraus <-> stacked Stiefel variable)
# ---------------------------------------------------------------------------
def stack_kraus(k_list: Sequence[np.ndarray]) -> np.ndarray:
	"""Stack r Kraus operators (d x d) vertically -> ((r*d) x d) Stiefel matrix."""
	return np.vstack(k_list).astype(np.complex128, copy=False)


def unstack_kraus(V: np.ndarray, d: int) -> List[np.ndarray]:
	"""Inverse of `stack_kraus`.

	Parameters
	----------
	V : (r*d, d) complex array with column orthonormality (V^† V = I_d)
	d : local dimension
	"""
	r = V.shape[0] // d
	return [V[k * d : (k + 1) * d, :] for k in range(r)]


def riemannian_grad_on_stiefel(V: np.ndarray, G: np.ndarray) -> np.ndarray:
	"""Project Euclidean gradient G onto tangent space at V (complex Stiefel).

	Tangent projection: G - V sym(V^† G)
	"""
	H = V.conj().T @ G
	sym = 0.5 * (H + H.conj().T)
	return G - V @ sym


def qr_retraction(X: np.ndarray) -> np.ndarray:
	"""Retract onto Stiefel via thin QR with positive diagonal signs."""
	Q, R = np.linalg.qr(X)
	diag = np.sign(np.real(np.diag(R)))
	diag[diag == 0] = 1.0
	return Q @ np.diag(diag)


# ---------------------------------------------------------------------------
# POVM & probe states
# ---------------------------------------------------------------------------
def build_sic_povm_qubit() -> List[np.ndarray]:
	"""Return the 4 tetra (SIC) POVM elements E_m for a qubit.

	Analytic construction: E_m = (1/4) (I + r_m · σ) with r_m the vertices of a
	regular tetrahedron. In exact arithmetic these are positive and satisfy
	Σ_m E_m = I. Numerically we build the raw set, then *whiten* by S^{-1/2}
	where S = Σ_m E_m (guaranteeing completeness without breaking Hermiticity
	or positivity). We avoid ad‑hoc eigenvalue clipping unless a (very unlikely)
	numerical negative < -1e-12 appears after whitening.
	"""
	# Tetrahedron Bloch vectors (normalised)
	r = np.array(
		[
			[1, 1, 1],
			[1, -1, -1],
			[-1, 1, -1],
			[-1, -1, 1],
		],
		dtype=float,
	) / np.sqrt(3.0)
	I2 = np.eye(2, dtype=np.complex128)
	sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
	sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
	sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
	sigmas = np.stack([sx, sy, sz], axis=0)
	povm_raw: List[np.ndarray] = []
	for v in r:
		M = I2.copy()
		for k in range(3):
			M += v[k] * sigmas[k]
		M *= 0.25  # (1/4)
		M = 0.5 * (M + M.conj().T)  # enforce Hermiticity symmetrically
		povm_raw.append(M)
	S = sum(povm_raw)
	# Whitening: S^{-1/2} E S^{-1/2}
	Se_vals, Se_vecs = np.linalg.eigh(0.5 * (S + S.conj().T))
	Se_vals_clipped = np.clip(Se_vals, 0.0, None)
	with np.errstate(divide='ignore'):
		inv_sqrt = np.diag(np.where(Se_vals_clipped > 0, Se_vals_clipped ** -0.5, 0.0))
	S_inv_sqrt = (Se_vecs @ inv_sqrt) @ Se_vecs.conj().T
	povm: List[np.ndarray] = []
	for E in povm_raw:
		Ew = S_inv_sqrt @ E @ S_inv_sqrt
		Ew = 0.5 * (Ew + Ew.conj().T)  # re-symmetrise post whitening
		# Optional safety: remove tiny negative eigenvalues due to rounding
		w, U = np.linalg.eigh(Ew)
		if w.min() < -1e-12:
			w = np.clip(w, 0.0, None)
			Ew = (U * w) @ U.conj().T
		povm.append(Ew)
	return povm


def random_pure_states(num: int, seed: Optional[int] = None) -> np.ndarray:
	"""Generate `num` Haar-random pure qubit state vectors (2, num)."""
	rng = np.random.default_rng(seed)
	# Complex normal components, then normalise
	vec = rng.normal(size=(2, num)) + 1j * rng.normal(size=(2, num))
	vec /= np.linalg.norm(vec, axis=0, keepdims=True)
	return vec.astype(np.complex128)


# ---------------------------------------------------------------------------
# Forward model (probs + NLL) & gradient
# ---------------------------------------------------------------------------
def forward_probs(K_list: Sequence[np.ndarray], probes: np.ndarray, povm: Sequence[np.ndarray]) -> np.ndarray:
	"""Compute probabilities p_{i,m} for each probe i and POVM element m.

	Parameters
	----------
	K_list : list of (2,2) Kraus matrices.
	probes : (2, S) complex array of state vectors |ψ_i>.
	povm   : list of 4 POVM elements (2x2) summing to I.

	Returns
	-------
	p : (S, M) array with M=len(povm).
	"""
	d, S = probes.shape
	M = len(povm)
	p = np.zeros((S, M), dtype=float)
	for i in range(S):
		psi = probes[:, i]
		rho = np.outer(psi, psi.conj())  # 2x2
		# Output density
		rho_out = sum(K @ rho @ K.conj().T for K in K_list)
		for m, E in enumerate(povm):
			p[i, m] = (E @ rho_out).trace().real
	# DO NOT renormalise or floor here: TP + POVM completeness should guarantee
	# non-negative rows summing to ~1 (up to FP noise). Keep model 'pure'.
	return p


def neg_log_likelihood(counts: np.ndarray, probs: np.ndarray) -> float:
	counts_safe = counts.astype(float)
	p_safe = np.clip(probs, 1e-14, None)
	return float(-np.sum(counts_safe * np.log(p_safe)))


def gradient_kraus(K_list: Sequence[np.ndarray], probes: np.ndarray, povm: Sequence[np.ndarray], counts: np.ndarray, probs: np.ndarray, eps: float = 1e-14) -> List[np.ndarray]:
	"""Euclidean gradient list matching formula:
		G_k = 2 Σ_i ( Σ_m W_{i,m} E_m ) K_k ρ_i
	  with W_{i,m} = N_i - n_{i,m} / p_{i,m}.
	"""
	d, S = probes.shape
	M = len(povm)
	N_i = counts.sum(axis=1, keepdims=True)  # (S,1)
	p_safe = np.maximum(probs, eps)
	W = N_i - counts / p_safe  # (S,M)

	# Precompute per probe: rho_i and S_i = Σ_m W_{i,m} E_m
	rho_list = [np.outer(probes[:, i], probes[:, i].conj()) for i in range(S)]
	S_list = []
	for i in range(S):
		acc = np.zeros((d, d), dtype=np.complex128)
		for m, E in enumerate(povm):
			acc += W[i, m] * E
		S_list.append(acc)

	grads: List[np.ndarray] = []
	for K in K_list:
		g = np.zeros_like(K)
		for i in range(S):
			g += S_list[i] @ K @ rho_list[i]
		grads.append(2.0 * g)
	return grads


# ---------------------------------------------------------------------------
# Optimiser (Barzilai-Borwein + backtracking on Stiefel manifold)
# ---------------------------------------------------------------------------
@dataclass
class MLEConfig:
	r: int = 2  # number of Kraus ops
	max_iters: int = 300
	step_init: float = 5e-2
	shrink: float = 0.5
	backtracks: int = 8
	armijo_c: float = 1e-4  # Armijo condition constant
	bb_alpha_min: float = 1e-6
	bb_alpha_max: float = 5e1
	tol_grad: float = 1e-6
	auto_stat_tol: bool = True  # adapt grad tolerance using shot noise heuristic
	stat_tol_factor: float = 0.5  # effective_tol = max(tol_grad, stat_tol_factor/sqrt(total_counts))
	plateau_rel_nll: float = 1e-9  # relative NLL improvement threshold
	plateau_window: int = 20       # iterations window for plateau detection
	step_floor: float = 1e-14      # minimum accepted step to consider progress
	max_reinits: int = 1           # reinitialisation attempts on plateau
	reinit_perturb_scale: float = 1e-2
	plateau_grad_factor: float = 5.0  # allow plateau exit if grad > factor*effective_tol
	verbose: bool = True
	seed: int = 0
	init: str = "identity"  # 'identity' | 'mixed' | 'random'
	capture_choi_every: int = 0   # >=1 to store Choi snapshots during optimisation
	max_choi_snapshots: int = 25


def initialise_kraus_qubit(r: int, mode: str, rng: np.random.Generator) -> List[np.ndarray]:
	"""Initial Kraus list for qubit.

	identity: K_0 = I, others zero (then orthonormalised -> identity channel)
	mixed   : Start near depolarising channel (K_k ~ scaled Pauli basis / sqrt(r))
	random  : random complex Gaussian blocks followed by column orthonormalisation.
	"""
	if r < 1:
		raise ValueError("Need at least one Kraus operator")
	if mode == "identity":
		K = [np.eye(2, dtype=np.complex128)] + [np.zeros((2, 2), dtype=np.complex128) for _ in range(r - 1)]
	elif mode == "mixed":
		paulis = [
			np.array([[1, 0], [0, 1]], dtype=np.complex128),
			np.array([[0, 1], [1, 0]], dtype=np.complex128),
			np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
			np.array([[1, 0], [0, -1]], dtype=np.complex128),
		]
		K = []
		for k in range(r):
			P = paulis[k % 4]
			# small random perturbation
			noise = 0.05 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
			K.append((P + noise) / np.sqrt(r))
	elif mode == "depolarising":
		# Exact completely depolarising (Pauli) set requires r>=4 for qubit.
		if r < 4:
			raise ValueError("'depolarising' init requires r>=4 for full Pauli set")
		paulis = [
			np.array([[1, 0], [0, 1]], dtype=np.complex128),
			np.array([[0, 1], [1, 0]], dtype=np.complex128),
			np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
			np.array([[1, 0], [0, -1]], dtype=np.complex128),
		]
		K = [P / 2.0 for P in paulis[:r]]  # each scaled so sum K†K = I
	elif mode == "random":
		K = []
		for _ in range(r):
			M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
			K.append(M)
	else:
		raise ValueError(f"Unknown init mode: {mode}")
	# Stiefel correction
	V = stack_kraus(K)
	V = qr_retraction(V)
	return unstack_kraus(V, 2)


def mle_qubit_channel(counts: np.ndarray, probes: np.ndarray, povm: Sequence[np.ndarray], cfg: MLEConfig, J_true: Optional[np.ndarray] = None) -> Dict[str, object]:
	rng = np.random.default_rng(cfg.seed)
	d = probes.shape[0]
	assert d == 2, "This toy implementation assumes qubit (d=2)."

	K = initialise_kraus_qubit(cfg.r, cfg.init, rng)
	K_initial = [k.copy() for k in K]
	V = stack_kraus(K)
	V = qr_retraction(V)
	K = unstack_kraus(V, d)

	probs = forward_probs(K, probes, povm)
	nll = neg_log_likelihood(counts, probs)

	# Adaptive statistical gradient tolerance (heuristic): O(1/sqrt(N_total))
	N_total = counts.sum()
	effective_tol_grad = max(cfg.tol_grad, cfg.stat_tol_factor / np.sqrt(max(N_total, 1))) if cfg.auto_stat_tol else cfg.tol_grad

	nll_hist = [nll]
	grad_norm_hist = []
	step_hist = []
	fidelity_hist: List[float] = []
	choi_snaps: List[np.ndarray] = []
	t0 = time.perf_counter()

	V_prev = None
	Rg_prev = None
	alpha = cfg.step_init

	reinits_used = 0
	plateau_counter = 0
	prev_nll = nll
	for it in range(1, cfg.max_iters + 1):
		# Gradient blocks & Riemannian projection
		G_blocks = gradient_kraus(K, probes, povm, counts, probs)
		G = stack_kraus(G_blocks)
		Rg = riemannian_grad_on_stiefel(V, G)
		grad_norm = float(np.sqrt(np.vdot(Rg, Rg).real))
		grad_norm_hist.append(grad_norm)
		if J_true is not None:
			# Compute process fidelity each iteration (cheap for d=2)
			J_iter = choi_from_kraus(K)
			fidelity_hist.append(process_fidelity(J_true, J_iter, d=probes.shape[0]))
		if cfg.verbose and (it == 1 or it % 25 == 0):
			msg = f"Iter {it:4d}  NLL={nll:.6f}  |grad|={grad_norm:.3e}  step={alpha:.2e}"
			if fidelity_hist:
				msg += f"  F={fidelity_hist[-1]:.6f}"
			print(msg)
		if grad_norm < effective_tol_grad:
			break

		# Plateau detection
		rel_impr = (prev_nll - nll) / max(abs(prev_nll), 1.0)
		if rel_impr < cfg.plateau_rel_nll and alpha < cfg.step_floor:
			plateau_counter += 1
		else:
			plateau_counter = 0
		prev_nll = nll
		if plateau_counter >= cfg.plateau_window and grad_norm > cfg.plateau_grad_factor * effective_tol_grad and reinits_used < cfg.max_reinits:
			# Reinitialise by small perturbation of current V then re-orthonormalise
			perturb = cfg.reinit_perturb_scale * (np.random.default_rng(cfg.seed + it).normal(size=V.shape) + 1j * np.random.default_rng(cfg.seed + it + 1).normal(size=V.shape))
			V = qr_retraction(V + perturb)
			K = unstack_kraus(V, d)
			probs = forward_probs(K, probes, povm)
			nll = neg_log_likelihood(counts, probs)
			plateau_counter = 0
			reinits_used += 1
			if cfg.verbose:
				print(f"[Reinit] Applied stochastic perturbation at iter {it}, new NLL={nll:.6f}")

		# Barzilai-Borwein step length (after first iteration)
		if V_prev is not None:
			S = V - V_prev
			Y = Rg - Rg_prev
			denom = np.vdot(S, Y).real
			numer = np.vdot(S, S).real
			if denom > 0:
				alpha_bb = numer / denom
				alpha = float(np.clip(alpha_bb, cfg.bb_alpha_min, cfg.bb_alpha_max))

		# Backtracking line search on retracted manifold (Armijo)
		V_prev = V
		Rg_prev = Rg
		descent = -Rg
		found = False
		cur_alpha = alpha
		for bt in range(cfg.backtracks):
			V_trial = qr_retraction(V + cur_alpha * descent)
			K_trial = unstack_kraus(V_trial, d)
			probs_trial = forward_probs(K_trial, probes, povm)
			nll_trial = neg_log_likelihood(counts, probs_trial)
			# Armijo condition: sufficient decrease
			if nll_trial <= nll - cfg.armijo_c * cur_alpha * (grad_norm ** 2):
				V = V_trial
				K = K_trial
				probs = probs_trial
				nll = nll_trial
				found = True
				break
			cur_alpha *= cfg.shrink
		step_hist.append(cur_alpha)
		if not found:
			# If no improvement, reduce alpha aggressively & continue
			alpha *= 0.1
		nll_hist.append(nll)
		if cfg.capture_choi_every and (it % cfg.capture_choi_every == 0):
			if len(choi_snaps) < cfg.max_choi_snapshots:
				choi_snaps.append(choi_from_kraus(K))

	total_time = time.perf_counter() - t0
	return {
		"Kraus": K,
		"Kraus_initial": K_initial,
		"V": V,
		"nll_history": np.array(nll_hist),
		"grad_norm_history": np.array(grad_norm_hist),
		"step_history": np.array(step_hist),
		"final_nll": nll,
		"runtime_sec": total_time,
		"choi_snapshots": np.array(choi_snaps) if choi_snaps else np.empty((0,4,4), dtype=np.complex128),
		"effective_tol_grad": effective_tol_grad,
		"reinits_used": reinits_used,
		"fidelity_history": np.array(fidelity_hist) if fidelity_hist else np.empty((0,), dtype=float),
	}


# ---------------------------------------------------------------------------
# Sampling synthetic data
# ---------------------------------------------------------------------------
def sample_counts(K_list: Sequence[np.ndarray], probes: np.ndarray, povm: Sequence[np.ndarray], shots_per_probe: int, seed: Optional[int] = None) -> np.ndarray:
	"""Sample counts for each probe from channel + POVM model."""
	rng = np.random.default_rng(seed)
	probs = forward_probs(K_list, probes, povm)
	S, M = probs.shape
	counts = np.zeros((S, M), dtype=int)
	for i in range(S):
		counts[i] = rng.multinomial(shots_per_probe, probs[i])
	return counts


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def choi_from_kraus(K_list: Sequence[np.ndarray]) -> np.ndarray:
	"""Return Choi matrix J (d^2 x d^2) using row-major vec grouping (m,p)."""
	d = K_list[0].shape[0]
	J4 = sum(np.einsum("mp,nq->mnpq", K, K.conj()) for K in K_list)
	J = np.transpose(J4, (0, 2, 1, 3)).reshape(d * d, d * d)
	return J


def tp_residual(K_list: Sequence[np.ndarray]) -> float:
	S = sum(K.conj().T @ K for K in K_list)
	return float(np.linalg.norm(S - np.eye(K_list[0].shape[0])))


def process_fidelity(Ja: np.ndarray, Jb: np.ndarray, d: int) -> float:
	"""Process fidelity F = (Tr sqrt( sqrt(Ja) Jb sqrt(Ja) ))^2 / d^2."""
	w, U = np.linalg.eigh(Ja)
	w_clipped = np.clip(w, 0.0, None)
	sqrtJa = (U * np.sqrt(w_clipped)) @ U.conj().T
	A = sqrtJa @ Jb @ sqrtJa
	A = 0.5 * (A + A.conj().T)
	wa, _ = np.linalg.eigh(A)
	wa_clipped = np.clip(wa, 0.0, None)
	return float((np.sum(np.sqrt(wa_clipped)) ** 2) / (d * d))


# ---------------------------------------------------------------------------
# Visualisations (in spirit of stiefel_toy)
# ---------------------------------------------------------------------------
def _ensure_mpl():  # pragma: no cover - runtime guard
	if not _HAVE_MPL:
		raise RuntimeError("matplotlib not available; install it to enable plotting")


def bloch_vector(psi: np.ndarray) -> np.ndarray:
	"""Return Bloch vector for pure qubit state |psi>."""
	sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
	sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
	sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
	rho = np.outer(psi, psi.conj())
	return np.array([
		np.real(np.trace(rho @ sx)),
		np.real(np.trace(rho @ sy)),
		np.real(np.trace(rho @ sz)),
	])


def plot_probe_distribution(probes: np.ndarray, ax=None):  # pragma: no cover - plotting
	_ensure_mpl()
	if ax is None:
		fig = plt.figure(figsize=(4,4))
		ax = fig.add_subplot(111, projection='3d')
	vecs = np.stack([bloch_vector(probes[:, i]) for i in range(probes.shape[1])])
	ax.scatter(vecs[:,0], vecs[:,1], vecs[:,2], c='C0', s=25)
	# Draw sphere wireframe
	u = np.linspace(0, 2*np.pi, 24)
	v = np.linspace(0, np.pi, 12)
	xs = np.outer(np.cos(u), np.sin(v))
	ys = np.outer(np.sin(u), np.sin(v))
	zs = np.outer(np.ones_like(u), np.cos(v))
	ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.5, alpha=0.5)
	ax.set_title('Probe Bloch vectors')
	ax.set_box_aspect([1,1,1])
	ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
	return ax


def plot_training_curves(res: Dict[str, object]):  # pragma: no cover - plotting
	_ensure_mpl()
	nll = res['nll_history']
	grad = res['grad_norm_history']
	steps = res['step_history']
	fig, axes = plt.subplots(1,3, figsize=(12,3.2), constrained_layout=True)
	axes[0].plot(nll, 'C0')
	axes[0].set_title('NLL'); axes[0].set_xlabel('Iter')
	axes[1].semilogy(grad, 'C1')
	axes[1].set_title('|grad|'); axes[1].set_xlabel('Iter')
	axes[2].semilogy(steps, 'C2')
	axes[2].set_title('Accepted step'); axes[2].set_xlabel('Update #')
	fig.suptitle('Optimisation diagnostics')
	fig.tight_layout()
	return fig, axes


def plot_choi_progress(J_true: np.ndarray, res: Dict[str, object]):  # pragma: no cover - plotting
	_ensure_mpl()
	snaps = res['choi_snapshots']
	if snaps.shape[0] == 0:
		print('No Choi snapshots captured (set capture_choi_every in config)')
		return None
	n = snaps.shape[0]
	cols = min(6, n)
	rows = math.ceil(n / cols)
	vmax = max(np.abs(J_true).max(), np.abs(snaps).max())
	fig, axes = plt.subplots(rows, cols, figsize=(2.2*cols, 2.2*rows), constrained_layout=True)
	axes = np.atleast_1d(axes).ravel()
	for i in range(n):
		im = axes[i].imshow(snaps[i].real, cmap='coolwarm', vmin=-vmax, vmax=vmax)
		axes[i].set_title(f'Iter snap {i+1}')
		axes[i].axis('off')
	for j in range(n, rows*cols):
		axes[j].axis('off')
	fig.colorbar(im, ax=axes.tolist(), shrink=0.7)
	fig.suptitle('Choi snapshots (Re part)')
	fig.tight_layout()
	return fig, axes


def plot_final_choi_comparison(J_true: np.ndarray, J_init: np.ndarray, J_est: np.ndarray):  # pragma: no cover - plotting
	_ensure_mpl()
	vmax = max(np.abs(J_true).max(), np.abs(J_init).max(), np.abs(J_est).max())
	mats = [J_true.real, J_init.real, J_est.real]
	titles = ['Truth Re', 'Init Re', 'Est Re']
	fig, axes = plt.subplots(1, 3, figsize=(9,3), constrained_layout=True)
	ims = []
	for ax, M, t in zip(axes, mats, titles):
		ims.append(ax.imshow(M, cmap='coolwarm', vmin=-vmax, vmax=vmax))
		ax.set_title(t); ax.axis('off')
	fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.7)
	fig.suptitle('Choi matrices (real part)')
	fig.tight_layout()
	return fig, axes


def plot_kraus_operators(K_list: Sequence[np.ndarray], title: str = 'Kraus operators'):  # pragma: no cover - plotting
	_ensure_mpl()
	r = len(K_list)
	fig, axes = plt.subplots(2, r, figsize=(2*r,4), constrained_layout=True)
	vmax = max(np.max(np.abs(K.real)) for K in K_list)
	for idx, K in enumerate(K_list):
		im0 = axes[0, idx].imshow(K.real, cmap='viridis', vmin=-vmax, vmax=vmax)
		axes[0, idx].set_title(f'Re K{idx}')
		axes[0, idx].axis('off')
		im1 = axes[1, idx].imshow(K.imag, cmap='viridis', vmin=-vmax, vmax=vmax)
		axes[1, idx].set_title(f'Im K{idx}')
		axes[1, idx].axis('off')
	fig.colorbar(im0, ax=axes[0].ravel().tolist(), shrink=0.6)
	fig.colorbar(im1, ax=axes[1].ravel().tolist(), shrink=0.6)
	fig.suptitle(title)
	fig.tight_layout()
	return fig, axes


def summary_figure(probes: np.ndarray, res: Dict[str, object], J_true: np.ndarray, J_init: np.ndarray, J_est: np.ndarray):  # pragma: no cover - plotting
	"""Single composite figure with:
	Row 1: Probe Bloch 3D (colspan=2), NLL, |grad| (log), step size (log)
	Row 2: Truth / Init / Est Choi (real part) + shared colorbar.
	"""
	_ensure_mpl()
	from matplotlib import gridspec
	fig = plt.figure(figsize=(18, 8), layout='constrained')
	gs = gridspec.GridSpec(2, 8, figure=fig)  # 8 columns

	# Panel: probe distribution (span 2 columns)
	ax_probe = fig.add_subplot(gs[0, 0:3], projection='3d')
	vecs = np.stack([bloch_vector(probes[:, i]) for i in range(probes.shape[1])])
	ax_probe.scatter(vecs[:,0], vecs[:,1], vecs[:,2], c='C0', s=28, depthshade=True)
	# sphere wireframe (coarse)
	u = np.linspace(0, 2*np.pi, 24)
	v = np.linspace(0, np.pi, 12)
	xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v)); zs = np.outer(np.ones_like(u), np.cos(v))
	ax_probe.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.4, alpha=0.4)
	ax_probe.set_title('Probe Bloch vectors')
	ax_probe.set_box_aspect([1,1,1])
	ax_probe.set_xlabel('x'); ax_probe.set_ylabel('y'); ax_probe.set_zlabel('z')

	# Training curves
	nll = res['nll_history']
	grad = res['grad_norm_history']
	steps = res['step_history'] if len(res['step_history']) else np.array([np.nan])
	# NLL
	ax_nll = fig.add_subplot(gs[0, 3])
	ax_nll.plot(nll, 'C0')
	ax_nll.set_title('NLL')
	ax_nll.set_xlabel('Iter')
	ax_nll.set_ylabel('Value')
	# Fidelity
	fid_hist = res.get('fidelity_history', np.empty((0,)))
	ax_fid = fig.add_subplot(gs[0, 4])
	if fid_hist.size:
		ax_fid.plot(fid_hist, 'C4')
		ax_fid.set_ylim(0.0, 1.01)
	else:
		ax_fid.text(0.5, 0.5, 'No fidelity history', ha='center', va='center')
	ax_fid.set_title('Process fidelity')
	ax_fid.set_xlabel('Iter')
	# Grad norm
	ax_grad = fig.add_subplot(gs[0, 5])
	ax_grad.semilogy(grad, 'C1')
	ax_grad.set_title('|grad|')
	ax_grad.set_xlabel('Iter')
	# Step sizes
	ax_step = fig.add_subplot(gs[0, 6])
	if np.all(np.isfinite(steps)):
		ax_step.semilogy(steps, 'C2')
	else:
		ax_step.plot(steps, 'C2')
	ax_step.set_title('Accepted step')
	ax_step.set_xlabel('Update #')
	# Empty small panel for legend / text stats
	ax_txt = fig.add_subplot(gs[0, 7])
	ax_txt.axis('off')
	stats_lines = [
		f"Final NLL: {res['final_nll']:.2f}",
		f"|grad|: {grad[-1]:.2e}",
		f"tol*: {res.get('effective_tol_grad', np.nan):.2e}",
		f"Steps: {len(nll)-1}",
	]
	ax_txt.text(0.0, 0.95, '\n'.join(stats_lines), va='top', ha='left', fontsize=9)

	# Choi panels (second row)
	vmax = max(np.abs(J_true).max(), np.abs(J_init).max(), np.abs(J_est).max())
	cm = 'coolwarm'
	ax_Jt = fig.add_subplot(gs[1, 0:2])
	ax_Ji = fig.add_subplot(gs[1, 2:4])
	ax_Je = fig.add_subplot(gs[1, 4:6])
	im_t = ax_Jt.imshow(J_true.real, cmap=cm, vmin=-vmax, vmax=vmax)
	ax_Jt.set_title('Truth Choi (Re)'); ax_Jt.axis('off')
	ax_Ji.imshow(J_init.real, cmap=cm, vmin=-vmax, vmax=vmax)
	ax_Ji.set_title('Init Choi (Re)'); ax_Ji.axis('off')
	ax_Je.imshow(J_est.real, cmap=cm, vmin=-vmax, vmax=vmax)
	ax_Je.set_title('Est Choi (Re)'); ax_Je.axis('off')
	# Shared colorbar occupying last column row 2
	ax_cb = fig.add_subplot(gs[1, 6:8])
	fig.colorbar(im_t, cax=ax_cb)
	ax_cb.set_title('Scale')
	fig.suptitle('Qubit Channel MLE Summary', fontsize=14)
	return fig


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main():
	d = 2
	S = 40                # number of probe states
	shots = 400           # counts per probe
	r = 4                 # Kraus rank (need >=4 for depolarising init)
	seed = 123

	probes = random_pure_states(S, seed=seed)
	povm = build_sic_povm_qubit()

	# Ground truth channel: slight amplitude damping style mixture
	gamma = 0.15
	K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]], dtype=np.complex128)
	K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
	K_true = [K0, K1]
	counts = sample_counts(K_true, probes, povm, shots_per_probe=shots, seed=seed + 1)

	cfg = MLEConfig(r=r, max_iters=400, init="depolarising", verbose=True, seed=seed,
					capture_choi_every=25, max_choi_snapshots=12)
	# Build truth Choi before optimisation for fidelity tracking
	J_true = choi_from_kraus(K_true)
	res = mle_qubit_channel(counts, probes, povm, cfg, J_true=J_true)

	# Diagnostics
	J_est = choi_from_kraus(res["Kraus"])
	# J_true already computed above
	fro_rel = np.linalg.norm(J_est - J_true) / np.linalg.norm(J_true)
	tp_err = tp_residual(res["Kraus"])
	# Choi eigen diagnostics (CP check)
	evals_est, _ = np.linalg.eigh(J_est)
	min_eig = float(evals_est.min())
	neg_eigs = int((evals_est < -1e-10).sum())
	# Process fidelity (normalized) F = (Tr sqrt( sqrt(J_true) J_est sqrt(J_true) ))^2 / d^2
	proc_F = process_fidelity(J_true, J_est, d)
	# Empirical vs fitted probability KL per probe
	probs_est = forward_probs(res["Kraus"], probes, povm)
	S = counts.sum(axis=1, keepdims=True)
	with np.errstate(divide='ignore', invalid='ignore'):
		p_emp = np.where(S > 0, counts / S, 0.0)
		kl_per = np.sum(np.where(p_emp > 0, p_emp * (np.log(p_emp + 1e-14) - np.log(probs_est + 1e-14)), 0.0), axis=1)
	kl_mean = float(kl_per.mean())
	kl_max = float(kl_per.max())

	print("--- Results ---")
	print(f"Final NLL: {res['final_nll']:.4f}")
	print(f"Iterations used: {len(res['nll_history'])-1}")
	print(f"Final grad norm: {res['grad_norm_history'][-1]:.3e}  (eff tol={res['effective_tol_grad']:.2e})")
	print(f"Trace-preserving residual: {tp_err:.2e}")
	print(f"Choi relative Frobenius error vs truth: {fro_rel:.3e}")
	print(f"Process fidelity: {proc_F:.6f}")
	print(f"Choi min eigenvalue: {min_eig:.3e} (neg eigs>{-1e-10}: {neg_eigs})")
	print(f"KL per-probe mean={kl_mean:.3e}  max={kl_max:.3e}")
	print(f"Reinitialisations used: {res['reinits_used']}")
	print(f"Runtime: {res['runtime_sec']:.2f} s")

	# Single summary figure
	if _HAVE_MPL:
		try:
			J_init = choi_from_kraus(res['Kraus_initial'])
			fig = summary_figure(probes, res, J_true, J_init, J_est)
			plt.show()
		except Exception as e:  # pragma: no cover
			print(f"Plotting failed: {e}")


if __name__ == "__main__":
	main()
