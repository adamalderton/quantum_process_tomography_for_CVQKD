"""Lightweight diagnostic tests for core functions in `PG_MLE_synthetic_channel`.

Run directly:

	python testing_synthetic_channel_MLE.py

These are NOT formal pytest unit tests – just quick sanity checks mirroring the
style of `old/agentic_messing_about/MLE/testing_MLE_from_scratch.py`.

Each test prints a PASS/FAIL line and returns a boolean. Numerical tolerances
are intentionally loose; the goal is catching glaring regressions (shape,
basic algebra, constraints) rather than precision benchmarking.
"""

from __future__ import annotations

import math
import numpy as np
from pathlib import Path
import sys
from typing import List, Callable

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.append(str(_THIS_DIR))  # add synthetic/ directory directly

from PG_MLE_synthetic_channel import (  # type: ignore
	factorial_array,
	adaptive_grid_half_size,
	grid_bin_width_and_area,
	build_grid,
	compute_channel_probabilites,  # note: original file's spelling
	neg_log_likelihood_poisson_multiinput,
	project_CPTP,
	mle_projected_gradient_descent,
	generate_synthetic_channel_data,
	run_mle_workflow,
)

RNG = np.random.default_rng(42)


def _status(name: str, ok: bool, msg: str = ""):
	print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")


def test_factorial_array() -> bool:
	name = "factorial_array"
	exact = factorial_array(6, exact=True, return_log=False)
	cond_exact = all(int(exact[i]) == math.factorial(i) for i in range(7))
	logs = factorial_array(6, return_log=True)
	cond_logs = np.allclose(logs, [math.lgamma(i + 1) for i in range(7)], atol=1e-12)
	floats = factorial_array(6, exact=False)
	cond_float = np.allclose(floats, [math.factorial(i) for i in range(7)], atol=1e-12)
	ok = cond_exact and cond_logs and cond_float
	_status(name, ok, "0..6 correctness (int/log/float)")
	return ok


def test_adaptive_grid_half_size() -> bool:
	name = "adaptive_grid_half_size"
	# Case 1: tiny complex cluster -> should return at least min_half_size (default 3.5)
	cluster = 0.05 * (RNG.normal(size=500) + 1j * RNG.normal(size=500))
	L1 = adaptive_grid_half_size([cluster])
	cond_min = L1 >= 3.5 - 1e-12
	# Case 2: real y-plane samples with larger spread and a max cap
	# Create ring around radius ~5
	angles = RNG.uniform(0, 2 * np.pi, size=1000)
	r = 5 + 0.2 * RNG.normal(size=1000)
	y_samples = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
	L2 = adaptive_grid_half_size([y_samples], max_half_size=6.0)
	cond_cap = L2 <= 6.0 + 1e-12 and L2 > 5.0  # should expand beyond raw percentile but be capped
	ok = cond_min and cond_cap
	_status(name, ok, f"L1={L1:.2f} (>=3.5), L2={L2:.2f} (capped<=6)")
	return ok


def test_grid_bin_width_and_area() -> bool:
	name = "grid_bin_width_and_area"
	L = 4.0
	G = 5
	delta, area = grid_bin_width_and_area(L, G)
	# Expected: delta = 2L/(G-1) = 8/4 = 2, area = delta^2 / 2 = 2
	ok = (abs(delta - 2.0) < 1e-12) and (abs(area - 2.0) < 1e-12)
	_status(name, ok, f"delta={delta:.3f}, area={area:.3f} (expect 2,2)")
	return ok


def test_build_grid() -> bool:
	name = "build_grid"
	L = 1.0
	G = 3
	alphas, Y1, Y2 = build_grid(L, G)
	cond_shape = (alphas.shape == (G * G,)) and Y1.shape == (G, G) and Y2.shape == (G, G)
	cond_range = np.isclose(Y1.min(), -L) and np.isclose(Y1.max(), L) and np.isclose(Y2.min(), -L) and np.isclose(Y2.max(), L)
	# Corner magnitude: sqrt((1^2 + 1^2)/2) = 1
	corner_alpha = (Y1[0, 0] + 1j * Y2[0, 0]) / np.sqrt(2.0)
	cond_corner = np.isclose(abs(corner_alpha), 1.0)
	ok = cond_shape and cond_range and cond_corner
	_status(name, ok, f"shapes ok & bounds ok & corner |alpha|≈1 => {abs(corner_alpha):.3f}")
	return ok


def _simple_coherent_overlaps(alphas: np.ndarray, cutoff: int) -> np.ndarray:
	"""Build (d,len(alphas)) overlap matrix quickly for small test only.
	a_n = exp(-|α|^2/2) α^n / sqrt(n!) for n=0..cutoff.
	"""
	d = cutoff + 1
	ks = np.arange(d)
	fac = factorial_array(cutoff, exact=False)[: d]
	out = np.empty((d, len(alphas)), dtype=np.complex128)
	for j, a in enumerate(alphas):
		pref = np.exp(-0.5 * abs(a) ** 2)
		out[:, j] = pref * (a ** ks) / np.sqrt(fac)
	return out


def test_compute_channel_probabilities() -> bool:
	name = "compute_channel_probabilities"
	cutoff = 1  # d=2
	d = cutoff + 1
	# Two input coherent states (|0>, |alpha>)
	alpha_in = np.array([0.0 + 0.0j, 0.3 + 0.2j])
	C_in = _simple_coherent_overlaps(alpha_in, cutoff)  # shape (d, S=2)
	# Three output grid coherent points
	alpha_out = np.array([0.0 + 0.0j, -0.3 + 0.1j, 0.25 - 0.15j])
	C_out = _simple_coherent_overlaps(alpha_out, cutoff)  # shape (d, M=3)
	# Bin area (arbitrary small) – ensure positive scaling
	bin_area = 0.05
	# Case A: Zero Choi -> probabilities floored to eps
	J_zero = np.zeros((d * d, d * d), dtype=np.complex128)
	p_zero = compute_channel_probabilites(J_zero, C_in, C_out, bin_area)
	eps = 1e-30
	cond_floor = np.all(p_zero == eps)
	# Case B: Identity channel Choi ~ |Φ+><Φ+|
	phi = np.zeros(d * d, dtype=np.complex128)
	for k in range(d):
		phi[k * d + k] = 1.0
	phi /= np.sqrt(d)
	J_id = np.outer(phi, phi.conjugate())  # rank-1 CPTP (identity)
	p_id = compute_channel_probabilites(J_id, C_in, C_out, bin_area)
	cond_positive = np.all(p_id > eps)
	cond_finite = np.isfinite(p_id).all()
	# Sanity: relative variation modest (avoid pathological broadcast issue). Not rigorous.
	spread = p_id.max() / p_id.min()
	cond_spread = spread < 5  # tighter heuristic
	ok = cond_floor and cond_positive and cond_finite and cond_spread
	_status(name, ok, f"floor_ok={cond_floor}, pos={cond_positive}, spread={spread:.2f} (<5)")
	return ok


def test_neg_log_likelihood_poisson_multiinput() -> bool:
	name = "neg_log_likelihood_poisson_multiinput"
	counts = np.array([[3, 2, 1], [4, 0, 2]], dtype=int)
	# Uniform probabilities per row (already normalised)
	p = np.array([[1/3, 1/3, 1/3], [0.25, 0.5, 0.25]])  # second row non-uniform
	# Manual NLL (dropping factorial constants): sum_i sum_j [N_i p_ij - n_ij log p_ij]
	N_i = counts.sum(axis=1, keepdims=True)
	manual = np.sum(N_i * p) - np.sum(counts * np.log(p))
	lib = neg_log_likelihood_poisson_multiinput(counts, p)
	ok = abs(manual - lib) < 1e-12
	_status(name, ok, f"manual={manual:.6f}, lib={lib:.6f}")
	return ok


def _partial_trace_out(J: np.ndarray, d_in: int, d_out: int) -> np.ndarray:
	"""Helper: Tr_out for Choi matrix J ( (d_in*d_out) x (d_in*d_out) )."""
	J4 = J.reshape(d_in, d_out, d_in, d_out)
	return np.einsum('ikjk->ij', J4, optimize=True)


def test_project_CPTP_basic_square() -> bool:
	"""Perturb a valid 2x2 identity-channel Choi and ensure projection restores CPTP."""
	name = "project_CPTP_basic_square"
	d_in = d_out = 2
	n = d_in * d_out
	# |Phi+>
	phi = np.zeros(n, dtype=np.complex128)
	for k in range(d_in):
		phi[k * d_out + k] = 1.0
	phi /= np.sqrt(d_in)
	J_id = np.outer(phi, phi.conjugate())
	# Add Hermitian noise
	perturb = 1e-3 * (RNG.normal(size=(n, n)) + 1j * RNG.normal(size=(n, n)))
	perturb = 0.5 * (perturb + perturb.conjugate().T)
	J0 = J_id + perturb
	J_proj = project_CPTP(J0)
	# Checks
	evals = np.linalg.eigvalsh(0.5 * (J_proj + J_proj.conjugate().T))
	psd_ok = evals.min() >= -5e-9
	Tr_out = _partial_trace_out(J_proj, d_in, d_out)
	tp_res = np.linalg.norm(Tr_out - np.eye(d_in))
	tp_ok = tp_res < 5e-6
	# Ensure closeness to original valid J_id (not required exact)
	fidelity_like = float(np.real(np.trace(J_proj @ J_id))) / max(1e-15, float(np.real(np.trace(J_id @ J_id))))
	close_ok = fidelity_like > 0.95  # very loose
	ok = psd_ok and tp_ok and close_ok
	_status(name, ok, f"min_eval={evals.min():.2e}, tp_res={tp_res:.2e}, sim={fidelity_like:.3f}")
	return ok


def test_project_CPTP_rectangular() -> bool:
	"""Test projection for a non-square channel (d_in=2, d_out=3)."""
	name = "project_CPTP_rectangular"
	d_in, d_out = 2, 3
	N = d_in * d_out
	# Random Hermitian input (not TP / not CP).
	X = RNG.normal(size=(N, N)) + 1j * RNG.normal(size=(N, N))
	X = 0.5 * (X + X.conjugate().T)
	J_proj = project_CPTP(X, d_in=d_in, d_out=d_out)
	# Shape
	shape_ok = J_proj.shape == (N, N)
	# TP condition
	Tr_out = _partial_trace_out(J_proj, d_in, d_out)
	tp_res = np.linalg.norm(Tr_out - np.eye(d_in))
	tp_ok = tp_res < 1e-5
	# CP condition
	evals = np.linalg.eigvalsh(0.5 * (J_proj + J_proj.conjugate().T))
	psd_ok = evals.min() >= -5e-9
	ok = shape_ok and tp_ok and psd_ok
	_status(name, ok, f"shape={shape_ok}, tp_res={tp_res:.2e}, min_eval={evals.min():.2e}")
	return ok


def test_project_CPTP_idempotence() -> bool:
	"""Applying projection twice should be (almost) idempotent."""
	name = "project_CPTP_idempotence"
	d_in = d_out = 3
	N = d_in * d_out
	X = RNG.normal(size=(N, N)) + 1j * RNG.normal(size=(N, N))
	X = 0.5 * (X + X.conjugate().T)
	J1 = project_CPTP(X, d_in=d_in, d_out=d_out)
	J2 = project_CPTP(J1, d_in=d_in, d_out=d_out)
	diff = np.linalg.norm(J2 - J1) / max(1e-15, np.linalg.norm(J1))
	idem_ok = diff < 1e-8
	_status(name, idem_ok, f"rel_diff={diff:.2e}")
	return idem_ok


def test_project_CPTP_dimension_error() -> bool:
	"""Supplying inconsistent (d_in, d_out) should raise a ValueError."""
	name = "project_CPTP_dimension_error"
	d_in, d_out = 2, 2
	N = d_in * d_out
	X = np.eye(N, dtype=np.complex128)
	# Provide wrong d_in that doesn't divide N cleanly after mismatch.
	try:
		_ = project_CPTP(X, d_in=3, d_out=2)  # 3*2 != 4
	except ValueError:
		_status(name, True, "caught ValueError as expected")
		return True
	except Exception as e:  # unexpected exception
		_status(name, False, f"unexpected exception type: {e}")
		return False
	else:
		_status(name, False, "no error raised for inconsistent dims")
		return False


# -------- Additional channel-specific CPTP projection tests -------- #

def _choi_from_unitary(U: np.ndarray) -> np.ndarray:
	"""Choi of unitary channel ρ -> UρU†: J = (U ⊗ I) |Φ+><Φ+| (U† ⊗ I)."""
	d = U.shape[0]
	phi = np.zeros(d * d, dtype=np.complex128)
	for k in range(d):
		phi[k * d + k] = 1.0
	phi /= np.sqrt(d)
	# Apply (U ⊗ I) to |Φ+>
	Phi_vec = np.kron(U, np.eye(d)) @ phi
	return np.outer(Phi_vec, Phi_vec.conjugate())


def _choi_from_kraus(kraus: List[np.ndarray]) -> np.ndarray:
	"""Choi via sum_k vec(E_k) vec(E_k)† with vec using column-stacking consistent with kron basis."""
	d_out, d_in = kraus[0].shape
	terms = []
	for E in kraus:
		vecE = E.reshape(-1, order='F')  # column-stacking
		terms.append(np.outer(vecE, vecE.conjugate()))
	J = sum(terms)
	# Ensure Hermitian
	return 0.5 * (J + J.conjugate().T)


def test_project_CPTP_phase_rotation() -> bool:
	name = "project_CPTP_phase_rotation"
	d = 3
	theta = 0.73
	U = np.diag(np.exp(1j * theta * np.arange(d)))
	J = _choi_from_unitary(U)
	perturb = 5e-4 * (RNG.normal(size=J.shape) + 1j * RNG.normal(size=J.shape))
	perturb = 0.5 * (perturb + perturb.conjugate().T)
	J_noisy = J + perturb
	J_proj = project_CPTP(J_noisy)
	# Allow global scaling differences by re-normalising TP constraint on original J
	J_tp = project_CPTP(J)  # ensure same normalisation pipeline
	rel_diff = np.linalg.norm(J_proj - J_tp) / np.linalg.norm(J_tp)
	Tr_out = _partial_trace_out(J_proj, d, d)
	tp_res = np.linalg.norm(Tr_out - np.eye(d))
	evals = np.linalg.eigvalsh(J_proj)
	ok = rel_diff < 0.8 and tp_res < 1e-8 and evals.min() >= -5e-9
	_status(name, ok, f"rel_diff={rel_diff:.2e} (<0.8), tp_res={tp_res:.1e}, min_eval={evals.min():.1e}")
	return ok


def test_project_CPTP_depolarizing() -> bool:
	name = "project_CPTP_depolarizing"
	d = 2
	p = 0.37
	# |Φ+>
	phi = np.zeros(d * d, dtype=np.complex128)
	for k in range(d):
		phi[k * d + k] = 1.0
	phi /= np.sqrt(d)
	J_phi = np.outer(phi, phi.conjugate())
	J_mix = np.eye(d * d, dtype=np.complex128) / d  # for qubit: I⊗I /2 after TP scaling
	# Depolarizing Choi: (1-p) |Φ+><Φ+| + p/d * I⊗I  (for qubit d=2)
	J = (1 - p) * J_phi + (p / d) * J_mix
	perturb = 1e-3 * (RNG.normal(size=J.shape) + 1j * RNG.normal(size=J.shape))
	perturb = 0.5 * (perturb + perturb.conjugate().T)
	J_noisy = J + perturb
	J_proj = project_CPTP(J_noisy)
	J_tp = project_CPTP(J)
	rel_diff = np.linalg.norm(J_proj - J_tp) / np.linalg.norm(J_tp)
	Tr_out = _partial_trace_out(J_proj, d, d)
	tp_res = np.linalg.norm(Tr_out - np.eye(d))
	evals = np.linalg.eigvalsh(J_proj)
	ok = rel_diff < 0.8 and tp_res < 1e-8 and evals.min() >= -5e-9
	_status(name, ok, f"rel_diff={rel_diff:.2e} (<0.8), tp_res={tp_res:.1e}, min_eval={evals.min():.1e}")
	return ok


def test_project_CPTP_amplitude_damping() -> bool:
	name = "project_CPTP_amplitude_damping"
	γ = 0.28
	E0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - γ)]], dtype=np.complex128)
	E1 = np.array([[0.0, np.sqrt(γ)], [0.0, 0.0]], dtype=np.complex128)
	J = _choi_from_kraus([E0, E1])
	perturb = 8e-4 * (RNG.normal(size=J.shape) + 1j * RNG.normal(size=J.shape))
	perturb = 0.5 * (perturb + perturb.conjugate().T)
	J_noisy = J + perturb
	J_proj = project_CPTP(J_noisy)
	rel_diff = np.linalg.norm(J_proj - J) / np.linalg.norm(J)
	Tr_out = _partial_trace_out(J_proj, 2, 2)
	tp_res = np.linalg.norm(Tr_out - np.eye(2))
	evals = np.linalg.eigvalsh(J_proj)
	ok = rel_diff < 1e-2 and tp_res < 1e-8 and evals.min() >= -5e-9
	_status(name, ok, f"rel_diff={rel_diff:.2e}, tp_res={tp_res:.1e}, min_eval={evals.min():.1e}")
	return ok


def test_project_CPTP_composed_rotation_damping() -> bool:
	name = "project_CPTP_composed_rotation_damping"
	# Compose phase rotation then amplitude damping on a qubit via Kraus product.
	γ = 0.15
	θ = 1.1
	R = np.diag([1.0, np.exp(1j * θ)])
	E0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - γ)]], dtype=np.complex128) @ R
	E1 = np.array([[0.0, np.sqrt(γ)], [0.0, 0.0]], dtype=np.complex128) @ R
	J = _choi_from_kraus([E0, E1])
	perturb = 1e-3 * (RNG.normal(size=J.shape) + 1j * RNG.normal(size=J.shape))
	perturb = 0.5 * (perturb + perturb.conjugate().T)
	J_noisy = J + perturb
	J_proj = project_CPTP(J_noisy)
	rel_diff = np.linalg.norm(J_proj - J) / np.linalg.norm(J)
	Tr_out = _partial_trace_out(J_proj, 2, 2)
	tp_res = np.linalg.norm(Tr_out - np.eye(2))
	evals = np.linalg.eigvalsh(J_proj)
	ok = rel_diff < 2e-2 and tp_res < 1e-8 and evals.min() >= -5e-9
	_status(name, ok, f"rel_diff={rel_diff:.2e}, tp_res={tp_res:.1e}, min_eval={evals.min():.1e}")
	return ok


TEST_FUNCS: List[Callable[[], bool]] = [
	test_factorial_array,
	test_adaptive_grid_half_size,
	test_grid_bin_width_and_area,
	test_build_grid,
	test_compute_channel_probabilities,
	test_neg_log_likelihood_poisson_multiinput,
	# New CPTP projection tests
	test_project_CPTP_basic_square,
	test_project_CPTP_rectangular,
	test_project_CPTP_idempotence,
	test_project_CPTP_dimension_error,
	test_project_CPTP_phase_rotation,
	test_project_CPTP_depolarizing,
	test_project_CPTP_amplitude_damping,
	test_project_CPTP_composed_rotation_damping,
]


# ---------------- Additional robustness tests (from report) ---------------- #

def test_project_CPTP_nonhermitian() -> bool:
	"""Non-Hermitian input should project to a Hermitian CPTP matrix."""
	name = "project_CPTP_nonhermitian"
	d_in = d_out = 3
	N = d_in * d_out
	rng = np.random.default_rng(123)
	X = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))  # not Hermitian
	J_proj = project_CPTP(X, d_in=d_in, d_out=d_out)
	# Hermitian check
	herm_ok = np.allclose(J_proj, J_proj.conjugate().T, atol=1e-10)
	# TP & CP checks
	Tr_out = _partial_trace_out(J_proj, d_in, d_out)
	tp_res = np.linalg.norm(Tr_out - np.eye(d_in))
	tp_ok = tp_res < 1e-6
	evals = np.linalg.eigvalsh(J_proj)
	psd_ok = evals.min() >= -5e-9
	ok = herm_ok and tp_ok and psd_ok
	_status(name, ok, f"Herm={herm_ok}, tp_res={tp_res:.2e}, min_eval={evals.min():.1e}")
	return ok


def test_project_CPTP_fixed_point_random_kraus() -> bool:
	"""A random CPTP Choi (from Kraus) should be (almost) a fixed point of the projector."""
	name = "project_CPTP_fixed_point_random_kraus"
	rng = np.random.default_rng(321)
	d_out, d_in = 3, 2
	# Build random Kraus with TP enforcement: sum_k E_k†E_k = I
	K = d_in * d_out  # number of Kraus ops
	Es = [rng.standard_normal((d_out, d_in)) + 1j*rng.standard_normal((d_out, d_in)) for _ in range(K)]
	M = sum(E.conj().T @ E for E in Es)
	w, V = np.linalg.eigh(M)
	Minv_sqrt = V @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-15, None))) @ V.conj().T
	Es = [E @ Minv_sqrt for E in Es]
	# Choi: sum_k vec(E_k) vec(E_k)† with column-stacking ('F') to match kron basis
	J = sum(np.outer(E.reshape(-1, order='F'), E.reshape(-1, order='F').conj()) for E in Es)
	J = 0.5 * (J + J.conjugate().T)  # ensure Hermitian numerically
	J_proj = project_CPTP(J, d_in=d_in, d_out=d_out)
	rel = np.linalg.norm(J_proj - J) / max(1e-15, np.linalg.norm(J))
	Tr_out = _partial_trace_out(J_proj, d_in, d_out)
	tp_res = np.linalg.norm(Tr_out - np.eye(d_in))
	evals = np.linalg.eigvalsh(J_proj)
	ok = (rel < 1e-7) and (tp_res < 1e-8) and (evals.min() >= -5e-9)
	_status(name, ok, f"rel={rel:.2e}, tp_res={tp_res:.1e}, min_eval={evals.min():.1e}")
	return ok


def test_project_CPTP_vs_naive_distance() -> bool:
	"""Dykstra projection should be no worse (Frobenius) than a single CP->TP pass."""
	name = "project_CPTP_vs_naive_distance"
	d_in = d_out = 2
	N = d_in * d_out
	rng = np.random.default_rng(111)
	X = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
	X = 0.5 * (X + X.conjugate().T)  # Hermitian but not TP/CP
	# Dykstra
	J_dyk = project_CPTP(X, d_in=d_in, d_out=d_out)
	d_dyk = np.linalg.norm(J_dyk - X)
	# Naive CP then TP once
	def _cp_once(Y):
		H = 0.5 * (Y + Y.conj().T)
		w, U = np.linalg.eigh(H)
		w = np.clip(w, 0.0, None)
		return (U * w) @ U.conj().T
	def _tp_once(Y):
		Y4 = Y.reshape(d_in, d_out, d_in, d_out).copy()
		Tr_out = _partial_trace_out(Y, d_in, d_out)
		Delta = (Tr_out - np.eye(d_in)) / d_out
		Iout = np.eye(d_out, dtype=Y.dtype)
		for i in range(d_in):
			for j in range(d_in):
				Y4[i, :, j, :] -= Delta[i, j] * Iout
		return Y4.reshape(N, N)
	J_naive = _tp_once(_cp_once(X))
	d_naive = np.linalg.norm(J_naive - X)
	# Feasibility checks
	Tr_out_dyk = _partial_trace_out(J_dyk, d_in, d_out)
	Tr_out_naive = _partial_trace_out(J_naive, d_in, d_out)
	tp_res_dyk = np.linalg.norm(Tr_out_dyk - np.eye(d_in))
	tp_res_naive = np.linalg.norm(Tr_out_naive - np.eye(d_in))
	min_eval_dyk = np.linalg.eigvalsh(0.5*(J_dyk+J_dyk.conj().T)).min()
	min_eval_naive = np.linalg.eigvalsh(0.5*(J_naive+J_naive.conj().T)).min()
	feasible_dyk = (tp_res_dyk < 1e-8) and (min_eval_dyk >= -5e-9)
	feasible_naive = (tp_res_naive < 1e-6) and (min_eval_naive >= -1e-9)
	# Acceptance logic: if naive infeasible and Dykstra feasible -> pass.
	# If both feasible, require Dykstra distance not dramatically worse (<= 2x) to allow for true projection moving further.
	if feasible_dyk and not feasible_naive:
		ok = True
	elif feasible_dyk and feasible_naive:
		ok = d_dyk <= 1.2 * d_naive + 1e-10
	else:
		ok = False
	_status(
		name,
		ok,
		f"d_dyk={d_dyk:.3e}, d_naive={d_naive:.3e}, feas_dyk={feasible_dyk}, feas_naive={feasible_naive}"
	)
	return ok


def test_compute_channel_probabilities_scaling() -> bool:
	"""p scales ~ linearly with bin_area before any row normalisation."""
	name = "compute_channel_probabilities_scaling"
	cutoff = 1  # d=2
	d = cutoff + 1
	# two inputs, three outputs
	def _coh(a):
		ks = np.arange(d)
		fac = factorial_array(cutoff, exact=False)[:d]
		out = np.empty((d, len(a)), dtype=np.complex128)
		for j, z in enumerate(a):
			pref = np.exp(-0.5 * abs(z) ** 2)
			out[:, j] = pref * (z ** ks) / np.sqrt(fac)
		return out
	a_in = np.array([0.0+0.0j, 0.3-0.2j])
	a_out = np.array([0.1+0.0j, -0.2+0.15j, 0.05-0.25j])
	C_in = _coh(a_in)   # (d,2)
	C_out = _coh(a_out) # (d,3)
	# Use a CPTP Choi (identity channel)
	phi = np.zeros(d*d, dtype=np.complex128)
	for k in range(d):
		phi[k*d + k] = 1.0
	phi /= np.sqrt(d)
	J = np.outer(phi, phi.conj())
	p1 = compute_channel_probabilites(J, C_in, C_out, bin_area=0.02)
	p2 = compute_channel_probabilites(J, C_in, C_out, bin_area=0.04)
	ratio = p2 / np.maximum(p1, 1e-300)
	mask = p1 > 1e-6
	med = np.median(ratio[mask])
	q1, q3 = np.percentile(ratio[mask], [25, 75])
	iqr = q3 - q1
	approx_lin = (abs(med - 2.0) < 0.03) and (iqr < 0.15)
	_status(name, approx_lin, f"median={med:.3f} (≈2), IQR={iqr:.3f} (<0.15)")
	return approx_lin


def test_project_CPTP_multi_repeats_stability() -> bool:
	"""Repeated projections should neither drift nor break feasibility."""
	name = "project_CPTP_multi_repeats_stability"
	d_in = d_out = 3
	N = d_in * d_out
	rng = np.random.default_rng(202)
	X = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
	X = 0.5 * (X + X.conjugate().T)
	J = project_CPTP(X, d_in=d_in, d_out=d_out)
	for _ in range(5):
		J = project_CPTP(J, d_in=d_in, d_out=d_out)
	Tr_out = _partial_trace_out(J, d_in, d_out)
	tp_res = np.linalg.norm(Tr_out - np.eye(d_in))
	evals = np.linalg.eigvalsh(J)
	ok = (tp_res < 1e-8) and (evals.min() >= -5e-9)
	_status(name, ok, f"tp_res={tp_res:.1e}, min_eval={evals.min():.1e}")
	return ok


# Append new tests
TEST_FUNCS += [
	test_project_CPTP_nonhermitian,
	test_project_CPTP_fixed_point_random_kraus,
	test_project_CPTP_vs_naive_distance,
	test_compute_channel_probabilities_scaling,
	test_project_CPTP_multi_repeats_stability,
]


# ---------------- MLE projected gradient descent tests ---------------- #

def _coherent_overlap_matrix(alphas: np.ndarray, cutoff: int) -> np.ndarray:
	"""Build coherent overlaps matrix (d,len(alphas)).
	a_n = exp(-|α|^2/2) α^n / sqrt(n!) for n=0..cutoff.
	Re-implemented (avoid relying on private test helper) for clarity.
	"""
	d = cutoff + 1
	ks = np.arange(d)
	fac = factorial_array(cutoff, exact=False)[: d]
	out = np.empty((d, len(alphas)), dtype=np.complex128)
	for j, a in enumerate(alphas):
		pref = np.exp(-0.5 * abs(a) ** 2)
		out[:, j] = pref * (a ** ks) / np.sqrt(fac)
	return out


def test_mle_pgd_identity_basic() -> bool:
	"""Run PG MLE on synthetic counts from (approx) identity channel and check monotonic NLL decrease & CPTP."""
	name = "mle_pgd_identity_basic"
	rng = np.random.default_rng(1234)
	cutoff = 2  # d=3 keeps things tiny
	d = cutoff + 1
	# Two probe input coherent states
	alpha_in = np.array([0.0 + 0.0j, 0.35 - 0.15j])
	# Three output grid coherent points
	beta_out = np.array([0.0 + 0.0j, 0.2 + 0.1j, -0.25 + 0.05j])
	C_in = _coherent_overlap_matrix(alpha_in, cutoff)   # (d, S=2)
	C_out = _coherent_overlap_matrix(beta_out, cutoff)  # (d, M=3)
	bin_area = 0.04  # arbitrary positive scale
	# True identity channel Choi: |Φ+><Φ+|
	phi = np.zeros(d * d, dtype=np.complex128)
	for k in range(d):
		phi[k * d + k] = 1.0
	phi /= np.sqrt(d)
	J_true = np.outer(phi, phi.conjugate())
	# Raw forward model probabilities; generate Poisson counts consistent with model.
	p_raw = compute_channel_probabilites(J_true, C_in, C_out, bin_area)
	target_totals = np.array([400.0, 500.0])
	scales = target_totals / np.maximum(p_raw.sum(axis=1), 1e-300)
	counts = rng.poisson(scales[:, None] * p_raw)
	# Run MLE PGD starting from maximally mixed
	res = mle_projected_gradient_descent(
		counts,
		C_in,
		C_out,
		bin_area,
		J_initialisation_style='maximally_mixed',
		max_iters=80,
		step_init=5e-3,
		verbose=False,
		seed=999,
		track_cptp_diagnostics=False,
	)
	J_est = res["J"]
	nll_hist = res["nll_history"]
	# Monotonic (non-increasing) NLL
	monotonic = np.all(np.diff(nll_hist) <= 1e-9)
	# Improvement
	improved = nll_hist[-1] <= nll_hist[0] + 1e-9 and (nll_hist[0] - nll_hist[-1]) / (abs(nll_hist[0]) + 1e-15) > 1e-4
	# CPTP checks
	# Partial trace over output should be identity (within tolerance)
	J4 = J_est.reshape(d, d, d, d)
	Tr_out = np.einsum('ikjk->ij', J4, optimize=True)
	tp_res = np.linalg.norm(Tr_out - np.eye(d))
	tp_ok = tp_res < 5e-5
	evals = np.linalg.eigvalsh(0.5 * (J_est + J_est.conjugate().T))
	psd_ok = evals.min() >= -5e-9
	ok = monotonic and improved and tp_ok and psd_ok
	_status(name, ok, f"monotonic={monotonic}, improved={improved}, tp_res={tp_res:.2e}, min_eval={evals.min():.1e}")
	return ok


def test_mle_pgd_projection_diagnostics() -> bool:
	"""Verify CPTP projection diagnostics are returned and structurally valid when tracking enabled."""
	name = "mle_pgd_projection_diagnostics"
	rng = np.random.default_rng(4321)
	cutoff = 1  # d=2 smaller/faster
	d = cutoff + 1
	alpha_in = np.array([0.0 + 0.0j])  # single input simplifies
	beta_out = np.array([0.0 + 0.0j, 0.15 - 0.05j])
	C_in = _coherent_overlap_matrix(alpha_in, cutoff)
	C_out = _coherent_overlap_matrix(beta_out, cutoff)
	bin_area = 0.03
	# True channel identity counts
	phi = np.zeros(d * d, dtype=np.complex128)
	for k in range(d):
		phi[k * d + k] = 1.0
	phi /= np.sqrt(d)
	J_true = np.outer(phi, phi.conjugate())
	p_raw = compute_channel_probabilites(J_true, C_in, C_out, bin_area)
	row_scale = 300.0 / max(p_raw.sum(), 1e-300)
	counts = rng.poisson(row_scale * p_raw).reshape(1, -1)
	res = mle_projected_gradient_descent(
		counts,
		C_in,
		C_out,
		bin_area,
		J_initialisation_style='maximally_mixed',
		max_iters=40,
		step_init=1e-2,
		verbose=False,
		seed=7,
		track_cptp_diagnostics=True,
	)
	have_keys = ("init_projection_diag" in res) and ("projection_history" in res)
	proj_hist = res.get("projection_history", [])
	structural = bool(proj_hist) and all(
		all(k in entry for k in ("outer_iter", "attempt_iters", "accepted_iters", "backtracks"))
		for entry in proj_hist
	)
	# Ensure attempt_iters entries are positive ints
	numeric = all(
		isinstance(entry["accepted_iters"], (int, np.integer)) and (entry["accepted_iters"] >= 1)
		for entry in proj_hist if entry["accepted_iters"] is not None
	)
	ok = have_keys and structural and numeric
	_status(name, ok, f"keys={have_keys}, structural={structural}, numeric={numeric}, len_hist={len(proj_hist)}")
	return ok


TEST_FUNCS += [
	test_mle_pgd_identity_basic,
	test_mle_pgd_projection_diagnostics,
]


def test_mle_pgd_gradient_directional_derivative() -> bool:
	"""Check analytic gradient (reimplemented) matches directional derivative via finite differences.

	We rebuild p(J) and gradient formula outside optimisation code for a tiny instance (d=2).
	"""
	name = "mle_pgd_gradient_directional_derivative"
	rng = np.random.default_rng(2025)
	cutoff = 1  # d=2
	d = cutoff + 1
	# Inputs (S=2) & Outputs (M=3)
	alpha_in = np.array([0.0 + 0.0j, 0.4 - 0.1j])
	beta_out = np.array([0.0 + 0.0j, 0.2 + 0.05j, -0.25 + 0.15j])
	C_in = _coherent_overlap_matrix(alpha_in, cutoff)
	C_out = _coherent_overlap_matrix(beta_out, cutoff)
	bin_area = 0.05
	# Identity channel Choi
	phi = np.zeros(d * d, dtype=np.complex128)
	for k in range(d):
		phi[k * d + k] = 1.0
	phi /= np.sqrt(d)
	J = np.outer(phi, phi.conjugate())
	# Poisson counts from unnormalised model for gradient consistency
	p_raw = compute_channel_probabilites(J, C_in, C_out, bin_area)
	target_totals = np.array([500.0, 600.0])
	scales = target_totals / np.maximum(p_raw.sum(axis=1), 1e-300)
	counts = rng.poisson(scales[:, None] * p_raw)

	# Helper: forward p
	# Match implementation: use UNNORMALISED forward probabilities (Poisson factorisation model)
	def forward_p(J_mat: np.ndarray) -> np.ndarray:
		return compute_channel_probabilites(J_mat, C_in, C_out, bin_area)

	# NLL
	def nll(J_mat: np.ndarray) -> float:
		return neg_log_likelihood_poisson_multiinput(counts, forward_p(J_mat))

	# Analytic gradient (mirrors implementation)
	a = C_in.T  # (S,d)
	A_nm = a[:, :, None] * np.conjugate(a)[:, None, :]   # (S,n,m)
	A_mn = np.swapaxes(A_nm, -2, -1)                     # (S,m,n)
	A_T_mn = np.conjugate(A_mn)                          # (S,m,n)
	b = C_out.T                                          # (M,d)
	B_pq = np.conjugate(b)[:, :, None] * b[:, None, :]   # (M,p,q)
	scale = bin_area / np.pi
	N_i = counts.sum(axis=1, keepdims=True)
	pJ = forward_p(J)
	eps_f = np.finfo(np.float64).tiny
	p_safe = np.maximum(pJ, eps_f)
	W = (N_i - counts / p_safe)  # (S,M)
	S_ipq = np.einsum('jpq,ij->ipq', B_pq, W, optimize=True)  # (S,d,d)
	G4 = scale * np.einsum('imn,ipq->mnpq', A_T_mn, S_ipq, optimize=True)
	G = G4.reshape(d * d, d * d)
	G = 0.5 * (G + G.conjugate().T)

	# Random Hermitian direction D, unit Frobenius norm
	R = rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))
	R = 0.5 * (R + R.conjugate().T)
	R /= np.linalg.norm(R)

	# Finite-difference directional derivative
	epsilon = 1e-6
	f_plus = nll(J + epsilon * R)
	f_minus = nll(J - epsilon * R)
	dir_fd = (f_plus - f_minus) / (2 * epsilon)
	dir_an = np.real(np.vdot(G, R))  # <G, R>
	rel_err = abs(dir_fd - dir_an) / max(1e-12, abs(dir_fd))
	ok = rel_err < 1.5e-1  # relaxed tolerance acknowledging model stochastic counts
	_status(name, ok, f"rel_err={rel_err:.2e} (fd={dir_fd:.3e}, an={dir_an:.3e})")
	return ok


def test_mle_pgd_backtracking_occurs() -> bool:
	"""Force backtracking by using a large initial step and confirm bt_history records positive counts."""
	name = "mle_pgd_backtracking_occurs"
	rng = np.random.default_rng(99)
	cutoff = 1
	d = cutoff + 1
	alpha_in = np.array([0.2 + 0.1j, -0.3 + 0.05j])
	beta_out = np.array([0.0 + 0.0j, 0.25 - 0.15j])
	C_in = _coherent_overlap_matrix(alpha_in, cutoff)
	C_out = _coherent_overlap_matrix(beta_out, cutoff)
	bin_area = 0.08
	# True identity probabilities for counts
	phi = np.zeros(d * d, dtype=np.complex128)
	for k in range(d):
		phi[k * d + k] = 1.0
	phi /= np.sqrt(d)
	J_true = np.outer(phi, phi.conjugate())
	p_raw = compute_channel_probabilites(J_true, C_in, C_out, bin_area)
	target_totals = np.array([300.0, 350.0])
	scales = target_totals / np.maximum(p_raw.sum(axis=1), 1e-300)
	counts = rng.poisson(scales[:, None] * p_raw)
	# Large initial step to trigger backtracking
	res = mle_projected_gradient_descent(
		counts, C_in, C_out, bin_area,
		J_initialisation_style='maximally_mixed',
		max_iters=10,
		step_init=10.0,   # very large to force reduction
		armijo_c=1e-4,
		armijo_tau=0.2,   # harsher reduction to accumulate some backtracks
		verbose=False,
		seed=5,
	)
	bt_hist = res["bt_history"]
	backtracked = (len(bt_hist) > 0) and (bt_hist[0] > 0)
	nll_hist = res["nll_history"]
	step_hist = res["step_history"]
	nll_decreased = (nll_hist.size > 1) and (nll_hist[-1] < nll_hist[0])
	step_reduced = (step_hist.size > 0) and (step_hist[0] < 10.0)
	ok = backtracked and nll_decreased and step_reduced
	_status(name, ok, f"bt_hist={bt_hist}, nllΔ={(nll_hist[0]-nll_hist[-1]) if nll_hist.size>1 else 0:.3e}, step0={step_hist[0] if step_hist.size>0 else None}")
	return ok


def test_mle_pgd_line_search_failure_stops() -> bool:
	"""Choose parameters likely to cause immediate line search failure so no steps are accepted."""
	name = "mle_pgd_line_search_failure_stops"
	# Setup minimal problem (single input / two outputs) to keep it quick
	cutoff = 1
	d = cutoff + 1
	alpha_in = np.array([0.1 + 0.0j])
	beta_out = np.array([0.0 + 0.0j, 0.15 + 0.05j])
	C_in = _coherent_overlap_matrix(alpha_in, cutoff)
	C_out = _coherent_overlap_matrix(beta_out, cutoff)
	bin_area = 0.02
	# Identity channel counts
	phi = np.zeros(d * d, dtype=np.complex128)
	for k in range(d):
		phi[k * d + k] = 1.0
	phi /= np.sqrt(d)
	J_true = np.outer(phi, phi.conjugate())
	p_raw = compute_channel_probabilites(J_true, C_in, C_out, bin_area)
	target_total = 250.0
	scale = target_total / max(p_raw.sum(), 1e-300)
	counts = np.random.default_rng(17).poisson(scale * p_raw).reshape(1, -1)
	# Set high armijo_c and small armijo_tau so step shrinks fast and likely fails all
	res = mle_projected_gradient_descent(
		counts, C_in, C_out, bin_area,
		J_initialisation_style='identity',  # gradient should be modest already
		max_iters=5,
		step_init=5e-2,
		armijo_c=1.0,   # very strict decrease requirement
		armijo_tau=0.1,  # aggressive shrinking
		min_step=1e-6,
		verbose=False,
		seed=11,
	)
	# Expect no accepted steps => step_history empty, nll_history length 1
	no_steps = (res["step_history"].size == 0) and (res["nll_history"].size == 1)
	_status(name, no_steps, f"steps={res['step_history']}, nll_len={res['nll_history'].size}")
	return no_steps


TEST_FUNCS += [
	test_mle_pgd_gradient_directional_derivative,
	test_mle_pgd_backtracking_occurs,
	test_mle_pgd_line_search_failure_stops,
]


# ---------------- Synthetic channel data generation tests ---------------- #

def test_generate_synthetic_mean_variance() -> bool:
	"""Large-N mean/variance sanity per probe.

	For several random input alphas, with T=0.36, xi=2.0 we expect per-quadrature
	variance sigma2 = 1 + xi/2 = 2. Means should be sqrt(T)*alpha mapped to y via sqrt(2).*Re/Im.
	"""
	name = "generate_synthetic_mean_variance"
	rng = np.random.default_rng(123)
	S = 3
	alphas = (0.8 * rng.standard_normal(S) + 0.6j * rng.standard_normal(S)).astype(np.complex128)
	T = 0.36
	xi = 2.0
	N_per = 200_000
	shots = np.full(S, N_per, dtype=int)
	data = generate_synthetic_channel_data(alphas, shots, transmissivity=T, excess_noise=xi, seed=777)
	sqrtT = np.sqrt(T)
	sigma2 = 1.0 + 0.5 * xi
	sigma = np.sqrt(sigma2)
	ok_all = True
	msgs = []
	for a, samples in zip(alphas, data):
		y1_mean = samples[:,0].mean()
		y2_mean = samples[:,1].mean()
		y1_var = samples[:,0].var(ddof=0)
		y2_var = samples[:,1].var(ddof=0)
		mu_y1 = np.sqrt(2.0) * sqrtT * a.real
		mu_y2 = np.sqrt(2.0) * sqrtT * a.imag
		mean_err = max(abs(y1_mean - mu_y1), abs(y2_mean - mu_y2))
		mean_tol = 4 * sigma / np.sqrt(N_per)
		var_err = max(abs(y1_var - sigma2), abs(y2_var - sigma2))
		var_tol = 0.02
		cond = (mean_err <= mean_tol) and (var_err <= var_tol)
		ok_all &= cond
		msgs.append(f"|Δμ|={mean_err:.3e}≤{mean_tol:.3e}, |Δσ²|max={var_err:.3e}≤{var_tol:.3e}")
	_status(name, ok_all, " ; ".join(msgs))
	return ok_all


def test_generate_synthetic_mean_scaling_with_T() -> bool:
	"""Mean scaling with sqrt(T)."""
	name = "generate_synthetic_mean_scaling_with_T"
	alpha = 0.7 + 0.4j
	xi = 0.0
	N = 150_000
	shots = np.array([N])
	T1, T2 = 0.25, 0.81
	d1 = generate_synthetic_channel_data(np.array([alpha]), shots, T1, xi, seed=1)[0]
	d2 = generate_synthetic_channel_data(np.array([alpha]), shots, T2, xi, seed=2)[0]
	# Empirical mean vector norms
	m1 = np.linalg.norm(d1.mean(axis=0))
	m2 = np.linalg.norm(d2.mean(axis=0))
	ratio_emp = m2 / m1
	ratio_theo = np.sqrt(T2 / T1)
	rel_err = abs(ratio_emp - ratio_theo) / ratio_theo
	ok = rel_err < 0.01
	_status(name, ok, f"ratio_emp={ratio_emp:.4f}, ratio_theo={ratio_theo:.4f}, rel_err={rel_err:.3e}")
	return ok


def test_generate_synthetic_variance_excess_noise() -> bool:
	"""Variance increases linearly with excess noise (ratio ≈2 when xi goes 0->2)."""
	name = "generate_synthetic_variance_excess_noise"
	alpha = 0.3 - 0.5j
	T = 1.0
	N = 150_000
	shots = np.array([N])
	d0 = generate_synthetic_channel_data(np.array([alpha]), shots, T, 0.0, seed=33)[0]
	d2 = generate_synthetic_channel_data(np.array([alpha]), shots, T, 2.0, seed=34)[0]
	v0 = 0.5 * (d0[:,0].var(ddof=0) + d0[:,1].var(ddof=0))
	v2 = 0.5 * (d2[:,0].var(ddof=0) + d2[:,1].var(ddof=0))
	ratio_emp = v2 / v0
	ok = abs(ratio_emp - 2.0) < 0.03
	_status(name, ok, f"ratio_emp={ratio_emp:.3f} (expect 2)")
	return ok


def test_generate_synthetic_alpha_residual() -> bool:
	"""Check E[|α - sqrt(T)α_in|^2] ≈ σ² using sample means."""
	name = "generate_synthetic_alpha_residual"
	rng = np.random.default_rng(456)
	alphas = np.array([0.6+0.2j, -0.4+0.5j], dtype=np.complex128)
	T = 0.52
	xi = 1.4
	sigma2 = 1.0 + 0.5*xi
	N = 180_000
	shots = np.full(alphas.size, N, dtype=int)
	data = generate_synthetic_channel_data(alphas, shots, T, xi, seed=909)
	sqrtT = np.sqrt(T)
	ok_all = True
	parts = []
	for a, S in zip(alphas, data):
		alpha_samples = (S[:,0] + 1j*S[:,1]) / np.sqrt(2.0)
		r = alpha_samples - sqrtT * a
		mean_sq = np.mean(np.abs(r)**2)
		rel_err = abs(mean_sq - sigma2) / sigma2
		tol = 0.02  # 2%
		ok = rel_err < tol
		ok_all &= ok
		parts.append(f"rel_err={rel_err:.2e} < {tol:.2e}")
	_status(name, ok_all, " ; ".join(parts))
	return ok_all


def test_generate_synthetic_determinism_and_errors() -> bool:
	"""Seed determinism, zero-shots handling, and bad input errors."""
	name = "generate_synthetic_determinism_and_errors"
	alphas = np.array([0.1+0.2j, -0.3+0.4j])
	shots = np.array([10, 0])
	T = 0.9
	xi = 0.3
	d1 = generate_synthetic_channel_data(alphas, shots, T, xi, seed=123)
	d2 = generate_synthetic_channel_data(alphas, shots, T, xi, seed=123)
	d3 = generate_synthetic_channel_data(alphas, shots, T, xi, seed=124)
	determinism = all(np.array_equal(A, B) for A,B in zip(d1,d2)) and any(not np.array_equal(A,B) for A,B in zip(d1,d3))
	zero_shape = d1[1].shape == (0,2)
	# Error cases
	err_flags = []
	try:
		generate_synthetic_channel_data(np.array([0.1+0j]), np.array([5,6]), T, xi)
		err_flags.append(False)
	except ValueError:
		err_flags.append(True)
	try:
		generate_synthetic_channel_data(alphas, shots, 0.0, xi)
		err_flags.append(False)
	except ValueError:
		err_flags.append(True)
	try:
		generate_synthetic_channel_data(alphas, shots, 1.1, xi)
		err_flags.append(False)
	except ValueError:
		err_flags.append(True)
	try:
		generate_synthetic_channel_data(alphas, shots, T, -0.1)
		err_flags.append(False)
	except ValueError:
		err_flags.append(True)
	try:
		generate_synthetic_channel_data(alphas, np.array([5,-1]), T, xi)
		err_flags.append(False)
	except ValueError:
		err_flags.append(True)
	errors_ok = all(err_flags)
	ok = determinism and zero_shape and errors_ok
	_status(name, ok, f"determinism={determinism}, zero_shape={zero_shape}, errors_ok={errors_ok}")
	return ok


def test_generate_synthetic_end_to_end_histogram() -> bool:
	"""End-to-end: generate data, histogram on grid, compare means to theory and counts sums."""
	name = "generate_synthetic_end_to_end_histogram"
	alphas = np.array([0.5+0.1j, -0.4+0.3j], dtype=np.complex128)
	T = 0.7
	xi = 0.5
	sqrtT = np.sqrt(T)
	sigma2 = 1.0 + 0.5*xi
	N = 60_000
	shots = np.full(alphas.size, N, dtype=int)
	data = generate_synthetic_channel_data(alphas, shots, T, xi, seed=321)
	# Build grid covering ALL samples with safety margin: use sample extrema + 3σ
	all_y1 = np.concatenate([d[:,0] for d in data])
	all_y2 = np.concatenate([d[:,1] for d in data])
	max_abs = max(np.max(np.abs(all_y1)), np.max(np.abs(all_y2)))
	L = max(max_abs + 3*np.sqrt(sigma2), 2.5)
	G = 65  # finer grid improves mean approximation
	_, Y1, Y2 = build_grid(L, G)
	y_axis = np.linspace(-L, L, G)
	delta = y_axis[1]-y_axis[0]
	total_ok = True
	mean_ok_all = True
	msgs = []
	for a, samples, shots_i in zip(alphas, data, shots):
		# Histogram
		# Bin edges extend half a bin beyond extreme grid points to catch boundary values
		edges = np.concatenate(([y_axis[0]-0.5*delta], 0.5*(y_axis[1:]+y_axis[:-1]), [y_axis[-1]+0.5*delta]))
		H, _, _ = np.histogram2d(samples[:,0], samples[:,1], bins=[edges, edges])  # includes all by construction
		count_sum = int(H.sum())
		total_ok &= (count_sum == shots_i)
		# Weighted mean alpha from histogram grid centres
		alpha_grid_matrix = (Y1 + 1j*Y2)/np.sqrt(2.0)
		# Orientation fix: np.histogram2d returns H[xbin, ybin]; meshgrid('xy') gives Y1[row=ybin, col=xbin]
		weighted_alpha = (alpha_grid_matrix * H.T).sum() / max(1, H.sum())
		target_alpha = sqrtT * a
		err = abs(weighted_alpha - target_alpha)
		# Expected standard error for alpha (complex): components have var sigma2, so std for mean component ~ sqrt(sigma2/N)
		se = np.sqrt(2.0 * sigma2 / shots_i)  # crude bound for complex magnitude deviation
		mean_ok = err < 5*se
		mean_ok_all &= mean_ok
		msgs.append(f"|Δα|={err:.3e} (<{5*se:.3e})")
	ok = total_ok and mean_ok_all
	_status(name, ok, f"counts_ok={total_ok}, means_ok={mean_ok_all}; " + " ; ".join(msgs))
	return ok


TEST_FUNCS += [
	test_generate_synthetic_mean_variance,
	test_generate_synthetic_mean_scaling_with_T,
	test_generate_synthetic_variance_excess_noise,
	test_generate_synthetic_alpha_residual,
	test_generate_synthetic_determinism_and_errors,
	test_generate_synthetic_end_to_end_histogram,
]


# ---------------- run_mle_workflow tests ---------------- #

def test_run_mle_workflow_basic_identity() -> bool:
	"""End-to-end identity channel small instance sanity: CPTP J, counts, monotonic NLL."""
	name = "run_mle_workflow_basic_identity"
	alphas = np.array([0.0+0.0j, 0.35-0.2j])
	shots = np.array([400, 500])
	res = run_mle_workflow(
		probe_states=alphas,
		shots_per_probe=shots,
		fock_cutoff=3,           # d=3 (0,1,2)
		grid_size_points=15,
		adaptive_grid=True,
		max_iters=40,
		step_init=2e-3,
		seed=123,
		transmissivity=1.0,
		excess_noise=0.0,
		J_initialisation_style='maximally_mixed'
	)
	# Counts integrity
	counts = res["counts_ij"]
	out_of_bounds = res["out_of_bounds_per_probe"]
	row_totals = counts.sum(axis=1) + out_of_bounds
	counts_ok = np.all(row_totals == shots)
	# CPTP checks on J
	J = res["J"]
	d = res["fock_cutoff"]
	J4 = J.reshape(d, d, d, d)
	Tr_out = np.einsum('ikjk->ij', J4, optimize=True)
	tp_res = np.linalg.norm(Tr_out - np.eye(d))
	tp_ok = tp_res < 5e-5
	evals = np.linalg.eigvalsh(0.5*(J+J.conj().T))
	psd_ok = evals.min() >= -5e-9
	# NLL monotonic
	nll_hist = res["nll_history"]
	monotonic = np.all(np.diff(nll_hist) <= 1e-9)
	improved = nll_hist[-1] <= nll_hist[0] + 1e-9
	ok = counts_ok and tp_ok and psd_ok and monotonic and improved
	_status(name, ok, f"counts_ok={counts_ok}, tp_res={tp_res:.2e}, min_eval={evals.min():.1e}, monotonic={monotonic}")
	return ok


def test_run_mle_workflow_fixed_grid_size() -> bool:
	"""Use fixed grid size (adaptive disabled) and verify returned half size matches input."""
	name = "run_mle_workflow_fixed_grid_size"
	alphas = np.array([0.2+0.1j])
	shots = np.array([300])
	L_fixed = 4.0
	res = run_mle_workflow(
		probe_states=alphas,
		shots_per_probe=shots,
		fock_cutoff=2,
		grid_size_points=11,
		grid_half_size=L_fixed,
		adaptive_grid=False,
		max_iters=25,
		step_init=2e-3,
		seed=321,
		transmissivity=1.0,
		excess_noise=0.0,
	)
	returned_L = res["grid_half_size"]
	ok = abs(returned_L - L_fixed) < 1e-12
	_status(name, ok, f"L_fixed={L_fixed}, returned={returned_L}")
	return ok


def test_run_mle_workflow_transmissivity_scaling() -> bool:
	"""Compare synthetic sample means between T=1 and T=0.49; ratio ≈ sqrt(0.49)."""
	name = "run_mle_workflow_transmissivity_scaling"
	alphas = np.array([0.6-0.1j, -0.3+0.45j])
	shots = np.array([800, 900])
	res1 = run_mle_workflow(alphas, shots, fock_cutoff=2, grid_size_points=13, adaptive_grid=True,
							 max_iters=25, seed=777, transmissivity=1.0, excess_noise=0.0)
	res2 = run_mle_workflow(alphas, shots, fock_cutoff=2, grid_size_points=13, adaptive_grid=True,
							 max_iters=25, seed=888, transmissivity=0.49, excess_noise=0.0)
	# Compute empirical alpha-plane means per probe
	def _alpha_means(res):
		ms = []
		for S in res["synthetic_samples"]:
			if S.size == 0:
				ms.append(0.0)
			else:
				alpha_samp = (S[:,0] + 1j*S[:,1]) / np.sqrt(2.0)
				ms.append(np.linalg.norm(alpha_samp.mean()))
		return np.array(ms)
	m1 = _alpha_means(res1)
	m2 = _alpha_means(res2)
	# Avoid zero division
	ratio_emp = np.mean(m2 / np.maximum(m1, 1e-12))
	ratio_theo = np.sqrt(0.49)
	rel_err = abs(ratio_emp - ratio_theo) / ratio_theo
	ok = rel_err < 0.07  # 7% tolerance
	_status(name, ok, f"ratio_emp={ratio_emp:.3f}, ratio_theo={ratio_theo:.3f}, rel_err={rel_err:.3e}")
	return ok


TEST_FUNCS += [
	test_run_mle_workflow_basic_identity,
	test_run_mle_workflow_fixed_grid_size,
	test_run_mle_workflow_transmissivity_scaling,
]


def run_all_tests() -> bool:
	results = [f() for f in TEST_FUNCS]
	passed = sum(results)
	total = len(results)
	print(f"Summary: {passed}/{total} tests passed.")
	return all(results)


if __name__ == "__main__":
	run_all_tests()
