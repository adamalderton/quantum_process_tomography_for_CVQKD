"""Unit tests for simple_model.py.

This file provides:
  * One basic unit test for every public helper/function in `simple_model.py`.
  * Physics / channel sanity tests for 9 canonical qubit channels (identity,
	rotation, depolarising, Pauli, phase flip, phase damping, amplitude
	damping, generalized amplitude damping, random unitary mixture) checking
	Choi TP, CP, trace, Born rule parity, and (where applicable) Bloch affine
	map parameters (T, t) and simple invariants.

Each test purposefully uses *very small* deterministic fixtures so the suite
executes quickly (<1s typically) while still exercising the code paths.

NOTE: The directory containing this file is *not* a package; we therefore add
the directory itself to sys.path to import `simple_model` directly without
adding __init__.py files to the project tree.
"""

from __future__ import annotations

import math
import os
import sys
import numpy as np
import pytest

# Allow direct import of simple_model.py
sys.path.append(os.path.dirname(__file__))

import simple_model as sm  # type: ignore

RTOL = 1e-10
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Helper utilities for tests
# ---------------------------------------------------------------------------
def pauli_matrices():
	sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
	sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
	sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
	return sx, sy, sz


def pauli_eigen_probes():
	"""Return 6 informationally convenient pure state vectors: |0>,|1>,|+>,|->,|+i>,|-i>."""
	zero = np.array([1.0, 0.0], dtype=np.complex128)
	one = np.array([0.0, 1.0], dtype=np.complex128)
	plus = (zero + one) / np.sqrt(2)
	minus = (zero - one) / np.sqrt(2)
	plus_i = (zero + 1j * one) / np.sqrt(2)
	minus_i = (zero - 1j * one) / np.sqrt(2)
	return np.stack([zero, one, plus, minus, plus_i, minus_i], axis=1)


def density_from_state(psi: np.ndarray) -> np.ndarray:
	return np.outer(psi, psi.conj())


def choi_born_probs(J: np.ndarray, probes: np.ndarray, povm):
	"""Compute probabilities via Choi: p_{i,m} = Tr[J (B_m ⊗ A_i^T)]."""
	d = 2
	p = np.zeros((probes.shape[1], len(povm)))
	for i in range(probes.shape[1]):
		A = density_from_state(probes[:, i]).T  # transpose for formula
		for m, B in enumerate(povm):
			op = np.kron(B, A)
			p[i, m] = np.real(np.trace(J @ op))
	# numeric stabilisation
	p = np.clip(p, 0, None)
	p /= p.sum(axis=1, keepdims=True)
	return p


def extract_bloch_affine(K_list):
	"""Return (T, t) for qubit channel using action on Bloch basis.

	For states ρ = (I + r·σ)/2, we compute images of basis vectors ex, ey, ez.
	t is image of 0 vector (channel applied to maximally mixed state).
	"""
	d = 2
	sx, sy, sz = pauli_matrices()
	I2 = np.eye(2, dtype=np.complex128)

	def apply(rho):
		return sum(K @ rho @ K.conj().T for K in K_list)

	rho_0 = I2 / 2
	rho_x = (I2 + sx) / 2
	rho_y = (I2 + sy) / 2
	rho_z = (I2 + sz) / 2
	out_0 = apply(rho_0)
	out_x = apply(rho_x)
	out_y = apply(rho_y)
	out_z = apply(rho_z)

	def bloch(rho):
		return np.array([
			np.real(np.trace(rho @ sx)),
			np.real(np.trace(rho @ sy)),
			np.real(np.trace(rho @ sz)),
		])

	r0 = bloch(out_0)
	rx = bloch(out_x) - r0
	ry = bloch(out_y) - r0
	rz = bloch(out_z) - r0
	T = np.stack([rx, ry, rz], axis=1)
	t = r0
	return T, t


# ---------------------------------------------------------------------------
# Unit tests for simple helper functions
# ---------------------------------------------------------------------------
def test_stack_and_unstack_kraus_roundtrip():
	K1 = np.eye(2, dtype=np.complex128)
	K2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
	V = sm.stack_kraus([K1, K2])
	parts = sm.unstack_kraus(V, 2)
	assert len(parts) == 2
	assert np.allclose(parts[0], K1)
	assert np.allclose(parts[1], K2)


def test_riemannian_grad_tangent_property():
	rng = np.random.default_rng(0)
	# random (4x2) then retract to Stiefel
	X = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
	V = sm.qr_retraction(X)
	G = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
	Rg = sm.riemannian_grad_on_stiefel(V, G)
	# Tangent condition: V^† Rg + Rg^† V = 0 (skew-Hermitian)
	cond = V.conj().T @ Rg + Rg.conj().T @ V
	assert np.allclose(cond, np.zeros_like(cond), atol=1e-10)


def test_qr_retraction_column_orthonormal():
	rng = np.random.default_rng(1)
	X = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
	Q = sm.qr_retraction(X)
	I = Q.conj().T @ Q
	assert np.allclose(I, np.eye(2), atol=1e-12)


def test_build_sic_povm_qubit_properties():
	povm = sm.build_sic_povm_qubit()
	S = sum(povm)
	# Completeness
	assert np.allclose(S, np.eye(2), atol=1e-12)
	# Positivity (eigenvalues >= 0)
	for E in povm:
		w, _ = np.linalg.eigh(0.5 * (E + E.conj().T))
		assert np.all(w >= -1e-12)


def test_random_pure_states_shape_and_norm():
	vecs = sm.random_pure_states(5, seed=42)
	assert vecs.shape == (2, 5)
	norms = np.sum(np.abs(vecs) ** 2, axis=0)
	assert np.allclose(norms, 1.0)


def test_forward_probs_identity_channel_rows_sum_to_one():
	povm = sm.build_sic_povm_qubit()
	probes = pauli_eigen_probes()
	K = [np.eye(2, dtype=np.complex128)]
	p = sm.forward_probs(K, probes, povm)
	assert p.shape == (probes.shape[1], len(povm))
	row_sums = p.sum(axis=1)
	assert np.allclose(row_sums, 1.0)


def test_neg_log_likelihood_uniform():
	counts = np.array([[1, 1, 1, 1]])  # one probe, 4 outcomes
	probs = np.full((1, 4), 0.25)
	nll = sm.neg_log_likelihood(counts, probs)
	# L = - Σ n_k log p_k = -4 * log(0.25)
	assert math.isclose(nll, -4 * math.log(0.25))


def test_gradient_kraus_zero_when_counts_match_probs():
	povm = sm.build_sic_povm_qubit()
	probes = pauli_eigen_probes()[:, :2]  # just |0>,|1>
	K = [np.eye(2, dtype=np.complex128)]
	p = sm.forward_probs(K, probes, povm)
	shots = 100
	counts = (p * shots).astype(int)  # exact synthetic counts (integer rounding ok)
	g_list = sm.gradient_kraus(K, probes, povm, counts, p)
	norm = sum(np.linalg.norm(g) for g in g_list)
	# Should be *very* small but may not be exactly zero due to integer rounding
	assert norm < 1e-6


def test_initialise_kraus_qubit_identity_mode():
	rng = np.random.default_rng(0)
	K = sm.initialise_kraus_qubit(2, "identity", rng)
	# First Kraus should be close to identity
	assert np.allclose(K[0], np.eye(2), atol=1e-12)
	# TP residual zero
	assert sm.tp_residual(K) < 1e-12


def test_initialise_kraus_qubit_depolarising_mode_rank4():
	rng = np.random.default_rng(0)
	K = sm.initialise_kraus_qubit(4, "depolarising", rng)
	# Sum K†K should be identity
	S = sum(Ki.conj().T @ Ki for Ki in K)
	assert np.allclose(S, np.eye(2), atol=1e-12)


def test_sample_counts_total_conserved():
	povm = sm.build_sic_povm_qubit()
	probes = pauli_eigen_probes()[:, :1]
	K = [np.eye(2, dtype=np.complex128)]
	counts = sm.sample_counts(K, probes, povm, shots_per_probe=50, seed=0)
	assert counts.sum() == 50


def test_choi_from_kraus_identity_properties():
	K = [np.eye(2, dtype=np.complex128)]
	J = sm.choi_from_kraus(K)
	# Trace of Choi = d
	assert np.isclose(np.trace(J).real, 2.0)
	# Rank 1 (within numerical tolerance)
	w, _ = np.linalg.eigh(J)
	assert (w > 1e-10).sum() == 1


def test_tp_residual_identity_zero():
	assert sm.tp_residual([np.eye(2, dtype=np.complex128)]) < 1e-12


def test_process_fidelity_self_equals_one():
	K = [np.eye(2, dtype=np.complex128)]
	J = sm.choi_from_kraus(K)
	F = sm.process_fidelity(J, J, d=2)
	assert math.isclose(F, 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_bloch_vector_of_zero_state():
	psi0 = np.array([1.0, 0.0], dtype=np.complex128)
	r = sm.bloch_vector(psi0)
	assert np.allclose(r, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_mle_qubit_channel_identity_fit_small():
	# Tiny optimisation to validate function runs & converges near identity
	povm = sm.build_sic_povm_qubit()
	probes = pauli_eigen_probes()
	K_true = [np.eye(2, dtype=np.complex128)]
	probs = sm.forward_probs(K_true, probes, povm)
	counts = (probs * 40).astype(int)  # small deterministic counts
	cfg = sm.MLEConfig(r=1, max_iters=60, verbose=False, init="identity", tol_grad=1e-7)
	J_true = sm.choi_from_kraus(K_true)
	res = sm.mle_qubit_channel(counts, probes, povm, cfg, J_true=J_true)
	assert res["final_nll"] >= 0
	assert res["grad_norm_history"][-1] < 1e-4  # loose
	assert sm.tp_residual(res["Kraus"]) < 1e-8


# ---------------------------------------------------------------------------
# Channel physics tests (9 channels)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
	"channel_name, kraus_builder, expected_T, expected_t",
	[
		(
			"identity",
			lambda: [np.eye(2, dtype=np.complex128)],
			lambda: np.eye(3),
			lambda: np.zeros(3),
		),
		(
			"rz_rotation",
			lambda: [np.array(
				[[np.exp(-1j * math.pi / 6), 0], [0, np.exp(1j * math.pi / 6)]],
				dtype=np.complex128,
			)],
			lambda: np.array([
				[math.cos(math.pi / 3), -math.sin(math.pi / 3), 0],
				[math.sin(math.pi / 3), math.cos(math.pi / 3), 0],
				[0, 0, 1],
			]),
			lambda: np.zeros(3),
		),
		(
			"depolarising_p0.2",
			lambda: [
				np.sqrt(0.8) * np.eye(2, dtype=np.complex128),
				np.sqrt(0.2 / 3) * np.array([[0, 1], [1, 0]], dtype=np.complex128),
				np.sqrt(0.2 / 3) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
				np.sqrt(0.2 / 3) * np.array([[1, 0], [0, -1]], dtype=np.complex128),
			],
			lambda: 0.8 * np.eye(3),
			lambda: np.zeros(3),
		),
		(
			"pauli_channel",
			lambda: [
				np.sqrt(1 - (0.1 + 0.05 + 0.2)) * np.eye(2, dtype=np.complex128),
				np.sqrt(0.1) * np.array([[0, 1], [1, 0]], dtype=np.complex128),
				np.sqrt(0.05) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
				np.sqrt(0.2) * np.array([[1, 0], [0, -1]], dtype=np.complex128),
			],
			lambda: np.diag([
				1 - 2 * (0.05 + 0.2),
				1 - 2 * (0.1 + 0.2),
				1 - 2 * (0.1 + 0.05),
			]),
			lambda: np.zeros(3),
		),
		(
			"phase_flip_p0.15",
			lambda: [
				np.sqrt(0.85) * np.eye(2, dtype=np.complex128),
				np.sqrt(0.15) * np.array([[1, 0], [0, -1]], dtype=np.complex128),
			],
			lambda: np.diag([1 - 2 * 0.15, 1 - 2 * 0.15, 1.0]),
			lambda: np.zeros(3),
		),
		(
			"phase_damping_gamma0.3",
			lambda: [
				np.array([[1, 0], [0, math.sqrt(1 - 0.3)]], dtype=np.complex128),
				np.sqrt(0.3) * np.array([[0, 0], [0, 1]], dtype=np.complex128),
			],
			lambda: np.diag([math.sqrt(1 - 0.3), math.sqrt(1 - 0.3), 1.0]),
			lambda: np.zeros(3),
		),
		(
			"amplitude_damping_gamma0.35",
			lambda: [
				np.array([[1, 0], [0, math.sqrt(1 - 0.35)]], dtype=np.complex128),
				math.sqrt(0.35) * np.array([[0, 1], [0, 0]], dtype=np.complex128),
			],
			lambda: np.diag([
				math.sqrt(1 - 0.35),
				math.sqrt(1 - 0.35),
				1 - 0.35,
			]),
			lambda: np.array([0.0, 0.0, 0.35]),
		),
		(
			"generalised_amplitude_damping",
			lambda: [
				math.sqrt(0.25)
				* np.array([[1, 0], [0, math.sqrt(1 - 0.3)]], dtype=np.complex128),
				math.sqrt(0.25)
				* np.array([[0, math.sqrt(0.3)], [0, 0]], dtype=np.complex128),
				math.sqrt(0.75)
				* np.array([[math.sqrt(1 - 0.3), 0], [0, 1]], dtype=np.complex128),
				math.sqrt(0.75)
				* np.array([[0, 0], [math.sqrt(0.3), 0]], dtype=np.complex128),
			],
			lambda: np.diag([
				math.sqrt(1 - 0.3),
				math.sqrt(1 - 0.3),
				1 - 0.3,
			]),
			lambda: np.array([0.0, 0.0, 0.3 * (1 - 2 * 0.25)]),
		),
		(
			"random_unitary_mixture_I_H_q0.4",
			lambda: [
				math.sqrt(0.6) * np.eye(2, dtype=np.complex128),
				math.sqrt(0.4)
				* (1 / math.sqrt(2))
				* np.array([[1, 1], [1, -1]], dtype=np.complex128),
			],
			lambda: np.array(
				[
					[1 - 0.4, 0, 0.4],
					[0, 1 - 2 * 0.4, 0],
					[0.4, 0, 1 - 0.4],
				]
			),
			lambda: np.zeros(3),
		),
	],
)
def test_channel_physics(channel_name, kraus_builder, expected_T, expected_t):
	probes = pauli_eigen_probes()
	povm = sm.build_sic_povm_qubit()
	K = kraus_builder()

	# Basic TP & CP checks
	assert sm.tp_residual(K) < 1e-10
	J = sm.choi_from_kraus(K)
	w, _ = np.linalg.eigh(0.5 * (J + J.conj().T))
	assert w.min() >= -1e-10
	assert np.isclose(np.trace(J).real, 2.0, atol=1e-10)

	# Born rule parity: Kraus forward probs vs Choi formula
	p1 = sm.forward_probs(K, probes, povm)
	p2 = choi_born_probs(J, probes, povm)
	assert np.allclose(p1, p2, atol=1e-10)

	# Bloch affine map comparisons
	T_obs, t_obs = extract_bloch_affine(K)
	T_exp = expected_T()
	t_exp = expected_t()
	assert np.allclose(T_obs, T_exp, atol=1e-8)
	assert np.allclose(t_obs, t_exp, atol=1e-8)

	# Extra invariant checks for selected channels
	if channel_name.startswith("phase_flip"):
		# Off-diagonal damping factor
		rho_plus = density_from_state((np.array([1, 0], dtype=np.complex128) + np.array([0, 1], dtype=np.complex128)) / np.sqrt(2))
		out = sum(Ki @ rho_plus @ Ki.conj().T for Ki in K)
		coherence = out[0, 1]
		lam = (T_exp[0, 0])  # = 1-2p
		assert np.isclose(coherence, lam / 2, atol=1e-8)  # Initial coherence 1/2

	if channel_name == "amplitude_damping_gamma0.35":
		rho1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
		out = sum(Ki @ rho1 @ Ki.conj().T for Ki in K)
		# Population of |1> decreases
		assert out[1, 1].real == pytest.approx(1 - 0.35, rel=1e-8)
		assert out[0, 0].real == pytest.approx(0.35, rel=1e-8)

	if channel_name == "generalised_amplitude_damping":
		# Fixed point z* = 1 - 2p = 0.5 should satisfy T z* + t = z*
		p = 0.25
		z_star = 1 - 2 * p
		# z' = T_zz z + t_z
		z_prime = T_obs[2, 2] * z_star + t_obs[2]
		assert np.isclose(z_prime, z_star, atol=1e-8)


def test_amplitude_damping_convergence_fixed_point():
	gamma = 0.35
	K = [
		np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128),
		math.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=np.complex128),
	]
	rho = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # |1><1|
	for _ in range(10):
		rho = sum(Ki @ rho @ Ki.conj().T for Ki in K)
	# Should approach ground state |0><0|
	assert rho[0, 0].real > 0.95
	assert rho[1, 1].real < 0.05


# ---------------------------------------------------------------------------
# End of tests
# ---------------------------------------------------------------------------

