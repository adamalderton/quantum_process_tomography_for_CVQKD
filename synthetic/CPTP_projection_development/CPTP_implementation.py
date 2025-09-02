"""Minimal CPTP projection (unnormalized Choi) via Dykstra.

Given J (N x N) with N=d_in*d_out, project onto
  C = { X >= 0, Tr_out X = I_{d_in} } under Frobenius norm.

Implements:
  project_TP(Y, d_in, d_out)
  project_CP(Y)
  project_CPTP_dykstra(J, d_in, d_out, ...)

Notes follow implementation_notes.md.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _validate_dims(J: Array, d_in: int, d_out: int):
	if d_in <= 0 or d_out <= 0:
		raise ValueError("d_in and d_out must be positive integers")
	N = d_in * d_out
	if J.shape != (N, N):
		raise ValueError(f"Expected matrix of shape {(N, N)}, got {J.shape}")


def partial_trace_out(X: Array, d_in: int, d_out: int) -> Array:
	"""Compute Tr_out X (returns d_in x d_in matrix).

	X viewed as blocks (i,j) each d_out x d_out; take trace of each block.
	"""
	# Reshape trick: (d_in, d_out, d_in, d_out)
	X_rs = X.reshape(d_in, d_out, d_in, d_out)
	return np.einsum("ikjk->ij", X_rs)


def project_TP(Y: Array, d_in: int, d_out: int) -> Array:
	"""Orthogonal projection onto affine TP set Tr_out X = I_{d_in}.

	P(Y) = Y - (I_out \otimes Delta), Delta = (Tr_out Y - I) / d_out.
	Implemented without forming Kronecker: subtract Delta from each
	output-diagonal block.
	"""
	N = d_in * d_out
	Tr_out_Y = partial_trace_out(Y, d_in, d_out)
	Delta = (Tr_out_Y - np.eye(d_in, dtype=Y.dtype)) / d_out
	# Subtract Delta from each output-diagonal block
	Y_corr = Y.copy()
	# Iterate over input indices (block indices)
	for i in range(d_in):
		for j in range(d_in):
			rs = slice(i * d_out, (i + 1) * d_out)
			cs = slice(j * d_out, (j + 1) * d_out)
			Y_corr[rs, cs] -= Delta[i, j] * np.eye(d_out, dtype=Y.dtype)
	return Y_corr


def project_CP(Y: Array, hermitize: bool = True, eps_clip: float = 1e-12) -> Array:
	"""Projection onto PSD cone (Hermitian part then eigen clip).

	True Euclidean projection: set negative eigenvalues to 0, keep positives.
	Then optionally zero out tiny numerical dust ( < eps_clip ).
	"""
	if hermitize:
		Y = 0.5 * (Y + Y.conj().T)
	w, U = np.linalg.eigh(Y)
	w_clipped = np.clip(w, 0.0, None)
	if eps_clip > 0:
		w_clipped[w_clipped < eps_clip] = 0.0
	Z = (U * w_clipped) @ U.conj().T
	return 0.5 * (Z + Z.conj().T)


def project_CPTP_dykstra(
	J: Array,
	d_in: int,
	d_out: int,
	max_iter: int = 500,
	tol: float = 1e-8,
	eig_clip: float = 1e-12,
	verbose: bool = False,
) -> tuple[Array, dict]:
	"""Project J onto CPTP set using Dykstra.

	Returns (X_proj, info dict).
	info: {'iterations', 'tp_residual', 'min_eig', 'delta_norm'}
	"""
	_validate_dims(J, d_in, d_out)
	N = d_in * d_out
	# Work in complex128 for safety
	X = J.astype(np.complex128, copy=True)
	R = np.zeros_like(X)
	S = np.zeros_like(X)
	prev_X = X.copy()
	for k in range(1, max_iter + 1):
		Yk = project_TP(X + R, d_in, d_out)
		R = X + R - Yk
		Zk = project_CP(Yk + S, hermitize=True, eps_clip=eig_clip)
		S = Yk + S - Zk
		X = Zk
		if k % 5 == 0 or k == 1:  # diagnostics cadence
			Tr_out = partial_trace_out(X, d_in, d_out)
			tp_res = np.linalg.norm(Tr_out - np.eye(d_in))
			evals = np.linalg.eigvalsh(0.5 * (X + X.conj().T))
			min_eig = float(evals[0])
			delta = np.linalg.norm(X - prev_X) / max(1.0, np.linalg.norm(prev_X))
			if verbose:
				print(
					f"iter {k}: tp_res={tp_res:.2e}, min_eig={min_eig:.2e}, rel_change={delta:.2e}"
				)
			if tp_res <= tol and min_eig >= -tol and delta <= tol:
				break
			prev_X = X.copy()
	Tr_out = partial_trace_out(X, d_in, d_out)
	tp_res = float(np.linalg.norm(Tr_out - np.eye(d_in)))
	evals = np.linalg.eigvalsh(0.5 * (X + X.conj().T))
	info = {
		"iterations": k,
		"tp_residual": tp_res,
		"min_eig": float(evals[0]),
		"delta_norm": float(np.linalg.norm(X - prev_X) / max(1.0, np.linalg.norm(prev_X))),
	}
	return X, info


def _random_cptp_choi(d_in: int, d_out: int, n_kraus: int | None = None, seed: int = 0) -> Array:
	"""Generate a random CPTP Choi matrix (unnormalized) via Kraus operators.

	Chooses A_k : C^{d_in} -> C^{d_out} with sum_k A_k^† A_k = I, then
	J = sum_k (I \otimes A_k) |Omega><Omega| (I \otimes A_k^†).
	Unnormalized maximally entangled |Omega> = sum_i |i,i>.
	"""
	rng = np.random.default_rng(seed)
	if n_kraus is None:
		n_kraus = d_in * d_out
	# Start with random Ginibre then enforce TP by polar correction
	As = []
	M = np.zeros((d_in, d_in), dtype=np.complex128)
	for _ in range(n_kraus):
		A = rng.normal(size=(d_out, d_in)) + 1j * rng.normal(size=(d_out, d_in))
		As.append(A)
		M += A.conj().T @ A
	# Normalize: want sum A_k^† A_k = I
	# Perform M^{-1/2} right scaling
	w, V = np.linalg.eigh(M)
	w_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(w, 1e-15, None)))
	Minv_sqrt = V @ w_inv_sqrt @ V.conj().T
	As = [A @ Minv_sqrt for A in As]
	# Build Choi
	# |Omega> (unnormalized)
	Omega = np.zeros((d_in * d_in,), dtype=np.complex128)
	for i in range(d_in):
		Omega[i * d_in + i] = 1.0
	J = np.zeros((d_in * d_out, d_in * d_out), dtype=np.complex128)
	for A in As:
		# (I \otimes A) |Omega>
		# Reshape Omega to (d_in, d_in) matrix form F with F_{ij} = delta_{ij}
		# vectorization identity: (I \otimes A) vec(I) = vec(A)
		vecA = A.reshape(-1)
		v = vecA  # already size d_out * d_in in column-major? Need consistent ordering.
		# Ensure ordering matches Choi convention used: we picked row-major earlier.
		# Use reshape then kron approach for clarity (small dims typical in tests).
		F = np.eye(d_in, dtype=np.complex128)
		v = (np.kron(np.eye(d_in), A) @ F.reshape(-1))
		J += np.outer(v, v.conj())
	return J


__all__ = [
	"partial_trace_out",
	"project_TP",
	"project_CP",
	"project_CPTP_dykstra",
	"_random_cptp_choi",
]
