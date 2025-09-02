"""Standalone sanity tests for CPTP projection implementation.

Run:
    python testing_cptp_projection.py

Mirrors the ad‑hoc style of `synthetic/testing_synthetic_channel_MLE.py` (no pytest).
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(_THIS_DIR))

from CPTP_implementation import (
    partial_trace_out,
    project_TP,
    project_CP,
    project_CPTP_dykstra,
    _random_cptp_choi,
)

RNG = np.random.default_rng(123)

def _status(name: str, ok: bool, msg: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")


def test_partial_trace_identity() -> bool:
    name = "partial_trace_identity"
    d = 3
    # Build |Phi> = sum_i |i,i>, J = |Phi><Phi| (unnormalized identity channel Choi for d_out=d_in)
    phi = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        phi[i * d + i] = 1.0
    J = np.outer(phi, phi.conjugate())
    Tr_out = partial_trace_out(J, d_in=d, d_out=d)
    ok = np.allclose(Tr_out, np.eye(d), atol=1e-12)
    _status(name, ok, f"||Tr_out - I||_F={np.linalg.norm(Tr_out-np.eye(d)):.2e}")
    return ok


def test_project_TP() -> bool:
    name = "project_TP"
    d_in, d_out = 2, 3
    N = d_in * d_out
    Y = RNG.normal(size=(N, N)) + 1j * RNG.normal(size=(N, N))
    Ytp = project_TP(Y, d_in, d_out)
    Tr_out = partial_trace_out(Ytp, d_in, d_out)
    ok = np.allclose(Tr_out, np.eye(d_in), atol=1e-10)
    _status(name, ok, f"res={np.linalg.norm(Tr_out-np.eye(d_in)):.2e}")
    return ok


def test_project_CP() -> bool:
    name = "project_CP"
    A = RNG.normal(size=(5,5)) + 1j * RNG.normal(size=(5,5))
    H = 0.5*(A + A.conjugate().T)  # Hermitian
    H[0,0] = -5.0  # ensure negative eigenvalue
    Z = project_CP(H)
    evals = np.linalg.eigvalsh(0.5*(Z+Z.conjugate().T))
    ok = evals.min() >= -1e-12
    _status(name, ok, f"min_eig={evals.min():.2e}")
    return ok


def test_dykstra_fixed_point() -> bool:
    name = "dykstra_fixed_point"
    d_in, d_out = 2, 2
    J = _random_cptp_choi(d_in, d_out, seed=0)
    X, info = project_CPTP_dykstra(J, d_in, d_out, tol=1e-10, max_iter=300)
    dist_rel = np.linalg.norm(X-J)/max(1, np.linalg.norm(J))
    ok = info['tp_residual'] < 1e-8 and info['min_eig'] > -1e-10 and dist_rel < 1e-8
    _status(name, ok, f"tp_res={info['tp_residual']:.1e}, min_eig={info['min_eig']:.1e}, rel_dist={dist_rel:.1e}")
    return ok


def test_dykstra_projection_changes_invalid() -> bool:
    name = "dykstra_projection_changes_invalid"
    d_in, d_out = 2, 2
    J = _random_cptp_choi(d_in, d_out, seed=1)
    # Break TP & positivity
    H = RNG.normal(size=J.shape) + 1j * RNG.normal(size=J.shape)
    H = 0.5*(H + H.conjugate().T)
    J_bad = J + 0.6*H + 0.5*np.eye(J.shape[0])
    X, info = project_CPTP_dykstra(J_bad, d_in, d_out, tol=1e-7, max_iter=500)
    Tr_out = partial_trace_out(X, d_in, d_out)
    tp_ok = np.linalg.norm(Tr_out - np.eye(d_in)) < 1e-6
    evals = np.linalg.eigvalsh(0.5*(X+X.conjugate().T))
    psd_ok = evals.min() >= -1e-8
    improved = np.linalg.norm(X-J) <= np.linalg.norm(J_bad-J) + 1e-9
    ok = tp_ok and psd_ok and improved
    _status(name, ok, f"tp_res={np.linalg.norm(Tr_out-np.eye(d_in)):.1e}, min_eig={evals.min():.1e}, improved={improved}")
    return ok


def test_scaling_larger_dim() -> bool:
    name = "scaling_larger_dim"
    d_in, d_out = 3, 2
    J = _random_cptp_choi(d_in, d_out, seed=5)
    # Deliberately perturb
    P = RNG.normal(size=J.shape) + 1j * RNG.normal(size=J.shape)
    P = 0.5*(P + P.conjugate().T)
    J_bad = J + 0.3*P
    X, info = project_CPTP_dykstra(J_bad, d_in, d_out, tol=1e-7, max_iter=600)
    Tr_out = partial_trace_out(X, d_in, d_out)
    tp_ok = np.linalg.norm(Tr_out - np.eye(d_in)) < 1e-6
    evals = np.linalg.eigvalsh(0.5*(X+X.conjugate().T))
    psd_ok = evals.min() >= -1e-8
    ok = tp_ok and psd_ok
    _status(name, ok, f"tp_res={np.linalg.norm(Tr_out-np.eye(d_in)):.1e}, min_eval={evals.min():.1e}, iters={info['iterations']}")
    return ok


def is_TP(X, d_in, d_out, tol=1e-9):
    T = partial_trace_out(X, d_in, d_out)
    return np.linalg.norm(T - np.eye(d_in)) <= tol


def min_eig(X):
    return float(np.linalg.eigvalsh(0.5 * (X + X.conj().T))[0])


def is_CP(X, tol=1e-10):
    return min_eig(X) >= -tol


def proj_density_matrix(R):
    R = 0.5 * (R + R.conj().T)
    w, U = np.linalg.eigh(R)
    # project eigenvalues onto simplex
    lam = w[::-1]
    csum = np.cumsum(lam)
    rho = lam - (csum - 1) / (np.arange(1, len(lam) + 1))
    k = np.argmax(rho > 0) + 1
    tau = (csum[k - 1] - 1) / k
    p = np.maximum(w - tau, 0)
    return (U * p) @ U.conj().T


def test_fixed_point_various_dims() -> bool:
    name = "fixed_point_various_dims"
    ok_all = True
    for d_in, d_out in [(2, 2), (3, 2)]:
        J = _random_cptp_choi(d_in, d_out, seed=10 + d_in + d_out)
        X, _ = project_CPTP_dykstra(J, d_in, d_out, tol=1e-10)
        dist = np.linalg.norm(X - J)
        ok = dist <= 1e-8 and is_TP(X, d_in, d_out, 1e-10) and is_CP(X, 1e-10)
        ok_all &= ok
    _status(name, ok_all, "all dims fixed point")
    return ok_all


def test_tp_not_cp():
    name = "tp_not_cp"
    d_in = d_out = 2
    J = _random_cptp_choi(d_in, d_out, seed=21)
    # Build direction with zero partial trace so TP is preserved
    H = np.random.default_rng(123).normal(size=J.shape) + 1j * np.random.default_rng(124).normal(size=J.shape)
    H = 0.5 * (H + H.conj().T)
    # Make zero partial trace: subtract its TP component minus identity part
    Tr_out_H = partial_trace_out(H, d_in, d_out)
    Delta = Tr_out_H / d_out  # since d_in=d_out=2
    K = H - np.kron(np.eye(d_out), Delta)
    # Scale to induce negativity
    t = 0.5 / (abs(min_eig(K)) + 1e-12)
    J_bad = J - (t * 1.2) * K  # subtract along K
    J_bad = project_TP(J_bad, d_in, d_out)  # enforce exact TP again (should be already)
    cond_setup = is_TP(J_bad, d_in, d_out) and (not is_CP(J_bad))
    X, _ = project_CPTP_dykstra(J_bad, d_in, d_out, tol=1e-8)
    ok = cond_setup and is_TP(X, d_in, d_out) and is_CP(X)
    _status(name, ok)
    return ok


def test_cp_not_tp():
    name = "cp_not_tp"
    d_in, d_out = 3, 2
    J = _random_cptp_choi(d_in, d_out, seed=3)
    Delta = np.diag([0.1, -0.05, -0.05])
    J_bad = J + np.kron(np.eye(d_out), Delta)
    # Ensure PSD by a CP projection (can disturb TP further but keeps CP)
    J_bad = project_CP(J_bad)
    cond_setup = is_CP(J_bad) and (not is_TP(J_bad, d_in, d_out))
    X, _ = project_CPTP_dykstra(J_bad, d_in, d_out)
    ok = cond_setup and is_TP(X, d_in, d_out) and is_CP(X)
    _status(name, ok)
    return ok


def test_non_hermitian_input():
    name = "non_hermitian_input"
    d_in, d_out = 2, 3
    J = _random_cptp_choi(d_in, d_out, seed=4)
    J = J + 1j * RNG.normal(size=J.shape)  # add skew part
    X, _ = project_CPTP_dykstra(J, d_in, d_out, tol=1e-8, max_iter=800)
    ok = np.allclose(X, X.conj().T, atol=1e-10) and is_TP(X, d_in, d_out, 1e-7) and is_CP(X)
    _status(name, ok)
    return ok


def test_idempotence():
    name = "idempotence"
    d_in, d_out = 3, 2
    J = _random_cptp_choi(d_in, d_out, seed=5)
    J_bad = J + 0.3 * RNG.standard_normal(size=J.shape)
    X1, _ = project_CPTP_dykstra(J_bad, d_in, d_out)
    X2, _ = project_CPTP_dykstra(X1, d_in, d_out)
    ok = np.linalg.norm(X2 - X1) <= 1e-8
    _status(name, ok)
    return ok


def test_din1_density_case():
    name = "din1_density_case"
    d_in, d_out = 1, 4
    R = RNG.standard_normal((d_out, d_out)) + 1j * RNG.standard_normal((d_out, d_out))
    R = 0.5 * (R + R.conj().T)
    X_dyk, _ = project_CPTP_dykstra(R, d_in, d_out, tol=1e-10, max_iter=800)
    X_closed = proj_density_matrix(R)
    ok = np.allclose(X_dyk, X_closed, atol=1e-7)
    _status(name, ok)
    return ok


def test_block_diagonal_cq_case():
    name = "block_diag_cq_case"
    d_in = d_out = 3
    rng = np.random.default_rng(9)
    blocks = [rng.standard_normal((d_out, d_out)) + 1j * rng.standard_normal((d_out, d_out)) for _ in range(d_in)]
    blocks = [0.5 * (B + B.conj().T) for B in blocks]
    J = np.zeros((d_in * d_out, d_in * d_out), dtype=complex)
    for i, B in enumerate(blocks):
        rs = slice(i * d_out, (i + 1) * d_out)
        cs = slice(i * d_out, (i + 1) * d_out)
        J[rs, cs] = B
    # Naive per-block density projection (not necessarily the global CPTP projection)
    J_block = np.zeros_like(J)
    for i, B in enumerate(blocks):
        rs = slice(i * d_out, (i + 1) * d_out)
        cs = slice(i * d_out, (i + 1) * d_out)
        J_block[rs, cs] = proj_density_matrix(B)
    X, _ = project_CPTP_dykstra(J, d_in, d_out)
    # Assert CPTP
    cond_cptp = is_TP(X, d_in, d_out) and is_CP(X)
    # Projection optimality: distance to J should not exceed naive candidate's distance by more than tiny tolerance.
    dist_proj = np.linalg.norm(X - J)
    dist_naive = np.linalg.norm(J_block - J)
    cond_opt = dist_proj <= dist_naive + 1e-8
    ok = cond_cptp and cond_opt
    _status(name, ok)
    return ok


def test_distance_vs_naive():
    name = "distance_vs_naive"
    d_in = d_out = 2
    rng = np.random.default_rng(11)
    J = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    X, _ = project_CPTP_dykstra(J, d_in, d_out, tol=1e-7)
    X_hand = project_TP(project_CP(J), d_in, d_out)
    ok = np.linalg.norm(X - J) <= np.linalg.norm(X_hand - J) + 1e-10
    _status(name, ok)
    return ok


def test_stress_sizes():
    name = "stress_sizes"
    ok_all = True
    for d_in, d_out in [(4, 3), (5, 2)]:
        N = d_in * d_out
        rng = np.random.default_rng(13 + d_in + d_out)
        J = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        X, _ = project_CPTP_dykstra(J, d_in, d_out, tol=1e-7)
        okd = is_TP(X, d_in, d_out, 1e-6) and is_CP(X, 1e-8)
        ok_all &= okd
    _status(name, ok_all)
    return ok_all


TESTS: List[Callable[[], bool]] = [
    test_partial_trace_identity,
    test_project_TP,
    test_project_CP,
    test_dykstra_fixed_point,
    test_dykstra_projection_changes_invalid,
    test_scaling_larger_dim,
    test_fixed_point_various_dims,
    test_tp_not_cp,
    test_cp_not_tp,
    test_non_hermitian_input,
    test_idempotence,
    test_din1_density_case,
    test_block_diagonal_cq_case,
    test_stress_sizes,
]


def run_all() -> bool:
    results = [t() for t in TESTS]
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed.")
    return all(results)

if __name__ == "__main__":
    run_all()
