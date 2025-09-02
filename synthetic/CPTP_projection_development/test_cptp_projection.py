import numpy as np
import pytest

from .CPTP_implementation import (
    project_TP,
    project_CP,
    project_CPTP_dykstra,
    partial_trace_out,
    _random_cptp_choi,
)


def test_partial_trace_identity_channel():
    d = 2
    # Choi of identity (unnormalized) = sum_{ij} |ii><jj| = d * projector onto maximally entangled? For unnormalized, Tr_out J = I.
    # Build via vec(I) vec(I)^
    psi = np.eye(d).reshape(-1)
    J = np.outer(psi, psi.conj())
    Tr_out = partial_trace_out(J, d_in=d, d_out=d)
    assert np.allclose(Tr_out, np.eye(d), atol=1e-10)


def test_project_TP_enforces_trace():
    d_in, d_out = 2, 3
    N = d_in * d_out
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Ytp = project_TP(Y, d_in, d_out)
    Tr_out = partial_trace_out(Ytp, d_in, d_out)
    assert np.allclose(Tr_out, np.eye(d_in), atol=1e-10)


def test_project_CP_psd():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    Y = A + A.conj().T  # Hermitian indefinite
    # Force some negative eigenvalues
    Y[0, 0] = -5
    Z = project_CP(Y)
    evals = np.linalg.eigvalsh(0.5 * (Z + Z.conj().T))
    assert evals.min() >= -1e-12


def test_dykstra_on_random_cptp_fixed_point():
    d_in, d_out = 2, 2
    J = _random_cptp_choi(d_in, d_out, seed=0)
    X, info = project_CPTP_dykstra(J, d_in, d_out, tol=1e-9, max_iter=300)
    # Should already be (near) fixed point
    assert info["tp_residual"] <= 1e-8
    evals = np.linalg.eigvalsh(0.5 * (X + X.conj().T))
    assert evals.min() >= -1e-10
    # Distance should be small
    assert np.linalg.norm(X - J) / max(1, np.linalg.norm(J)) < 1e-8


def test_dykstra_projects_invalid():
    d_in, d_out = 2, 2
    J = _random_cptp_choi(d_in, d_out, seed=1)
    # Perturb to violate TP and positivity
    rng = np.random.default_rng(10)
    H = rng.normal(size=J.shape) + 1j * rng.normal(size=J.shape)
    H = 0.5 * (H + H.conj().T)
    J_bad = J + 0.2 * H + 0.5 * np.eye(J.shape[0])  # likely off constraints
    X, info = project_CPTP_dykstra(J_bad, d_in, d_out, tol=1e-7, max_iter=400)
    Tr_out = partial_trace_out(X, d_in, d_out)
    assert np.allclose(Tr_out, np.eye(d_in), atol=1e-6)
    evals = np.linalg.eigvalsh(0.5 * (X + X.conj().T))
    assert evals.min() >= -1e-8


if __name__ == "__main__":
    pytest.main([__file__])
