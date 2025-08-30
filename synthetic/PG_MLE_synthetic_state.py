"""Projected-Gradient (PG) MLE for synthetic single-mode quantum state tomography.

This file mirrors the structure of `rrr_MLE_synthetic_state.py` but replaces the
RρR (EM-style) update loop with a first-order projected-gradient optimizer for the
same Poisson factorised negative log-likelihood described in the accompanying
`PG_MLE_state_estimation_notes.md`.

Key points:
	* All helper functions (grid building, sampling, histogramming, coherent
	  overlaps, probability forward model, NLL, projection) are IMPORTED from
	  the existing RRR implementation; that file is left unmodified.
	* Only the iterative scheme (mle_pg) differs: we form the full gradient
	  ∇L(ρ) = Σ_j (N_tot - n_j / p_j(ρ)) E_j  (+ entropy term if τ>0) and take
	  Armijo-backtracked steps followed by projection onto the PSD trace-one set.
	* Public workflow functions intentionally reuse the same names
	  (`run_mle_workflow`, `run_default_tests`, `main_cli`) so importing *from
	  this module* shadows the RRR versions and seamlessly swaps the optimizer.

Usage:
	from PG_MLE_synthetic_state import run_mle_workflow
	out = run_mle_workflow(state_type='coherent', ...)

CLI behaviour matches the RRR script; running this file directly with no
arguments executes the default batch test (PG variant).
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional

# Re-use ALL existing utilities (sampling, grid, probability, projection, etc.)
from rrr_MLE_synthetic_state import (  # noqa: F401,F403
	# sampling / grids / numerics
	build_grid,
	histogram_on_grid,
	adaptive_grid_half_size,
	sample_pure_coherent_state,
	sample_number_state,
	build_coherent_state_matrix,
	initialise_rho,
	true_rho_coherent,
	analytic_Q_coherent_state,
	analytic_Q_number_state,
	Q_from_rho,
	# probability + objective helpers
	compute_model_probabilities,
	neg_log_likelihood_poisson,
	# linear-algebra helpers
	_project_psd_trace_one,
	_hermitize,
)

matplotlib.use("Agg")

#################### PROJECTED GRADIENT CORE ###################

def _rank1_sum(C: np.ndarray, weights: np.ndarray) -> np.ndarray:
	"""Efficiently build Σ_j w_j |α_j><α_j| given C[:,j]=<n|α_j>.

	Args:
		C: (N+1, M) coherent overlap matrix.
		weights: (M,) real weights w_j.

	Returns:
		(N+1, N+1) Hermitian PSD matrix.
	"""
	if C.ndim != 2:
		raise ValueError("C must be 2D")
	w = np.asarray(weights, dtype=np.float64)
	if w.ndim != 1 or w.size != C.shape[1]:
		raise ValueError("weights shape mismatch")
	CW = C * w  # broadcast over rows
	R = CW @ C.conjugate().T
	return 0.5 * (R + R.conjugate().T)


def _entropy_gradient(rho: np.ndarray, floor: float = 1e-18) -> np.ndarray:
	"""Compute grad of Tr(ρ log ρ) = log ρ + I with eigen floor."""
	Xh = _hermitize(rho)
	vals, vecs = np.linalg.eigh(Xh)
	vals = np.clip(vals, floor, None)
	log_rho = (vecs * np.log(vals)) @ vecs.conjugate().T
	return log_rho + np.eye(rho.shape[0], dtype=np.complex128)


def mle_pg(
	counts_2d: np.ndarray,
	C: np.ndarray,
	bin_area: float,
	init_rho: np.ndarray,
	*,
	max_iters: int = 300,
	tol: float = 1e-6,
	entropy_tau: float = 0.0,
	step_init: float = 1.0,
	armijo_beta: float = 0.5,
	armijo_c: float = 1e-4,
	min_step: float = 1e-12,
	verbose: bool = False,
) -> Tuple[np.ndarray, List[float]]:
	"""Projected-gradient MLE loop (Poisson factorisation) with Armijo line search.

	Objective (optionally regularised):
		L(ρ) = PoissonNLL(ρ) + τ Tr(ρ log ρ)

	Gradient (inside truncation):
		∇L = Σ_j (N_tot - n_j / p_j(ρ)) E_j + τ (log ρ + I)
		with E_j = (ΔA/π) |α_j><α_j|.

	Args mirror the RRR routine for easy swapping.
	Returns final ρ and NLL history (recorded *without* the entropy term for comparability).
	"""

	if counts_2d.ndim != 2:
		raise ValueError("counts_2d must be 2D")
	if bin_area <= 0:
		raise ValueError("bin_area must be positive")
	if C.ndim != 2:
		raise ValueError("C must be (N+1,M)")
	if C.shape[0] != init_rho.shape[0]:
		raise ValueError("Dimension mismatch between C and init_rho")

	rho = _project_psd_trace_one(init_rho)
	n_vec = counts_2d.ravel(order="C").astype(np.float64)
	N_total = float(n_vec.sum())
	nll_history: List[float] = []

	# Initial objective
	p = compute_model_probabilities(rho, C, bin_area)
	nll = neg_log_likelihood_poisson(counts_2d, p, N_total=N_total, include_constant=False)
	nll_history.append(float(nll))

	# For Armijo reuse gradient norm etc.
	for it in range(1, max_iters + 1):
		# Forward probabilities & gradient weights
		p = compute_model_probabilities(rho, C, bin_area)  # (M,)
		eps = 1e-30
		p_safe = np.clip(p, eps, None)
		# weights for Σ_j (N_tot - n_j / p_j) E_j
		w_grad = (N_total - n_vec / p_safe) * (bin_area / np.pi)
		G = _rank1_sum(C, w_grad)
		if entropy_tau > 0.0:
			G = G + entropy_tau * _entropy_gradient(rho)

		grad_norm_sq = float(np.vdot(G, G).real)
		if grad_norm_sq <= 1e-30:
			if verbose:
				print("PG: gradient norm tiny -> stop")
			break

		# Armijo line search
		step = step_init
		base_nll = nll
		accepted = False
		while step >= min_step:
			trial = rho - step * G  # gradient descent step
			trial = _project_psd_trace_one(trial)
			p_trial = compute_model_probabilities(trial, C, bin_area)
			nll_trial = neg_log_likelihood_poisson(counts_2d, p_trial, N_total=N_total, include_constant=False)
			# Regularised objective for acceptance test
			if entropy_tau > 0.0:
				# add τ Tr(ρ log ρ) for both current and trial
				reg_current = base_nll + entropy_tau * _entropy_value(rho)
				reg_trial = nll_trial + entropy_tau * _entropy_value(trial)
			else:
				reg_current = base_nll
				reg_trial = nll_trial
			if reg_trial <= reg_current - armijo_c * step * grad_norm_sq:
				accepted = True
				rho = trial
				nll = nll_trial  # store *unregularised* NLL like RRR version
				break
			step *= armijo_beta
		if not accepted:
			if verbose:
				print("PG: line search failed to decrease objective; stopping")
			break

		nll_history.append(float(nll))

		# Relative improvement check (using unregularised NLL for parity)
		prev = nll_history[-2]
		curr = nll_history[-1]
		if prev - curr >= 0:
			rel = (prev - curr) / (abs(prev) + 1e-18)
			if rel < tol:
				if verbose:
					print(f"PG: tol reached at iter {it} (rel {rel:.2e})")
				break
	return rho, nll_history


def _entropy_value(rho: np.ndarray, floor: float = 1e-18) -> float:
	"""Compute Tr(ρ log ρ) safely."""
	Xh = _hermitize(rho)
	vals, _ = np.linalg.eigh(Xh)
	vals = np.clip(vals, floor, None)
	return float(np.sum(vals * np.log(vals)))


#################### WORKFLOW (mirrors RRR API) ###################

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
	"""End-to-end PG MLE driver (API compatible with RRR variant)."""
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

	# 2. Window size
	if grid_half_size is None:
		grid_half_size = adaptive_grid_half_size(samples_alpha, padding=padding)

	# 3. Grid
	alphas_grid, _, _ = build_grid(grid_half_size, grid_side_points)

	# 4. Histogram
	Q_hat, counts_2d, coverage_hist, counts_in, bin_area = histogram_on_grid(
		samples_alpha, grid_half_size, grid_side_points, padding=padding
	)

	# 5. Coherent overlaps
	C = build_coherent_state_matrix(alphas_grid.ravel(order="C"), fock_cutoff)

	# 6. Initial rho
	if init_method.strip().lower().replace('-', '_') == "alpha_bar":
		alpha_bar = samples_alpha.mean()
		init_rho = initialise_rho(fock_cutoff, method="alpha_bar", alpha_bar=alpha_bar)
	else:
		init_rho = initialise_rho(fock_cutoff, method="maximally_mixed")

	# 7. PG loop
	rho_est, nll_history = mle_pg(
		counts_2d=counts_2d,
		C=C,
		bin_area=bin_area,
		init_rho=init_rho,
		max_iters=max_iters,
		tol=tol,
		entropy_tau=entropy_tau,
		verbose=verbose,
	)

	# True rho & fidelity
	true_rho = None
	fidelity = None
	if state_key == "coherent":
		true_rho, _tail = true_rho_coherent(coherent_alpha, fock_cutoff)
		vals, vecs = np.linalg.eigh(true_rho)
		psi = vecs[:, np.argmax(vals)]
		fidelity = float(np.real(np.vdot(psi, rho_est @ psi)))
	else:
		if number_k <= fock_cutoff:
			basis_vec = np.zeros(fock_cutoff + 1, dtype=np.complex128)
			basis_vec[number_k] = 1.0
			fidelity = float(np.real(np.vdot(basis_vec, rho_est @ basis_vec)))
			true_rho = np.outer(basis_vec, basis_vec.conjugate())
		else:
			fidelity = 0.0

	if verbose:
		print("--- PG MLE Summary ---")
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
		print("-----------------------")

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

	if plot:
		try:
			import os
			from matplotlib.colors import TwoSlopeNorm
			G = grid_side_points
			Q_recon = Q_from_rho(alphas_grid, rho_est)
			if state_key == "coherent":
				Q_true = analytic_Q_coherent_state(alphas_grid, coherent_alpha)
				identifier = f"pg_a={coherent_alpha.real:.2f}{'+' if coherent_alpha.imag>=0 else ''}{coherent_alpha.imag:.2f}i"
			else:
				Q_true = analytic_Q_number_state(alphas_grid, number_k)
				identifier = f"pg_n={number_k}"
			fig, axes = plt.subplots(3, 3, figsize=(12, 12))
			extent = (-grid_half_size, grid_half_size, -grid_half_size, grid_half_size)

			def im(ax, data, title, cmap="viridis", vmin=None, vmax=None):
				im_ = ax.imshow(data, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
				ax.set_title(title)
				ax.set_xlabel("y1")
				ax.set_ylabel("y2")
				fig.colorbar(im_, ax=ax, fraction=0.046, pad=0.04)

			if true_rho is not None:
				im(axes[0,0], Q_true, "True Q(α)")
			else:
				axes[0,0].set_title("True Q(α) unavailable")
				axes[0,0].axis('off')
			im(axes[0,1], Q_hat, "Histogram Q̂(α)")
			im(axes[0,2], Q_recon, "Reconstructed Q(α)")

			if true_rho is not None:
				v_rho = np.max(np.abs(true_rho))
				im2 = axes[1,0].imshow(np.abs(true_rho), origin='lower', cmap='viridis', vmin=0, vmax=v_rho)
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
			idx = np.arange(len(diag_est))
			if true_rho is not None:
				diag_true = np.real(np.diag(true_rho))
				width = 0.4
				ax_diag.bar(idx - width/2, diag_true, width=width, label='true')
				ax_diag.bar(idx + width/2, diag_est, width=width, label='est')
			else:
				ax_diag.bar(idx, diag_est, width=0.6, label='est')
			ax_diag.legend()
			ax_diag.set_title("Diag(ρ) comparison")
			ax_diag.set_xlabel('n')
			ax_diag.set_ylabel('Population')

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

			fig.suptitle(f"PG MLE Reconstruction ({identifier})  samples={num_samples}", fontsize=14)
			fig.tight_layout(rect=[0,0,1,0.97])
			results_dir = os.path.join(os.path.dirname(__file__), 'results')
			os.makedirs(results_dir, exist_ok=True)
			filepath = os.path.join(results_dir, f"{identifier}.png")
			fig.savefig(filepath, dpi=160)
			plt.close(fig)
			result['figure_path'] = filepath
			if verbose:
				print(f"Saved figure -> {filepath}")
		except Exception as e:  # noqa: BLE001
			if verbose:
				print(f"Plotting failed: {e}")

	return result


def _add_cli_arguments(parser):  # mirror original CLI
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
	parser.add_argument("--max-iters", type=int, default=500, help="Maximum PG iterations")
	parser.add_argument("--tol", type=float, default=1e-6, help="Relative NLL improvement tolerance")
	parser.add_argument("--no-verbose", action="store_true", help="Suppress printed summary")
	parser.add_argument("--seed", type=int, default=None, help="Random seed")
	parser.add_argument("--plot", action="store_true", help="Generate 3x3 diagnostic figure and save to results/")
	return parser


def main_cli():
	import argparse
	parser = _add_cli_arguments(argparse.ArgumentParser(description="Heterodyne PG MLE"))
	args = parser.parse_args()
	rng = np.random.default_rng(args.seed) if args.seed is not None else None
	coherent_alpha = complex(args.alpha_real, args.alpha_imag)
	return run_mle_workflow(
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


#################### DEFAULT TEST BATTERY (PG) ###################

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
	summary_filename: str = "pg_test_summary.png",
	init_method: str = "alpha_bar",
) -> Dict[str, object]:
	"""PG analogue of the RRR batch test (coherent grid + number states)."""
	rng = np.random.default_rng(seed)
	results: Dict[str, object] = {}

	fock_fidelities = []
	fock_iterations = []
	fock_final_nll = []
	for k in range(5):
		if verbose:
			print(f"[PG Batch] Fock state k={k}")
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

	re_vals = np.linspace(-coherent_range, coherent_range, 5)
	im_vals = np.linspace(-coherent_range, coherent_range, 5)
	coh_fidelity = np.zeros((5,5), dtype=float)
	coh_iters = np.zeros((5,5), dtype=float)
	coh_final_nll = np.zeros((5,5), dtype=float)
	for i, re in enumerate(re_vals):
		for j, im in enumerate(im_vals):
			alpha0 = complex(re, im)
			if verbose:
				print(f"[PG Batch] Coherent α={alpha0:.3g}")
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

	if make_panel_figs:
		try:
			import os
			from matplotlib.colors import Normalize
			fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12,5))
			ks = np.arange(5)
			bar = ax_left.bar(ks, fock_fidelities, color='tab:blue', alpha=0.7, label='Fidelity')
			ax_left.set_ylim(0, 1.05)
			ax_left.set_xlabel('Fock k')
			ax_left.set_ylabel('Fidelity')
			ax_left.set_title('PG: Fock states reconstruction')
			ax2 = ax_left.twinx()
			ax2.plot(ks, fock_iterations, color='tab:orange', marker='o', label='Iterations')
			ax2.set_ylabel('Iterations')
			lines, labels = [], []
			for h in [bar]:
				lines.append(h)
				labels.append('Fidelity')
			l2, = ax2.plot([], [], color='tab:orange', marker='o')
			lines.append(l2)
			labels.append('Iterations')
			ax_left.legend(lines, labels, loc='lower center')
			im_show = ax_right.imshow(coh_fidelity, origin='lower', cmap='viridis',
									  extent=(re_vals[0], re_vals[-1], im_vals[0], im_vals[-1]),
									  aspect='equal', vmin=0, vmax=1)
			ax_right.set_xlabel('Re(α)')
			ax_right.set_ylabel('Im(α)')
			ax_right.set_title('PG: Coherent grid fidelity (iterations)')
			fig.colorbar(im_show, ax=ax_right, fraction=0.046, pad=0.04, label='Fidelity')
			for ii, re in enumerate(re_vals):
				for jj, imv in enumerate(im_vals):
					ax_right.text(re, imv, f"{int(coh_iters[ii,jj])}", ha='center', va='center', color='white' if coh_fidelity[ii,jj] < 0.5 else 'black', fontsize=8)
			fig.suptitle(f'PG MLE Batch Summary  samples/state={num_samples}  init={init_method}', fontsize=14)
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
		print("PG batch test complete.")
	return results


if __name__ == "__main__":
	import sys
	if len(sys.argv) == 1:
		run_default_tests(make_panel_figs=True, verbose=True)
	else:
		main_cli()

