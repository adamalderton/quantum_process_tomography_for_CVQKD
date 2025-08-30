import numpy as np
from typing import Dict, Tuple, List, Optional

################ CHANNEL RECONSTRUCTION MACHINERY ################

def factorial_array():
    # TODO: From state estimation code.
    pass

def build_grid():
    # TODO: Adapt from state estimation code.
    pass


def compute_channel_probabilites(
    J: np.ndarray,
    C_inputs: np.ndarray,
    C_outputs: np.ndarray,
    bin_area: float,
    eps: float = 1e-30
) -> np.ndarray:
    """
    Forward model for a *channel*: p_{i,j}(J) = Tr[ J * (E_j ⊗ |α_i><α_i|^T) ],
    where E_j = (bin_area/π) |α_j><α_j| (heterodyne POVM per output bin).
    
    Reminder: S is the number of input states, M = G^2 is the number of output bins.
    
    Args:
        J: Choi matrix of the channel, shape (d*d, d*d)
        C_inputs: Coherent state overlaps for input states, shape (d, S). Columns are a^(i).
        C_outputs: Coherent state overlaps for output states, shape (d, M). Columns are b^(j).
        bin_area: Area of each output bin in phase space (dual-homodyne grid).
        eps: Small constant to avoid numerical issues.

    Returns:
        p: Probability matrix, shape (S, M), where p[i][j] = p_{i,j}(J).
    """
    # TODO: implement efficient batched contractions using rank-1 factors from C_in, C_out.
    # Shape notes:
    #   a_i = C_in[:, i], b_j = C_out[:, j]
    #   F_ij = (ΔA/π) * (b_j b_j†) ⊗ (a_i a_i†)^T
    #   p_ij = Tr[J F_ij]
    pass

def neg_log_likelihood_poisson_multiinput(
    counts_ij: np.ndarray,
    p_ij: np.ndarray
) -> float:
    """
    Poisson factorised NLL over all inputs i and bins j.

    Args:
        counts_ij: Measured counts, shape (S, M).
        p_ij: Probability matrix from the channel model, shape (S, M). \sum_j p_ij = 1 for each i.

    Returns:
        nll: Negative log-likelihood value.
    """
    # TODO: Vectorised NLL.
    pass

def project_CPTP(
    J: np.ndarray,
    add_eps: float = 1e-10
) -> np.ndarray:
    """
    Project a Choi matrix J onto the set of CPTP maps.

    Reminder: CPTP = Completely Positive and Trace Preserving.
    This is the same as Positive Semi-Definite (PSD) and Trace-Preserving (TP).

    Args:
        J: Hermitian Choi matrix to be projected, shape (d*d, d*d).
        add_eps: Small constant to add to eigenvalues to stabilise inverses.
    
    Returns:
        J_proj: Projected Choi matrix, shape (d*d, d*d).
    """
    # TODO: Hermitise → eigendecompose → clamp → compute partial trace over output → congruence scaling.
    pass

################ OPTIMISATION MACHINERY ################

def mle_projected_gradient_descent(
    counts_ij: np.ndarray,
    C_inputs: np.ndarray,
    C_outputs: np.ndarray,
    bin_area: float,
    J_initialisation_style: str = 'maximally_mixed',
    transmissivity: Optional[float] = None,
    excess_noise: Optional[float] = None,
    max_iters: int = 100,
    step_init: float = 1e-3,
    min_step: float = 1e-8,
    armijo_c: float = 1e-4,
    armijo_tau: float = 0.5,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Perform MLE of a quantum channel using projected gradient descent.

    Projected gradient descent loop:
    - Initialise J (CPTP).
    - Until convergence or max_iters:
        - Compute p_ij(J) using the channel forward model.
        - Compute NLL and gradient.
        - Backtracking line search to find step size satisfying Armijo condition.
        - Gradient step and project back to CPTP.

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
        armijo_c: Armijo condition constant.
        armijo_tau: Step size reduction factor for backtracking line search.
        verbose: Whether to print progress.
        seed: Random seed for reproducibility.

    Returns:
        results: Dictionary containing:
            - 'J': Estimated Choi matrix of the channel, shape (d*d, d*d).
            - 'history': NLL values, step size, back tracks, etc.
    """
    # TODO: Initialise J, main loop with gradient computation, line search, projection, logging.
    pass

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
    quadrature variance is 1/2), divide all SNU variances by 2 and use α = (y1 + i y2) / 2; under that
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
    # TODO: For each input state, sample shots from the corresponding Gaussian distribution.
    pass

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
    J_initialisation_style: str = 'maximally_mixed',
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
        J_initialisation_style: Style for initial Choi matrix. Options:
            - 'identity': J = |Φ+><Φ+|, where |Φ+> = (1/√d) ∑_k |k,k>.
            - 'maximally_mixed': J = I / d^2
            - 'known_Gaussian': Gaussian CPTP channel with transmissivity and excess noise (see below).
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
    # TODO: Define probe state set, synthetic channel (identity for now), generate synthetic data.
    # TODO: Grid + histogram (adapt histogram_on_grid from state estimation). Define grid 'adaptively' by looking at output data variance.
    # TODO: Calculate C_inputs, C_outputs by adapting build_coherent_state_matrix from state estimation, for channel estimation.
    # TODO: Call mle_projected_gradient_descent with prepared data.
    # TODO: Visualisations
    pass

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
    pass

def main():
    # TODO: Parse command line args, call run_default_tests or run_mle_workflow.
    pass

if __name__ == "__main__":
    main()