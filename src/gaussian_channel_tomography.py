import os
import numpy as np
import pandas as pd
from numpy.polynomial.laguerre import laggauss
from scipy.special import eval_laguerre, factorial, gammaln
from rich.console import Console
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  

console = Console()

def coherent_state_ground_truth_rho(alpha: complex, N: int) -> np.ndarray:
    """Full |α⟩⟨α| truncated at Fock N."""
    n = np.arange(N + 1)
    coeff = np.exp(-abs(alpha)**2 / 2) * alpha**n / np.sqrt(factorial(n))
    return np.outer(np.conjugate(coeff), coeff)          # (N+1, N+1)

def load_qkd_data(file_path: str) -> dict:
    """
    Loads an SNU-style QKD export and returns a dict with:
      V_mod   -> float
      xi      -> float   (xi_calibration)
      T       -> float   (T_calibration)
      alice_x -> numpy array of Alice_X_SNU_Quant
      alice_p -> numpy array of Alice_P_SNU_Quant
      bob_x   -> numpy array of Bob_X_SNU
      bob_p   -> numpy array of Bob_P_SNU
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    if not os.path.exists(f"data/{base_name}.npz"):
        console.log(f"Loading {file_path}...")

        metadata = {}
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                content = line[1:].strip()
                if ':' not in content:
                    continue
                key, val = content.split(':', 1)
                try:
                    metadata[key.strip()] = float(val)
                except ValueError:
                    pass

        df = pd.read_csv(file_path, comment='#')

        # Unpack dictionary to qkd_data
        qkd_data = {}
        qkd_data['V_mod'] = metadata.get('V_mod')
        qkd_data['xi'] = metadata.get('xi_calibration')
        qkd_data['T'] = metadata.get('T_calibration')
        qkd_data['alice_x'] = df['Alice_X_SNU_Quant'].to_numpy()
        qkd_data['alice_p'] = df['Alice_P_SNU_Quant'].to_numpy()
        qkd_data['bob_x'] = df['Bob_X_SNU'].to_numpy()
        qkd_data['bob_p'] = df['Bob_P_SNU'].to_numpy()

        # Save the data to a .npz file
        np.savez(f"data/{base_name}.npz",
            V_mod=qkd_data['V_mod'],
            xi=qkd_data['xi'],
            T=qkd_data['T'],
            alice_x=qkd_data['alice_x'],
            alice_p=qkd_data['alice_p'],
            bob_x=qkd_data['bob_x'],
            bob_p=qkd_data['bob_p']
        )

        # Return the data as a dictionary
        return qkd_data
    
    # If the .npz file already exists, load it (much quicker)
    else:
        console.log(f"Loading precomputed data from {base_name}.npz...")
        return np.load(f"data/{base_name}.npz")
        
def precomputation(R: int, P: int, N: int, alphas: np.ndarray, d: int) -> dict:
    """
    Build and return **all** static arrays required by the reconstruction
    pipeline.

    Parameters
    ----------
    R, P : int
        Radial and angular grid sizes.
    N    : int
        Fock-cutoff (highest photon number).
    alphas : (A,) complex ndarray
        Probe-state amplitudes alpha_i
    d    : int
        Maximum total degree of the polynomial fit.

    Returns
    -------
    dict
        A dictionary whose keys match the bullet-points of the
        pre-computation section.  Shapes are noted in comments.
    """

    # ------------------------------------------------------------------
    # 1.  Gauss–Laguerre mesh (u, φ) and weights
    # ------------------------------------------------------------------
    u_mesh, w_gl = laggauss(R)          # both length R, float64
    u_mesh = u_mesh.astype(np.float64, copy=False)
    w_gl   = w_gl.astype(np.float64, copy=False)

    phi = np.linspace(0.0, 2.0*np.pi, P, endpoint=False, dtype=np.float64)  # (P,)

    # ------------------------------------------------------------------
    # 2.  Hermite table  H_{k,m}  with 0 ≤ k ≤ 2N
    # ------------------------------------------------------------------
    k_vals = np.arange(0, 2*N + 1, dtype=int)           # (2N+1,)
    # H = eval_hermite(k_vals[:, None], u_mesh[None, :] / 2.0)  # (2N+1, R)
    H = eval_laguerre(k_vals[:, None], u_mesh[None, :])

    # ------------------------------------------------------------------
    # 3.  FFT basis   exp(-i k φ_n)
    # ------------------------------------------------------------------
    k_max = N + 1
    k_idx = np.arange(0, k_max + 1, dtype=int)[:, None]        # (k_max+1, 1)
    fft_basis = np.exp(-1j * k_idx * phi[None, :])  # (k_max+1, P)

    # ------------------------------------------------------------------
    # 4.  Polynomial design matrix  Φ_{i,p,q}
    # ------------------------------------------------------------------
    A = alphas.shape[0]
    p_idx, q_idx = np.meshgrid(np.arange(d + 1), np.arange(d + 1), indexing='ij')
    mono_exponents = (p_idx.flatten(), q_idx.flatten())         # two 1-D arrays
    M = (d + 1) ** 2
    Φ = np.empty((A, M), dtype = np.complex128)                           # (A, M)

    for s, (p, q) in enumerate(zip(*mono_exponents)):
        Φ[:, s] = (alphas ** p) * (alphas.conj() ** q)

    # ------------------------------------------------------------------
    # 5.  Factorials and helpers  (0…N)
    # ------------------------------------------------------------------
    fact        = factorial(np.arange(N + 1), exact=False)      # float64
    sqrt_fact   = np.sqrt(fact)
    inv_fact    = 1.0 / fact

    # ------------------------------------------------------------------
    # 6.  Prefactor for ρ_{mn}
    # ------------------------------------------------------------------
    m_idx, n_idx = np.meshgrid(np.arange(N + 1),
                                np.arange(N + 1),
                                indexing='ij')
    epsilon = ((-1) ** (m_idx + n_idx)).astype(float)       # parity sign

    # compute log-prefactor
    log_num  = 0.5*(gammaln(m_idx+1) + gammaln(n_idx+1))
    log_den  = np.log(2.0) + gammaln(m_idx + n_idx + 1)
    log_pref = log_num - log_den

    # safe prefactor array
    prefactor = epsilon * np.exp(log_pref)    # shape (N+1,N+1)

    # ------------------------------------------------------------------
    # 7.  Package and ship
    # ------------------------------------------------------------------
    return {
        # mesh
        "u": u_mesh,                     # (R,)
        "phi": phi,                      # (P,)
        "w_GL": w_gl,                    # (R,)

        # Hermite and FFT
        "H_hermite": H.astype(np.complex128),    # (2N+1, R)
        "fft_basis": fft_basis,          # (k_max+1, P)

        # polynomial design
        "Phi": Φ,                        # (A, (d+1)^2)
        "p_idx": mono_exponents[0],      # (M,)  (optionally useful later)
        "q_idx": mono_exponents[1],      # (M,)

        # factorial helpers
        "fact": fact,                    # (N+1,)
        "sqrt_fact": sqrt_fact,          # (N+1,)
        "inv_fact": inv_fact,            # (N+1,)

        # ρ-prefactor
        "pref_rho": prefactor.astype(np.complex128)  # (N+1, N+1)
    }

def extract_valid_probe_states(alice_x: np.ndarray, alice_p: np.ndarray, bob_x: np.ndarray, bob_p: np.ndarray, probe_min: int) -> pd.DataFrame:
    """
    Returns a pandas DataFrame with only those rows whose (alice_x, alice_p)
    probe-state appears at least probe_min times.
    """
    # sanity check
    if not (alice_x.shape == alice_p.shape == bob_x.shape == bob_p.shape):
        raise ValueError("Inputs must all be the same length.")
    total = alice_x.size

    # build the DataFrame
    df = pd.DataFrame({
        "alice_x": alice_x,
        "alice_p": alice_p,
        "bob_x":   bob_x,
        "bob_p":   bob_p,
    })

    # how many unique probe-states?
    n_states = df.groupby(["alice_x","alice_p"]).ngroups
    console.log(f"Found {n_states} distinct probe-states in {total} samples.")

    console.log(f"Filtering out probe-states with fewer than {probe_min} samples...")

    # filter out small groups
    df_filt = df.groupby(["alice_x","alice_p"]) \
                .filter(lambda g: len(g) >= probe_min)

    kept = df_filt.groupby(["alice_x","alice_p"]).ngroups
    console.log(f"Keeping {kept} probe-states with ≥{probe_min} reps.")
    console.log(f"After filtering: {len(df_filt)} total samples remain.")

    return df_filt

def _histogram_Q(
    bob_x: np.ndarray,
    bob_p: np.ndarray,
    u_mesh: np.ndarray,          # (R,)
    phi_mesh: np.ndarray         # (P,)
) -> np.ndarray:
    """
    Build the discretised Q-function  Q[r_m, phi_n]  on the (u,phi) quadrature
    mesh by simple binning.  The output has shape (R, P) and integrates to 1.
    """
    # polar coordinates for every shot
    r   = np.sqrt(bob_x**2 + bob_p**2)
    phi = np.mod(np.arctan2(bob_p, bob_x), 2*np.pi)

    # the theory uses  u = 2 r^2   (Laguerre quadrature variable)
    u = 2.0 * r**2

    # turn the mesh points into bin edges once
    u_edges   = np.concatenate([[0.0], 0.5*(u_mesh[1:] + u_mesh[:-1]), [np.inf]])
    phi_edges = np.concatenate([[0.0], phi_mesh + (phi_mesh[1]-phi_mesh[0]), [2*np.pi]])

    hist, _, _ = np.histogram2d(u, phi, bins=[u_edges, phi_edges])   # (R, P)
    return hist / hist.sum()                                            # normalise

def _angular_fft(Q: np.ndarray, k_max: int) -> np.ndarray:
    """
    Return  Q_k[u_m, k]  for  k = 0..k_max  (shape: (k_max+1, R)).
    """
    # FFT over the *second* axis (phi index).  Divide by P for an orthonormal DFT.
    Qk_full = np.fft.fft(Q, axis=1) / Q.shape[1]                  # (R, P)
    return np.ascontiguousarray(Qk_full[:, :k_max+1].T)           # (k_max+1, R)

def _radial_contraction(Qk: np.ndarray,
                        H_hermite: np.ndarray,
                        w_GL: np.ndarray,
                        pref_rho: np.ndarray,
                        N: int,
                        u_mesh: np.ndarray) -> np.ndarray:
    """
    Compute rho[m,n] for a single probe from its Q_k radial slices.

    Parameters
    ----------
    Qk         : (N+1, R)  – FFT slices for k = 0…N  (positive k only)
    H_hermite  : (2N+1, R) – H_{ℓ}(u_m/2) for ℓ = 0…2N
    w_GL       : (R,)      – Gauss–Laguerre weights
    pref_rho   : (N+1,N+1) – ε_{mn} √m!√n! / (2 (m+n)!)
    N          : int       – Fock cutoff
    u_mesh     : (R,)      – Gauss–Laguerre mesh points
    """
    k_max = N                               # range we really need
    # ------------------------------------------------------------
    # Step 1 : radial integrals  I[ℓ]  for ℓ = 0…2N,
    #          where ℓ = m+n and k = n-m = ℓ - 2m
    # ------------------------------------------------------------
    I = np.zeros((2*N+1,), dtype=np.complex128)

    for k in range(-k_max, k_max+1):        # k = -N…+N
        # pick the stored slice: Qk_pos = Qk[|k|]
        Qk_pos = Qk[abs(k)]
        if k < 0:                           # negative k comes from complex-conjugate
            Q_slice = np.conj(Qk_pos)
        else:
            Q_slice = Qk_pos

        # matching Hermite row index is   ℓ = k + N + m   but we don't yet know m.
        # The easy trick: H_{ℓ}(u) only depends on ℓ, so integrate for *all* ℓ at once
        # by dotting the whole (2N+1,R) table with w_GL weighed slice.
        #
        # But because each k uses a *single* ℓ = n-m shift, the cheaper way is:
        ell = k + N              # shift range -N…N → 0…2N
        if 0 <= ell <= 2*N:
            # compensate the weight mismatch: multiply by e^{+u/2}
            I[ell] += np.dot(w_GL, np.exp(u_mesh / 2.0) * Q_slice * H_hermite[ell])

    # ------------------------------------------------------------
    # Step 2 : map (m,n) to k = n-m and ℓ = m+n and multiply prefactor
    # ------------------------------------------------------------
    rho = np.empty((N+1, N+1), dtype=np.complex128)
    for m in range(N+1):
        for n in range(N+1):
            k = n - m                          # -N … +N
            ell = m + n                        # 0 … 2N
            rho[m, n] = pref_rho[m, n] * I[ell]

    return rho

def process_single_probe(idx: int,
                         alpha: complex,
                         df: pd.DataFrame,
                         bob_cols=("bob_x", "bob_p"),
                         *,
                         u_mesh, phi_mesh, w_GL,
                         H_hermite, pref_rho, N) -> tuple[int, np.ndarray]:
    """
    Worker function — runs in a separate process when we parallelise.

    Returns
    -------
    idx          : the probe index (so the caller can re-insert in order)
    rho_i        : (N+1, N+1) complex ndarray for this probe
    """
    # 1) slice the dataframe very cheaply
    mask = (df["alice_x"].values == alpha.real) & (df["alice_p"].values == alpha.imag)
    grp  = df.loc[mask, list(bob_cols)]

    # 2) build Q(u_m, phi_n)
    Q = _histogram_Q(grp["bob_x"].to_numpy(),
                     grp["bob_p"].to_numpy(),
                     u_mesh,
                     phi_mesh)                                    # (R, P)

    # 3) angular FFT (only need k = 0 … N+1)
    Qk = 2*np.pi * _angular_fft(Q, k_max=N+1)                               # (k_max+1, R)

    # 4) radial contraction → rho_{mn}
    rho_i = _radial_contraction(Qk, H_hermite, w_GL, pref_rho, N, u_mesh)

    # 5) return the index and the computed density matrix. idx is needed for multiprocessing.
    return idx, rho_i

def main():
    console.log("### Starting...")
    # Parameters
    plot = True # Set to False to disable plotting
    R = 80   # Radial grid size
    P = 80   # Angular grid size
    N = 35  # Fock cutoff
    d = 5     # Maximum polynomial degree
    probe_state_instance_minimum = 1000 # The minimum number of data points for each probe state (x_in, p_in)

    console.log("Loading QKD data...")
    qkd_data = load_qkd_data("data/10km_real_snu_data.csv")

    # Unpack the data
    V_mod = qkd_data['V_mod']
    xi = qkd_data['xi']
    T = qkd_data['T']
    alice_x_raw = qkd_data['alice_x']
    alice_p_raw = qkd_data['alice_p']
    bob_x_raw = qkd_data['bob_x']
    bob_p_raw = qkd_data['bob_p']

    console.log("### Extracting valid probe states...")
    
    probe_states_file = "data/probe_states.npz"
    if os.path.exists(probe_states_file):
        console.log(f"### Loading precomputed probe_states from {probe_states_file}...")

        probe_states = np.load(probe_states_file)["probe_states"]
        probe_counts = np.load(probe_states_file)["probe_counts"]
        probe_counts = pd.DataFrame(probe_counts, columns=["alice_x", "alice_p", "n_shots"])

        console.log("### probe_states loaded successfully.")

    else:
        console.log("### probe_states.npz not found. Extracting and saving probe_states...")

        # Returns a pandas DataFrame with ["alice_x", "alice_p", "bob_x", "bob_p"].
        df = extract_valid_probe_states(alice_x_raw, alice_p_raw, bob_x_raw, bob_p_raw, probe_state_instance_minimum)
        probe_states = (df["alice_x"] + 1j * df["alice_p"]).drop_duplicates().to_numpy()
        
        # build a DataFrame of counts
        probe_counts = (
            df.groupby(['alice_x', 'alice_p'])
                .size()
                .reset_index(name='n_shots')
        )

        # save BOTH arrays
        np.savez(probe_states_file,
                    probe_states=probe_states,
                    probe_counts=probe_counts.values)
        console.log("### probe_states saved successfully.")

    if plot:
        console.log("### Plotting probe states...")

        sc = plt.scatter(
            probe_counts['alice_x'],
            probe_counts['alice_p'],
            c = probe_counts['n_shots'],
            cmap = 'viridis',           # any perceptually uniform map works
            norm = LogNorm(),           # try removing this if the colour contrast is too flat
            s = 20,                     # marker size
            edgecolors='none'           # cleaner look
        )

        plt.gca().set_aspect('equal')   # keep the phase-space axes square
        plt.xlabel(r'Alice $x$  (SNU)')
        plt.ylabel(r'Alice $p$  (SNU)')
        plt.title('Probe-state distribution\n(colour = # shots)')
        plt.colorbar(sc, label='samples per probe state')

        plt.tight_layout()
        plt.show()

    console.log("### Precomputing static arrays...")
    precomputation_results = precomputation(R, P, N, probe_states, d)

    # Unpack the precomputation results
    u = precomputation_results["u"]
    phi = precomputation_results["phi"]
    w_GL = precomputation_results["w_GL"]
    H_hermite = precomputation_results["H_hermite"]
    fft_basis = precomputation_results["fft_basis"]
    Phi = precomputation_results["Phi"] # Note that this is the polynomial design matrix, not the angular grid. (Phi =\= phi)
    p_idx = precomputation_results["p_idx"]
    q_idx = precomputation_results["q_idx"]
    fact = precomputation_results["fact"]
    sqrt_fact = precomputation_results["sqrt_fact"]
    inv_fact = precomputation_results["inv_fact"]
    pref_rho = precomputation_results["pref_rho"]

    console.log("### Preprocessing done. Ready to compute density matrices...")

    ######################### PREPROCESSING DONE #########################

    # Extract shape of probe_states
    A = len(probe_states)

    # Check if rho_all.npz already exists
    rho_file_path = "data/rho_all.npz"
    if os.path.exists(rho_file_path):
        console.log(f"### Loading precomputed rho_all from {rho_file_path}...")
        rho_all = np.load(rho_file_path)["rho_all"]
        console.log("### rho_all loaded successfully.")
    else:
        console.log("### rho_all.npz not found. Proceeding with computation...")

        console.log("### Processing each probe state (alpha_i)...")
        # Extract shape of probe_states

        rho_all  = np.empty((A, N+1, N+1), dtype=np.complex128)

        # Put everything that is read-only into a single kwargs dict so that we
        # forward just one argument instead of many.
        common_kw = dict(
            u_mesh    = u,
            phi_mesh  = phi,
            w_GL      = w_GL,
            H_hermite = H_hermite,
            pref_rho  = pref_rho,
            N         = N
        )

        # TODO: Multiprocessing of this loop
        for i, alpha in enumerate(probe_states):
            _, rho_i = process_single_probe(i, alpha, df, **common_kw)
            rho_all[i] = rho_i

            if i % 10 == 0:
                console.log(f"  → finished {i+1}/{A}")

            # Manually change i range to see different probe states
            if plot and (i < 5):
                mask = (
                    (df["alice_x"] == alpha.real) &
                    (df["alice_p"] == alpha.imag)
                )
                x_vals = df.loc[mask, "bob_x"]
                p_vals = df.loc[mask, "bob_p"]

                plt.figure(figsize=(6,6))
                plt.scatter(x_vals, p_vals, s=5, alpha=0.7)
                plt.plot(alpha.real, alpha.imag, marker='X', color='red', markersize=12, label = "input probe state")
                # Add another marker for the expected output
                plt.plot(T * alpha.real, T * alpha.imag, marker='X', color='blue', markersize=12, label = "Expected output")
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                plt.xlabel('Bob $x$ (SNU)')
                plt.ylabel('Bob $p$ (SNU)')
                plt.title(f'Probe #{i} — $\\alpha={alpha.real:.2f}{alpha.imag:+.2f}i$')
                plt.gca().set_aspect('equal')
                plt.grid(True, linestyle='--', alpha=0.7)  # Add a grid
                plt.tight_layout()
                plt.legend()
                plt.show()

        console.log("### Finished processing all probe states.")

        # Save rho_all to an npz file
        console.log("### Saving rho_all to file...")
        np.savez("data/rho_all.npz", rho_all=rho_all)
        console.log("### rho_all saved successfully.")

    if plot:
        console.log("### Visualising rho_mn for the first 5 probe states...")
        for i in range(5):
            rho_i = rho_all[i]

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Left plot: Reconstructed rho_i
            im1 = axes[0].imshow(np.abs(rho_i), origin='lower', cmap='viridis', interpolation='none')
            trace_reconstructed = np.trace(rho_i)
            axes[0].set_title(f'Reconstructed $\\rho_{{mn}}$ — Probe #{i}\n$\\alpha_in={probe_states[i].real:.2f}{probe_states[i].imag:+.2f}i$, Trace={trace_reconstructed:.4f}')
            axes[0].set_xlabel(r'$n$')
            axes[0].set_ylabel(r'$m$')
            fig.colorbar(im1, ax=axes[0], label=r'$|\rho_{mn}|$')

            # Right plot: Expected output rho
            expected_alpha = probe_states[i] * T
            rho_expected = coherent_state_ground_truth_rho(expected_alpha, N)
            im3 = axes[1].imshow(np.abs(rho_expected), origin='lower', cmap='viridis', interpolation='none')
            trace_expected = np.trace(rho_expected)
            axes[1].set_title(f'Expected $\\rho_{{mn}}$ — Probe #{i}\n$\\alpha={expected_alpha.real:.2f}{expected_alpha.imag:+.2f}i$, Trace={trace_expected:.4f}')
            axes[1].set_xlabel(r'$n$')
            axes[1].set_ylabel(r'$m$')
            fig.colorbar(im3, ax=axes[1], label=r'$|\rho_{mn}|$')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()