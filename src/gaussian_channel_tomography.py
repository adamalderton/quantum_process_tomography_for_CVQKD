import os
import numpy as np
import pandas as pd
from numpy.polynomial.laguerre import laggauss
from scipy.special import eval_hermite, factorial, gammaln
from rich.console import Console

console = Console()

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
    H = eval_hermite(k_vals[:, None], u_mesh[None, :] / 2.0)  # (2N+1, R)

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

def main():
    console.log("### Starting...")
    # Parameters
    A = 1000  # Number of probe states
    R = 128   # Radial grid size
    P = 128   # Angular grid size
    N = 100   # Fock cutoff
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
    # Returns a pandas DataFrame with ["alice_x", "alice_p", "bob_x", "bob_p"].
    df = extract_valid_probe_states(alice_x_raw, alice_p_raw, bob_x_raw, bob_p_raw, probe_state_instance_minimum)
    
    # Determine the unique probe states, solely for passing to the function below. They will be iterated over
    # more neatly later.
    probe_states = (df["alice_x"] + 1j * df["alice_p"]).drop_duplicates().to_numpy()

    console.log("### Precomputing static arrays...")
    precomputation_results = precomputation(R, P, N, probe_states, d)

    # Unpack the precomputation results
    u_mesh = precomputation_results["u"]
    phi = precomputation_results["phi"]
    w_GL = precomputation_results["w_GL"]
    H_hermite = precomputation_results["H_hermite"]
    fft_basis = precomputation_results["fft_basis"]
    Phi = precomputation_results["Phi"]
    p_idx = precomputation_results["p_idx"]
    q_idx = precomputation_results["q_idx"]
    fact = precomputation_results["fact"]
    sqrt_fact = precomputation_results["sqrt_fact"]
    inv_fact = precomputation_results["inv_fact"]
    pref_rho = precomputation_results["pref_rho"]

if __name__ == "__main__":
    main()