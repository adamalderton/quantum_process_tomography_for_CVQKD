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

def process_single_probe():
    pass

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

    # TODO: Neatly perform pre-processing here

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

        # TODO: Multiprocessing of this loop
        for i, alpha in enumerate(probe_states):
            _, rho_i = process_single_probe(i, alpha, df)
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