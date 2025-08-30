import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from numpy.polynomial.laguerre import laggauss
from scipy.special import eval_hermite, factorial

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
  
def load_probe_state_data(
        qkd_file="data/10km_real_snu_data.csv",
        probe_states_file="data/probe_states.npz",
        tol=1e-9
):
    """
    Return Bob's (x, p) for the *first* probe state stored in probe_states.npz.
    Also plots them if `plot=True`.
    """
    # ------------------------------------------------------------------
    # 1. read the probe list, pick α₀
    # ------------------------------------------------------------------
    ps_data = np.load(probe_states_file)
    alpha_0  = ps_data["probe_states"][2]       # complex128
    console.log(f"Using probe state α₀ = {alpha_0.real:+.3f}{alpha_0.imag:+.3f}i")

    # ------------------------------------------------------------------
    # 2. load the full shot list (fast thanks to the .npz cache)
    # ------------------------------------------------------------------
    d = load_qkd_data(qkd_file)    # dict-like (np.load if cached)

    # ------------------------------------------------------------------
    # 3. build a mask that selects exactly that α  (use tolerance!)
    # ------------------------------------------------------------------
    alice_x = d["alice_x"]
    alice_p = d["alice_p"]

    mask = (np.isclose(alice_x, alpha_0.real, atol=tol) &
            np.isclose(alice_p, alpha_0.imag, atol=tol))

    x_vals = d["bob_x"][mask]
    p_vals = d["bob_p"][mask]
    nshots = x_vals.size
    console.log(f"Extracted {nshots} shots for α₀")

    return alpha_0, x_vals, p_vals, d['T']

if __name__ == "__main__":
    plot = True
    