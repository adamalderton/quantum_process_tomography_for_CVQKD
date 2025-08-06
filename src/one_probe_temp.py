import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from numpy.polynomial.laguerre import laggauss
from scipy.special import eval_hermite, factorial

console = Console()

RADIAL_GRID_SIZE = 30
RADIAL_MAX_SNU = 7
ANGULAR_GRID_SIZE = 80
FOCK_CUTOFF = 40

# Pre-computed factorial lookup
fl = factorial(np.arange(2*FOCK_CUTOFF + 1))

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
    alpha_0, x, p, T = load_probe_state_data()

    # Generate discretised evaluation points
    u_vals = np.linspace(0, 2 * RADIAL_MAX_SNU**2, RADIAL_GRID_SIZE) # Upper limit is 2 * RADIAL_MAX_SNU^2 so as counter the variable change
    phi_vals = np.linspace(0.0, 2.0 * np.pi, ANGULAR_GRID_SIZE, endpoint = False) # Don't include 2\pi (therefore, endpoint = False)

    # Calculate the edges for the purposes of the histogram
    u_edges = np.concatenate([[0.0], 0.5 * (u_vals[1:] + u_vals[:-1]), [np.inf]])
    phi_edges = np.concatenate([[0.0], phi_vals + (phi_vals[1] - phi_vals[0]), [2*np.pi]])

    if plot:
        # Replace the last radial edge (∞) by something finite so that pcolormesh works (for plotting purposes)
        u_edges_plot = u_edges.copy()
        if np.isinf(u_edges_plot[-1]):
            u_edges_plot[-1] = u_edges_plot[-2] + (u_edges_plot[-2] - u_edges_plot[-3])

        # Convert u → r (u = 2 r² ⇒ r = √(u / 2))
        r_edges = np.sqrt(u_edges_plot / 2.0)
        phi_edges_plot = phi_edges

        # Visualize x and p as a 2D histogram
        plt.figure(figsize=(6, 6))
        plt.hist2d(x, p, bins=50, cmap='viridis', density=True, range=[[-7, 7], [-7, 7]])
        plt.colorbar(label='Density')
        plt.xlabel('Bob $x$ (SNU)')
        plt.ylabel('Bob $p$ (SNU)')
        plt.title('2D Histogram of Bob\'s $x$ and $p$')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    # Building Husimi Q-Function histogram
    husimi_Q_histogram = Q_histogram(x, p, u_edges, phi_edges)

    if plot:
        # Build a grid of cell-edges
        R_edges, PHI_edges = np.meshgrid(r_edges, phi_edges_plot, indexing="ij")

        # Convert those edges to Cartesian coordinates
        X_edges = R_edges * np.cos(PHI_edges)
        Y_edges = R_edges * np.sin(PHI_edges)

        # Plotting
        fig = plt.figure(figsize=(12, 5))

        # Polar view
        ax_pol = fig.add_subplot(1, 2, 1, projection="polar")
        pcm_pol = ax_pol.pcolormesh(PHI_edges, r_edges, husimi_Q_histogram, shading="auto", cmap="viridis")
        ax_pol.set_title("Husimi $Q$‑function (polar)")
        fig.colorbar(pcm_pol, ax=ax_pol, pad=0.1, label="Probability")

        # Cartesian view
        ax_cart = fig.add_subplot(1, 2, 2)
        pcm_cart = ax_cart.pcolormesh(X_edges, Y_edges, husimi_Q_histogram, shading="auto", cmap="viridis")
        ax_cart.set_aspect("equal")
        ax_cart.set_xlim(-RADIAL_MAX_SNU, RADIAL_MAX_SNU)
        ax_cart.set_ylim(-RADIAL_MAX_SNU, RADIAL_MAX_SNU)
        ax_cart.set_xlabel(r"$x\;(\mathrm{SNU})$")
        ax_cart.set_ylabel(r"$p\;(\mathrm{SNU})$")
        ax_cart.set_title("Husimi $Q$‑function (Cartesian)")
        fig.colorbar(pcm_cart, ax=ax_cart, pad=0.1, label="Probability")

        plt.tight_layout()
        plt.show()

    # Compute the angular FFT. Need to divide by ANGULAR_GRID_SIZE due to numpy FFT convention (and we need to preserve the norm)
    husimi_Q_angular_FFT = np.fft.fft(husimi_Q_histogram, axis = 1) / ANGULAR_GRID_SIZE

    if plot:
        # Visualize the angular FFT result
        plt.figure(figsize=(10, 6))
        plt.imshow(
            np.abs(husimi_Q_angular_FFT),
            extent=[0, 2 * np.pi, 0, RADIAL_GRID_SIZE],
            aspect='auto',
            cmap='viridis',
            origin='lower'
        )
        plt.colorbar(label='Magnitude of FFT')
        plt.xlabel('Angular Frequency (rad)')
        plt.ylabel('Radial Index')
        plt.title('Magnitude of Angular FFT of Husimi Q-Function')
        plt.tight_layout()
        plt.show()

    # With the Q function FFT found, we now need to integrate the FFT for each m, n \leq FOCK_CUTOFF.
    # For this, it is useful to pre-compute the Hermite polynomials such that the N'th hermite polynomial evaluated of u_vals/2.0 is found via hermite_lookup[N]
    hermite_lookup = eval_hermite(np.arange(2*FOCK_CUTOFF + 1).reshape((2*FOCK_CUTOFF + 1, 1)), u_vals / 2.0)

    # Visualize the Hermite polynomial lookup table
    plt.figure(figsize=(10, 6))
    for n in range(0, 2 * FOCK_CUTOFF + 1, max(1, (2 * FOCK_CUTOFF) // 10)):
        plt.plot(u_vals, hermite_lookup[n], label=f"H_{n}(u/2)")
    plt.xlabel("u")
    plt.ylabel("Hermite Polynomial Value")
    plt.title("Hermite Polynomials $H_n(u/2)$ for $n=0$ to $2N$")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # rho_mn = np.empty((FOCK_CUTOFF + 1, FOCK_CUTOFF + 1), dtype = np.complex128)
    # for m in range(FOCK_CUTOFF + 1):
    #     for n in range(0, m + 1):
    #         integral = (RADIAL_MAX_SNU / RADIAL_GRID_SIZE) * np.sum(husimi_Q_angular_FFT[:, n - m] * hermite_lookup[n + m])
    #         factorial_prefactor = (np.sqrt(fl[m] * fl[n])) / (2 * fl[m + n])
    #         rho_mn[m][n] = _epsilon_prefactor(m, n) * factorial_prefactor * integral
            
    #         if (m != n):
    #             rho_mn[n][m] = np.conjugate(rho_mn[m][n])

    # # Plotting result. LEFT: Reconstructed result. RIGHT: Expected result
    # fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    # expected_alpha = T * alpha_0
    # rho_expected = coherent_state_ground_truth_rho(expected_alpha, FOCK_CUTOFF)

    # im1 = axes[0].imshow(np.abs(rho_mn), origin = "lower", cmap = "viridis", interpolation = "none")
    # axes[0].set_title(f"Reconstructed $\\rho_{{mn}}$. $\\alpha_{{in}} = {alpha_0.real:.2f} + {alpha_0.imag:.2f}i$, trace = {np.trace(rho_mn):.2f}")
    # axes[0].set_xlabel(f"$n$")
    # axes[0].set_ylabel(f"$m$")
    # fig.colorbar(im1, ax = axes[0], label = f"|\\rho_{{mn}}|")

    # im2 = axes[1].imshow(np.abs(rho_expected), origin = "lower", cmap = "viridis", interpolation = "none")
    # axes[1].set_title(f"Expected $\\rho_{{mn}}$. $\\alpha = {expected_alpha.real:.2f} + {expected_alpha.imag:.2f}i$, trace = {np.trace(rho_expected):.2f}")
    # axes[1].set_xlabel(f"$n$")
    # axes[1].set_ylabel(f"$m$")
    # fig.colorbar(im2, ax = axes[1], label = f"$|\\rho_{{mn}}|$")

    # plt.tight_layout()
    # plt.show()

    # print(rho_mn[0][0])
    # print(rho_mn[0][1])
    # print(rho_mn[1][0])
    
    # print(rho_mn[0][-1])

