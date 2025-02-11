import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from data_dir import data_dir

def load_losses_and_vmods():
    """
        From data_dir, load the losses and optimised V_mods for the given losses.
        The losses are in dB, and the optimal V_mods are in shot noise units.


        Returns:
            np.ndarray, np.ndarray: The losses and optimal V_mods of the channel.
    """
    data = pd.read_csv(data_dir + "m1_asymptotic_GBSR_vs_SR.csv", sep = " ")
    data.describe()

    losses = np.array(data["loss"])
    optimal_v_mods = np.array(data["SR_v_mod"])

    return losses, optimal_v_mods

def generate_pure_loss_x_and_p_values(loss_db, optimal_v_mod, num_samples):
    """
        # TODO
    """
    # Convert loss to linear scale
    loss = 10 ** (-loss_db / 10)

    # Convert to transmittance
    transmittance = 1.0 - loss

    # Generate the covariance matrix
    a = optimal_v_mod + 1.0
    b = (transmittance * optimal_v_mod) + 1.0 # + xi (\xi = 0.0 as it's pure loss)
    c = np.sqrt(transmittance * (a**2 - 1.0))

    covariance_matrix = np.array([[a, c], [c, b]])

    # Define random variable
    rv = multivariate_normal([0.0, 0.0], covariance_matrix)
    
    # Generate samples
    samples = rv.rvs(num_samples)

    x_synthetic = samples[:, 0]
    p_synthetic = samples[:, 1]

    return x_synthetic, p_synthetic

    