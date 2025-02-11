import numpy as np
import scipy as sp
from numba import njit
from scipy.special import factorial
import matplotlib.pyplot as plt
import os

class CVQKDChannelTomographySynthetic():
    """
        CVQKD channel tomography class
    """
    def __init__(self, cutoff: int):
        """
        Initializes the Gaussian channel tomography class.

        Parameters:
            cutoff (int): The cutoff parameter for the Fock basis.
        """
        self.cutoff = cutoff
        if cutoff > 255:
            raise ValueError("Cutoff parameter cannot exceed 255 due to memory constraints.")

        self.process_tensor_shape = (cutoff, cutoff, cutoff, cutoff)

        # Generate m, n, j, k meshgrid for the process tensor. Store as jnp.uint8 to save memory. Note this limits cutoff to 255.
        self.m_mesh, self.n_mesh, self.j_mesh, self.k_mesh = np.meshgrid(np.arange(cutoff), np.arange(cutoff), np.arange(cutoff), np.arange(cutoff))
        
    def generate_pure_loss_process_tensor(self, loss: float) -> np.ndarray:
        """
            Generates a process tensor for a pure loss channel.

            \\Epsilon_{jk}^{mn} = \\sqrt{\\frac{m!n!}{j!k!}} \\frac{\\eta^{j+k}(1 - \\eta^2)^{m-j}}{(m-j)!} \\delta_{m-j, n-k}

            Parameters:
                loss (float): The loss parameter bounded 0 and 1, where 0 means no loss and 1 means complete loss. Of course, NOT dB loss.

            Returns:
                np.ndarray: The process tensor representing the pure loss channel.
        """

        # Due to the delta function, the process tensor is sparse. # TODO: Implement sparse version with either scipy.sparse or
        # jax.experimental.sparse. However, manually found process tensor is assumed sparse, so won't save much compared to dense.

        eta = 1.0 - loss # Represents the transmissivity of the channel: \rho = \ket{\eta \alpha} \bra{\eta \alpha}

        def process_tensor_element(m, n, j, k, eta):
            # Check delta condition
            delta_condition = (m - j) == (n - k)
            
            # Check denominator condition. (Physically relevant, can't have negative photons)
            denominator_condition = (m - j) >= 0

            valid_indices = np.where(delta_condition & denominator_condition)

            m_fac = factorial(m[valid_indices])
            n_fac = factorial(n[valid_indices])
            j_fac = factorial(j[valid_indices])
            k_fac = factorial(k[valid_indices])
            m_minus_j_fac = factorial(m[valid_indices] - j[valid_indices])

            result = np.zeros_like(m, dtype=float)
            result[valid_indices] = (
                np.sqrt(m_fac * n_fac / (j_fac * k_fac))
                * (eta ** (j[valid_indices] + k[valid_indices]))
                * ((1 - eta ** 2) ** (m[valid_indices] - j[valid_indices]))
                / m_minus_j_fac
            )
            return result
    
        m, n, j, k = np.meshgrid(
            np.arange(self.cutoff),
            np.arange(self.cutoff),
            np.arange(self.cutoff),
            np.arange(self.cutoff),
            indexing='ij'
        )

        process_tensor = process_tensor_element(m, n, j, k, eta)

        return process_tensor

    def load_pure_loss_data(self, filename, data_dir):
        # Load the process tensor from data_dir + filename produced by the synthetic notebook
        full_path = os.path.join(data_dir, filename)
        process_tensor = np.load(full_path)
        return process_tensor

if __name__ == "__main__":
    cutoff = 100
    
    # ct = CVQKDChannelTomographySynthetic(cutoff)
    # theoretical_process_tensor = ct.generate_pure_loss_process_tensor(0.5)

    # # Given \Epsilon_{jk}^{mn}, find the 'diagonal' elements \Epsilon_{kk}^{mm}
    # # This is the probability of the channel outputting k photons given m input photons.
    
    # theoretical_diagonal_elements = np.einsum('kkmm->km', theoretical_process_tensor)

    # data_dir = data_dir.split("\\")[:-2]
    # data_dir = "\\".join(data_dir) + "\\"
    # filename = "synthetic_data_20km_loss.csv"
    # synthetic_data = ct.load_pure_loss_data(filename, data_dir, allow_pickle=True)

    # # Show column headings of synthetic data
    # print(synthetic_data[0])



