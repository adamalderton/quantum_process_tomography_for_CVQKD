import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax import jit
import jax.scipy as jsp
from jax.scipy.special import factorial
from jax.scipy.stats import multivariate_normal
import time
import matplotlib.pyplot as plt

class CVQKDChannelTomography():
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
        self.m_mesh, self.n_mesh, self.j_mesh, self.k_mesh = jnp.meshgrid(jnp.arange(cutoff), jnp.arange(cutoff), jnp.arange(cutoff), jnp.arange(cutoff))
        
    # @jit
    def generate_pure_loss_process_tensor(self, loss: float) -> jnp.ndarray:
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

        # @jit
        def process_tensor_element(m, n, j, k, eta):
            delta = ((m - j) == (n - k))

            m_fac = factorial(m)
            n_fac = factorial(n)
            j_fac = factorial(j)
            k_fac = factorial(k)
            m_minus_j_fac = factorial(m - j)

            return (
                delta
                * jnp.sqrt(m_fac * n_fac / (j_fac * k_fac))
                * (eta ** (j + k))
                * ((1 - eta ** 2) ** (m - j))
                / m_minus_j_fac
            )

        # # Vectorize the function over all indices
        # vectorized_epsilon = vmap(
        #     vmap(
        #         vmap(
        #             vmap(process_tensor_element, in_axes=(0, 0, 0, 0, None)),  # Batch over m
        #             in_axes=(0, 0, 0, 0, None)                                 # Batch over n
        #         ),
        #         in_axes=(0, 0, 0, 0, None)                                     # Batch over j
        #     ),
        #     in_axes=(0, 0, 0, 0, None)                                         # Batch over k
        # )

        # # Compute the process tensor
        # process_tensor = vectorized_epsilon(self.m_mesh, self.n_mesh, self.j_mesh, self.k_mesh, eta)
        # Initialize the process tensor with zeros
        process_tensor = np.zeros(self.process_tensor_shape)

        # Compute the process tensor using nested for loops
        for m in range(self.cutoff):
            for n in range(self.cutoff):
                for j in range(self.cutoff):
                    for k in range(self.cutoff):
                        process_tensor[m, n, j, k] = process_tensor_element(m, n, j, k, eta)

        return process_tensor

if __name__ == "__main__":
    cutoff = 6
    ct = CVQKDChannelTomography(cutoff)
    process_tensor = ct.generate_pure_loss_process_tensor(0.5)

    print(process_tensor.shape)

    # Given \Epsilon_{jk}^{mn}, find the 'diagonal' elements \Epsilon_{kk}^{mm}
    # This is the probability of the channel outputting k photons given m input photons.
    
    diagonal_elements = np.einsum('kkmm->km', process_tensor)

    # Plot the diagonal elements.
    # plt.imshow(diagonal_elements)
    # plt.xlabel("m")
    # plt.ylabel("k")
    # plt.title("$\epsilon_{mm}^{kk}$")
    # plt.colorbar()
    # plt.show()

    np.set_printoptions(precision=2, suppress=False)
    print(diagonal_elements)
