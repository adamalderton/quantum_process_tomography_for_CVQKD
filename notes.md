**Goal**: Reconstruct the *process tensor* $\mathcal{E}_{jk}^{mn}$ of a single‐mode quantum channel $\mathcal{E}$ (acting on optical modes, for example), using data gathered from sending *random* coherent states $|\alpha\rangle$ through the channel and measuring them by dual‐homodyne (heterodyne) detection.

---

## 1. Overview: The Process Tensor in the Fock Basis

This is to estimate the process tensor of an arbitrary quantum process $\mathcal{E}$ using coherent states randomly selected from those used as part of a CV-QKD system. This is in the interest of using quantum process tomography (QPT) to characterise the dynamics of the quantum channel between Alice and Bob, so as to quantify the extent to which a non-Gaussian attack may have been carried out.

A single‐mode quantum channel (completely positive, trace‐preserving map) $\mathcal{E}$ can be characterized by its *process‐tensor elements* in the Fock basis:
$$
  \mathcal{E}_{jk}^{mn}
  \;=\;
  \langle j \,|\, \mathcal{E} (| m\rangle \langle n | ) \,|\, k \rangle
  \quad\text{for }j,k,m,n=0,1,2,\dots
$$
Equivalently, if we send the basis operator $|m\rangle\langle n|$ through $\mathcal{E}$, these coefficients give the resulting Fock‐basis matrix elements.

### 1.1 Coherent‐State Formula

A powerful identity from (Rahimi‐Keshari et al. 2011, Lobino et al. 2008) shows that one can recover $\mathcal{E}_{jk}^{mn}$ using only *coherent‐state* probes.  Specifically, if
$$
  \rho_{\mathcal{E}}(\alpha)
  \;=\;
  \mathcal{E}\bigl(|\alpha\rangle\langle\alpha|\bigr),
$$
is the channel’s output when the input is the coherent state $|\alpha\rangle$, then
$$
  \mathcal{E}_{jk}^{mn}
  \;=\;
  \frac{1}{\sqrt{m!\,n!}}
  \;\biggl[
    \frac{\partial^m}{\partial\alpha^m}\,
    \frac{\partial^n}{\partial\overline{\alpha}^n}
    \Bigl(
      e^{|\alpha|^2}\,
      \langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle
    \Bigr)
  \biggr]_{\alpha=0}.
$$
Here, $\bra{j} \rho_\mathcal{E}(\alpha) \ket{k}$ are complex matrix elements (a matrix across $j$ and $k$ photon numbers) of the output state $\rho_\mathcal{E}(\alpha)$. These are individual functions of $\rho_\mathcal{E}(\alpha)$. Thus **the entire** process tensor is encoded in how $\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle$ varies with $\alpha$ around $\alpha = 0$ (the vacuum state).

---

## 2. Data Collection in a CV‐QKD Setting

### 2.1 Random Modulation of Coherent States

In general quantum‐process tomography, one might choose $\{\alpha_i\}$ systematically.  
However, in **CV‐QKD**:
- Alice’s transmitter sends coherent states $|\alpha\rangle$ where $\alpha$ is *randomly sampled* from some (e.g. Gaussian) distribution.  
- We do *not* have the freedom to pick $\alpha$ deterministically.

**Key Idea**: *Bin* (or cluster) the randomly chosen $\alpha$ values.  
1. Partition the complex plane into bins $\mathcal{B}_1,\dots,\mathcal{B}_K$. These bins should be fine in phase space such that $|\alpha_2 - \alpha_1| < \epsilon$, for some small epsilon.
2. Whenever $\alpha$ falls into bin $\mathcal{B}_i$, label that shot “$\alpha_i$.”  
3. Collect enough samples in each bin that you can treat all those states as effectively $|\alpha_i\rangle$.

This gives you a discrete set of “effective” coherent states $\{\alpha_i\}$, each with multiple measurement shots.

### 2.2 Dual‐Homodyne (Heterodyne) Measurement

The standard measurement in many CV‐QKD protocols is a **dual‐homodyne** or **heterodyne** detection.  The outcome $\beta\in\mathbb{C}$ is distributed according to the *Husimi-Q* function:
$$
  P_{\rho}(\beta)
  \;=\;
  \tfrac{1}{\pi}\,\langle\beta|\rho|\beta\rangle,
  \quad
  |\beta\rangle\text{ is a coherent state.}
$$

Hence, if the channel output is $\rho_{\mathcal{E}}(\alpha_i)$, then measuring it many times yields samples $\{\beta_k\}$ from
$$
  P_{\rho_{\mathcal{E}}(\alpha_i)}(\beta)
  \;=\;
  \tfrac{1}{\pi}\,\langle\beta|\rho_{\mathcal{E}}(\alpha_i)|\beta\rangle.
$$

---

## 3. Reconstructing $\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle$

### 3.1 $\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle$ vs $\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle$

[[../../../__source_notes/Rahimi-Keshari-2011|Rahimi-Keshari-2011]] and [[../../../__source_notes/Lobino-2008|Lobino-2008]] performed this process while being able to do a 'standard' form of QPT. In our case, we can only use quadrature measurements discarded as used in the parameter estimation process. Therefore, we have to take a bit more care.

[[../../../__source_notes/Rahimi-Keshari-2011|Rahimi-Keshari-2011]] constructs the full output state $\rho_\mathcal{E}(\alpha)$ by interpolating a set of fully reconstructed states $\rho_\mathcal{E}(\alpha_i)$, each reconstructed using homodyne tomography. This requires the repeated deterministic sending of $\alpha_i \forall i$. This may be a well defined grid in phase space. Of course, in the case of using CV-QKD states, this is not possible as the states must be random.

Therefore, due to the random modulation of the states used, we can attempt to reconstruct the matrix elements $\bra{j} \rho_\mathcal{E}(\alpha_i) \ket{k}$ for a collection of $\alpha_i$ in the Fock basis directly rather than reconstructing the entire state density matrix $\rho_\mathcal{E}(\alpha)$ first. We can then fit continuous functions to $\bra{j} \rho_\mathcal{E}(\alpha_i) \ket{k}$.

We must first calculate $\bra{j} \rho_\mathcal{E}(\alpha_i) \ket{k}$ at specific locations in emitted state phase space defined by $\alpha_i$ to then build up to a polynomial fit of $\bra{j} \rho_\mathcal{E}(\alpha) \ket{k}$. This is done by discretising the emitted coherent states in phase space, across bins defined by $|\alpha|$ and $\arg(\alpha)$. That is, as the coherent states are random (due to their use in a CV-QKD system) we can group very similar coherent states up to an $\epsilon$ if $|\alpha_1 - \alpha_2| < \epsilon$, where $\epsilon$ is small, but big enough to group enough coherent states. If we are fitting a polynomial of degree $n$ (= 9? or so), we need to (randomly?) select 10 ($n + 1$) bins from which to use measured $x$ and $p$ data.

We will later discuss how to group these measurements of $\bra{j} \rho_\mathcal{E}(\alpha_i) \ket{k}$ together, but first we discuss how to measure $\bra{j} \rho_\mathcal{E}(\alpha_i) \ket{k}$.

Ultimately, we want the *Fock‐basis matrix elements*.  From the known expansion of a coherent state in the Fock basis,
$$
  |\beta\rangle
  \;=\;
  e^{-|\beta|^2/2}
  \sum_{n=0}^{\infty}
    \frac{\beta^n}{\sqrt{n!}}\;|n\rangle,
$$
one finds
$$
  \langle\beta|\rho_{\mathcal{E}}(\alpha_i)|\beta\rangle
  \;=\;
  e^{-|\beta|^2}
  \sum_{j,k=0}^{\infty}
    \frac{(\beta^*)^j}{\sqrt{j!}}
    \,\frac{\beta^k}{\sqrt{k!}}
    \,\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle.
$$

Thus the measurement distribution is
$$
  P_{\rho_{\mathcal{E}}(\alpha_i)}(\beta)
  \;=\;
  \frac{1}{\pi}\,\langle\beta|\rho_{\mathcal{E}}(\alpha_i)|\beta\rangle
  \;=\;
  \frac{e^{-|\beta|^2}}{\pi}
  \sum_{j,k=0}^{\infty}
    \frac{(\beta^*)^j}{\sqrt{j!}}\,\frac{\beta^k}{\sqrt{k!}}
    \,\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle.
$$

### 3.2 Finite‐Dimensional Truncation

In practice, we truncate at some photon number $N$.  We only treat the matrix elements with $0 \le j,k \le N$.  Then we have $(N+1)^2$ unknown complex numbers $\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle$.  [[../../../__source_notes/Rahimi-Keshari-2011|Rahimi-Keshari-2011]] discusses the implications of this Hilbert space truncation further. Other CV-QKD papers discuss the Hilbert truncation in the context of security

### 3.3 Setting Up the Linear Inversion

4. Make a 2D histogram or density estimate of the measured $\beta$‐outcomes (the dual‐homodyne data) to get $\hat{P}_{\alpha_i}(\beta)$.  
5. Notice that each $\hat{P}_{\alpha_i}(\beta_m)$ is a *linear* combination of the unknown matrix elements $\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle$.  
6. Formally, we can write:
$$
  \hat{P}_{\alpha_i}(\beta_m)
  \;\approx\;
  \sum_{j=0}^N\,\sum_{k=0}^N
    \Bigl[
      \tfrac{e^{-|\beta_m|^2}}{\pi}\,
      \tfrac{(\beta_m^*)^j}{\sqrt{j!}}\,\tfrac{\beta_m^k}{\sqrt{k!}}
    \Bigr]
    \;\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle.
$$

7. Collect these values into a matrix equation $\mathbf{\hat{P}}_{\alpha_i} = \mathbf{M} \,\mathbf{r}_{\alpha_i}$, where:
   - $\mathbf{\hat{P}}_{\alpha_i}$ is a vector of your measured Q‐function values (one entry per $\beta_m$ bin).  
   - $\mathbf{r}_{\alpha_i}$ is the vector of unknowns $\{\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle\}$.  
   - $\mathbf{M}$ is the “design matrix” from the known basis functions.

Solving that linear system (via least squares or a more sophisticated method) yields estimates for $\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle$.  Repeat for **each** bin $\alpha_i$.

**Later**: Discuss least squares vs MLE vs other frequentist approaches to get error bars on the estimates.

---

## 4. Fitting $\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle$ as a Function of $\alpha$

After doing the above for each bin $\alpha_i$, you have discrete points $\alpha_i \mapsto \langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle$.  But to apply the formula for $\mathcal{E}_{jk}^{mn}$, you need a smooth (or at least polynomial) function in $\alpha$. As discussed in [[../../../__source_notes/Rahimi-Keshari-2011|Rahimi-Keshari-2011]], $\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle$ is an entire function (therefore it can be expanded as a convergent power series everywhere), such that it can be expanded as a polynomial. Also, if we expect Gaussian or near-Gaussian maps, only a low order polynomial is needed although the order of the polynomial $d$ is a free parameter.

Expanding $\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle$ by a **bivariate polynomial** in $\alpha$ and $\bar{\alpha}$:
$$
  \langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle
  \;\approx\;
  \sum_{p=0}^d \sum_{q=0}^d\,
    C_{p,q}^{(j,k)}\,\alpha^p\,\overline{\alpha}^q.
$$

You choose a polynomial degree $d$ large enough to capture non‐Gaussianities but small enough to avoid overfitting.  Then you solve a (complex) linear‐regression problem for $\{C_{p,q}^{(j,k)}\}$:
$$
  \bigl(\alpha_i,\,\langle j|\rho_{\mathcal{E}}(\alpha_i)|k\rangle\bigr)
  \;\longmapsto\;
  C_{p,q}^{(j,k)}.
$$
(Optionally add Tikhonov/ridge regularization if the data is noisy.)

---

## 5. Extracting $\mathcal{E}_{jk}^{mn}$ from $C_{p,q}^{(j,k)}$

Once you have obtained the fitted polynomial coefficients $C_{p,q}^{(j,k)}$ from your regression—i.e. from fitting
$$
\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle \approx \sum_{p=0}^d \sum_{q=0}^d C_{p,q}^{(j,k)}\,\alpha^p\,\overline{\alpha}^q,
$$
you can compute the process-tensor elements $\mathcal{E}_{jk}^{mn}$ directly using the final formula (see appendix below):
$$
\mathcal{E}_{jk}^{mn} = \sqrt{m!\,n!}\,\sum_{r=0}^{\min(m,n)} \frac{C_{m-r,\,n-r}^{(j,k)}}{r!}\,.
$$

In practice, this is done as follows:
1. **For each Fock basis operator** $|m\rangle\langle n|$ (with $m,n$ running from 0 up to the chosen truncation level $N$), identify the corresponding coefficients $C_{p,q}^{(j,k)}$ for each output matrix element (indexed by $j$ and $k$).
2. **For each combination** of $(m,n)$ and for each $(j,k)$, perform the summation over $r$ from 0 to $\min(m,n)$ by setting
   - $p = m - r$ and $q = n - r$, and then
   - computing the term $\dfrac{C_{m-r,\,n-r}^{(j,k)}}{r!}$.
3. **Multiply the summed value** by $\sqrt{m!\,n!}$ to yield the final element:
   $$
   \mathcal{E}_{jk}^{mn} = \sqrt{m!\,n!}\,\sum_{r=0}^{\min(m,n)} \frac{C_{m-r,\,n-r}^{(j,k)}}{r!}\,.
   $$
4. **Repeat for all indices** $j,k,m,n$ required to fully characterize the process tensor.

Optionally, if you have uncertainties from the regression, you can propagate these through the summation to obtain error estimates on each $\mathcal{E}_{jk}^{mn}$. This step-by-step procedure allows you to convert the fitted polynomial (which encodes the $\alpha$-dependence of the output states) into the process-tensor representation of the quantum channel.

# Uncertainties Estimation

**COME BACK TO THIS**. This will entail propagating uncertainties from calculating $C_{p,q}^{(j,k)}$.
# Appendix: Derivation of Process Tensor Formula

We start from the coherent‐state formula for the process‐tensor element in the Fock basis:
$$
\mathcal{E}_{jk}^{mn} = \frac{1}{\sqrt{m! \, n!}}
\left[
\frac{\partial^m}{\partial\alpha^m}\,
\frac{\partial^n}{\partial\overline{\alpha}^n}
\Bigl(
e^{|\alpha|^2}\,
\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle
\Bigr)
\right]_{\alpha=0}\,.
$$
Here, the channel output (for input $|\alpha\rangle$) has been expanded in the Fock basis via
$$
\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle
=\sum_{p,q} C_{p,q}^{(j,k)}\,\alpha^p\,\overline{\alpha}^q\,,
$$
with coefficients $C_{p,q}^{(j,k)}$ determined by the process.

Our goal is to show that
$$
\boxed{
\mathcal{E}_{jk}^{mn} = \sqrt{m!\, n!}\,\sum_{r=0}^{\min(m,n)} \frac{C_{m-r,\,n-r}^{(j,k)}}{r!}\,.
}
$$

---

## Step 1. Expand the Exponential Factor

Recall the Taylor expansion of the exponential:
$$
e^{|\alpha|^2} = e^{\alpha\overline{\alpha}} = \sum_{r=0}^\infty \frac{(\alpha\,\overline{\alpha})^r}{r!}\,.
$$

---

## Step 2. Multiply by the Expansion of $\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle$

Inserting the expansion for $\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle$, we have:
$$
e^{|\alpha|^2}\,\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle
=\left(\sum_{r=0}^\infty \frac{(\alpha\,\overline{\alpha})^r}{r!}\right)
\left(\sum_{p,q} C_{p,q}^{(j,k)}\,\alpha^p\,\overline{\alpha}^q\right)\,.
$$

This can be written as a double sum:
$$
e^{|\alpha|^2}\,\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle
=\sum_{r=0}^\infty \sum_{p,q} \frac{C_{p,q}^{(j,k)}}{r!}\,\alpha^{p+r}\,\overline{\alpha}^{q+r}\,.
$$

---

## Step 3. Apply the Wirtinger Derivatives

We need to compute
$$
D = \frac{\partial^m}{\partial\alpha^m}\,
\frac{\partial^n}{\partial\overline{\alpha}^n}\left(
\sum_{r=0}^\infty \sum_{p,q} \frac{C_{p,q}^{(j,k)}}{r!}\,\alpha^{p+r}\,\overline{\alpha}^{q+r}
\right)\,.
$$

Because the derivatives are linear, we can differentiate term‐by‐term. Notice that differentiating a term $\alpha^{p+r}$ $m$ times and $\overline{\alpha}^{q+r}$ $n$ times yields a nonzero result **only** if
$$
p + r = m \quad \text{and} \quad q + r = n\,.
$$

In that case, we have:
$$
\frac{\partial^m}{\partial\alpha^m}\,\alpha^{m} = m! \quad \text{and} \quad \frac{\partial^n}{\partial\overline{\alpha}^n}\,\overline{\alpha}^{n} = n!\,.
$$

Thus, the only contributions come from terms where
$$
p = m - r \quad \text{and} \quad q = n - r\,.
$$
Furthermore, we must have $r \le m$ and $r \le n$; that is, $r$ runs from 0 to $\min(m,n)$.

Taking the derivatives and evaluating at $\alpha = 0$ then gives:
$$
\left[\frac{\partial^m}{\partial\alpha^m}\,
\frac{\partial^n}{\partial\overline{\alpha}^n}\left(
e^{|\alpha|^2}\langle j|\rho_{\mathcal{E}}(\alpha)|k\rangle
\right)\right]_{\alpha=0}
=\sum_{r=0}^{\min(m,n)} \frac{C_{m-r,\,n-r}^{(j,k)}}{r!}\; m!\, n!\,.
$$

---

## Step 4. Multiply by the Prefactor and Simplify

Recall the original definition:
$$
\mathcal{E}_{jk}^{mn} = \frac{1}{\sqrt{m! \, n!}}\,\left[\text{derivatives}\right]_{\alpha=0}\,.
$$

Substitute our derivative result:
$$
\mathcal{E}_{jk}^{mn} = \frac{1}{\sqrt{m! \, n!}}\Bigl( m!\, n!\,\sum_{r=0}^{\min(m,n)} \frac{C_{m-r,\,n-r}^{(j,k)}}{r!}\Bigr)\,.
$$

Simplify by cancelling:
$$
\mathcal{E}_{jk}^{mn} = \sqrt{m!\, n!}\,\sum_{r=0}^{\min(m,n)} \frac{C_{m-r,\,n-r}^{(j,k)}}{r!}\,.
$$

---

## Final Result

Thus, we have derived the desired r‑sum formula:
$$
\boxed{
\mathcal{E}_{jk}^{mn} = \sqrt{m!\, n!}\,\sum_{r=0}^{\min(m,n)} \frac{C_{m-r,\,n-r}^{(j,k)}}{r!}\,.
}
$$
This completes the detailed derivation.
