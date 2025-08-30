# State Estimation Notes using Maximum-Likelihood Estimation

# Maximum-Likelihood Quantum State Tomography with Dual-Homodyne (Heterodyne) Data

## Self-contained notes on the $R\rho R$ (EM) method in a Fock-truncated model

These notes lay out the full statistical model and the iterative $R\rho R$ algorithm you’re using to reconstruct a density operator $\rho$ from dual-homodyne data, with all objects written explicitly in the truncated Fock basis. They mirror (and justify) the structure of your code.

---

## 1) Measurement model: ideal heterodyne POVM and the $Q$-function

An ideal dual-homodyne (heterodyne) detection realizes the continuous POVM

$$
\{\Pi(\alpha)\}_{\alpha\in\mathbb{C}}, \qquad 
\Pi(\alpha)=\frac{1}{\pi}\,|\alpha\rangle\langle\alpha|,
$$

where $|\alpha\rangle$ is a canonical coherent state. For a quantum state $\rho$,

$$
p(\alpha\mid \rho)=\operatorname{Tr}\!\left[\rho\,\Pi(\alpha)\right]
=\frac{1}{\pi}\,\langle\alpha|\rho|\alpha\rangle
=:Q_\rho(\alpha),
$$

i.e. the heterodyne outcome density is exactly the Husimi $Q$-function.

### Truncation

We reconstruct $\rho$ in the finite subspace $\mathcal{H}_N=\operatorname{span}\{|0\rangle,\dots,|N\rangle\}$ with projector $P$. The model then uses the *projected* POVM

$$
\Pi_N(\alpha)=\frac{1}{\pi}\,P\,|\alpha\rangle\langle\alpha|\,P,
$$

and $\rho$ is a PSD, trace-one matrix on $\mathcal{H}_N$. Completeness is no longer exact:
$\int \Pi_N(\alpha)\,d^2\alpha \neq I_N$, which is one source of (controllable) modeling error.

---

## 2) Coherent-state overlaps in the Fock basis

Write $\rho$ in the Fock basis as $\rho=\sum_{m,n=0}^N \rho_{mn}\,|m\rangle\langle n|$. The coherent-state amplitudes are

$$
\langle n|\alpha\rangle
= e^{-|\alpha|^2/2}\,\frac{\alpha^n}{\sqrt{n!}},
\qquad n=0,\dots,N.
$$

Define the “coherent-state matrix” $C\in\mathbb{C}^{(N+1)\times M}$ for a set of $M$ phase-space points $\{\alpha_j\}_{j=1}^M$ by

$$
C[n,j]=\langle n|\alpha_j\rangle
= e^{-|\alpha_j|^2/2}\,\frac{\alpha_j^n}{\sqrt{n!}}.
$$

Then the (unbinned) model intensity at $\alpha_j$ is

$$
\langle\alpha_j|\rho|\alpha_j\rangle = 
\sum_{m,n} \rho_{mn}\,C[m,j]^*\,C[n,j]
= \big(C^\dagger \rho\, C\big)_{jj}.
$$

---

## 3) Discretization: grids, bins, and the forward model

We approximate the continuous POVM by a regular Cartesian grid $(y_1,y_2)\in[-L,L]^2$ of size $G\times G$ and map to $\alpha=(y_1+i y_2)/\sqrt{2}$. If the step is $\Delta = 2L/(G-1)$, the area element in $\alpha$ is

$$
\Delta A = \frac{\Delta^2}{2}.
$$

The discretized POVM elements (“bins”) are

$$
E_j \;\approx\; \frac{\Delta A}{\pi}\,|\alpha_j\rangle\langle\alpha_j|\quad (j=1,\dots,M,\; M=G^2).
$$

Given $\rho$, the model *bin mass* is

$$
p_j(\rho)\;=\;\operatorname{Tr}(\rho E_j)
=\frac{\Delta A}{\pi}\,\langle\alpha_j|\rho|\alpha_j\rangle
=\frac{\Delta A}{\pi}\,\big(C^\dagger \rho\, C\big)_{jj}.
$$

In vector form, with $C$ defined above, your code computes these as

$$
p(\rho) = \frac{\Delta A}{\pi}\;\mathrm{diag}(C^\dagger \rho\, C).
$$

**Coverage.** Because we are on a finite window $[-L,L]^2$, the total mass $S(\rho):=\sum_j p_j(\rho)$ need not equal $1$. When the window is wide enough, $S(\rho)\approx 1$; otherwise one can (optionally) include an “outside” bin to capture the complement mass. Your implementation proceeds without the outside bin, which is fine when coverage is high.

---

## 4) Statistical model for the counts

Let $n_j$ be the number of samples falling in bin $j$. Two equivalent views are common:

* **Multinomial model** (fixed total $N$):
  $ \mathbf{n}\sim\mathrm{Multinomial}(N,\;\pi_j)$ with
  $\pi_j(\rho) = p_j(\rho)/\sum_\ell p_\ell(\rho)$.

* **Independent Poisson factorization** (used in your code):
  $n_j\sim\mathrm{Poisson}(\mu_j)$ with $\mu_j = N_\text{tot}\,p_j(\rho)$.

The Poisson negative log-likelihood (ignoring constants) is

$$
\mathcal{L}(\rho)=
N_\text{tot} \sum_j p_j(\rho)
-\sum_j n_j\log p_j(\rho)
-\Big(\sum_j n_j\Big)\log N_\text{tot}.
$$

Since $N_\text{tot}$ is fixed by the data, minimizing $\mathcal{L}$ is equivalent to maximizing the usual ML objective. (With perfect coverage, the first term is constant; with finite windows it keeps the estimate from pushing mass outside the grid.)

---

## 5) ML extremal equation and the $R\rho R$ fixed point

Let $\ell(\rho)=\sum_j n_j\log p_j(\rho)$ be the log-likelihood (up to constants and possibly the coverage term). Taking a Fréchet derivative and enforcing $\operatorname{Tr}\rho=1$ via a Lagrange multiplier yields the extremal (KKT) condition

$$
R(\rho)\,\rho \,=\, \rho, 
\qquad
R(\rho) \;=\; \sum_j \frac{n_j}{p_j(\rho)}\,E_j.
$$

Intuitively, $R$ *reweights* each POVM element by the ratio of observed to predicted mass in that bin. Any ML solution in the interior of the PSD cone satisfies this fixed-point equation (together with positivity and trace-one).

---

## 6) EM / $R\rho R$ iteration (Hradil / $R\rho R$ algorithm)

A monotone EM-style scheme to solve the fixed point is:

$$
\boxed{
\begin{aligned}
R_t &= \sum_j \frac{n_j}{p_j(\rho_t)}\,E_j,\\[2pt]
\tilde{\rho}_{t+1} &= R_t\,\rho_t\,R_t,\\[2pt]
\rho_{t+1} &= \frac{\tilde{\rho}_{t+1}}{\operatorname{Tr}\tilde{\rho}_{t+1}}.
\end{aligned}}
$$

* This is the celebrated $R\rho R$ update.
* It can be derived from EM majorization of the concave log by Jensen’s inequality; it increases the likelihood (for the pure multinomial case and complete POVM; small deviations can occur with coverage loss or regularization).

### Discrete/bin form used in code

With $E_j = \frac{\Delta A}{\pi}|\alpha_j\rangle\langle\alpha_j|$ and the coherent-state matrix $C$,

$$
R_t \;=\; \sum_{j=1}^M 
\frac{n_j}{p_j(\rho_t)} \frac{\Delta A}{\pi}\,
|\alpha_j\rangle\langle\alpha_j|
\;=\;(C W)\,C^\dagger,
$$

where the diagonal weights are

$$
W_{jj}=\frac{n_j}{p_j(\rho_t)}\frac{\Delta A}{\pi}.
$$

In matrix code, that is `CW = C * weights; R = CW @ C.conj().T` (and then Hermitize to suppress round-off asymmetry).

**Numerical guards.** In practice we floor the model masses $p_j(\rho_t)\ge \varepsilon$ to prevent divisions by zero, clip tiny negatives from round-off, and re-project to the physical set (PSD + trace-one) after the sandwich.

---

## 7) Optional entropy regularization

To promote mixedness or stabilize ill-posed cases, add a von Neumann entropy penalty $\tau\,\operatorname{Tr}(\rho\log\rho)$. A simple proximal-like tweak is

$$
R_t \;\leftarrow\; R_t \;-\; \tau\big(\log \rho_t + I\big),
$$

followed by the usual $\tilde{\rho}_{t+1}=R_t\rho_t R_t$ and projection. This may sacrifice strict EM monotonicity but often improves robustness.

---

## 8) Computing the forward map and diagnostics

### (a) Model bin masses

$$
p_j(\rho)=\frac{\Delta A}{\pi}\,\big(C^\dagger \rho\, C\big)_{jj},\quad j=1,\dots,M.
$$

### (b) $Q$-function on an arbitrary mesh

For any $\alpha$,

$$
Q_\rho(\alpha) 
= \frac{1}{\pi}e^{-|\alpha|^2}
\sum_{m,n=0}^N \rho_{mn}\,
\frac{(\alpha^*)^{m}\alpha^{n}}{\sqrt{m!\,n!}}.
$$

This is how your plotting routine compares “true” vs reconstructed $Q$.

### (c) Negative log-likelihood (Poisson factorization)

$$
\mathcal{L}(\rho)=
N_\text{tot} \sum_j p_j(\rho)
-\sum_j n_j\log p_j(\rho)
\quad(+\text{const}),
$$

tracked per iteration to monitor convergence.

---

## 9) Algorithm in one glance (matches your `run_mle_workflow`)

1. **Grid & histogram.** Choose half-width $L$ (possibly adaptively from the samples), grid size $G$, compute $\Delta A=\Delta^2/2$, and the counts $n_j$ by binning.
2. **Build $C$.** For each bin center $\alpha_j$, set $C[n,j]=e^{-|\alpha_j|^2/2}\alpha_j^n/\sqrt{n!}$.
   *Stable recurrence:* $C[n+1,\cdot]=C[n,\cdot]\cdot \alpha/\sqrt{n+1}$.
3. **Initialize $\rho_0$.** E.g. truncated coherent projector around the sample mean $\bar{\alpha}$, or maximally mixed $I/(N+1)$.
4. **Iterate:**
   a) $p(\rho_t)=\frac{\Delta A}{\pi}\,\mathrm{diag}(C^\dagger \rho_t C)$ (floored).
   b) $R_t=(C W)C^\dagger$ with $W_{jj}= \dfrac{n_j}{p_j(\rho_t)}\dfrac{\Delta A}{\pi}$.
   c) (Optional) Regularize: $R_t\leftarrow R_t-\tau(\log\rho_t + I)$.
   d) $\tilde{\rho}_{t+1}=R_t\,\rho_t\,R_t$.
   e) **Project to** PSD & trace-one (eigenvalue clamping $\ge 0$, renormalize trace).
   f) Check NLL decrease and stop when relative improvement $<$ tolerance.
5. **Outputs/diagnostics.** Final $\rho$, NLL trace, reconstructed $Q$, diagonals, fidelity to a known ground truth (when available).

---

## 10) Practical considerations

* **Coverage/windowing.** If $\sum_j p_j(\rho)$ drifts appreciably from $1$, widen the window or include an explicit “outside” bin $E_\emptyset = I - \sum_j E_j$ with count $n_\emptyset = N_\text{tot}-\sum_j n_j$. Then

  $$
  R_t \gets \sum_j \frac{n_j}{p_j(\rho_t)}E_j + \frac{n_\emptyset}{p_\emptyset(\rho_t)}E_\emptyset.
  $$

  Your implementation omits this for simplicity; the adaptive half-width largely mitigates the issue.
* **Physicality.** The eigen-projection step guarantees PSD and trace-one even with numerical noise and tiny negative eigenvalues.
* **Initialization.** A coherent projector at $\bar{\alpha}$ often accelerates convergence compared with maximally mixed.
* **Stability.** Always floor $p_j(\rho_t)$ by a tiny $\varepsilon$ (e.g. $10^{-30}$) before forming weights; Hermitize $R_t$.
* **Cost.** Per iteration, the dominant contractions are $C^\dagger \rho C$ and $(C W)C^\dagger$, both $\mathcal{O}\big((N{+}1)^2 M\big)$ with $M=G^2$.
* **When does EM increase the likelihood?** With a complete (properly normalized) POVM and the multinomial model it is monotone. Finite windows, Poisson modeling with the mass term, and entropy regularization may mildly break strict monotonicity—hence your NLL check is the right stopping rule.

---

## 11) Mapping code ↔ equations

* `C = build_coherent_state_matrix(...)` $\leftrightarrow$ $C[n,j]=\langle n|\alpha_j\rangle$.
* `bin_area` $\leftrightarrow$ $\Delta A$.
* `compute_model_probabilities(rho,C,bin_area)` $\leftrightarrow$ $p_j(\rho)=\frac{\Delta A}{\pi}\,\mathrm{diag}(C^\dagger \rho C)$.
* `compute_R_operator(counts_2d, p, C, bin_area)` $\leftrightarrow$ $R=(C W)C^\dagger$, $W_{jj}=\dfrac{n_j}{p_j}\dfrac{\Delta A}{\pi}$.
* `rho_new = R @ rho @ R; rho_new = _project_psd_trace_one(rho_new)` $\leftrightarrow$ $\tilde{\rho}=R\rho R$, then PSD + trace-one projection.
* `neg_log_likelihood_poisson(...)` $\leftrightarrow$ $\mathcal{L}(\rho)$ above.
* `Q_from_rho` $\leftrightarrow$ $Q_\rho(\alpha)$ formula in §8(b).

---

## 12) Derivation sketch of the $R\rho R$ update

Start from the (unbinned) log-likelihood for samples $\{\alpha_i\}_{i=1}^N$:

$$
\ell(\rho) = \sum_{i=1}^N \log \operatorname{Tr}\!\big(\rho \Pi(\alpha_i)\big),\qquad \Pi(\alpha)=\frac1\pi|\alpha\rangle\langle\alpha|.
$$

Using concavity of $\log$ and Jensen, build a surrogate at $\rho_t$:

$$
\log \operatorname{Tr}(\rho X) 
\;\ge\; \log \operatorname{Tr}(\rho_t X)
+ \frac{\operatorname{Tr}\big((\rho-\rho_t)X\big)}{\operatorname{Tr}(\rho_t X)}.
$$

Summing $X=\Pi(\alpha_i)$ over the data produces the linearized bound

$$
\ell(\rho)\;\ge\; \mathrm{const} + \operatorname{Tr}\!\Big(R_t\,\rho\Big),
\qquad
R_t := \sum_{i=1}^N \frac{\Pi(\alpha_i)}{\operatorname{Tr}(\rho_t \Pi(\alpha_i))}.
$$

Maximizing the RHS over PSD, trace-one $\rho$ is a constrained quadratic problem whose stationary condition is $R_t\rho=\rho R_t = \lambda \rho$. The update
$\tilde{\rho}_{t+1}=R_t \rho_t R_t$ (followed by normalization) enforces this structure and increases the bound, which in turn increases the true likelihood under the cited conditions. The binned case is obtained by grouping repeated $\alpha$’s, giving the weights $n_j$.

---

## 13) Minimal pseudocode (dimension-agnostic)

```text
Given: counts {n_j}, grid {α_j}, ΔA, cutoff N.

Build C[n,j] = exp(-|α_j|^2/2) α_j^n / sqrt(n!),  n=0..N.
Init ρ₀ ≽ 0 with Tr ρ₀ = 1  (e.g., truncated coherent at ᾱ or I/(N+1)).

for t = 0,1,2,...
    p_j = (ΔA/π) * (C^† ρ_t C)_{jj}    (floor by ε)
    W_jj = (n_j / p_j) * (ΔA/π)
    R_t = (C W) C^†                      (Hermitize)
    if τ>0:  R_t ← R_t - τ (log ρ_t + I)
    ρ̃ = R_t ρ_t R_t
    ρ_{t+1} = Proj_{PSD, Tr=1}(ρ̃)
    check NLL decrease; stop if relative improvement < tol
end
return ρ
```

---

### Final remarks

* With enough shots and a sufficiently large window, the method yields consistent ML estimates within the truncated model.
* The coherent-state matrix recurrence and PSD projection you use are numerically robust choices.
* If you ever need to account for loss/inefficiency or known Gaussian noise, simply replace the ideal $E_j$ by the appropriate noisy POVM kernels; the $R\rho R$ machinery stays the same.
