Below are self-contained, rigorous notes for **projected-gradient (PG) maximum-likelihood estimation of a single quantum state** from **histogrammed dual-homodyne (heterodyne) data** in a **Fock-truncated** model. No priors or regularizers are included.

---

# 1. Physical & statistical model

## 1.1 State, truncation, and constraints

Fix a single bosonic mode and a Fock cutoff $N$. Work on

$$
\mathcal{H}_N=\operatorname{span}\{|0\rangle,\dots,|N\rangle\},\qquad d=N+1.
$$

The unknown state is a density operator

$$
\rho\in\mathbb{C}^{d\times d},\qquad \rho\succeq 0,\quad \operatorname{Tr}\rho=1.
\tag{C}
$$

## 1.2 Dual-homodyne POVM and discretization

Ideal dual-homodyne implements the heterodyne POVM $\{\Pi(\alpha)\}_{\alpha\in\mathbb{C}}$,

$$
\Pi(\alpha)=\frac{1}{\pi}\,|\alpha\rangle\langle\alpha|.
$$

To handle large datasets efficiently, discretize phase space on a square grid in $(y_1,y_2)\in[-L,L]^2$ with $G$ points per axis and step $\Delta=2L/(G-1)$. Map grid points to coherent labels by

$$
\alpha=\frac{y_1+i y_2}{\sqrt{2}},\qquad \Delta A=\frac{\Delta^2}{2}.
$$

Each grid point $\alpha_j$ (with index $j=1,\dots,M$, $M=G^2$) defines the **discretized POVM element**

$$
E_j=\frac{\Delta A}{\pi}\,|\alpha_j\rangle\langle\alpha_j|\ \succeq\ 0.
\tag{1}
$$

## 1.3 Histogrammed data and forward probabilities

Collect $N_{\rm tot}$ dual-homodyne shots from $\rho$, histogrammed into counts $n_j\in\mathbb{N}$ over the grid ($\sum_j n_j=N_{\rm tot}$). The **model bin mass** is

$$
p_j(\rho)=\operatorname{Tr}(\rho\,E_j)
=\frac{\Delta A}{\pi}\,\langle \alpha_j|\rho|\alpha_j\rangle \;\ge 0.
\tag{2}
$$

Because the grid/window is finite, the **coverage**

$$
S(\rho):=\sum_{j=1}^M p_j(\rho)=\operatorname{Tr}\!\Big(\rho\sum_{j}E_j\Big)
$$

need not equal 1 (it approaches 1 as the window enlarges and the discretization refines).

## 1.4 Likelihood (Poisson factorization)

Model the histogram counts by independent Poisson variables,

$$
n_j\sim\mathrm{Poisson}\!\big(\mu_j\big),\qquad \mu_j=N_{\rm tot}\,p_j(\rho).
$$

Ignoring constants independent of $\rho$, the **negative log-likelihood (NLL)** is

$$
\mathcal{L}(\rho)=
\sum_{j=1}^M\Big[N_{\rm tot}\,p_j(\rho)-n_j\log p_j(\rho)\Big]
= N_{\rm tot}\sum_j \operatorname{Tr}(\rho E_j)
-\sum_j n_j \log \operatorname{Tr}(\rho E_j).
\tag{3}
$$

Each summand $N_{\rm tot}p - n\log p$ is convex for $p>0$; since $p_j(\rho)$ is affine in $\rho$,

$$
\boxed{\ \mathcal{L}(\rho)\ \text{is convex on the spectrahedron } \{\rho\succeq 0,\ \operatorname{Tr}\rho=1\}\ }.
$$

---

# 2. Efficient forward map in the Fock basis

Write coherent overlaps

$$
c^{(j)}_n=\langle n|\alpha_j\rangle
= e^{-|\alpha_j|^2/2}\,\frac{\alpha_j^n}{\sqrt{n!}},\qquad n=0,\dots,N,
$$

and the rank-1 matrix $B^{(j)}\in\mathbb{C}^{d\times d}$ with

$$
B^{(j)}_{mn}=c^{(j)}_m\,c^{(j)\,*}_n,\qquad
E_j=\frac{\Delta A}{\pi}\,B^{(j)}.
$$

Then

$$
p_j(\rho)=\operatorname{Tr}(\rho E_j)=\frac{\Delta A}{\pi}\sum_{m,n=0}^{N}\rho_{nm}\,B^{(j)}_{mn}.
\tag{4}
$$

This can be evaluated with a single contraction (e.g. `einsum`) for all $j$. **Histogramming** ensures the cost scales with $M$ (number of bins), not with raw shots.

---

# 3. Projected-gradient (PG) MLE

## 3.1 Gradient and Hessian

From (3)–(4), using the Hilbert–Schmidt inner product $\langle X,Y\rangle=\operatorname{Tr}(X^\dagger Y)$,

$$
\nabla \mathcal{L}(\rho)=\sum_{j=1}^M\Big(N_{\rm tot}-\frac{n_j}{p_j(\rho)}\Big)E_j.
\tag{5}
$$

For numerical stability we **floor** the probabilities

$$
p_j(\rho)\ \leftarrow\ \max\{p_j(\rho),\varepsilon\},\qquad \varepsilon>0.
\tag{6}
$$

The Hessian is the positive semidefinite linear operator

$$
\nabla^2\mathcal{L}(\rho)[H]=\sum_{j=1}^M\frac{n_j}{p_j(\rho)^2}\ \operatorname{Tr}(H E_j)\,E_j,
\tag{7}
$$

which confirms convexity.

## 3.2 Projection to the state set $\{\rho\succeq 0,\ \operatorname{Tr}\rho=1\}$

Given a Hermitian $X$, the **Euclidean projection** onto the trace-one PSD cone is obtained by eigen-decomposition $X=V\Lambda V^\dagger$ and **projecting the eigenvalue vector** $\lambda=\operatorname{diag}\Lambda$ onto the probability simplex

$$
\Delta_d=\{w\in\mathbb{R}^d:\; w_k\ge 0,\ \sum_k w_k=1\}.
$$

Let $\hat w=\operatorname{proj}_{\Delta_d}(\lambda)$ (the standard “water-filling”/thresholding). Then

$$
\Pi_{\rm state}(X)=V\,\operatorname{diag}(\hat w)\,V^\dagger.
\tag{8}
$$

*(In practice, many implementations use the simpler “clip negatives, then renormalize trace”; (8) is the exact Euclidean projection.)*

## 3.3 The PG iteration

Choose any feasible $\rho_0$ (e.g. $I/d$). For $t=0,1,2,\dots$:

$$
\begin{aligned}
&\textbf{(i) Probabilities:}\quad p_j\gets \operatorname{Tr}(\rho_t E_j)\ \ (\text{then floor via (6)}).\\
&\textbf{(ii) Gradient:}\qquad G_t\gets \sum_{j=1}^M\Big(N_{\rm tot}-\frac{n_j}{p_j}\Big)E_j.\\
&\textbf{(iii) Step:}\qquad\quad \tilde \rho\gets \rho_t-\eta_t\,G_t.\\
&\textbf{(iv) Projection:}\quad \rho_{t+1}\gets \Pi_{\rm state}(\tilde \rho).
\end{aligned}
\tag{PG}
$$

**Step size $\eta_t$.** Any standard choice works:

* **Armijo backtracking** (monotone descent): choose $\eta_t$ by halving until

  $$
  \mathcal{L}\!\big(\Pi_{\rm state}(\rho_t-\eta_t G_t)\big)
  \le \mathcal{L}(\rho_t)-c\,\eta_t\,\langle G_t,G_t\rangle
  \quad (c\in(0,1)).
  $$
* **Barzilai–Borwein (BB)**: $\displaystyle \eta_t=\frac{\langle \Delta \rho_{t-1},\Delta \rho_{t-1}\rangle}{\langle \Delta \rho_{t-1},\Delta G_{t-1}\rangle}$ (clipped to $[\eta_{\min},\eta_{\max}]$).
* **Conservative fixed step** when counts are large and well-covered.

**Stopping.** Stop when the relative decrease

$$
\frac{\mathcal{L}(\rho_t)-\mathcal{L}(\rho_{t+1})}{|\mathcal{L}(\rho_t)|+\epsilon}
$$

is below a tolerance, or $\|G_t\|_F$ is small.

**Convergence.** Since $\mathcal{L}$ is convex and smooth on the interior and the feasible set is convex, PG with Armijo (or diminishing steps) produces a sequence $\{\rho_t\}\subset$ feasible that converges to a **global** minimizer of $\mathcal{L}$.

---

# 4. Multinomial variant (optional)

If you prefer exact per-row normalization (useful when the window is very wide), set

$$
\pi_j(\rho)=\frac{p_j(\rho)}{\sum_k p_k(\rho)},\qquad
\mathcal{L}_{\rm Mult}(\rho)=-\sum_j n_j\log \pi_j(\rho).
$$

Then

$$
\nabla \mathcal{L}_{\rm Mult}(\rho)
=\sum_j\Big(\frac{N_{\rm tot}}{\sum_k p_k(\rho)}-\frac{n_j}{p_j(\rho)}\Big)E_j,
$$

and use the same PG template and projection. (When coverage $S(\rho)\approx 1$, this coincides with the Poisson gradient up to a tiny constant.)

---

# 5. Complexity & implementation notes

* **Forward pass.** Computing all $p_j$ via (4) is $O(M d^2)$ with good BLAS locality; histogramming makes runtime depend on the **number of bins** $M$, not on raw shot count.
* **Projection.** One $d\times d$ eigen-decomposition per iteration ($O(d^3)$) and a $d$-dimensional simplex projection (linear-time after sorting).
* **Compared to $R\rho R$.** Both require the same forward pass and an eigen-decomposition; $R\rho R$ additionally computes a **sandwich** $R\rho R$ (two $d\times d$ multiplies with a dense $R$), whereas PG performs a single axpy $ \rho-\eta G$, which is cheaper—hence PG typically achieves faster wall-clock convergence.

---

# 6. Minimal pseudocode (mathematical)

**Precompute**

* Coherent overlaps $c^{(j)}_n$ and matrices $B^{(j)}=c^{(j)}(c^{(j)})^\dagger$.
* POVM elements $E_j=(\Delta A/\pi)B^{(j)}$.

**Loop (PG)**

1. $p_j\leftarrow \max\{\operatorname{Tr}(\rho E_j),\varepsilon\}$.
2. $G\leftarrow \sum_j (N_{\rm tot}-n_j/p_j)\,E_j$.
3. $\tilde\rho\leftarrow \rho-\eta G$.
4. $\rho\leftarrow \Pi_{\rm state}(\tilde\rho)$ via (8).
5. Check NLL; stop on tolerance.

Return $\rho$.

---

# 7. Truncation & modeling error

All operators live in $\mathcal{H}_N$. The recovered $\rho$ is the true state projected onto $\mathcal{H}_N$. If the state’s photon-number distribution is predominantly below $N$, the truncation bias is small; increasing $N$ reduces this bias at $O(d^3)$ eigen-cost per iteration.

---

## Takeaway

The **projected-gradient** method solves the **convex** Poisson MLE for a single state using only: (i) the same fast **histogram-based** forward model you already implemented, and (ii) an exact **projection** to the trace-one PSD cone. It avoids the multiplicative sandwich of $R\rho R$, supports mini-batching/acceleration, and is well-suited to **latency-aware** state tomography.
