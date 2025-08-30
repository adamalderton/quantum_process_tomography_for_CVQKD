Below are self-contained, rigorous notes for **projected-gradient (PG) maximum-likelihood estimation of a quantum process** from **histogrammed dual-homodyne (heterodyne) data** in a **Fock-truncated** model. No priors (no Tikhonov, no entropy) are assumed.

---

# 1. Physical and statistical model

## 1.1 Process, Choi matrix, and CPTP constraints

Let $\mathcal{E}:\mathcal{L}(\mathcal{H}_{\rm in})\!\to\!\mathcal{L}(\mathcal{H}_{\rm out})$ be a single-mode quantum channel. Fix a Fock cutoff $N$ and set $d=N{+}1$ for both input and output spaces. The **Choi matrix** (Choi–Jamiołkowski operator) of $\mathcal{E}$ is

$$
J \;=\; (I_{\rm out}\!\otimes\!\mathcal{E})\big(|\Omega\rangle\langle\Omega|\big)\ \in\ \mathcal{L}(\mathcal{H}_{\rm out}\!\otimes\!\mathcal{H}_{\rm in}),
\quad
|\Omega\rangle=\sum_{n=0}^{N}|n\rangle_{\rm out}\!\otimes |n\rangle_{\rm in}.
$$

The channel is **completely positive and trace preserving (CPTP)** iff

$$
J \succeq 0
\quad\text{and}\quad
\operatorname{Tr}_{\rm out} J \,=\, I_{\rm in}.
\tag{CPTP}
$$

(We choose the convention “out $\otimes$ in”.)

The Choi identity giving the channel action is

$$
\mathcal{E}(X)
=\operatorname{Tr}_{\rm in}\!\big[J \,(I_{\rm out}\!\otimes X^{\mathsf T})\big],
\quad\text{equivalently}\quad
\operatorname{Tr}\!\big(Y\,\mathcal{E}(X)\big)
=\operatorname{Tr}\!\big[J\,(Y\!\otimes\!X^{\mathsf T})\big]
\tag{1}
$$

for any $X\in\mathcal{L}(\mathcal{H}_{\rm in})$, $Y\in\mathcal{L}(\mathcal{H}_{\rm out})$, with transpose $(\cdot)^{\mathsf T}$ taken in the input Fock basis.

## 1.2 Dual-homodyne measurement and discretization

A dual-homodyne measurement realises the heterodyne POVM $\{\Pi(\alpha)\}_{\alpha\in\mathbb{C}}$ with

$$
\Pi(\alpha)=\frac{1}{\pi}\,|\alpha\rangle\langle\alpha|.
$$

To process large datasets efficiently, we **histogram** outcomes on a square grid in $(y_1,y_2)\in[-L,L]^2$ with $G$ points per axis and step $\Delta=2L/(G-1)$. We map

$$
\alpha \;=\; \frac{y_1+i y_2}{\sqrt{2}},
\quad\text{so that}\quad
\Delta A \;=\; \frac{\Delta^2}{2}
$$

is the area element in $\alpha$-space. Each grid point (bin) $\alpha_j$ corresponds to the discretized POVM element

$$
E_j \;=\; \frac{\Delta A}{\pi}\,|\alpha_j\rangle\langle\alpha_j|\ \succeq\ 0,
\quad j=1,\dots,M,\ \ M=G^2.
\tag{2}
$$

## 1.3 Probe set and binned data

Let $\{|\alpha_i\rangle\}_{i=1}^S$ be the **input coherent probes**. For each input $i$, we collect $N_i$ shots and histogram the output onto the grid, giving **counts** $n_{ij}\in\mathbb{N}$ with $\sum_j n_{ij}=N_i$.

By (1) and (2), the **model bin probability (mass)** for input $i$, bin $j$ is

$$
p_{ij}(J)\;=\;\operatorname{Tr}\!\big[J\,F_{ij}\big],
\qquad
F_{ij}\;:=\;E_j\otimes |\alpha_i\rangle\langle\alpha_i|^{\mathsf T}\ \succeq\ 0.
\tag{3}
$$

Thus $p_{ij}$ is **affine** in $J$ and nonnegative for all CPTP $J$.

## 1.4 Likelihood

We use the **Poisson factorization** for histogrammed counts (one may also use per-row multinomial; both are equivalent in practice):

$$
n_{ij}\sim \mathrm{Poisson}\!\big(\mu_{ij}\big),\qquad
\mu_{ij}=N_i\,p_{ij}(J).
$$

Ignoring constants, the **negative log-likelihood (NLL)** is

$$
\mathcal{L}(J)
=\sum_{i=1}^S\sum_{j=1}^M \Big[\,N_i\,p_{ij}(J)\;-\;n_{ij}\log p_{ij}(J)\,\Big].
\tag{4}
$$

Because $p_{ij}(J)=\operatorname{Tr}(J F_{ij})$ is affine, each term $N_i p - n\log p$ is convex in $p>0$; hence:

> **Proposition 1 (Convexity).** $\mathcal{L}(J)$ is convex on the convex set $\{J\succeq 0,\ \operatorname{Tr}_{\rm out}J=I\}$.

Therefore any local minimum is global.

---

# 2. Efficient forward model in the Fock basis

Write coherent overlaps

$$
a^{(i)}_n=\langle n|\alpha_i\rangle
= e^{-|\alpha_i|^2/2}\,\frac{\alpha_i^n}{\sqrt{n!}},\quad
b^{(j)}_n=\langle n|\alpha_j\rangle.
$$

Define rank-1 Fock matrices

$$
A^{(i)}\in\mathbb{C}^{d\times d},\quad A^{(i)}_{nm}=a^{(i)}_n\,a^{(i)\,*}_m,
\qquad
B^{(j)}\in\mathbb{C}^{d\times d},\quad B^{(j)}_{pq}=b^{(j)\,*}_p\,b^{(j)}_q,
$$

so $|\alpha_i\rangle\langle\alpha_i|=(A^{(i)})$ and $E_j=(\Delta A/\pi)\,B^{(j)}$. Using (3),

$$
p_{ij}(J)
=\frac{\Delta A}{\pi}\sum_{m,n,p,q=0}^{N}
J_{(mn),(pq)}\,A^{(i)}_{nm}\,B^{(j)}_{pq}.
\tag{5}
$$

This can be evaluated with one or two `einsum`-style contractions. **Histogramming** makes the cost scale with the number of **bins** ($S\times M$), not with the number of **shots** ($\sum_i N_i$).

---

# 3. Projected-gradient (PG) MLE

## 3.1 Gradient

From (4) and (3),

$$
\nabla \mathcal{L}(J)
=\sum_{i,j}\Big(N_i-\frac{n_{ij}}{p_{ij}(J)}\Big)F_{ij}.
\tag{6}
$$

(Use the Hilbert–Schmidt inner product $\langle X,Y\rangle=\operatorname{Tr}(X^\dagger Y)$.)

The Hessian is the positive semidefinite linear operator

$$
\nabla^2 \mathcal{L}(J)[H]
=\sum_{i,j}\frac{n_{ij}}{p_{ij}(J)^2}\,\operatorname{Tr}(H F_{ij})\,F_{ij},
\tag{7}
$$

confirming convexity. For numerical stability, we **floor** the probabilities

$$
p_{ij}(J)\leftarrow \max\{p_{ij}(J),\,\varepsilon\},
\quad \varepsilon>0,
\tag{8}
$$

to avoid $\log 0$ and division by zero.

## 3.2 Projection to the CPTP set

Given an arbitrary Hermitian $X$, we project in two steps:

**(a) PSD clamp.** Eigendecompose $X=V\Lambda V^\dagger$, set $\Lambda\leftarrow \max(\Lambda,0)$, and reconstruct $\hat X\succeq 0$.

**(b) Trace-preserving renormalization.** Let

$$
\sigma \;=\; \operatorname{Tr}_{\rm out}\hat X\ \in\ \mathbb{C}^{d\times d}.
$$

If $\sigma\succ 0$, set

$$
\Pi_{\rm TP}(\hat X)
\;=\; (I_{\rm out}\!\otimes \sigma^{-1/2})\,\hat X\,(I_{\rm out}\!\otimes \sigma^{-1/2}).
\tag{9}
$$

Then $\Pi_{\rm TP}(\hat X)\succeq 0$ and $\operatorname{Tr}_{\rm out}\Pi_{\rm TP}(\hat X)=I_{\rm in}$.
If $\sigma$ is not strictly PD, replace $\sigma\leftarrow \sigma+\delta I$ with a tiny $\delta>0$.

*(Remark.)* The linear **orthogonal projection** onto the affine space $\{X:\operatorname{Tr}_{\rm out}X=I\}$ is

$$
\Pi_{\rm aff}(X)=X-\frac{1}{d_{\rm out}}\big(\operatorname{Tr}_{\rm out}X-I\big)\otimes I_{\rm out},
$$

but it does not preserve positivity in general; (9) is preferred because it preserves PSD by congruence.

## 3.3 The PG iteration

Given an initial $J_0$ satisfying (CPTP) (e.g. maximally mixed Choi), repeat for $t=0,1,2,\dots$:

$$
\begin{aligned}
&\textbf{(i) Probabilities:}\quad p_{ij}\gets \operatorname{Tr}(J_t F_{ij})\ \ (\text{then floor by (8)}).\\
&\textbf{(ii) Gradient:}\qquad G_t\gets \sum_{i,j}\Big(N_i-\frac{n_{ij}}{p_{ij}}\Big)F_{ij}.\\
&\textbf{(iii) Step:}\qquad\quad \tilde J\gets J_t-\eta_t\,G_t.\\
&\textbf{(iv) Projection:}\quad J_{t+1}\gets \Pi_{\rm TP}\big(\Pi_{\rm PSD}(\tilde J)\big).
\end{aligned}
\tag{PG}
$$

**Step size.** Any of the following ensures descent and convergence to the MLE:

* **Armijo backtracking:** choose $\eta_t$ by halving until

  $$
  \mathcal{L}\!\big(\Pi_{\rm TP}\!\circ\!\Pi_{\rm PSD}(J_t-\eta_t G_t)\big)
  \;\le\; \mathcal{L}(J_t)\;-\;c\,\eta_t\,\langle G_t,\,G_t\rangle
  $$

  holds for some $c\in(0,1)$.
* **Barzilai–Borwein (BB):** set

  $$
  \eta_t=\frac{\langle \Delta J_{t-1},\Delta J_{t-1}\rangle}{\langle \Delta J_{t-1},\Delta G_{t-1}\rangle}
  \quad\text{clipped to }[\eta_{\min},\eta_{\max}],
  $$

  with $\Delta J_{t-1}=J_{t}-J_{t-1}$, $\Delta G_{t-1}=G_{t}-G_{t-1}$.
* **Conservative fixed step:** small $\eta$ (empirically tuned) often suffices thanks to projection.

**Stopping.** Terminate when the relative NLL decrease

$$
\frac{\mathcal{L}(J_t)-\mathcal{L}(J_{t+1})}{|\mathcal{L}(J_t)|+\epsilon}
$$

falls below a tolerance, or when $\|G_t\|_F$ is small.

> **Theorem 1 (Convergence).** With Armijo backtracking (or any standard diminishing-stepsize rule), the PG sequence $\{J_t\}$ remains in the CPTP set and converges to a global minimizer of $\mathcal{L}$ over CPTP.
> *Sketch.* $\mathcal{L}$ is convex and continuously differentiable on the relative interior of the CPTP set; the projection mapping is nonexpansive; Armijo guarantees sufficient decrease and summability of steps; cluster points exist and satisfy first-order optimality, hence global optimality by convexity.

---

# 4. Complexity and implementation notes

* **Forward evaluation (shared by all methods).** Computing all $p_{ij}$ via (5) dominates when $S\!\times\!M$ is large; histogramming makes this scale with the number of **bins**, not shots.
* **PG vs multiplicative EM.** Both PG and EM require the **same** probability pass and the **same** CPTP projection. EM additionally computes a **sandwich** $RJR$ (two dense $D\times D$ multiplies with $D=d^2$), which is $O(D^3)$ and quickly dominates for moderate $d$. PG **avoids** those multiplies, which is why it is typically faster per unit progress.
* **Mini-batch PG.** You may replace the full $(i,j)$ sums in (PG-ii) by a random subset (“mini-batch”) to cut wall-clock time drastically. Backtracking or slightly smaller fixed steps keep it stable.
* **Numerical guards.** Always floor $p_{ij}$ by $\varepsilon$ (8); Hermitize any accumulated matrices to suppress round-off asymmetry; add a tiny $\delta I$ before $\sigma^{-1/2}$ in (9) if needed.

---

# 5. Multinomial variant (optional)

If you prefer per-input normalization, set

$$
\pi_{ij}(J)=\frac{p_{ij}(J)}{\sum_{k}p_{ik}(J)},\qquad
\mathcal{L}_{\rm Mult}(J)=\sum_{i}\sum_{j}-\,n_{ij}\log \pi_{ij}(J).
$$

Then

$$
\nabla \mathcal{L}_{\rm Mult}(J)=\sum_{i}\sum_{j}\Big(\tfrac{N_i}{\sum_k p_{ik}}-\tfrac{n_{ij}}{p_{ij}}\Big)\,F_{ij},
$$

and the same PG template and projection apply verbatim. (This variant enforces per-row normalisation but is slightly more coupled across bins within each $i$.)

---

# 6. Relation to state tomography pieces you already have

* Your coherent-state matrix builder already yields the vectors $a^{(i)}$, $b^{(j)}$ and hence $A^{(i)}$, $B^{(j)}$ used in (5).
* Your state $R\rho R$ code uses the same discretized POVM $E_j$ and the same $\Delta A$ bookkeeping; PG reuses all of that.
* Your PSD projection routine generalises directly; the only addition is the **trace-preserving** congruence (9).

---

# 7. Minimal pseudocode (mathematical)

**Precompute**

* $A^{(i)}=a^{(i)}(a^{(i)})^\dagger$ for inputs $i=1..S$.
* $B^{(j)}=b^{(j)}(b^{(j)})^\dagger$ and $E_j=(\Delta A/\pi)B^{(j)}$ for bins $j=1..M$.
* $F_{ij}=E_j\otimes (A^{(i)})^{\mathsf T}$ implicitly (never form explicitly; use contractions).

**Loop**

1. $p_{ij}\leftarrow \frac{\Delta A}{\pi}\sum_{mnpq}J_{(mn),(pq)}A^{(i)}_{nm}B^{(j)}_{pq}$, then $p_{ij}\leftarrow \max(p_{ij},\varepsilon)$.
2. $G\leftarrow \sum_{i,j}\big(N_i-\frac{n_{ij}}{p_{ij}}\big)F_{ij}$ (accumulate via rank-1 structure).
3. $\tilde J\leftarrow J-\eta G$.
4. PSD clamp: $\hat J=\Pi_{\rm PSD}(\tilde J)$.
5. TP renorm: $\sigma=\operatorname{Tr}_{\rm out}\hat J$, $J\leftarrow (I\otimes \sigma^{-1/2})\hat J(I\otimes \sigma^{-1/2})$.
6. Check NLL decrease; stop when below tolerance.

---

# 8. Truncation and modeling error

All operators live in the truncated space $\mathcal{H}_N$. The recovered $J$ therefore represents $\mathcal{E}$ **projected** to that subspace. If the true channel preserves low photon numbers (typical in weakly-noisy links), the truncation bias is small; otherwise, increasing $N$ reduces the bias at the cost of $D=d^2$ scaling in the projection/EVD steps.

---

## Takeaway

* The PG method solves the **convex** MLE problem over the **convex** CPTP set using only: (i) the same fast, histogram-based forward model you already use, and (ii) a **PSD clamp + TP renormalization** projection.
* It avoids the expensive multiplicative **sandwich** $RJR$ of EM, supports **mini-batches**, and converges reliably with standard step-rules—making it well-suited to **latency-aware** environments.
