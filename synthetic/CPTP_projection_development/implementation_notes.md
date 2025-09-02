# Closed-form notes — projecting a Choi matrix onto the CPTP set (Frobenius metric)

Let $\mathcal H_{\text{in}},\mathcal H_{\text{out}}$ have dimensions $d_{\text{in}},d_{\text{out}}$.
Let $N=d_{\text{in}}d_{\text{out}}$ and identify Choi matrices with $N\times N$ operators on $\mathcal H_{\text{in}}\otimes\mathcal H_{\text{out}}$.

We use the **unnormalized** Choi convention:

$$
\text{CPTP set } \mathcal C=\{X\succeq 0,\ \operatorname{Tr}_{\text{out}}X=I_{\text{in}}\}.
$$

The desired projection is the unique minimizer

$$
\Pi_{\mathcal C}(J)\;=\;\arg\min_{X\in\mathcal C}\ \tfrac12\|X-J\|_F^2,
\qquad \langle A,B\rangle=\operatorname{Tr}(A^\dagger B).
$$

---

## Core linear–algebra identities

1. **Partial trace pairing.** For any $X$ on $\mathcal H_{\text{in}}\otimes\mathcal H_{\text{out}}$ and $\Lambda$ on $\mathcal H_{\text{in}}$,

$$
\langle I_{\text{out}}\!\otimes\!\Lambda,\; X\rangle
=\langle \Lambda,\; \operatorname{Tr}_{\text{out}}X\rangle.
$$

2. **Block form for $\operatorname{Tr}_{\text{out}}$.** If you write $X$ as a $d_{\text{in}}\times d_{\text{in}}$ block matrix with blocks $X_{ij}\in\mathbb C^{d_{\text{out}}\times d_{\text{out}}}$, then

$$
\operatorname{Tr}_{\text{out}}X=\big[\operatorname{Tr}(X_{ij})\big]_{i,j=1}^{d_{\text{in}}}.
$$

3. **Reshape trick.** If you reshape $X$ as $X\in\mathbb C^{d_{\text{in}}\times d_{\text{out}}\times d_{\text{in}}\times d_{\text{out}}}$, then

$$
\big(\operatorname{Tr}_{\text{out}}X\big)_{i,j}=\sum_{k=1}^{d_{\text{out}}} X_{i,k,j,k}.
$$

---

## Closed-form projections onto the two constraint sets

We decompose $\mathcal C=\mathcal A\cap\mathcal K$ where
$\mathcal A=\{X:\operatorname{Tr}_{\text{out}}X=I_{\text{in}}\}$ (affine TP set) and $\mathcal K=\{X\succeq 0\}$ (CP cone).

### A) Orthogonal projection onto $\mathcal A$ (TP constraint)

Given any $Y$, the Frobenius projection $P_{\mathrm{TP}}(Y)$ solves
$\min_X \|X-Y\|_F^2$ s.t. $\operatorname{Tr}_{\text{out}}X=I_{\text{in}}$.
Using the pairing identity, first-order optimality gives

$$
\boxed{\;
P_{\mathrm{TP}}(Y)=Y-\big(I_{\text{out}}\otimes \Delta\big),\qquad
\Delta=\tfrac{1}{d_{\text{out}}}\Big(\operatorname{Tr}_{\text{out}}Y-I_{\text{in}}\Big).
\;}
$$

*(Normalized Choi convention: replace $I_{\text{in}}$ by $I_{\text{in}}/d_{\text{in}}$ in $\Delta$.)*

**Implementation note.** Never explicitly build a Kronecker if memory is tight: “subtract $I_{\text{out}}\otimes\Delta$” means subtract the same $\Delta$ from each of the $d_{\text{out}}$ diagonal output blocks.

### B) Orthogonal projection onto $\mathcal K$ (CP constraint)

Given any $Y$, the Euclidean projection onto the PSD cone is

$$
\boxed{\;
P_{\mathrm{CP}}(Y)=U\,\mathrm{diag}\!\big(\max(\lambda,0)\big)\,U^\dagger,
\;}
$$

where $H=(Y+Y^\dagger)/2=U\,\mathrm{diag}(\lambda)\,U^\dagger$ is a Hermitian eigendecomposition.
(Discard the skew-Hermitian part; **clip** negative eigenvalues.)

**Numerical hygiene.** After clipping, re-symmetrize: $Z\leftarrow(Z+Z^\dagger)/2$. Zero tiny negatives with a threshold (e.g. $10^{-12}$).

---

## Dykstra’s projection onto $\mathcal A\cap\mathcal K$ (all steps closed-form)

**Inputs.** Choi matrix $J\in\mathbb C^{N\times N}$, integers $d_{\text{in}},d_{\text{out}}$, tolerance $\varepsilon$.
**Init.** $X_0=J,\; R_0=0,\; S_0=0$.

For $k=0,1,2,\dots$ repeat:

$$
\begin{aligned}
\text{(1) TP step:}\quad &Y_k = P_{\mathrm{TP}}(X_k+R_k), &&
R_{k+1}=X_k+R_k-Y_k,\\[2pt]
\text{(2) CP step:}\quad &Z_k = P_{\mathrm{CP}}(Y_k+S_k), &&
S_{k+1}=Y_k+S_k-Z_k,\\[2pt]
\text{(3) Update:}\quad &X_{k+1}=Z_k.\;\;
\end{aligned}
$$

**Stop** when, e.g.,

$$
\|\operatorname{Tr}_{\text{out}}X_k-I_{\text{in}}\|_F\le \varepsilon,\quad
\lambda_{\min}(X_k)\ge -\varepsilon,\quad
\frac{\|X_k-X_{k-1}\|_F}{\max(1,\|X_{k-1}\|_F)}\le \varepsilon.
$$

**Guarantee.** For closed convex sets in a Hilbert space, Dykstra converges to the **metric projection** onto $\mathcal A\cap\mathcal K$, i.e. the **nearest** CPTP Choi to $J$ in Frobenius norm. (Plain alternating projections lack this optimality.)

**Work per iteration.**

* One Hermitian eigendecomposition of size $N$ in $P_{\mathrm{CP}}$ (dominant).
* One partial trace + structured correction in $P_{\mathrm{TP}}$.

---

## KKT view (sanity check / certificate)

The projection solves

$$
\min_{X\succeq0}\tfrac12\|X-J\|_F^2\ \ \text{s.t.}\ \ \operatorname{Tr}_{\text{out}}X=I_{\text{in}}.
$$

KKT conditions: $\exists\;\Lambda=\Lambda^\dagger,\;S\succeq0$ such that

$$
X-J+(I_{\text{out}}\!\otimes\!\Lambda)-S=0,\quad
\operatorname{Tr}_{\text{out}}X=I_{\text{in}},\quad
\langle S,X\rangle=0.
$$

Equivalently $X=\big(J-(I_{\text{out}}\!\otimes\!\Lambda)\big)_+$ with $\operatorname{Tr}_{\text{out}}X=I_{\text{in}}$.
Dykstra implicitly searches for the $\Lambda$ that enforces the TP constraint at the PSD **positive-part** solution.

---

## Practical implementation details

* **Hermitization.** At every CP step, set $H=\tfrac12(Z+Z^\dagger)$ before eigen-solve; set $Z\leftarrow\tfrac12(Z+Z^\dagger)$ after clipping.
* **Partial trace efficiently.** Use block traces or reshape-sum (see identities above). Complexity is $O(d_{\text{in}}^2d_{\text{out}})$.
* **Building $I_{\text{out}}\otimes\Delta$ efficiently.** Subtract $\Delta$ from each output-diagonal block of size $d_{\text{out}}\times d_{\text{out}}$; no explicit Kronecker materialization required.
* **Tolerances.** Typical $\varepsilon\in[10^{-8},10^{-6}]$. Clip eigenvalues below $10^{-12}$ to zero to avoid negative “dust.”
* **Scaling & units.** If using the **normalized** Choi ($\operatorname{Tr}_{\text{out}}X=I_{\text{in}}/d_{\text{in}}$), change only the target in $\Delta$.
* **Warm starts.** If projecting a sequence $\{J^{(t)}\}$ (e.g., inside an outer optimization), set $X_0= \Pi_{\mathcal C}(J^{(t-1)})$ to cut iterations.

---

## Special cases with literal one-shot formulas

* **Density matrices** ($d_{\text{in}}=1$).
  Diagonalize $J=U\operatorname{diag}(\lambda)U^\dagger$.
  Project eigenvalues onto the probability simplex:
  $p=\operatorname{proj}_\Delta(\lambda)$, then $X^\star=U\operatorname{diag}(p)U^\dagger$.
* **Block-diagonal “CQ/QC” Chois.**
  If $J=\bigoplus_{a=1}^{d_{\text{in}}} B_a$, the problem decouples into $d_{\text{in}}$ density-matrix projections:
  $B_a^\star=\operatorname{proj}_{\{\succeq0,\ \operatorname{Tr}=1\}}(B_a)$.

---

## Minimal checklist to verify the result

After termination, check:

1. $X=X^\dagger$ (to machine precision).
2. $X\succeq 0$ (all eigenvalues $\ge -\varepsilon$; after clipping, $\ge 0$).
3. $\operatorname{Tr}_{\text{out}}X=I_{\text{in}}$ (Frobenius residual $\le \varepsilon$).
4. $\|X-J\|_F$ is non-increasing across iterations (diagnostic sanity).

---

These notes give all formulas needed to implement a **CPTP projection** numerically: two **closed-form** projections (TP and PSD) combined via **Dykstra’s corrections** to yield the **exact Frobenius projection** onto the CPTP set.
