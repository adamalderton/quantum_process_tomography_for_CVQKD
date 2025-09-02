totally — let’s rebuild the “Stiefel/Kraus” approach from the ground up, slowly and self-contained. i’ll define every object, show why this keeps you **CPTP at every iterate**, and give you a clean step-by-step recipe you can drop into your optimizer.

# 0) symbols and shapes

* $d$: Hilbert space dimension (your Fock cutoff).
* $r$: chosen Kraus rank (an integer you pick; start small).
* Inputs: coherent states $\{\lvert\alpha_i\rangle\}_{i=1}^S$ with vectors $a_i\in\mathbb{C}^d$, projectors $A_i=a_i a_i^\dagger$.
* Outputs (bins): $\{\lvert\beta_j\rangle\}_{j=1}^M$ with vectors $b_j\in\mathbb{C}^d$, POVM elements $B_j=b_j b_j^\dagger$.
* Heterodyne scale $c>0$ (often $\Delta A/\pi$).
* Counts $n_{ij}$ with exposure $N_i$.
* We estimate the quantum channel $\mathcal{E}$.

# 1) channels, Choi matrices, CPTP

A quantum channel $\mathcal{E}$ (from $d$ to $d$) is **completely positive and trace preserving (CPTP)**. Its **Choi matrix** $J\in\mathbb{C}^{d^2\times d^2}$ encodes the map and satisfies

$$
J \succeq 0 \quad\text{(CP)},\qquad \mathrm{Tr}_{\text{out}}(J)=I_d \quad\text{(TP)}.
$$

Given input $A_i$ and output effect $B_j$, the probability predicted by $J$ is

$$
p_{ij}(J)=c\,\mathrm{Tr}\!\big[J\,(B_j\otimes A_i^{\mathsf T})\big].
$$

# 2) Kraus form and how Stiefel enters

Every CPTP channel admits a **Kraus decomposition**

$$
\mathcal{E}(\rho)=\sum_{k=1}^r K_k\,\rho\,K_k^\dagger,\qquad K_k\in\mathbb{C}^{d\times d},
$$

with the **trace-preserving** constraint

$$
\sum_{k=1}^r K_k^\dagger K_k = I_d. \tag{TP}
$$

Now **stack** the Kraus operators vertically into one tall matrix:

$$
V=\begin{bmatrix}K_1\\K_2\\\vdots\\K_r\end{bmatrix}\in \mathbb{C}^{(rd)\times d}.
$$

Then the TP condition above is exactly

$$
V^\dagger V = I_d .
$$

The set of all complex $ (rd)\times d$ matrices with orthonormal columns is the **(complex) Stiefel manifold**:

$$
\mathrm{St}(rd,d)\;=\;\{V\in\mathbb{C}^{(rd)\times d}:V^\dagger V=I_d\}.
$$

**Key point:** if you constrain your optimization variable to live on $\mathrm{St}(rd,d)$, you automatically satisfy TP at every iterate. And **CP is automatic** too, because the Choi matrix built from Kraus operators is a Gram matrix:

$$
J=\sum_{k=1}^r \mathrm{vec}(K_k)\,\mathrm{vec}(K_k)^\dagger \;\succeq\;0 .
$$

(Here $\mathrm{vec}(\cdot)$ stacks matrix columns into a vector.)

So: **optimize over $V\in\mathrm{St}(rd,d)$** instead of over $J$ with projections. Every $V$ corresponds to a **CPTP** channel; no post-step CPTP projection is needed.

# 3) forward probabilities without forming $J$

You never need to materialize $J$ during training. Using the identity
$\mathrm{vec}(X)^\dagger(B\otimes A^{\mathsf T})\mathrm{vec}(X)=\mathrm{Tr}\!\left(A\,X^\dagger B X\right)$,

$$
\boxed{ \quad
p_{ij}(V) \;=\; c\sum_{k=1}^r \mathrm{Tr}\!\big(A_i\,K_k^\dagger B_j K_k\big)
\;=\; c\sum_{k=1}^r \big|\,b_j^\dagger K_k a_i\,\big|^2.
\quad}
$$

That last form is fast and numerically friendly: it’s just squared moduli of complex inner products.

# 4) the loss and its gradient w\.r.t. $K_k$

With Poisson counts $n_{ij}\sim\mathrm{Poisson}(N_i p_{ij})$, the negative log-likelihood (dropping constants) is

$$
\mathcal{L}(V)=\sum_{i,j}\Big(N_i\,p_{ij}(V)-n_{ij}\log p_{ij}(V)\Big).
$$

Define the weights

$$
W_{ij}=N_i-\frac{n_{ij}}{p_{ij}(V)}.
$$

Collect, for each input $i$, the **output-side accumulator**

$$
S_i \;=\; \sum_{j=1}^M W_{ij}\,B_j\quad \in \mathbb{C}^{d\times d}.
$$

Then the Euclidean gradient w\.r.t. each Kraus operator is

$$
\boxed{\quad
\nabla_{K_k}\mathcal{L}(V)\;=\;2c \sum_{i=1}^S S_i\,K_k\,A_i ,
\quad}
$$

where $A_i=a_i a_i^\dagger$.
(Reason: $p_{ij}$ is a sum of $|b_j^\dagger K_k a_i|^2$; differentiate that quadratic in $K_k$ using standard complex/Wirtinger calculus. The factor 2 appears because the loss is real.)

Stack these block-gradients to make the Euclidean gradient $G_V\in\mathbb{C}^{(rd)\times d}$ for the tall variable $V$.

# 5) optimizing **on** the Stiefel manifold

You can’t just do $V\leftarrow V-\alpha G_V$, because that would break $V^\dagger V=I$. Use **Riemannian gradient descent**:

**(a) Project to the tangent space.**
The tangent space at $V$ is $T_V\mathrm{St}=\{Z: V^\dagger Z + Z^\dagger V=0\}$.
Project the Euclidean gradient $G_V$ to get the **Riemannian gradient**:

$$
\boxed{\quad
\mathrm{grad}\,\mathcal{L}(V)\;=\; G_V - V\,\mathrm{sym}(V^\dagger G_V),\qquad
\mathrm{sym}(X)=\tfrac{1}{2}(X+X^\dagger).
\quad}
$$

**(b) Take a step in the tangent direction.**
Form a trial update $\tilde V = V - \alpha\,\mathrm{grad}\,\mathcal{L}(V)$ with step size $\alpha>0$ (BB/Armijo, etc.).

**(c) Retract back to the manifold.**
Use the **polar retraction** (numerically robust):

$$
\boxed{\quad
V_{\text{new}} \;=\; \tilde V\,\big(\tilde V^\dagger \tilde V\big)^{-1/2}.
\quad}
$$

Equivalently, you can QR-orthonormalize the columns: $\tilde V=QR$ with $Q^\dagger Q=I$; take $V_{\text{new}}=Q$.

This guarantees $V_{\text{new}}^\dagger V_{\text{new}}=I$ (**TP preserved**), and since the channel defined by any $V$ is Kraus-form CP, every iterate remains **CPTP**.

# 6) full “recipe” you can implement

**inputs:** $a_i, b_j, N_i, n_{ij}, c, d$.
**hyper-param:** Kraus rank $r$.

**init:**

* Set $K_1=I_d$, $K_{k>1}=0$.
* Stack into $V=[K_1;\dots;K_r]\in\mathrm{St}(rd,d)$.

**loop:** until convergence

1. **Forward probs:**
   $p_{ij}= c\sum_{k=1}^r \big|\,b_j^\dagger K_k a_i\,\big|^2$ (clip by an $\epsilon$ if needed).
   Loss $\mathcal{L}=\sum_{i,j}(N_i p_{ij}-n_{ij}\log p_{ij})$.
2. **Weights & accumulators:**
   $W_{ij}=N_i-\dfrac{n_{ij}}{p_{ij}}$, then $S_i=\sum_j W_{ij} B_j$.
3. **Euclidean gradient blocks:**
   $\nabla_{K_k}=2c \sum_i S_i K_k A_i$ for $k=1,\dots,r$.
   Stack to $G_V$.
4. **Riemannian gradient:**
   $\mathrm{grad}\,\mathcal{L}=G_V - V\,\mathrm{sym}(V^\dagger G_V)$.
5. **Step + retraction:**
   Choose $\alpha$ (e.g., Barzilai–Borwein with backtracking);
   $\tilde V = V - \alpha\,\mathrm{grad}\,\mathcal{L}$;
   $V\leftarrow \tilde V (\tilde V^\dagger \tilde V)^{-1/2}$.
6. **Stop** when relative loss improvement or $\|\mathrm{grad}\,\mathcal{L}\|_F$ is small.

**return:** either the Kraus operators $\{K_k\}$ (they’re your model), or the Choi $J=\sum_k \mathrm{vec}(K_k)\mathrm{vec}(K_k)^\dagger$.

# 7) practical notes

* **Why this stays CPTP:**
  – TP: enforced by $V^\dagger V=I$ and retraction;
  – CP: automatic from Kraus/Gram construction.
* **Rank $r$:** start with $r=1$ or $2$. If the fit stalls high, increase $r$.
* **Costs:** the heavy ops are matrix–vector products to form $p_{ij}$ and the retraction (a $d\times d$ matrix inverse square root). No eigen-projections, no Dykstra.
* **Numerics:** keep $p_{ij}\ge \epsilon$ to avoid $\log 0$; you can reuse the same $\epsilon$ you use now.

---

that’s the whole Stiefel story: **optimize over a tall, column-orthonormal matrix $V$** whose blocks are the Kraus operators. orthonormal columns $\Leftrightarrow$ TP; Kraus $\Rightarrow$ CP. optimize with a **Riemannian gradient + retraction**, and every iterate is **CPTP by construction**.

---------------------

Yes—what you’re running is (finite-dimensional, coarse-grained) **maximum-likelihood quantum process tomography**. You’re estimating the **Choi matrix** $\hat J$ of an unknown CPTP map $\mathcal E$ from preparation/measurement data, and you’re doing it in a way that keeps every iterate CPTP.

Here’s the clean mapping to quantum channel estimation.

# 1) Observation model (Born rule in Choi form)

Prepare probe states $A_i=\lvert\alpha_i\rangle\!\langle\alpha_i\rvert$ (coherent states, truncated to $d$ Fock levels).
Measure with a discretized heterodyne POVM on a $G\times G$ grid: $E_j=\tfrac{\Delta A}{\pi}\,\lvert\beta_j\rangle\!\langle\beta_j\rvert$.

For a CPTP channel with Choi matrix $J\in\mathbb C^{d^2\times d^2}$,

$$
p_{ij}(J)=\operatorname{Tr}\!\big[J\,(E_j\otimes A_i^{\mathsf T})\big]
\quad\text{(Born rule in Choi picture).}
$$

You collect counts $n_{ij}\sim \text{Poisson}(N_i\,p_{ij})$. The **negative log-likelihood** is

$$
\mathcal L(J)=\sum_{i,j}\Big(N_i\,p_{ij}(J)-n_{ij}\log p_{ij}(J)\Big).
$$

# 2) The estimation problem (MLE over the CPTP set)

The “true” MLE is

$$
\min_{J}\ \mathcal L(J)\quad\text{s.t.}\quad J\succeq 0,\ \ \operatorname{Tr}_{\rm out}J=I_d.
$$

Two important facts:

* $\mathcal L$ is **convex in $J$** (it’s linear minus a sum of $\log$ composed with an affine map of $J$), and the CPTP set is convex ⇒ the MLE in $J$ is a convex program with a unique global optimum (on your truncated/binned model).
* Identifiability: if the operators $\{E_j\otimes A_i^{\mathsf T}\}$ **span** the Hermitian space on $\mathbb C^{d}\otimes\mathbb C^{d}$ (informational completeness within the truncation), the MLE $\hat J$ is unique in that space. With coherent probes and a sufficiently dense heterodyne grid, you’re *approximately* informationally complete on the chosen cutoff $d$.

# 3) Why you see a Choi matrix even though you optimize over Kraus

In the **Stiefel/Kraus** implementation you used, you don’t store $J$ directly. You parameterize the channel by Kraus operators $\{K_k\}_{k=1}^r$ stacked into a tall matrix $V$ with $V^\dagger V=I_d$ (the complex **Stiefel** constraint). That ensures:

* **TP**: $V^\dagger V=I_d$ ⇔ $\sum_k K_k^\dagger K_k=I_d$.
* **CP**: $J(V)=\sum_k \mathrm{vec}(K_k)\mathrm{vec}(K_k)^\dagger\succeq 0$.

The **forward model** is then

$$
p_{ij}(V)=\frac{\Delta A}{\pi}\sum_{k=1}^r\left|\,b_j^\dagger K_k\, a_i\,\right|^2,
$$

and you minimize $\mathcal L(V)=\mathcal L(J(V))$ with **Riemannian gradient descent** on the Stiefel manifold (tangent projection + retraction). When you finish, you can form the **estimated Choi**

$$
\hat J \;=\; \sum_{k=1}^r \mathrm{vec}(\hat K_k)\,\mathrm{vec}(\hat K_k)^\dagger .
$$

So yes—the output of your run is exactly the **Choi matrix that best explains your data** (within the truncation/grid and chosen Kraus rank).