# Mathematics of Incremental Computation: Foundations for VNNUE

Research notes for the VNNUE ("Very easily updatable Neural Network") paper.
Scope: the algebraic, numerical and algorithmic facts that explain why NNUE can be updated cheaply, and which other known results could give more expressive evaluation per unit of update cost.

Tags used below:
- **[V]**: bibliographic data checked online during this session (publisher, DBLP, ACM DL, Project Euclid, VLDB site).
- **[K]**: standard textbook fact or a citation I am confident of but did not re-check this session. The paper should re-check these before submission.
- **[UNVERIFIED]**: a detail (usually an exact engine constant or a Mathlib lemma name) that has to be checked before it is used.

Notation:
- x ∈ {0,1}^n is the sparse binary feature vector of a position.
- W ∈ R^{M×n} is the feature-transformer matrix, with columns w_i.
- b is the bias, and A(x) = b + Wx is the accumulator.
- A move m sends x to x' = x + Δx. The removed set R and the added set S are the supports of the negative and positive parts of Δx.

---

## 0. Baseline: what NNUE actually does (facts to anchor everything else)

- Origin: Yu Nasu, "Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi", appeal document for the 28th World Computer Shogi Championship (2018). Original in Japanese; English translation by D. Klein (2021). [V]
- In the Stockfish nnue-pytorch documentation (official-stockfish/nnue-pytorch, `docs/nnue.md`) [V]:
  - The simple "A" feature set has 64·6·2 = 768 inputs. A legal position has at most 32 of them nonzero.
  - "A move can change at most 4 inputs (castling), and the average should be below 3."
  - HalfKP has 64·64·5·2 = 40960 features, indexed by (own king square, piece square, piece type, colour). The figure 41024 is a shogi leftover that adds 64 unused features.
  - When the king moves, every HalfKP feature of that perspective changes, so the accumulator is refreshed from scratch.
  - There are two accumulators, one per perspective. Sometimes one is updated while the other is refreshed.
  - Update rule: copy the previous accumulator, subtract the columns w_i for i ∈ R, add the columns w_j for j ∈ S. Refresh rule: start from b and add w_i for every active i.
  - With floats, rounding error builds up, but accumulators are stored on the search stack and never "un-updated". The error is therefore O(MAX_DEPTH). With int16 quantization the updates are exact, and the only risk is overflow.
  - Architecture history: 256x2→32→32→1 changed later to 1024x2→8→32→1. Input sparsity is about 0.1%. Widths above 256 need several passes because AVX2 has too few registers.
- HalfKAv2_hm (the later Stockfish feature set): 32 king buckets with horizontal mirroring × 11 piece planes (the two kings merged into one plane) × 64 squares = 22528 features per perspective. [UNVERIFIED: re-check against Stockfish `src/nnue/features/half_ka_v2_hm.h`]
- Recent large-net L1 widths (2560 to 3072) and the "Finny table" accumulator-refresh cache, which stores per-king-bucket accumulators and the board each was computed for, so a refresh becomes a diff against the cached board. [UNVERIFIED: cite Stockfish commits/release notes]

Cost model used throughout (per perspective, in vector adds of length M):
- incremental update: |R| + |S|, which is 2 to 4 for chess and below 3 on average;
- full refresh: |active features|, at most 32 (HalfKA) or 30 (HalfKP, which excludes kings).
- The speed-up is about 32/3 ≈ 10× on the first layer. The first layer dominates when M ≫ width of L2.

---

## 1. Linear maps and abelian-group homomorphisms

**Theorem 1.1 (linearity ⇒ O(|Δx|) update).** Let φ : Z^n → G be a homomorphism into an abelian group G (e.g. G = Z^M or R^M with φ(x) = Wx). Then for any x and x':

  φ(x') = φ(x) + φ(x' − x) = φ(x) + Σ_{j∈S} φ(e_j) − Σ_{i∈R} φ(e_i).

The proof is one line, from additivity. The cost is |R| + |S| group operations and does not depend on n or on the number of active features. [K: elementary algebra]

Remarks that matter for VNNUE:

1. **Only additivity and inverses are used.** The accumulator can be any abelian group or module: Z^M with wraparound (int16), (Z/2)^64 (Zobrist), R^M, a tropical semiring, and so on.
   - Removal needs inverses. In a commutative *monoid* that is not a group (for example (R, max) or a bitwise OR), additions can be done incrementally but removals cannot. This is the formal reason NNUE needs a linear (sum) aggregator and cannot use max-pooling over pieces.
2. **Commutativity is what removes the ordering cost.** For a non-commutative group G with an aggregate Π_{i in order} g_i^{x_i}, changing one factor in the middle needs either recomputation or a balanced-tree (segment-tree) structure with O(log n) work per change. Fenwick trees (Section 6) also need invertibility. Rule: commutative + invertible ⇒ O(1) per change; associative only ⇒ O(log n) per change.
3. **Exactness under quantization.** In Z/2^16 Z the update is exact even after intermediate overflow, because the homomorphism property holds in the quotient ring. What matters is that the *true* value fits in int16 when it is read through the activation. This is a clean Lean target: `ZMod (2^16)` is a commutative ring, and casting from ℤ is a ring homomorphism.
4. **Path independence.** Since φ(x) depends only on x, any sequence of moves that returns to the same x gives the same accumulator in exact arithmetic. This is the property Zobrist hashing relies on (Section 3). It does **not** hold for floats; the O(depth) error bound in §0 comes from this.

---

## 2. Finite differencing (Paige & Koenig 1982) and the calculus of changes

**Paige & Koenig (1982)**, "Finite Differencing of Computable Expressions", ACM TOPLAS 4(3):402–454. [K]
- Idea: keep a costly expression E = f(x₁,…,x_k) available as a stored variable while its inputs change. Each modification dx_i comes with a *derivative* code fragment that restores E for O(|dx|) work instead of recomputing f.
- This generalizes classical strength reduction (Cocke, Allen, Kennedy) from arithmetic to set-theoretic SETL expressions.
- Their key rules are the "chain rules" for composing derivatives, and the observation that sums, set cardinalities and set unions with counts have cheap derivatives.

**Cai, Giarrusso, Rendel, Ostermann (2014)**, "A Theory of Changes for Higher-Order Languages: Incrementalizing λ-Calculi by Static Differentiation", PLDI 2014, pp. 145–155, DOI 10.1145/2594291.2594304; arXiv:1312.0658. [V]

- **Change structure:** a tuple (V, Δ, ⊕, ⊖), where Δv is the set of valid changes for v, ⊕ : (v:V) → Δv → V, and ⊖ : V → (v:V) → Δv, with the law v ⊕ (u ⊖ v) = u.
- **Derivative of f : A → B:** a map f′ : (a:A) → Δa → Δ(f a) such that

    f(a ⊕ da) = f(a) ⊕ f′(a, da).

- **Main result:** a static program transformation `Derive` that maps every simply-typed λ-term to its derivative, proved correct (machine-checked in Agda). Function types get a change structure pointwise: df ∈ Δ(A→B) with (f ⊕ df)(a) = f(a) ⊕ df(a, 0_a).
- **Chain rule** (Lean-checkable): (g ∘ f)′(a, da) = g′(f a, f′(a, da)).

**Polynomial finite differences.** For p ∈ R[x] of degree d, the forward difference Δ_h p(x) = p(x+h) − p(x) has degree d − 1 (leading coefficient d·h·c_d), and Δ_h^{d+1} p = 0. [K: classical; Newton 1687 / Boole 1860]

Multivariate Boolean form, which is the one that matters here:
- Every f : {0,1}^n → R has a unique multilinear representation f(x) = Σ_{T⊆[n]} c_T Π_{i∈T} x_i (Möbius inversion / Fourier–Walsh expansion; see O'Donnell 2014, *Analysis of Boolean Functions*, CUP). [K]
- Flipping x_j from 0 to 1 changes f by the discrete derivative

    ∂_j f(x) = Σ_{T∋j} c_T Π_{i∈T\{j}} x_i,

  which is multilinear of degree deg f − 1 and does not depend on x_j.
- **Consequence:** a degree-d evaluation needs, per changed feature, the evaluation of a degree-(d−1) function. NNUE's first layer is the case d = 1, where the derivative is a constant column w_j.
- For d = 2 the derivative is *linear* in x, so it can itself be maintained by an accumulator. This is the step from NNUE to factorization machines (Section 7).

---

## 3. Zobrist hashing (1970) as a homomorphism

**Zobrist (1970)**, "A New Hashing Method with Application for Game Playing", Tech. Rep. #88, Computer Sciences Dept., University of Wisconsin, April 1970; reprinted in ICCA Journal 13(2):69–73, 1990. [K]

- Each (piece, square) feature i gets a random key z_i ∈ (Z/2)^64, and H(x) = ⊕_i x_i z_i.
- H is a group homomorphism (Z/2)^n → (Z/2)^64, so the incremental update is H(x') = H(x) ⊕ ⊕_{i ∈ R∪S} z_i. In characteristic 2, add and remove are the same operation.
- **Collision bound:** for independent uniform keys and x ≠ y, Pr[H(x) = H(y)] = 2^{−64}. The reason is that H(x) ⊕ H(y) = H(x ⊕ y) with x ⊕ y ≠ 0, and a nonzero XOR of uniform keys is uniform.
- **Lesson for VNNUE:** NNUE's accumulator is "Zobrist over R instead of over GF(2)". It is the same homomorphism pattern with learned keys instead of random ones.
- A random-projection reading also links NNUE to the Johnson–Lindenstrauss lemma (Johnson & Lindenstrauss 1984, Contemp. Math. 26:189–206 [K]). A random W ∈ R^{M×n} with M = O(ε^{−2} log N) roughly preserves the distances between any N positions, which says a wide first layer is "free" capacity in the hash sense.

---

## 4. Incremental and self-adjusting computation (Acar et al.)

- Acar, Blelloch, Harper, "Adaptive Functional Programming", POPL 2002, pp. 247–259. Journal version in ACM TOPLAS 28(6):990–1034, 2006. [K]
- Acar's PhD thesis, "Self-Adjusting Computation", CMU 2005. [K]

**Mechanism.** At run time the program records a *dynamic dependence graph* of modifiable references. When an input changes, *change propagation* re-executes only the affected reads, in topological or time order.
- The cost is proportional to the trace distance between the old and new executions (Acar's "trace stability"). It is not proportional to the input size.
- Memoization and time-ordered propagation (order-maintenance data structures; Dietz & Sleator 1987, STOC) make the overhead O(1) amortized per affected node, up to log factors depending on the variant.

**Application: delta propagation past the first layer.** Let a = A(x) and h = σ(a), with σ = ClippedReLU (clamp to [0, 127] after scaling). The next layer is y = W₂ h + b₂. Then

  Δy = W₂ (σ(a + Δa) − σ(a)),

and σ(a + Δa)_k − σ(a)_k = 0 whenever Δa_k = 0, *or* when a_k and a_k + Δa_k fall in the same saturated region (both ≤ 0 or both ≥ 127).

- So nnz(Δh) ≤ nnz(Δa) ≤ M, and the update costs O(nnz(Δh) · |L2|) instead of O(M · |L2|).
- This is the self-adjusting "only recompute what changed" principle applied one layer deeper. Stockfish already exploits the *sparsity of h* (zeros) in the L1→L2 product. Exploiting the sparsity of *Δh* is a different and complementary idea.
- Whether nnz(Δh) is small has to be **measured**. It depends on how many accumulator coordinates a single column w_j moves across a clamp boundary.

---

## 5. Sherman–Morrison–Woodbury low-rank updates

- Sherman & Morrison, "Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix", Ann. Math. Statist. 21(1):124–127, 1950, DOI 10.1214/aoms/1177729893. Earlier abstract: Ann. Math. Statist. 20:621, 1949. [V]
- Woodbury, "Inverting Modified Matrices", Memorandum Report 42, Statistical Research Group, Princeton University, 1950. [K]

**Identities.**
- Sherman–Morrison: if A is invertible and 1 + vᵀA⁻¹u ≠ 0, then

    (A + u vᵀ)⁻¹ = A⁻¹ − (A⁻¹ u vᵀ A⁻¹) / (1 + vᵀ A⁻¹ u).

- Woodbury: (A + U C V)⁻¹ = A⁻¹ − A⁻¹ U (C⁻¹ + V A⁻¹ U)⁻¹ V A⁻¹.
- Matrix determinant lemma: det(A + uvᵀ) = (1 + vᵀA⁻¹u) det A.
- Cost: O(n²) per rank-1 update instead of O(n³) re-inversion; O(n²k + k³) per rank-k update.
- Mathlib has the Woodbury identity and the determinant lemma; the candidate names are `Matrix.add_mul_mul_inv_eq_sub` and `Matrix.det_add_col_mul_row` or `Matrix.det_one_add_col_mul_row`. [UNVERIFIED: names must be checked in the current Mathlib]

**Relevance.** Chess evaluation does not invert matrices. The *structural* lesson still carries over: a low-rank perturbation of a base operator can be applied through cached products. The NNUE analogue is the king-bucket refresh problem.

- HalfKA uses W_k, a separate M × 704 block for each king bucket k. A king move changes k and forces a refresh costing about 32·M.
- **Low-rank bucket factorization:** W_k = W₀ + U_k V, with W₀ ∈ R^{M×704} shared, V ∈ R^{r×704} shared, and U_k ∈ R^{M×r}. Then

    W_k x = W₀ x + U_k (V x).

- Both W₀x (M-dimensional) and z = Vx (r-dimensional) are king-independent, so they update incrementally on every move, king moves included. A king move costs one M×r matrix–vector product U_k z, which is r column-adds instead of about 32.
- With r = 8 that is a 4× cheaper king move. Bucket-specific expressivity is limited to rank r.
- This is the Woodbury idea (base + low-rank correction through cached products) applied to forward evaluation. It is provable in Lean as associativity and distributivity of matrix multiplication.

---

## 6. Fenwick trees and prefix aggregates

- Fenwick, "A New Data Structure for Cumulative Frequency Tables", Software: Practice and Experience 24(3):327–336, 1994. [K]
- Ryabko (1989/1992) described an equivalent structure earlier. [UNVERIFIED citation details]

**Structure.** An implicit tree over an array a[1..n] in an abelian group, where node i stores Σ_{j ∈ (i − lowbit(i), i]} a_j. Point update and prefix query both take O(log n). Range sums need the group inverse: sum(l..r) = P(r) − P(l−1).

**Relevance to engines.** Stockfish evaluates *lazily*. When an accumulator is needed, it walks back up the search stack to the nearest computed ancestor and applies the chain of deltas, or it refreshes.

- With per-ply delta vectors d_1, …, d_t, the accumulator at ply t is A_0 + Σ d_i, which is a prefix sum along the search path.
- A Fenwick/skip-list layout over the search stack would allow O(log t) "jumps". In practice t is tiny (one to a few plies since the last evaluation), so the gain is negligible.
- The value for the paper is negative but rigorous: it explains why no data structure beyond a stack is needed. The amortized cost per evaluated node is already O(average delta chain length · |Δx|).

---

## 7. Factorization machines: pairwise interactions in O(k) per change

- Rendle, "Factorization Machines", IEEE ICDM 2010, pp. 995–1000, DOI 10.1109/ICDM.2010.127. [K]

**Model.** ŷ(x) = w₀ + Σ_i w_i x_i + Σ_{i<j} ⟨v_i, v_j⟩ x_i x_j, with v_i ∈ R^k.

**Identity (Rendle 2010, Lemma 3.1).**

  Σ_{i<j} ⟨v_i, v_j⟩ x_i x_j = ½ ( ‖Σ_i v_i x_i‖² − Σ_i ‖v_i‖² x_i² ).

- Proof: ‖Σ v_i x_i‖² = Σ_{i,j} ⟨v_i, v_j⟩ x_i x_j. Split the sum into the diagonal and two copies of the strict upper triangle. This fits in a few lines of Lean with `Finset.sum_mul_sum` and `Finset.sum_comm`, plus a diagonal/off-diagonal split.
- Direct evaluation is O(k n²); the identity gives O(k n), or O(k · nnz(x)) for sparse x.

**Incremental version (new for chess, derived here).** For binary x we have x_i² = x_i. Maintain:
- s = Σ_i v_i x_i ∈ R^k, an ordinary NNUE-style accumulator;
- q = Σ_i ‖v_i‖² x_i, a scalar accumulator.

Then P(x) = ½(‖s‖² − q). Adding feature j (x_j : 0 → 1):

  ΔP = ⟨v_j, s⟩ (s taken *before* the addition), s ← s + v_j.

Removing j: s ← s − v_j, then ΔP = −⟨v_j, s⟩ (s taken *after* the removal).

- So all ½·32·31 = 496 pairwise piece–piece interactions among the active features are maintained exactly at O(k) per changed feature: one dot product plus one vector add.
- HalfKP is a *restricted* degree-2 model: it covers only king × piece pairs, with a full (not factorized) table, 40960 features. The FM covers *all* piece × piece pairs over 768 base features, with rank-k factorized weights. The full table would have 768² / 2 ≈ 295k pair weights per output; the FM uses 768·k parameters per output.

**Vector-valued FM / bilinear accumulator.** To output an M-dimensional feature vector instead of a scalar, use P_m(x) = ½(‖S_m x‖² − …) for m = 1..M. The cost is M·k per change. A cheaper alternative is the elementwise square of the accumulator, h = (Wx)⊙(Wx), which is already quadratic in x and costs nothing beyond computing the activation. Stockfish's pairwise multiplication of accumulator halves ("SqrCReLU / pairwise mul") is exactly a degree-2 FM-like interaction computed *after* a linear accumulator. [UNVERIFIED: exact Stockfish activation naming and history]

**Theorem template (degree-2 incrementality, Lean target).** For any bilinear form B and accumulator s(x) = Σ v_i x_i,

  B(s(x+e_j), s(x+e_j)) − B(s(x), s(x)) = B(v_j, s(x)) + B(s(x), v_j) + B(v_j, v_j).

---

## 8. Higher-order interactions: CP decomposition and Newton's identities

- Hitchcock, "The Expression of a Tensor or a Polyadic as a Sum of Products", J. Math. Phys. 6:164–189, 1927. [K]
- Carroll & Chang, "Analysis of Individual Differences in Multidimensional Scaling via an N-way Generalization of 'Eckart–Young' Decomposition", Psychometrika 35:283–319, 1970. [K]
- Harshman, "Foundations of the PARAFAC Procedure", UCLA Working Papers in Phonetics 16:1–84, 1970. [K]
- Survey: Kolda & Bader, "Tensor Decompositions and Applications", SIAM Review 51(3):455–500, 2009. [K]

**CP form.** A symmetric order-d interaction tensor T has CP form T = Σ_{r=1}^R λ_r a_r^{⊗d}. Then

  Σ_{i_1<…<i_d} T_{i_1…i_d} x_{i_1}⋯x_{i_d} = Σ_r λ_r e_d(a_{r,1}x_1, …, a_{r,n}x_n),

where e_d is the d-th elementary symmetric polynomial. (The diagonal is not counted, so this holds for the strictly-upper-triangular part.)

**Newton–Girard identities.** With power sums p_m = Σ_i y_i^m:
- k·e_k = Σ_{i=1}^k (−1)^{i−1} e_{k−i} p_i;
- e_2 = (p_1² − p_2)/2;
- e_3 = (p_1³ − 3p_1p_2 + 2p_3)/6.

Mathlib formalizes these as `MvPolynomial.mul_esymm_eq_sum` and `MvPolynomial.psum_eq_mul_esymm_sub_sum` in `Mathlib/RingTheory/MvPolynomial/NewtonIdentities.lean`. [UNVERIFIED: exact names; the file exists in Mathlib as of 2023/24]

**Incremental consequence.** For binary x, each p_m^{(r)} = Σ_i a_{r,i}^m x_i is a *linear* function of x, so it is an accumulator. Maintaining p_1..p_d for R components costs d·R scalar adds per changed feature, after which e_d is evaluated in O(d²) per component.

- **Theorem:** an order-d interaction model of CP-rank R is updatable in O(dR) per feature change and evaluable in O(d²R), regardless of the number of active pieces.
- The FM identity in §7 is the case d = 2: e_2 = (p_1² − p_2)/2 with p_1 = ⟨a_r, x⟩ and p_2 = Σ a_{r,i}² x_i.
- Rendle's FM is the vector-valued rank-k case. Blondel et al. (NeurIPS 2016, "Higher-Order Factorization Machines") use the ANOVA kernel / DP recursion, which is the same e_d. [K]

---

## 9. DBSP and Z-sets: a general algebra for incremental view maintenance

- Budiu, Chajed, McSherry, Ryzhyk, Tannen, "DBSP: Automatic Incremental View Maintenance for Rich Query Languages", PVLDB 16(7):1601–1614, 2023, DOI 10.14778/3587136.3587137. VLDB 2023 Best Paper. Extended version: VLDB Journal 34(4), 2025, DOI 10.1007/s00778-025-00922-y. [V]
- The "Best Paper" status is [K]; re-verify it.

**Definitions.**
- A Z-set over a domain D is a finitely supported function D → Z. Z-sets form an abelian group, and relational "multisets with deletions" are Z-sets.
- Streams are maps N → G into an abelian group G.
- Integration: I(s)_t = Σ_{i≤t} s_i. Differentiation: D(s)_t = s_t − s_{t−1}. These are inverse to each other.
- The incrementalization of a stream operator Q is Q^Δ = D ∘ Q ∘ I.

**Key theorems (paper, Sections 3–4).**
1. **Chain rule:** (Q₁ ∘ Q₂)^Δ = Q₁^Δ ∘ Q₂^Δ.
2. **Linear operators** (group homomorphisms lifted pointwise) are their own incremental versions: Q^Δ = Q. This is the formal statement of "the NNUE first layer is free to incrementalize".
3. **Bilinear operators** (joins, products), with z⁻¹ the delay operator:

     (a × b)^Δ = Δa × Δb + z⁻¹(I a) × Δb + Δa × z⁻¹(I b).

   This is the discrete product rule, the same algebra as the FM update in §7. With a = b = s and × = dot product it gives ΔP = 2⟨s_prev, Δs⟩ + ‖Δs‖², which is twice the FM update because DBSP counts ordered pairs.
4. **Non-linear operators** (distinct, aggregation) need extra state. They are incrementalized by maintaining the integral and computing D(Q(I(·))) only where the support changed.

**Mapping to an evaluation network.**
- The position stream is x_t, and the move deltas are D(x)_t. The feature transformer is linear ⇒ Q^Δ = Q.
- Bilinear layers have the 3-term product-rule update.
- Activations σ are "non-linear operators": their incremental cost is proportional to the support of the change in their input (§4).
- DBSP thus gives a *compositional cost calculus*. A network is "very easily updatable" exactly when it is a composition of linear operators, bilinear operators with low-dimensional operands, and pointwise non-linearities applied to sparsely changing inputs.

---

## 10. Monoid and group actions on states

- Moves do not form a group on positions: they are partial and state-dependent. The feature deltas still form an action of the free abelian group Z^n on Z^n by translation, and φ is equivariant: φ(x + δ) = φ(x) + φ(δ).
- Make/unmake gives each move a two-sided inverse on the reachable state, (x + δ) − δ = x. NNUE avoids "unmake" arithmetic by restoring from the stack (§0).
- **Group-action view of evaluation with an accumulator.**
  - Let G act on a state space X and on an accumulator module V. A map φ : X → V is *equivariant* if φ(g·x) = ρ(g) φ(x).
  - NNUE: G = Z^n, X = Z^n, V = R^M, ρ(δ) = translation by Wδ (an affine action).
  - Allowing ρ to be a general *linear* representation (ρ(δ) = matrices) gives "multiplicative" accumulators, A' = ρ(m)·A. Each update then costs an M×M matrix–vector product (O(M²)), unless ρ(m) is diagonal or a permutation.
  - Diagonal ρ: A'_k = A_k · exp(c_k(m)). This is a log-domain linear accumulator, so it is no more expressive than the additive one: it is the additive one with an exp activation.
  - Permutation ρ: A' = P_m A. This is how symmetry transforms act (§11) and costs O(M) memory moves.

---

## 11. Symmetry and group equivariance (Cohen & Welling 2016)

- Cohen & Welling, "Group Equivariant Convolutional Networks", ICML 2016, PMLR 48:2990–2999. [K]
- Main idea: G-convolutions with weight tying W_{g·i} = ρ(g) W_i give exact equivariance f(g·x) = ρ′(g) f(x) for discrete groups such as p4m (rotations and reflections). Parameter efficiency improves by a factor of up to |G| at no extra compute.

**Chess symmetries.**
- **Colour flip combined with a vertical (rank) mirror** is an exact symmetry of chess with side to move swapped. NNUE already exploits it: both perspectives share one W, and the features are computed in the mover's frame.
- **Horizontal (file) mirror a↔h** is exact when neither side has castling rights. HalfKAv2_hm ("hm" = horizontal mirror) uses it to fold 64 king squares into 32 buckets by mirroring the board so the king is always on files e–h. [UNVERIFIED: exact mirror convention]
- **Full dihedral group D4 (order 8)** is exact for pawnless positions with no castling rights. This is the symmetry used to compress Nalimov/Syzygy endgame tablebases. [K]

**Implication.** For a G-invariant evaluation one can tie columns, w_{g·i} = P_g w_i (P_g a permutation of accumulator coordinates). Then the accumulator for g·x is P_g A(x), obtained by an O(M) permutation with no feature walk.

- A VNNUE could keep **one** accumulator and derive the mirrored-board accumulator by permutation. On a king crossing the d/e file, which triggers a mirror change in HalfKAv2_hm, the cost would be an O(M) permute instead of a 32·M refresh.
- This needs the column ties to hold *exactly*, which is a training constraint. The parameter savings would follow the Cohen–Welling pattern.

---

## 12. Competitive analysis: when to refresh vs. update

- Sleator & Tarjan, "Amortized Efficiency of List Update and Paging Rules", CACM 28(2):202–208, 1985. Move-to-front is 2-competitive for list update; LRU/FIFO are k-competitive for paging, which is optimal for deterministic algorithms. [K]
- Karlin, Manasse, Rudolph, Sleator, "Competitive Snoopy Caching", Algorithmica 3:79–119, 1988. Introduces the term "competitive" and gives the rent-or-buy argument. [K]
- Karlin, Manasse, McGeoch, Owicki, "Competitive Randomized Algorithms for Nonuniform Problems", Algorithmica 11(6):542–571, 1994. Randomized ski rental achieves e/(e−1) ≈ 1.582. [K]

**Ski rental, precisely.** Renting costs 1 per day, buying costs B. The deterministic "rent B−1 days, then buy" strategy is (2 − 1/B)-competitive, and that is optimal. The randomized optimum is e/(e−1) as B → ∞.

**Engine mapping.** Three decisions fit this framework:
- **Lazy accumulators.** Pay incremental deltas ("rent") along a line of play that may never be evaluated, or refresh at the evaluated node ("buy").
- **Finny-table caching per king bucket.** Choose which bucket states to keep.
- **King-move refresh vs. incremental-from-cache.** Diff against the cached board for that bucket (cost about |board Δ|) or start from bias (cost about 32).

The search tree is not adversarial, so the empirical average matters more than worst-case ratios. Competitive analysis still gives *provable* worst-case guarantees for a VNNUE refresh policy.
- Example policy: "walk back up to B/c plies, else refresh" is 2-competitive against the offline optimum, where c is the per-ply delta cost and B the refresh cost.
- The ratio follows from the standard ski-rental proof and is Lean-provable as an inequality on natural numbers.

---

## Implications for VNNUE (concrete design ideas)

Each idea is labelled with the evidence class it would need: **[Lean]** formal proof, **[Exp]** experiment, **[Cite]** literature.

1. **Additive-group principle as the definition of "updatable".** [Lean]
   - Formally define a layer as *k-updatable* if f(x + δ) − f(x) can be computed in O(k·|δ|) from stored state.
   - Prove: linear ⇒ 1-updatable; bilinear on k-dimensional accumulators ⇒ O(k)-updatable (DBSP product rule); CP rank-R, order-d ⇒ O(dR)-updatable (Newton identities); compositions follow the chain rule (Cai et al.; DBSP).
   - This gives VNNUE a precise mathematical meaning instead of a slogan.

2. **Factorized pairwise piece-interaction head ("FM accumulator").** [Lean + Exp]
   - Add to the NNUE accumulator a small rank-k accumulator s = Σ v_i x_i over the 768 base features, plus the scalar q.
   - Output ½(‖s‖² − q), or a few such heads. Cost: about k + 1 extra adds per changed feature, plus one dot product.
   - It captures *all* 496 piece–piece pairs, whereas HalfKP/HalfKA capture only king–piece pairs.
   - Lean: the FM identity and the incremental ΔP = ⟨v_j, s⟩ rule, over a commutative ring, which covers the int-quantized case.
   - Experiment: Elo or loss versus cost at k ∈ {8, 16, 32}.

3. **Order-3 interactions through power-sum accumulators.** [Lean + Exp]
   - Maintain p_1, p_2, p_3 per component and evaluate e_3 = (p_1³ − 3p_1p_2 + 2p_3)/6, which models piece triples (e.g. battery/pin geometry) at 3R adds per change.
   - Watch out for the numerical range: p_1³ grows fast, and int16 is not enough, so int32 or float is needed for this head.

4. **Low-rank king conditioning to remove king-move refreshes.** [Lean + Exp]
   - Use W_k = W₀ + U_k V (the Woodbury-style split from §5). King moves then cost r column-adds instead of about 32.
   - The bucket accumulator is exactly reconstructible: W_k x = W₀x + U_k z. That is a Lean lemma about `Matrix.mul_assoc` and `Matrix.add_mul`.
   - Measure: the fraction of nodes that need a refresh, the time per node, and the accuracy loss compared with full HalfKA buckets.
   - Hybrid: full buckets for a few king regions plus low-rank modulation within each.

5. **Second-layer delta propagation (self-adjusting).** [Exp, with a Lean lemma]
   - Exploit the sparsity of Δσ(a), not just of σ(a). The Lean lemma: σ(a+Δ)_k = σ(a)_k when Δ_k = 0 or both values are on the same saturated side.
   - Experiment: histogram of nnz(Δh)/M per move on real search trees. The idea only pays off if this is ≪ 1. It may also motivate training regularizers that encourage saturation (sparse columns w_j).

6. **Exact symmetry-tied columns.** [Lean + Exp]
   - Constrain w_{mirror(i)} = P w_i so that mirrored accumulators are a permutation P·A, an O(M) operation.
   - Lean: if W is P-equivariant, then A(g·x) = P A(x).
   - Experiment: whether the constraint costs strength, compared with HalfKAv2_hm's learned mirror.

7. **Overflow-safe quantized accumulators by construction.** [Lean]
   - Prove the ZMod(2^16) homomorphism claim: update order does not matter, and the result equals the true integer sum whenever the true sum lies in [−2^15, 2^15).
   - Add a bound lemma: |b_m| + Σ_{top-32 |w_{m,i}|} < 2^15 ⇒ no overflow for any legal position (at most 32 active features). This gives a *checkable certificate* for each trained net, generated by a script and proved by `decide` or `norm_num` on the per-row bound.

8. **Provably good refresh policy.** [Lean + Exp]
   - Model lazy-accumulator walking as ski rental: walk back ≤ ⌊B/c⌋ plies, else refresh. Lean-prove the 2-competitive bound, and measure it against Stockfish's current heuristic.

9. **Negative results worth stating.** [Lean]
   - (a) Max-pooling or other non-invertible aggregators cannot support O(1) removals in general. A counter-example in (ℕ, max) shows the delta is not determined by the old aggregate and the removed element.
   - (b) Non-commutative aggregators need Ω(log n)-style structure. Cite segment trees and Fenwick; the general lower bound is in Fredman & Saks 1989 (STOC), "The cell probe complexity of dynamic data structures" [K].
   - (c) Deep stacks beyond the first two layers gain nothing from incrementality, since σ destroys linearity and the downstream widths are small.

---

## Candidate Lean 4 / Mathlib theorem list

| # | Statement | Mathlib ingredients |
|---|---|---|
| L1 | `φ : ℤⁿ →+ G`, φ(x+δ)=φ x+φ δ; sparse sum form | `AddMonoidHom.map_add`, `Finset.sum_add_distrib` |
| L2 | Zobrist: XOR hash is an additive hom on `Fin n → ZMod 2` | `LinearMap`, `ZMod 2` |
| L3 | Quantized accumulator: cast ℤ → ZMod(2^16) commutes with updates; equals true value in range | `Int.cast`, `ZMod.intCast_cast`, `ZMod.valMinAbs` |
| L4 | FM identity Σ_{i<j}⟨v_i,v_j⟩x_ix_j = ½(‖Σv_ix_i‖² − Σ‖v_i‖²x_i²) | `Finset.sum_mul_sum`, `Finset.sum_product`, off-diagonal split |
| L5 | FM incremental rule ΔP = ⟨v_j,s⟩ for x_j: 0→1 | algebra on L4 |
| L6 | Newton e_2, e_3 in terms of power sums ⇒ order-3 CP update | `MvPolynomial.psum`, `MvPolynomial.esymm`, Newton identities file |
| L7 | Discrete product rule (DBSP bilinear incrementalization) | `LinearMap.BilinForm` / ring algebra |
| L8 | Chain rule for change structures (Cai et al.) | custom structure, `Function.comp` |
| L9 | Low-rank bucket: (W₀+U_kV)x = W₀x + U_k(Vx) | `Matrix.add_mulVec`, `Matrix.mulVec_mulVec` |
| L10 | Sherman–Morrison/Woodbury (cite Mathlib lemma) | `Matrix.SchurComplement` / Woodbury lemma [UNVERIFIED name] |
| L11 | Polynomial forward difference lowers degree by 1 | `Polynomial.natDegree`, `Polynomial.comp (X + C h)` |
| L12 | Boolean discrete derivative of multilinear degree-d is degree ≤ d−1 | `MvPolynomial.totalDegree` |
| L13 | Clamp-saturation lemma for Δ-sparsity | `max`/`min` lemmas |
| L14 | Equivariant tying ⇒ A(g·x) = P A(x) | `Matrix.mulVec`, permutation matrices |
| L15 | Ski-rental deterministic (2 − 1/B) bound | `Nat`/`ℚ` inequalities |

---

## References

1. Nasu, Y. (2018). *Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi.* 28th World Computer Shogi Championship appeal document, Computer Shogi Association (in Japanese). English translation by D. Klein (2021), https://github.com/asdfjkl/nnue. [V]
2. Official Stockfish (2020–). *nnue-pytorch documentation, docs/nnue.md.* https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md. [V]
3. Paige, R., & Koenig, S. (1982). Finite Differencing of Computable Expressions. *ACM TOPLAS* 4(3):402–454. DOI 10.1145/357172.357177. [K; DOI UNVERIFIED]
4. Acar, U. A., Blelloch, G. E., & Harper, R. (2002). Adaptive Functional Programming. *POPL 2002*, 247–259. Journal version: *ACM TOPLAS* 28(6):990–1034, 2006. [K]
5. Acar, U. A. (2005). *Self-Adjusting Computation.* PhD thesis, Carnegie Mellon University, CMU-CS-05-129. [K; report number UNVERIFIED]
6. Zobrist, A. L. (1970). *A New Hashing Method with Application for Game Playing.* Tech. Rep. 88, Computer Sciences Dept., Univ. of Wisconsin–Madison. Reprinted in *ICCA Journal* 13(2):69–73, 1990. [K]
7. Sherman, J., & Morrison, W. J. (1950). Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix. *Ann. Math. Statist.* 21(1):124–127. DOI 10.1214/aoms/1177729893. [V]
8. Sherman, J., & Morrison, W. J. (1949). Adjustment of an inverse matrix corresponding to changes in the elements of a given column or a given row of the original matrix (abstract). *Ann. Math. Statist.* 20:621. [V]
9. Woodbury, M. A. (1950). *Inverting Modified Matrices.* Memorandum Report 42, Statistical Research Group, Princeton University. [K]
10. Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience* 24(3):327–336. [K]
11. Cai, Y., Giarrusso, P. G., Rendel, T., & Ostermann, K. (2014). A Theory of Changes for Higher-Order Languages: Incrementalizing λ-Calculi by Static Differentiation. *PLDI 2014*, 145–155. DOI 10.1145/2594291.2594304. arXiv:1312.0658. [V]
12. Budiu, M., Chajed, T., McSherry, F., Ryzhyk, L., & Tannen, V. (2023). DBSP: Automatic Incremental View Maintenance for Rich Query Languages. *PVLDB* 16(7):1601–1614. DOI 10.14778/3587136.3587137. [V]
13. Budiu, M., et al. (2025). DBSP: automatic incremental view maintenance for rich query languages. *The VLDB Journal* 34(4). DOI 10.1007/s00778-025-00922-y. [V]
14. Hitchcock, F. L. (1927). The Expression of a Tensor or a Polyadic as a Sum of Products. *J. Math. Phys.* 6:164–189. [K]
15. Carroll, J. D., & Chang, J.-J. (1970). Analysis of Individual Differences in Multidimensional Scaling via an N-way Generalization of "Eckart–Young" Decomposition. *Psychometrika* 35:283–319. [K]
16. Harshman, R. A. (1970). Foundations of the PARAFAC Procedure: Models and Conditions for an "Explanatory" Multi-modal Factor Analysis. *UCLA Working Papers in Phonetics* 16:1–84. [K]
17. Kolda, T. G., & Bader, B. W. (2009). Tensor Decompositions and Applications. *SIAM Review* 51(3):455–500. [K]
18. Rendle, S. (2010). Factorization Machines. *IEEE ICDM 2010*, 995–1000. DOI 10.1109/ICDM.2010.127. [K]
19. Blondel, M., Fujino, A., Ueda, N., & Ishihata, M. (2016). Higher-Order Factorization Machines. *NeurIPS 2016*. [K]
20. Cohen, T. S., & Welling, M. (2016). Group Equivariant Convolutional Networks. *ICML 2016*, PMLR 48:2990–2999. [K]
21. Sleator, D. D., & Tarjan, R. E. (1985). Amortized Efficiency of List Update and Paging Rules. *CACM* 28(2):202–208. [K]
22. Karlin, A. R., Manasse, M. S., Rudolph, L., & Sleator, D. D. (1988). Competitive Snoopy Caching. *Algorithmica* 3:79–119. [K]
23. Karlin, A. R., Manasse, M. S., McGeoch, L. A., & Owicki, S. (1994). Competitive Randomized Algorithms for Nonuniform Problems. *Algorithmica* 11(6):542–571. [K]
24. Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. *Contemporary Mathematics* 26:189–206. [K]
25. O'Donnell, R. (2014). *Analysis of Boolean Functions.* Cambridge University Press. [K]
26. Dietz, P., & Sleator, D. (1987). Two Algorithms for Maintaining Order in a List. *STOC 1987*, 365–372. [K]
27. Fredman, M. L., & Saks, M. E. (1989). The Cell Probe Complexity of Dynamic Data Structures. *STOC 1989*, 345–354. [K]
28. Mathlib Community. *Mathlib4*, `Mathlib/RingTheory/MvPolynomial/NewtonIdentities.lean`, `Mathlib/LinearAlgebra/Matrix/SchurComplement.lean`. https://github.com/leanprover-community/mathlib4. [UNVERIFIED: file paths and lemma names]
