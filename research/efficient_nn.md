# Efficient and structured architectures: which nonlinear interactions stay updatable?

Research notes for the VNNUE project. Topic: efficient/structured neural network architectures and what they mean for an evaluator that is updated incrementally.

Status tags used below:
- **[CITED]**: taken from the referenced source. I checked the bibliographic data by web search on 2026-11-09.
- **[ALGEBRA]**: an elementary identity. It is stated so it can be proved in Lean 4/Mathlib, and it should be proved before it appears in the paper.
- **[UNVERIFIED]**: I believe it is true but could not confirm it against a primary source in this session.
- **[MEASURE]**: a quantitative claim that has to come from our own experiment.

---

## 0. Setup and cost model

Let `U` be the universe of binary input features. For HalfKAv2_hm, `|U| = 32 king buckets × 11 piece planes × 64 squares = 22,528`. The trainer code has `NUM_PLANES_REAL = 64*11` [CITED: nnue-pytorch `halfka_v2.py`]. A position defines an active set `A ⊆ U` with `|A| ≤ 32` for piece-square features. A move changes it to `A' = (A \ R) ∪ D`, with removed set `R` and added set `D`, where `k = |R| + |D|`. Typical values per perspective:
- quiet move: k = 2
- capture: k = 3
- castling: k = 4

A king move that changes the king bucket forces a refresh (standard NNUE behaviour; see Nasu 2018 and the nnue-pytorch docs).

**Definition (updatable statistic).** A map `T : 2^U → M` is *O(k·m)-updatable* if `T(A')` can be computed from `T(A)`, `R` and `D` in `O(k·m)` operations, where `m` is the size of the representation of `T`.

**Fact (NNUE accumulator) [ALGEBRA].** For `a(A) = b + Σ_{i∈A} W_i` with `W_i ∈ ℝ^d`:

    a(A') = a(A) − Σ_{i∈R} W_i + Σ_{j∈D} W_j        (R ⊆ A, D ∩ (A\R) = ∅)

This costs `k·d` additions. The full NNUE cost per evaluated node is `k·d` (update) plus `C_head` (a dense head from `d` to L2 and onward). Stockfish master as of 2026-11-09 has:
- `L1 = 1024`, `L2 = 32`, `L3 = 32`, `LayerStacks = 32`
- head: `fc_0 = AffineTransformSparseInput<1024, 32>`, then `SqrClippedReLU` and `ClippedReLU` in parallel (concatenated to 64), then `fc_1: 64→32`, again both activations, and `fc_2: 128→1`
- a skip term `fc_0_out[30] − fc_0_out[31]`
- feature sets: `PSQFeatureSet = K32H3`, `ThreatFeatureSet = FullThreats`, `PairFeatureSet = PP_3Wide`

[CITED: Stockfish `src/nnue/nnue_architecture.h`, master, fetched 2026-11-09]

**Key structural point.** Incremental update works because `T` is a *group homomorphism* from the free abelian group on `U` (with formal sums of features) into `(ℝ^d, +)`. Removal needs inverses, so you need a group, not just a monoid. Max-pooling `T(A) = max_{i∈A} φ(i)` is a monoid homomorphism into `(ℝ, max)`. It supports insertion in O(m) but not deletion: deleting the argmax forces a rescan, unless you keep a multiset or heap (O(log|A|) per coordinate) [ALGEBRA]. This is the main dividing line for every architecture below.

**Characterization (proposed theorem, Lean target).** Let `φ : U → ℝ^m`. Every function of the form `g(Σ_{i∈A} φ(i))` is O(k·m)-updatable, with `g` evaluated afterwards. Conversely, any `T` that is "additively updatable" (meaning `T(A ∪ {j}) − T(A)` depends only on `j`, for all `A ∌ j`) satisfies `T(A) = T(∅) + Σ_{i∈A} φ(i)` with `φ(j) := T({j}) − T(∅)`. The proof is induction on `Finset`. So **every** updatable nonlinear interaction must be expressed as a *nonlinear readout of additive accumulators*. The design question is then which accumulators `φ` and which readout `g` give the most expressive interactions per unit of `m`.

---

## 1. Factorized / virtual features (NNUE training trick)

**Fact [CITED: nnue-pytorch docs, "Feature factorization"].** During training, a "virtual" feature set is added next to the real one. Each virtual feature is denser and redundant with one or more real features, for example `P` (piece-square, ignoring the king) shared across all king buckets. Because they are redundant, their weights "can be merged into the real feature weights once training is done". HalfKAv2_hm is trained as `HalfKAv2_hm^` (the factorized version, the default), with `NUM_PLANES_VIRTUAL = 64*12`. Stockfish has used HalfKAv2_hm since early August 2021, and it halves the feature transformer relative to HalfKAv2 by mirroring so the king is always on files e–h.

**Merge identity [ALGEBRA, Lean target].** Let `π : U → V` map each real feature to its virtual feature. Training uses `a(A) = b + Σ_{i∈A} (R_i + Q_{π(i)})`. Then

    Σ_{i∈A} (R_i + Q_{π(i)}) = Σ_{i∈A} W_i ,   W_i := R_i + Q_{π(i)}

This is exact because the first layer is linear. The factorization therefore costs nothing at inference and only changes the *parametrization*, which affects the optimization dynamics and the effective prior (weight sharing across king buckets). The identity extends to any number of factor sets, `W_i = R_i + Σ_s Q^{(s)}_{π_s(i)}`, so it is a rank-structured reparametrization of `W ∈ ℝ^{d×|U|}`: `W = R + Σ_s Q^{(s)} P_s`, where `P_s` is the 0/1 aggregation matrix of `π_s`.

**Caveat.** Merging interacts with quantization. The merged `W_i` must fit int16, which is why the docs discuss "accounting for virtual layers in the feature transformer quantization scheme". Bounding `|W_i| ≤ |R_i| + Σ|Q|` is the formal statement.

---

## 2. Low-rank factorization

**Fact.** `W = U V` with `U ∈ ℝ^{d×r}` and `V ∈ ℝ^{r×|U|}`. Parameter-prediction and low-rank papers show large redundancy in trained weight matrices: Denil et al. (2013) predicted more than 95% of weights from a small subset; Jaderberg, Vedaldi & Zisserman (2014) and Sainath et al. (2013) report several-fold speedups. LoRA (Hu et al. 2021) uses `ΔW = BA`.

**Incremental consequence [ALGEBRA].** `a = U(Vx)`. One option is to accumulate `z = Vx ∈ ℝ^r` (cost `k·r`) and then compute `Uz` (cost `d·r`, dense, every evaluation). That is only a win when `r ≪ d` *and* the readout can work directly on `z`. The other option, precomputing `W_i = U V_i`, gives back the ordinary NNUE. **Conclusion:** low rank reduces *storage/cache footprint* (`|U|·r + d·r` instead of `|U|·d`), but by itself it does not reduce update arithmetic below `k·d` unless the downstream layer is fused: `fc_0 ∘ σ ∘ U` cannot be fused because of `σ`. The structured `W = R + QP` from §1 is the useful variant, since it is low-rank-plus-sparse in parameter space while inference stays at `k·d`.

Cache effect: the weight rows touched per move are `k·d·2 bytes` (int16). With `d=1024` and `k=3` that is 6 KiB, from a table of `22,528×1024×2 B ≈ 46 MB` (PSQ part only). L2/L3 misses on weight rows dominate update cost in practice [MEASURE]. This is the motivation for the explicit prefetch parameters (`prefetchBase`, `prefetchStride`) in Stockfish's `append_changed_indices` [CITED: `full_threats.h`, `pp_3wide.h`].

---

## 3. Hashing trick and product keys

**Feature hashing [CITED: Weinberger et al. 2009].** `φ_j(x) = Σ_{i : h(i)=j} ξ(i) x_i`, with hash `h : U → [M]` and sign hash `ξ : U → {±1}`. The hashed inner product is unbiased: `E[⟨φ(x), φ(x')⟩] = ⟨x, x'⟩`, with variance `O(1/M)` and exponential tail bounds. **Updatability:** `φ` is linear in `x`, so a hashed accumulator `Σ_{i∈A} W_{h(i)} ξ(i)` stays O(k·d)-updatable [ALGEBRA]. Hashing lets you use a *huge* (even unbounded) virtual feature space, such as all piece pairs or all (attacker, victim, square) triples, with a fixed table of `M` rows. The cost is collisions, which act as noise.

**Product keys [CITED: Lample et al. 2019].** A key set `K = K_1 × K_2` with `|K_1| = |K_2| = √N` makes exact top-k nearest-key search over `N` keys cost `O(√N · d_q)` instead of `O(N · d_q)`. Their memory layer had up to about 1M slots and used `k` selected values. **Updatability:** the query `q(x)` is a dense function of the input, so the top-k selection changes discontinuously and nothing is incremental. However, the *values* are a sparse table lookup, the same as an NNUE weight row. A product-key layer applied to the *output* of an accumulator is just an expensive head.

**Takeaway.** Hashing is compatible with incrementality. Learned routing (product keys, MoE gating on dense activations) generally is not.

---

## 4. Explicit second-order features: what Stockfish already does

**Fact [CITED: Stockfish master `full_threats.h`].** `FullThreats` has `Dimensions = 59,808` and `MaxActiveDimensions = 256`. An index is `make_index(perspective, attacker piece, from, to, attacked piece, ksq)`, so each feature is an (attacker, victim) **pairwise relation** that is gated by attack geometry. A 6×6 attacker/target piece-type map excludes some pairs. Pawn diagonal threats only target knights and rooks, because pawn–pawn relations are handled separately.

**Fact [CITED: Stockfish master `pp_3wide.h`].** `PP_3Wide` is a **pawn-pair** feature set: `PawnIds = 2·48 = 96`, `Dimensions = 96·95/2 = 4,560`, and indices start at `IndexBase = 59,808` (after the threats). The diff type is `DirtyPawnPairs`. The meaning of "3Wide" (presumably a 3-file window) is not documented in the header [UNVERIFIED].

**Interpretation.** These are *explicitly enumerated* second-order features: `x_{ij} = x_i x_j · g(i,j)`. They are still accumulated linearly, so updates stay additive. The update cost, however, is the number of *pairs* that change, which can be tens per move. One slider move can change many attack relations, while a quiet move changes only 2 piece-square features. Costs per move:
- explicit pair features: `O(Δpairs · d)` with `Δpairs = O(|A|)`
- factorized pair interactions (§5): `O(k · r)`

---

## 5. Factorization machines: second- and higher-order interactions at accumulator cost

**FM model [CITED: Rendle 2010].**

    ŷ(x) = w_0 + Σ_i w_i x_i + Σ_{i<j} ⟨v_i, v_j⟩ x_i x_j ,   v_i ∈ ℝ^r

**Rendle's identity [CITED: Rendle 2010, Lemma 3.1; ALGEBRA, Lean target].**

    Σ_{i<j} ⟨v_i,v_j⟩ x_i x_j = ½ Σ_{f=1}^{r} [ (Σ_i v_{i,f} x_i)^2 − Σ_i v_{i,f}^2 x_i^2 ]

For binary `x` with active set `A`, write `s(A) = Σ_{i∈A} v_i ∈ ℝ^r` and `q(A) = Σ_{i∈A} ‖v_i‖²`. Then

    Σ_{ {i,j} ⊆ A, i≠j } ⟨v_i, v_j⟩ = ½ ( ‖s(A)‖² − q(A) )

`s` and `q` are both additive accumulators, so **all `|A|(|A|−1)/2` pairwise interactions are maintained in O(k·r) per move, plus O(r) for the readout**. This is exactly the NNUE accumulator with a quadratic readout. Rendle states linear-time evaluation, `O(r·n)` in the number of nonzeros. Incrementally it is `O(r·k)`.

**Incremental readout of the squared norm [ALGEBRA].** If `s' = s + δ`, then `‖s'‖² = ‖s‖² + 2⟨s,δ⟩ + ‖δ‖²`. Even the readout can be updated in `O(r)`.

**Field-aware FM [CITED: Juan et al. 2016].** `Σ_{i<j} ⟨v_{i,F(j)}, v_{j,F(i)}⟩ x_i x_j` with `F` fields. For chess, natural fields are piece types, or (own/enemy × piece type) giving 12 fields. Define `S_{F,G} = Σ_{i∈A, F(i)=F} v_{i,G}`. Then [ALGEBRA, Lean target]:

    FFM(A) = Σ_{F<G} ⟨S_{F,G}, S_{G,F}⟩ + Σ_F ½( ‖S_{F,F}‖² − Σ_{i∈A,F(i)=F} ‖v_{i,F}‖² )

This needs `|F|²` accumulators of width `r`. A move touching a feature in field `F` updates the `|F|` accumulators `S_{F,·}` (cost `|F|·r` per changed feature), and the readout changes by `Σ_G ⟨δ_{F,G}, S_{G,F}⟩` (cost `|F|·r`). With 12 fields and `r=32`, that is 384 numbers per changed feature, comparable to a 384-wide NNUE.

**Higher order: ANOVA kernel / Newton–Girard [CITED: Blondel et al. 2016 NeurIPS; ALGEBRA, Lean target].** The degree-`m` interaction `Σ_{i_1<…<i_m} Π_t x_{i_t} ⟨v_{i_1},…,v_{i_m}⟩_f` (a coordinate-wise product summed over `f`) equals `Σ_f e_m(v_{·,f} x)`, where `e_m` is the elementary symmetric polynomial. By Newton's identities with power sums `p_t = Σ_{i∈A} v_{i,f}^t`:

    e_1 = p_1
    e_2 = (p_1^2 − p_2)/2
    e_3 = (p_1^3 − 3 p_1 p_2 + 2 p_3)/6
    m·e_m = Σ_{t=1}^{m} (−1)^{t−1} e_{m−t} p_t

Each `p_t` is an additive accumulator. **Degree-`m` all-subset interactions therefore cost `m` accumulators of width `r`, i.e. an update of O(k·m·r).** Mathlib has Newton's identities for multivariate symmetric polynomials (`MvPolynomial.psum`, `MvPolynomial.esymm`, and the Newton identities file `Mathlib/RingTheory/MvPolynomial/NewtonIdentities.lean`) [UNVERIFIED exact name; check before use]. **Corollary (fundamental theorem of symmetric polynomials, char 0):** every symmetric polynomial in the per-feature embeddings is a polynomial in finitely many power sums. So "updatable" (§0) and "symmetric polynomial readout" coincide at the polynomial level.

**Polynomial networks [CITED: Blondel et al. 2016 ICML; Chrysos et al. 2020].** Π-nets express outputs as high-degree polynomials using Hadamard products of linear maps, e.g. `x_{n+1} = (Λ_n z) ⊙ x_n + x_n`. A Hadamard product of two *accumulators* `(W^1 x) ⊙ (W^2 x)` has `d` coordinates, each a **rank-1 quadratic form** `x^T w^1_c (w^2_c)^T x` [ALGEBRA]. This is the second-order structure that Stockfish's feature-transformer *pairwise multiplication* produces: the accumulator is split into two halves, each half is clipped, and they are multiplied elementwise [UNVERIFIED: version/date of introduction; the nnue-pytorch docs describe it]. With clipping, the result is a piecewise-quadratic function of `x`, still computed only from the additive accumulator.

---

## 6. Piecewise-linear networks

**Facts [CITED].** ReLU networks compute continuous piecewise-linear (CPWL) functions. Conversely, every CPWL function `ℝ^n → ℝ` is representable by a ReLU network of depth at most `⌈log₂(n+1)⌉ + 1` (Arora et al. 2018). The number of linear regions can grow exponentially with depth (Montúfar et al. 2014). ClippedReLU `min(max(x,0),1)` equals `ReLU(x) − ReLU(x−1)`, so CPWL is preserved [ALGEBRA].

**Incremental consequence [ALGEBRA].** Inside one linear region (a fixed activation pattern `P`), `f(x+δ) = f(x) + J_P δ`. If `δ` (sparse, `k` nonzeros) does not change the pattern, the output changes by `J_P δ`, which costs `k` columns of the Jacobian. In NNUE, however, each `W_i` is dense in `ℝ^d`, so after one move *every* accumulator coordinate changes and the activation pattern of layer 1 can flip anywhere. The sparsity of the input change does **not** propagate past the first layer.

**Exploited activation sparsity [CITED: Stockfish `AffineTransformSparseInput`].** `fc_0` skips zero inputs after ClippedReLU. This is *activation* sparsity, not *change* sparsity.

---

## 7. Delta networks and incremental inference in other fields

- **Delta Networks [CITED: Neil, Lee, Delbruck & Liu 2017].** Each neuron transmits only when `|Δ activation|` exceeds a threshold Θ. The next layer then updates `z_t = z_{t−1} + W Δx_t`, at a cost proportional to the number of changed units. Reported savings: a 5.7× speedup on WSJ speech with negligible accuracy loss, and about 100× fewer RNN operations on a CNN-RNN steering model.
- **Recurrent Residual Module [CITED: Pan et al. 2018].** Uses the linearity of conv/FC layers, `W x_t = W x_{t−1} + W(x_t − x_{t−1})`, on sparse (thresholded) frame differences. About 2× average speedup on standard CNNs, and 9× on XNOR-nets.
- **DeltaCNN [CITED: Parger et al. 2022].** End-to-end sparse delta propagation through *all* CNN layers, including nonlinear ones. It keeps per-layer input/output buffers so that `σ(a + Δa) − σ(a)` is computed only where `Δa ≠ 0`, and it does **not** accumulate error over time (an exact update after one dense initial frame). Up to 7× over cuDNN.
- **Linear attention [CITED: Katharopoulos et al. 2020].** `Attn_i = φ(Q_i)^T S / φ(Q_i)^T z` with `S = Σ_j φ(K_j) V_j^T` and `z = Σ_j φ(K_j)`. `S` and `z` are additive accumulators over tokens ("Transformers are RNNs"). If the tokens are pieces, then inserting/removing a token costs `O(r·d_v)` [ALGEBRA].
- **Fixed-query softmax pooling [ALGEBRA, Lean target].** Use a learned query `q`, as in Set Transformer's PMA (Lee et al. 2019). Let `N(A) = Σ_{j∈A} e^{⟨q,k_j⟩} v_j` and `Z(A) = Σ_{j∈A} e^{⟨q,k_j⟩}`. Both are additive, so `softmax-pool(A) = N(A)/Z(A)` is **exactly updatable**. Full self-attention, where queries depend on other tokens, is not, because `O(|A|)` pairs change per move. Numerics: `e^{⟨q,k_j⟩}` has to be precomputed per feature (it depends only on `j`), so the weights become `w_j = e^{s_j}` and `w_j v_j`. This is just an accumulator with an extra scalar channel `Z`, followed by a division. The precision of `Z` in int32 needs checking [MEASURE].
- **Zobrist hashing [CITED: Zobrist 1970].** The original incremental "feature accumulator" in game programming: `H(A') = H(A) ⊕ ⊕_{i∈R∪D} z_i`. It is a group homomorphism into `(𝔽₂^{64}, ⊕)`, where every element is its own inverse. NNUE generalizes this pattern from `𝔽₂` to `ℤ` (int16).

**Exact sparse delta propagation through layer 2 (DeltaCNN-style) [ALGEBRA].** Let `h = σ(a)`. Layer 2 computes `z = W² h`. If only coordinates `C = {c : σ(a'_c) ≠ σ(a_c)}` change, then `z' = z + Σ_{c∈C} W²_c (h'_c − h_c)`, which costs `|C|·L2` instead of `d·L2`. With ClippedReLU, coordinates that are saturated (0 or 1) both before and after do not change. **[MEASURE]:** the distribution of `|C|/d` per move for a trained Stockfish-like net. If it is typically below about 0.3, fc_0 can be made incremental too.

---

## 8. Mixture of experts and bucketing

**Sparse MoE [CITED: Shazeer et al. 2017].** `y = Σ_e G(x)_e E_e(x)` with noisy top-k gating, `G(x) = Softmax(KeepTopK(H(x), k))`. Up to about 137B parameters, with an LSTM backbone and up to thousands of experts. Compute per example scales with `k`, not with the number of experts.

**NNUE bucketing as hard MoE [CITED: Stockfish].** `LayerStacks = 32` heads, selected by a cheap discrete function of the position (historically piece count). PSQT buckets give a separate linear output per bucket. King buckets on the *input side* (HalfKA) are a hard gate on which weight table is used. The two kinds behave differently:
- **Output-side gating** (head selection by a discrete statistic of `A`) costs nothing for updates. The accumulator is shared, and only the selected head runs.
- **Input-side gating** `a = Σ_{i∈A} W^{(b(A))}_i` breaks additivity when `b` changes. When `b(A') ≠ b(A)`, a refresh costs `|A'|·d`. Accumulator caches keyed by bucket ("Finny tables" in engine-programming jargon [UNVERIFIED: origin]) reduce this to a diff against the cached position for that bucket.
- **Soft/continuous gating** on a dense hidden state makes every expert's input dense, so experts must sit after the accumulator.

**Design rule [ALGEBRA].** A gate is update-compatible iff it is a function of updatable statistics (§0). Its switch frequency sets the amortized refresh cost: `E[cost] = k·d + Pr[switch]·(refresh cost)`.

---

## 9. Universal approximation and the Kolmogorov–Arnold representation

**Universal approximation [CITED].**
- Cybenko (1989): finite sums `Σ_j α_j σ(w_j^T x + θ_j)` with continuous sigmoidal `σ` are dense in `C([0,1]^n)`.
- Hornik (1991): density holds for any bounded nonconstant `σ`, in `L^p(μ)` and in `C(X)` for compact `X`.
- Leshno, Lin, Pinkus & Schocken (1993): for locally bounded, piecewise continuous `σ`, density holds **iff `σ` is not a polynomial**.
- Barron (1993): an `O(C_f²/n)` squared-error rate with `n` sigmoidal units for functions with finite first Fourier moment `C_f`.

**Implications.**
1. An accumulator with a non-polynomial readout (NNUE) is universal as `d → ∞`. A *fixed-degree polynomial readout* (pure FM/HOFM) is **not** universal over `ℝ^n`, by Leshno. Pure FM must therefore be combined with a CPWL/clipped nonlinearity.
2. **The binary domain is special [ALGEBRA, Lean target].** On `{0,1}^n` every function is a multilinear polynomial: the Fourier–Walsh expansion, `f(x) = Σ_{S⊆[n]} \hat f(S) Π_{i∈S} x_i` (O'Donnell 2014). Degree `≤ m` truncations are exactly what the ANOVA / power-sum accumulators of §5 compute in low-rank form. So "degree-m low-rank Walsh approximation" is the precise expressivity class of an m-th order VNNUE readout.
3. **Width-1 universality on finite domains (warning) [ALGEBRA].** If `φ : U → ℝ` is injective on subsets, e.g. `φ(i) = 2^i`, then `A ↦ Σ_{i∈A} φ(i)` is injective and any `f = g ∘ Σφ`. So "universality" over chess positions is trivial and tells us nothing. The real constraints are **precision** (int16 accumulators allow at most 2^16 distinct values per coordinate) and **smoothness/learnability of `g`**. Arguments in the paper should therefore be about quantized capacity and approximation *rate*, not about density.

**Kolmogorov–Arnold theorem [CITED: Kolmogorov 1957; Arnold 1957].** Every continuous `f : [0,1]^n → ℝ` can be written

    f(x_1,…,x_n) = Σ_{q=0}^{2n} Φ_q ( Σ_{p=1}^{n} ψ_{q,p}(x_p) )

with continuous univariate `Φ_q` and `ψ_{q,p}`. The inner functions can be made independent of `f` (Kolmogorov; Sprecher 1965 reduced them to shifts and scalings of one function). Girosi & Poggio (1989) argued the theorem is "irrelevant" for learning because the inner functions are highly non-smooth; Kůrková (1991) responded "Kolmogorov's theorem is relevant".

**Structural link to NNUE [ALGEBRA, Lean target].** For binary inputs, `ψ(x_p) = ψ(0) + x_p(ψ(1) − ψ(0))`, so every inner sum is an **affine accumulator** `c_q + Σ_{p∈A} w_{q,p}`. The KA representation restricted to the cube is therefore exactly "a width-(2n+1) additive accumulator followed by a sum of univariate readouts". This is a one-hidden-layer NNUE with *learnable per-unit activations* `Φ_q` and a fixed sum. **KANs [CITED: Liu et al. 2024/ICLR 2025]** make `Φ` and `ψ` learnable B-splines on edges. For VNNUE, the first KAN layer on binary inputs collapses to a standard accumulator (`ψ` is affine on {0,1}). Only the **readout activations** benefit from KAN-style learnable splines, and those can be implemented as small lookup tables over the quantized accumulator value (int16 → int8 LUT). Such a LUT is an exact representation of any univariate function on a finite grid.

---

## 10. Invariance, equivariance and weight sharing

**Deep Sets [CITED: Zaheer et al. 2017].** For a countable universe, a set function is permutation-invariant iff it has the form `ρ(Σ_{x∈X} φ(x))` (Thm 2). The NNUE accumulator is exactly sum-pooling Deep Sets over pieces, with `φ(piece, square, king) = W_i`. **Limitation [CITED: Wagstaff et al. 2019]:** for continuous `φ, ρ`, the latent dimension must be at least the maximum set size to represent all continuous permutation-invariant functions. For chess (`|A| ≤ 32`, finite universe) this does not bind, but it suggests `d ≥ 32` per independent "set channel" as a lower bound heuristic.

**Group equivariance [CITED: Cohen & Welling 2016].** Weight sharing over a symmetry group `G` reduces parameters by `|G|` and imposes `f(g·x) = ρ(g) f(x)`. Chess symmetries that NNUE already uses:
- Colour flip plus vertical mirror gives *perspective* weight sharing: the same `W` for both sides, with the board flipped.
- Horizontal mirror when castling rights are gone. HalfKAv2_hm mirrors so the king is on files e–h. It is a *partial* symmetry (castling breaks it) applied by input canonicalization, which halves the feature transformer.

In a VNNUE context, canonicalization costs a refresh when the canonical frame flips (the king crosses the d/e boundary). This is the same amortized-refresh trade-off as §8.

---

## 11. Summary table: interaction structures and update cost

| Structure | Exact identity | Accumulators | Update cost per move | Readout |
|---|---|---|---|---|
| Linear (NNUE L1) | `a = b + Σ_A W_i` | d | k·d | dense head |
| Virtual/factorized features | `W_i = R_i + Σ_s Q^{(s)}_{π_s(i)}` | d | k·d (merged) | same |
| Hashed features | `Σ_A ξ(i) W_{h(i)}` | d | k·d | same |
| Explicit pairs (Threats, PP) | `Σ_{(i,j)} x_i x_j W_{ij}` | d | Δpairs·d | same |
| FM (2nd order, rank r) | `½(‖s‖² − q)` | r+1 | k·(r+1) | O(r) |
| FFM (F fields) | `Σ_{F<G}⟨S_{FG},S_{GF}⟩ + …` | F²·r | k·F·r | O(F²·r), or incremental O(k·F·r) |
| HOFM degree m | Newton–Girard on `p_1..p_m` | m·r | k·m·r | O(m²·r) |
| Hadamard of accumulators | `(W¹x)⊙(W²x)`: rank-1 quadratics | 2d | 2k·d | d |
| Fixed-query softmax pool | `N/Z` | d_v+1 | k·(d_v+1) | d_v divisions |
| Linear attention | `φ(q)^T S / φ(q)^T z` | r·d_v + r | k·r·d_v | depends on q |
| Max-pool | `max_A φ(i)` | d (+heap) | deletion not O(1) | — |
| Full self-attention | softmax(QK^T)V | — | Ω(|A|·k·d) | not updatable |

---

## Implications for VNNUE

Concrete, testable design ideas, ordered by expected payoff. Every identity should go to Lean (`Finset.sum` lemmas, `Finset.sum_insert`, `Finset.sum_erase`, `inner_add_left`, Newton identities). Every performance claim needs to be measured.

1. **FM-augmented accumulator ("quadratic channel").** Next to the `d`-wide NNUE accumulator `a`, keep `s ∈ ℤ^r` (`r ≈ 16–64`) and the scalar `q`, and add `½(‖s‖² − q)` (or `r'` such channels) to the head input. This captures all `C(|A|,2) ≤ 496` piece-pair interactions at an extra update cost of `k·(r+1)`, under 6% of `k·1024` for `r=64`. **Lean:** Rendle identity over `Finset`, plus the update lemma `‖s+δ‖² = ‖s‖² + 2⟨s,δ⟩ + ‖δ‖²`. **Experiment:** compare loss and Elo against explicit pawn-pair features (PP_3Wide, 4,560 dims) at equal compute.

2. **Field-aware pair channels with chess fields.** Use fields = (side × piece type) (12). FFM accumulators `S_{F,G}` (`12×12×r`), with the readout maintained incrementally via `Σ_G⟨δ_{F,G}, S_{G,F}⟩`. This is a learned, low-rank generalization of `FullThreats` that does not depend on attack geometry (so it is cheaper to compute diffs). The two are complementary: explicit threats encode geometry, FFM encodes "what pieces co-exist".

3. **Power-sum (Newton–Girard) accumulators for degree-3 interactions.** Keep coordinate-wise `p_1, p_2, p_3` (`3r` integers) and compute `e_2, e_3` exactly in the readout. The theorem for the paper: *"Any symmetric polynomial readout of degree ≤ m over per-feature embeddings is O(k·m·r)-updatable."* It follows from Newton's identities, which Mathlib has. Integer overflow bounds (int32 for `p_3` with int8 embeddings) should also be proved in Lean.

4. **Updatability characterization theorem (main theoretical contribution).** Formalize: (i) `T` is additively updatable ⟺ `T(A) = T(∅) + Σ_{i∈A} φ(i)`; (ii) deletion in O(m) requires a group, so max-pooling is excluded (counterexample in Lean); (iii) consequently, every O(k·m)-updatable evaluator is `g ∘ (additive accumulators)`. VNNUE's novelty then lies entirely in the choice of `φ` (structured, factorized, hashed, power sums) and `g` (CPWL + low-degree polynomial + LUT activations).

5. **Exact delta propagation into fc_0 (DeltaCNN-style).** Store `h = clip(a)` and `fc_0` pre-activations. After an update, find coordinates whose clipped value changed and apply `z += W²_c Δh_c`. The update is exact (no drift), because integer arithmetic is associative and commutative, which is provable in Lean over `ℤ`. Payoff depends on [MEASURE] `|C|/d`. Encourage it in training with a penalty on `E|Δh|_0` between parent/child positions, the analogue of the delta-network training-time sparsity in Neil et al. 2017.

6. **Block-sparse feature columns.** Constrain each `W_i` to `s ≪ d` nonzero coordinates, chosen per piece-type block. Update cost drops to `k·s`, and §5's delta propagation gets guaranteed `|C| ≤ k·s`. This trades expressivity for speed and is a natural ablation axis.

7. **Hashed virtual feature spaces in training.** Generalize the factorizer: add hashed virtual features for pair or triple patterns at training time only. Merging is exact as long as the virtual features are functions of a *single* real feature (§1). Pair-hashed features cannot be merged into single-feature weights. They stay at inference, with O(Δpairs·d) cost, and the hash keeps the table bounded.

8. **Output-side MoE over richer gates.** Keep 32 layer stacks but gate on an *updatable* statistic richer than piece count, e.g. `argmax` of a small accumulator projection, or a material-signature hash. Free under the §8 rule, because only the head changes.

9. **Fixed-query softmax pooling heads.** A few learned queries `q_h` give attention-like "which pieces matter" pooling at accumulator cost: `N_h/Z_h`, with `e^{⟨q_h,k_j⟩}` folded into per-feature weights at export. Open issues are quantized division and the dynamic range of `Z` [MEASURE].

10. **LUT (KAN-style) readout activations.** Replace `ClippedReLU`/`SqrClippedReLU` on the 32-wide `fc_0` output with learned per-neuron univariate functions, as splines in training and int8 LUTs at inference. KA theory says the expressive burden can sit in univariate readouts over additive inner sums. Cost: 32 table lookups. Stockfish's current dual activation (`ClippedReLU` ∥ `SqrClippedReLU`) is already a 2-basis instance of this.

**Things to avoid** (provably non-updatable or with refresh costs): dense learned routing before the accumulator, full self-attention over pieces, max-pooling with deletions, and input-side gates that switch often.

---

## References

Format: Author(s) (Year). Title. Venue. Identifier/URL. Status.

1. Arnold, V. I. (1957). On functions of three variables. *Doklady Akademii Nauk SSSR* 114:679–681. [UNVERIFIED page numbers]
2. Arora, R., Basu, A., Mianjy, P., Mukherjee, A. (2018). Understanding Deep Neural Networks with Rectified Linear Units. *ICLR 2018*. arXiv:1611.01491. [UNVERIFIED this session; well known]
3. Barron, A. R. (1993). Universal approximation bounds for superpositions of a sigmoidal function. *IEEE Trans. Information Theory* 39(3):930–945. [UNVERIFIED this session]
4. Blondel, M., Fujino, A., Ueda, N., Ishihata, M. (2016). Higher-Order Factorization Machines. *NeurIPS 2016*. arXiv:1607.07195. [UNVERIFIED this session]
5. Blondel, M., Ishihata, M., Fujino, A., Ueda, N. (2016). Polynomial Networks and Factorization Machines: New Insights and Efficient Training Algorithms. *ICML 2016*. arXiv:1607.08810. [UNVERIFIED this session]
6. Chrysos, G., Moschoglou, S., Bouritsas, G., Panagakis, Y., Deng, J., Zafeiriou, S. (2020). Π-nets: Deep Polynomial Neural Networks. *CVPR 2020*. arXiv:2003.03828. [UNVERIFIED this session]
7. Cohen, T., Welling, M. (2016). Group Equivariant Convolutional Networks. *ICML 2016*, PMLR 48. arXiv:1602.07576. [UNVERIFIED this session]
8. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems* 2(4):303–314. [UNVERIFIED this session; standard]
9. Denil, M., Shakibi, B., Dinh, L., Ranzato, M., de Freitas, N. (2013). Predicting Parameters in Deep Learning. *NeurIPS 2013*. arXiv:1306.0543. [UNVERIFIED this session]
10. Girosi, F., Poggio, T. (1989). Representation properties of networks: Kolmogorov's theorem is irrelevant. *Neural Computation* 1(4):465–469. [UNVERIFIED this session]
11. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks* 4(2):251–257. [UNVERIFIED this session; standard]
12. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685 (ICLR 2022). [UNVERIFIED this session]
13. Jaderberg, M., Vedaldi, A., Zisserman, A. (2014). Speeding up Convolutional Neural Networks with Low Rank Expansions. *BMVC 2014*. arXiv:1405.3866. [UNVERIFIED this session]
14. Juan, Y., Zhuang, Y., Chin, W.-S., Lin, C.-J. (2016). Field-aware Factorization Machines for CTR Prediction. *ACM RecSys 2016*. [UNVERIFIED this session]
15. Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *ICML 2020*. arXiv:2006.16236. [UNVERIFIED this session]
16. Kolmogorov, A. N. (1957). On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition. *Doklady Akademii Nauk SSSR* 114:953–956. [UNVERIFIED page numbers]
17. Kůrková, V. (1991). Kolmogorov's theorem is relevant. *Neural Computation* 3(4):617–622. [UNVERIFIED this session]
18. Lample, G., Sablayrolles, A., Ranzato, M., Denoyer, L., Jégou, H. (2019). Large Memory Layers with Product Keys. *NeurIPS 2019*. arXiv:1907.05242. [UNVERIFIED this session]
19. Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., Teh, Y. W. (2019). Set Transformer. *ICML 2019*. arXiv:1810.00825. [UNVERIFIED this session]
20. Leshno, M., Lin, V. Ya., Pinkus, A., Schocken, S. (1993). Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. *Neural Networks* 6(6):861–867. [UNVERIFIED this session; standard]
21. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., Tegmark, M. (2024/2025). KAN: Kolmogorov–Arnold Networks. arXiv:2404.19756; *ICLR 2025* (openreview Ozo7qJ5vZi). Verified.
22. Montúfar, G., Pascanu, R., Cho, K., Bengio, Y. (2014). On the Number of Linear Regions of Deep Neural Networks. *NeurIPS 2014*. arXiv:1402.1869. [UNVERIFIED this session]
23. Nasu, Y. (2018). ƎUИИ: Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi. Ziosoft Computer Shogi Club / 28th World Computer Shogi Championship appeal document. https://www.apply.computer-shogi.org/wcsc28/appeal/the_end_of_genesis_T.N.K.evolution_turbo_type_D/nnue.pdf ; English translation by D. Klein (2021), https://github.com/asdfjkl/nnue. Verified.
24. Neil, D., Lee, J. H., Delbruck, T., Liu, S.-C. (2017). Delta Networks for Optimized Recurrent Network Computation. *ICML 2017*, PMLR 70:2584–2593. https://proceedings.mlr.press/v70/neil17a.html ; arXiv:1612.05571. Verified.
25. O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge University Press. [UNVERIFIED this session; standard]
26. official-stockfish (2020–2026). nnue-pytorch documentation, "NNUE" (feature factorization, HalfKAv2_hm). https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html ; source `halfka_v2.py`. Verified.
27. official-stockfish (2026). Stockfish source, `src/nnue/nnue_architecture.h`, `src/nnue/features/full_threats.h`, `src/nnue/features/pp_3wide.h` (master, fetched 2026-11-09). https://github.com/official-stockfish/Stockfish. Verified.
28. Pan, B., Lin, W., Fang, X., Huang, C., Zhou, B., Lu, C. (2018). Recurrent Residual Module for Fast Inference in Videos. *CVPR 2018*, pp. 1536–1545. arXiv:1802.09723. Verified.
29. Parger, M., Tang, C., Twigg, C. D., Keskin, C., Wang, R., Steinberger, M. (2022). DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos. *CVPR 2022*, pp. 12487–12496. arXiv:2203.03996. Verified.
30. Rendle, S. (2010). Factorization Machines. *IEEE ICDM 2010*, pp. 995–1000. [UNVERIFIED page numbers]
31. Sainath, T. N., Kingsbury, B., Sindhwani, V., Arisoy, E., Ramabhadran, B. (2013). Low-rank matrix factorization for deep neural network training with high-dimensional output targets. *ICASSP 2013*. [UNVERIFIED this session]
32. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR 2017*. arXiv:1701.06538. [UNVERIFIED this session; standard]
33. Sprecher, D. A. (1965). On the structure of continuous functions of several variables. *Trans. AMS* 115:340–355. [UNVERIFIED]
34. Wagstaff, E., Fuchs, F., Engelcke, M., Posner, I., Osborne, M. A. (2019). On the Limitations of Representing Functions on Sets. *ICML 2019*. arXiv:1901.09006. [UNVERIFIED this session]
35. Weinberger, K., Dasgupta, A., Langford, J., Smola, A., Attenberg, J. (2009). Feature Hashing for Large Scale Multitask Learning. *ICML 2009*. arXiv:0902.2206. [UNVERIFIED this session; standard]
36. Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., Smola, A. (2017). Deep Sets. *NeurIPS 2017*. arXiv:1703.06114. [UNVERIFIED this session]
37. Zobrist, A. L. (1970). A New Hashing Method with Application for Game Playing. Technical Report 88, Computer Sciences Dept., University of Wisconsin. [UNVERIFIED this session; standard]
