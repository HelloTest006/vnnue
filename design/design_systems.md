# VNNUE design: systems-first variant ("VNNUE-S")

Status: design draft, 2026-11-09. Angle: **reduce the real cost of keeping the first layer current**, measured as memory traffic, refreshes and SIMD lanes, while keeping the evaluation exactly incrementally updatable and keeping plain NNUE as a special case.

Evidence tags:
- **[Lean]**: listed in Section 9 and to be proved in Lean 4 + Mathlib.
- **[Exp]**: has to be measured by us.
- **[V]/[K]/[UNVERIFIED]**: inherited from the research notes (`research/*.md`). [V] was checked online, [K] is a standard reference that still needs re-checking, [UNVERIFIED] must not be used as fact.

The search-side notes (`research/search.md`) were never written because the disk was full. Nothing below depends on them. The Elo-conversion protocol in Section 10 is stated generically for that reason.

---

## 0. Baseline being generalized

NNUE as used in Stockfish (HalfKAv2_hm feature transformer, FT) [V: nnue.md §4–6; quant.md §1]:
- Per perspective c, there are N = 32 · 704 = 22,528 binary features (mirrored own-king bucket κ ∈ K, |K| = 32) × (11 piece planes × 64 squares).
- The accumulator is a_c = b + Σ_{i∈A_c} W_i ∈ ℤ^L, with int16 lanes and L = 1024 on current master.
- The output stage is pairwise-product clipped activations into a sparse 1024→32 layer and a 32-way layer stack.
- Extra feature families (Full_Threats, PP_3Wide) have their own accumulators [V].
- Update cost per perspective, counted in column operations (one L-wide add or subtract):
  - quiet move: 2
  - capture: 3
  - castling: 4
  - own-king move: refresh, costing |S| ≤ 32 columns, or |C ⊖ S| with a Finny cache [V].
- Lazy evaluation walks back to the last computed accumulator and replays the plies forward [V: asteri.sm].
- Quantization is safe only because the trainer clamps weights so that "no combination of possible active features can exceed the maximum value" [V: nnue-pytorch docs].

Systems facts that drive the design:
1. **FT updates are memory-bound.** Arithmetic intensity is I ≈ (a+r)/(2(a+r)+4) ≤ 0.5 op/byte [DERIVED, quant.md §6.2].
2. **The FT table is far larger than the private caches.** The table is 46 MB (22,528 × 1024 × 2 B) per weight block [DERIVED]. Each column is 2 KB, which is 32 cache lines.
3. **Refreshes are the expensive tail.** They are caused by king moves, d/e mirror crossings and material-block changes [V].

The VNNUE-S levers below each target one of these facts. The output stage (pairwise product, hidden stack, layer stacks) is **unchanged**. VNNUE-S only changes how the accumulator is represented and maintained.

---

## 1. Algebraic core

Let F be a finite feature universe, and let x ∈ ℤ^F be the indicator of the active set A (a position) or a signed **delta** δ ∈ ℤ^F. Let G be a commutative ring of lane values: ℤ for the specification, ZMod 2^16 for the implementation. Given W : F → G^L and b ∈ G^L, define

  acc(x) = b + Σ_{i∈F} x_i · W_i.

Linear part. Φ(x) := Σ x_i W_i is an additive group homomorphism ℤ^F → G^L. Therefore:
- **Delta rule:** acc(x + δ) = acc(x) + Φ(δ).
- **Path independence:** for any sequence δ_1, …, δ_t with x_t = x_0 + Σ_j δ_j, folding the deltas gives acc(x_t), whatever the order or grouping. [Lean T1]
- **Delta of a set change:** for sets U → S, δ = 1_{S∖U} − 1_{U∖S}, so the update costs |S ⊖ U| columns. [Lean T1, T9]

Everything below is an instance of this algebra plus a cost model.

---

## 2. Components of VNNUE-S

### 2.1 Factored king conditioning (Woodbury-style bucket split)

NNUE stores a full block W_κ ∈ ℤ^{L×704} for every king bucket. VNNUE-S writes the block for bucket κ as

  W_κ = W⁰ + R_{ρ(κ)} + U_κ V,

where:
- W⁰ ∈ ℤ^{L×704} is shared by all buckets;
- ρ : K → P maps the 32 king buckets onto |P| coarse **regions**, each with a full-rank residual R_p ∈ ℤ^{L×704};
- V ∈ ℤ^{r×704} is shared, and U_κ ∈ ℤ^{L×r} is per bucket, with a small rank r (e.g. 8–32).

**Strict generalization.** Taking ρ = id (|P| = 32), W⁰ = 0 and r = 0 gives exactly HalfKAv2_hm. Taking |P| = 1 and r = 0 gives the king-independent "A"-style table. Any HalfKAv2_hm table is reachable, and the design adds a continuum of cheaper points. [Lean T8b]

**Auxiliary accumulator.** z(x) = Vx ∈ ℤ^r, maintained per perspective at k·r scalar adds per move. For r ≤ 16 this fits in one AVX2 int16 register.

**King-transition identity** [Lean T8]. For buckets κ, κ' in the same region and positions x, x', write M_κ := W⁰ + R_{ρ(κ)} + U_κ V for the merged table. Then

  M_{κ'} x' = M_κ x + M_{κ'}(x' − x) + (U_{κ'} − U_κ)(V x).

A king move inside a region therefore costs k columns of M_{κ'} plus r columns of (U_{κ'} − U_κ) scaled by the stored z, with no refresh. The identity holds over any commutative ring, so it also holds in ZMod 2^16 (Section 2.4). Only the final value has to fit in int16.

Two storage modes, chosen by measurement [Exp]:

| Mode | Stored | Per-move cost | Per-eval extra |
|---|---|---|---|
| **Merged** | M_κ for all κ (as NNUE) + U_κ, V | k cols (+ r cols on intra-region king move) | 0 |
| **Factored** | (W⁰+R_p) per region + U_κ, V | k cols of region table, king moves inside a region cost k | r cols (U_κ z, multiply–add) |

In factored mode the state is (a_base, z) with a_base = (W⁰ + R_p)x. The evaluated accumulator is a = a_base + U_κ z.
- Any rounding or shift needed to put U_κ z on the FT scale is applied **at readout**, as a deterministic function of the exact state. The state never drifts.
- Merged mode needs integer factors whose scales multiply to the FT scale (s_U · s_V = s), so that M_κ is exact. This is a QAT constraint and a risk (Section 11).

### 2.2 Net-delta path compression (lazy catch-up)

On the search stack, ply j stores a sparse signed delta δ_j (dirty pieces). To evaluate at ply t from the nearest materialized ancestor u:
- **NNUE** replays t − u fused updates. Each reads k_j columns and loads and stores the accumulator.
- **VNNUE-S** computes the net delta Δ = Σ_{j=u+1}^{t} δ_j over sparse index lists (a small hash or sort over at most Σk_j ≤ 4(t−u) indices). It then applies Δ in **one** tile pass.
  - Terms cancel: a piece that moves a→b→c contributes −a +c, and b vanishes. A move-and-return contributes nothing.
  - Correctness: by T1 the result equals sequential replay. [Lean T2]
  - Column count: |supp Δ| ≤ Σ_j |supp δ_j|. [Lean T2]
  - Traffic: with column bytes c = L·β_w and accumulator bytes a = L·β_a, a fused pass moves |supp Δ|·c + 2a bytes, versus Σ_j (k_j c + 2a) sequentially. The saving is at least 2a(t−u−1) bytes. [Lean T3]
- If the path crosses an intra-region king move, the deltas are split at the bucket change and joined with the T8 correction term, using z stored at the junction ply.

### 2.3 Source selection: provably-no-worse refresh

When the bucket changes, or a catch-up is long, the accumulator for the current active set S (bucket κ) can be built from any **source** σ with a stored exact accumulator over set U_σ. The candidates are:
1. the nearest materialized stack ancestor (net delta, Section 2.2);
2. the Finny entry for κ (cost |C_κ ⊖ S|) [V: Stockfish refresh table];
3. the Finny entry for any κ'' with ρ(κ'') = ρ(κ), plus a rank-r correction (cost |C_{κ''} ⊖ S| + r; the entry also stores z(C_{κ''}));
4. the empty set (cost |S|).

Policy: pick the argmin of cost(σ). This choice is made offline at evaluation time, since every cost is computable from bitboards before any column is touched.

Guarantees:
- **Correct for every source.** Building from any σ gives exactly acc(S), whatever the cache contents. [Lean T9]
- **Never worse than Stockfish.** min over a superset ≤ min over a subset, and Stockfish's candidates {Finny κ, empty} are a subset. [Lean T10]
- **Cost as a distance.** |· ⊖ ·| is a metric on Finsets (triangle inequality), so routing through an intermediate source never beats the direct diff in column count. [Lean T10]

### 2.4 Wrap-around exact integer accumulators with certificates

Specification:
- Lane values live in **ZMod 2^16** (wrapping `vpaddw`, `vpsubw`, `vpmullw`).
- The readout decodes the lane value y with dec(y) = y.val if y.val < 2^15, else y.val − 2^16.

Theorem T4 [Lean]: if the true integer value v of a lane at the *final* position satisfies −2^15 ≤ v < 2^15, then dec(cast v) = v, whatever overflow happened along the update path. The cast ℤ → ZMod 2^16 is a ring homomorphism, so this covers T1, T2 and T8, including the products in U_κ z.

Contrast: saturating int16 addition is **not** associative, e.g. sat(sat(32767+1) − 1) = 32766 while sat(32767 + sat(1 − 1)) = 32767. Saturating updates are therefore path-dependent. [Lean T5]

**Per-lane slot certificate.** For each output lane ℓ:
- Partition the feature indices into groups G_1..G_m, e.g. by (colour, piece type), with capacities c_t equal to the maximum number of simultaneously active features in G_t on a legal board. The capacities must respect promotions. A safe choice is a cap per colour of 16 pieces and per-type caps of at most 10 (knights, bishops, rooks), 9 (queens) and 8 (pawns), which are true of legal positions [K: rules of chess].
- Let M_t⁺ = max(0, max_{i∈G_t} W_{ℓ,i}) and M_t⁻ = min(0, min_{i∈G_t} W_{ℓ,i}).
- Then b_ℓ + Σ_t c_t M_t⁻ ≤ acc_ℓ(A) ≤ b_ℓ + Σ_t c_t M_t⁺ for every admissible A. [Lean T6]

An offline script checks the certificate for a trained net in O(L · |F|). The global bound |acc| ≤ |b| + |A|·max|W| [Lean T7] is the coarse special case.

Systems payoff:
- (i) It replaces folklore clamps with a checkable condition.
- (ii) It lets the trainer **raise the FT scale s** (e.g. 255 or 511) wherever the certificate still fits. The rounding bound (n+1)/(2s) [Lean T11] shrinks as s grows.
- (iii) It justifies **i8 column storage widened to i16 lanes** (β_w = 1, as Stockfish already does for threats [V]) for any feature family whose certificate fits. This halves column traffic.

### 2.5 Symmetry-tied columns (optional, memory)

Perspective sharing (colour flip plus vertical mirror) is already exact in NNUE.

VNNUE-S optionally ties the region tables under the horizontal mirror μ:

  (W⁰ + R_p)_{μ(i)} = P (W⁰ + R_p)_i,

where P is a **block-local lane permutation** (a permutation inside each 16-lane SIMD block, i.e. one `vpshufb`-class shuffle). Then acc(μ·A) = P·acc(A) when Pb = b [Lean T12]. A d/e-file crossing then costs one O(L) shuffle pass instead of a refresh.

The constraint halves the stored region tables. Whether it costs accuracy is an open question to answer by experiment [Exp]. HalfKAv2_hm's canonicalization is the "untied" alternative.

### 2.6 Materialization policy (ski rental)

Lazy evaluation trades two costs:
- **Rent.** Leaving an interior ply u un-materialized means each later catch-up through u pays c_u ≥ 1 extra column operations, its share of the net delta.
- **Buy.** Materializing u costs B_u, one pass plus a store.

The number n of future descendants that will need u is unknown online. The break-even rule "materialize at the B-th use" costs ALG(n) = n·c for n < B and (B−1)c + Bc otherwise.

With unit costs: B·ALG(n) ≤ (2B − 1)·OPT(n) for every n, where OPT(n) = min(n, B). No deterministic threshold does better in the worst case. [Lean T13]

The search tree is not adversarial, so the worst-case ratio is only a safety net. The tuned threshold is set by measurement [Exp].

---

## 3. Full VNNUE-S evaluation (per perspective c)

1. The state is (a_c, z_c, κ_c), plus optional threat and pair accumulators (unchanged from NNUE, but under the same certificate and net-delta machinery).
2. Readout: â_c = dec(a_c) (merged mode) or dec(a_c + U_κ z_c) (factored mode). Then the NNUE output stage: pairwise-product clamp, sparse 1024→32, layer stack.
3. Update on a move: choose the source by Section 2.3. The default is the parent with a k-column delta, plus the T8 correction on an intra-region king move.

---

## 4. Update cost per move type (per perspective, L-wide column operations)

Notation:
- k: piece-feature changes;
- r: rank (scalar z updates cost k·r scalar adds, i.e. ≈ k/ (L/r) columns, negligible);
- C: Finny cache set.

| Move type | NNUE (HalfKAv2_hm + Finny) | VNNUE-S merged | VNNUE-S factored |
|---|---|---|---|
| Quiet, non-king | 2 | 2 | 2 (+r at eval) |
| Capture / en passant / capture-promotion | 3 | 3 | 3 (+r at eval) |
| Promotion | 2 | 2 | 2 (+r at eval) |
| Opponent king move (our view) | 2–3 | 2–3 | 2–3 |
| Own king move, same region | min(\|S\|, \|C_κ' ⊖ S\|) | min(k + r, \|C_κ' ⊖ S\|, …) | k (+r at eval) |
| Castling (own) | min(\|S\|, \|C ⊖ S\|) | min(4 + r, …) if the region is unchanged | 4 (+r at eval) |
| Region or mirror change | min(\|S\|, \|C ⊖ S\|) | min over Section 2.3 candidates (≤ NNUE) | same (≤ NNUE); with 2.5 tied: 1 shuffle pass + k |
| t-ply catch-up | Σ_j k_j cols + t acc load/stores | \|supp Σδ_j\| cols + 1 load/store | same |

Here k + r < |S| whenever r < |S| − k, e.g. r = 8 against a typical middlegame |S| ≈ 25 [Exp: |S| and |C ⊖ S| distributions].

Bytes per update are cost·L·β_w + 2·L·β_a, with β_a = 2 and β_w ∈ {1, 2} chosen per family by the certificate.

---

## 5. Memory cost (L = 1024, β_w = 2 unless noted)

| Configuration | Feature-transformer bytes |
|---|---|
| NNUE HalfKAv2_hm, one material block | 32·704·1024·2 ≈ 46.1 MB [DERIVED] |
| VNNUE-S merged, r = 16 | 46.1 MB + U: 32·16·1024·2 ≈ 1.05 MB + V: 704·16·2 ≈ 22.5 KB (≈ +2.3%) |
| VNNUE-S factored, \|P\| = 4, r = 16 | 4·704·1024·2 ≈ 5.77 MB + 1.07 MB ≈ 6.8 MB (≈ 6.7× smaller) |
| Factored + mirror tying (2.5) | ≈ 2.9 + 1.07 ≈ 4.0 MB |
| Any of the above with i8 columns (certificate permitting) | ≈ ½ of the column part |

Small tables also make Finny-style caches and repeated column reads more likely to hit in L2 or L3. This is an [Exp] claim, to be measured with hardware counters (LLC misses per node, bytes per node).

---

## 6. Why each lever is mathematically safe

All levers are reparametrizations or reorderings of an exact integer linear map, followed by a decode that only needs the final value in range:
- **2.1** is distributivity and associativity of matrix products (T8);
- **2.2** is commutativity and associativity of the group operation (T1, T2);
- **2.3** is "any source plus its delta" (T9) together with min monotonicity (T10);
- **2.4** is ring-homomorphism plus injectivity of the decode on a window of width 2^16 (T4), with range certificates T6/T7;
- **2.5** is reindexing a finite sum (T12).

None of these changes the function computed. Only the **trained** parameters (r, |P|, tying, i8 scale) change what the net can express. That is the question for experiment.

---

## 7. Relation to prior work (for the paper)

- **Zobrist hashing** (1970) is the (ℤ/2)^64 instance of acc [K].
- **Finite differencing** (Paige & Koenig 1982) [K], **the change calculus** (Cai et al. PLDI 2014) [V] and **DBSP** (Budiu et al. PVLDB 2023) [V] give the "linear operators are self-incremental" principle used in T1/T2.
- **Sherman–Morrison–Woodbury** (1950) [V/K] is the base-plus-low-rank pattern of 2.1.
- **Ski rental and competitive analysis**: Karlin et al. 1988 [K]; Sleator & Tarjan 1985 [K].
- **Integer-only inference**: Jacob et al. CVPR 2018 [K].
- **Roofline model**: Williams et al. CACM 2009 [K].
- **Group-equivariant weight tying**: Cohen & Welling ICML 2016 [K].
- **Stockfish refresh tables**: commit 49ef4c9 [V].
- **i8 threat weights**: SFNNv10 [V].

---

## 8. What is new, stated carefully

- Stockfish already has pieces 1–3 below in some form: incremental updates, Finny caches, lazy catch-up, i8 threat weights and wrapping adds. **Whether Stockfish uses wrapping adds is [UNVERIFIED] in current source.**
- New in VNNUE-S:
  1. the low-rank king transition (2.1), which removes intra-region refreshes;
  2. net-delta cancellation across plies (2.2) [nnue.md §8.5 notes no engine is known to do this; UNVERIFIED];
  3. the source-selection superset with rank correction (2.3);
  4. per-lane slot certificates replacing global clamps (2.4);
  5. block-local-permutation mirror tying (2.5);
  6. the ski-rental materialization rule (2.6);
  7. machine-checked proofs of all the correctness and op-count claims.

---

## 9. Lean 4 / Mathlib theorem list

All theorems are over finite types, ℤ, ℚ, ℕ or ZMod (2^16), with `Fin n` indices and `Finset` sums. `G` is a commutative ring or `AddCommGroup`.

- **T1 (accumulator homomorphism, path independence).**
  - Define `acc (b : Fin L → G) (W : Fin N → Fin L → G) (x : Fin N → ℤ) := b + ∑ i, x i • W i`.
  - Then `acc b W (x + δ) = acc b W x + ∑ i, δ i • W i`.
  - For any `l : List (Fin N → ℤ)`, `l.foldl (fun a δ => a + ∑ i, δ i • W i) (acc b W x) = acc b W (x + l.sum)`.
- **T2 (net-delta cancellation).**
  - Sequential application of `l` equals the single application of `l.sum` (from T1).
  - `(Finset.univ.filter (fun i => l.sum i ≠ 0)).card ≤ (l.map (fun δ => (Finset.univ.filter (fun i => δ i ≠ 0)).card)).sum`.
- **T3 (fused-pass traffic, ℕ).** For `ks : List ℕ` of length t ≥ 1 and `knet ≤ ks.sum`: `knet*c + 2*a + 2*a*(t-1) ≤ (ks.map (fun k => k*c + 2*a)).sum`.
- **T4 (wrap-around exactness).**
  - Define `dec (y : ZMod 65536) : ℤ := if y.val < 32768 then y.val else y.val - 65536`.
  - Then `∀ v : ℤ, -32768 ≤ v → v < 32768 → dec (v : ZMod 65536) = v`.
  - Corollary with T1: the decoded ZMod accumulator after any update list equals the ℤ accumulator of the final position whenever the latter is in range.
- **T5 (saturation is path-dependent).** With `sat v := max (-32768) (min 32767 v)`: `sat (sat (32767 + 1) + (-1)) ≠ sat (32767 + sat (1 + (-1)))`, proved by `decide` or `norm_num`.
- **T6 (slot certificate).**
  - Setup: `g : Fin N → Fin m` (group map), capacities `cap : Fin m → ℕ`, weights `w : Fin N → ℤ`.
  - If `A : Finset (Fin N)` satisfies `∀ t, (A.filter (g · = t)).card ≤ cap t`, then `∑ i in A, w i ≤ ∑ t, cap t * Mplus t`, where `Mplus t := max 0 (sup over g⁻¹ t of w)`.
  - The symmetric lower bound holds with `Mminus`.
- **T7 (global range).** If `|w i| ≤ M` for all i, then `|b + ∑ i in A, w i| ≤ |b| + A.card * M`. Hence `|b| + n*M < 32768` and `A.card ≤ n` imply the decoded value is exact (with T4).
- **T8 (low-rank king transition).** For matrices over a commutative ring R (instantiated at ℤ and ZMod 65536), with `M κ := W0 + Rp + U κ * V`:
  - `(M κ').mulVec x' = (M κ).mulVec x + (M κ').mulVec (x' - x) + (U κ' - U κ).mulVec (V.mulVec x)`.
  - **T8b (factored = merged; generalization):** `(W0 + Rp + U*V).mulVec x = W0.mulVec x + Rp.mulVec x + U.mulVec (V.mulVec x)`, and with `U = 0`, `W0 = 0` the model equals the per-bucket table `Rp`.
- **T9 (any-source correctness).** For `S U : Finset (Fin N)`: `∑ i in S, W i = ∑ i in U, W i + ∑ i in S \ U, W i - ∑ i in U \ S, W i`, and the number of column operations is `(S \ U).card + (U \ S).card = (symmDiff S U).card`.
- **T10 (source selection).**
  - (a) `(symmDiff A C).card ≤ (symmDiff A B).card + (symmDiff B C).card`.
  - (b) For nonempty `C₁ ⊆ C₂` of candidates and a cost `f`: `C₂.inf' _ f ≤ C₁.inf' _ f`. Hence the VNNUE-S policy is never worse than the Stockfish candidate set.
- **T11 (quantized-sum rounding error, ℝ).** If `|q i - s * w i| ≤ 1/2` for all i ∈ A, `|qb - s*b| ≤ 1/2` and `s > 0`, then `|(qb + ∑ i in A, q i)/s - (b + ∑ i in A, w i)| ≤ (A.card + 1)/(2*s)`.
- **T12 (permutation equivariance).**
  - Setup: `σ : Equiv.Perm (Fin N)`, `P : Equiv.Perm (Fin L)`. Suppose `W (σ i) = W i ∘ P.symm` for all i and `b ∘ P.symm = b`.
  - Then `acc` of the image set `A.map σ` equals `(acc A) ∘ P.symm`.
- **T13 (ski rental, ℕ).**
  - Define `alg B n := if n < B then n else 2*B - 1` and `opt B n := min n B`.
  - Upper bound: `B ≥ 1 → B * alg B n ≤ (2*B - 1) * opt B n`.
  - Lower bound: for every threshold t with `algT t n := if n ≤ t then n else t + B`, `B * algT t (t+1) ≥ (2*B - 1) * opt B (t+1)`.
- **T14 (SIMD pair safety, ℤ).** `0 ≤ a₁ ≤ 127`, `0 ≤ a₂ ≤ 127`, `|w₁| ≤ 127` and `|w₂| ≤ 127` imply `|a₁*w₁ + a₂*w₂| ≤ 32258`. The witness `2*255*127 = 64770 > 32767` shows the 0..255 range is unsafe for `pmaddubsw` [instruction semantics: Intel SDM, K].

---

## 10. What needs experiments (cannot be proved)

1. **Accuracy cost** of r, |P|, mirror tying and i8 columns: validation loss, then fixed-node matches against the NNUE baseline trained on identical data.
2. **Path statistics** on real search trees:
   - distributions of k, |S| and |C ⊖ S|;
   - the net-delta ratio |supp Σδ| / Σ|supp δ|;
   - refresh frequency per cause (king move, mirror, region, material block);
   - reuse counts n for materialization (2.6).
3. **Speed:** nodes per second, bytes per node and LLC misses per node, measured with perf counters on AVX2, AVX-512 and ARM NEON.
4. **Playing strength:** fixed-time SPRT at two time controls. Elo depends on both speed and evaluation quality, so no Elo is claimed without these runs.
5. **Certificate tightness:** how much larger an FT scale the per-lane certificate admits than the global clamp, on a trained net.

---

## 11. Risks

- The low-rank king residual may lose accuracy against full HalfKA buckets. Fallback: |P| = 32 recovers NNUE exactly.
- Merged mode needs exact integer factorization at scale (s_U·s_V = s), which constrains QAT. Factored mode avoids this but adds r columns per evaluation.
- Net-delta bookkeeping (sort or hash of index lists) costs scalar overhead that may cancel the column savings when t is 1–2. This must be profiled.
- The certificate depends on correct legal-position capacities. Over-tight caps (e.g. ignoring promotions) would be unsound. The script must use provably valid caps.
- Mirror tying with a block-local P may be too restrictive to train well.
