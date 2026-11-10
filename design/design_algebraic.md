# VNNUE design, algebra-first angle (VNNUE-A)

Status: design draft, 2026-11-09. Evidence tags: **[Lean]** = to be proved in Lean 4 + Mathlib (theorem list in Section 7); **[Exp]** = must be measured; **[Cite]** = from a source listed in the research notes (`research/nnue.md`, `research/incremental_math.md`, `research/quant.md`, `research/efficient_nn.md`); **[UNVERIFIED]** = not yet checked. No playing-strength claim is made anywhere in this document.

## 0. Thesis

NNUE works because its first layer is a group homomorphism φ: ℤ^N → ℤ^L. A move changes O(1) features, so updating it costs O(1) column operations (Nasu 2018 [Cite]). The algebra-first question is which *non-linear* functions of the board can be maintained with the same kind of exact, integer, path-independent update at a small extra cost per changed feature.

The classical answer comes from the calculus of finite differences, specifically its Boolean form. If f: {0,1}^N → ℤ is a multilinear polynomial of degree d, flipping one feature changes f by a polynomial of degree d−1. When that derivative has a low-rank factorization, it can be maintained by linear accumulators (Paige & Koenig 1982; Cai et al. 2014; Budiu et al. 2023, DBSP product rule [Cite]). Three exact identities provide such factorizations:

1. Rendle's factorization-machine identity (Rendle 2010 [Cite]) for degree 2.
2. The Newton–Girard identities for symmetric degree 3 (CP/PARAFAC: Hitchcock 1927; Carroll & Chang 1970; Harshman 1970 [Cite]).
3. Base-plus-low-rank factorization (the Woodbury pattern: Sherman & Morrison 1950; Woodbury 1950 [Cite]) for king-conditioned blocks, which turns refreshes into rank-r corrections.

The bookkeeping layer is a delta monoid (actually a group): ℤ-valued finitely supported functions, the Z-sets of DBSP. In this group, deltas along a search path telescope and cancel before any memory is touched.

VNNUE-A is NNUE plus four channels that are exactly incrementally updatable, plus a refresh and replay calculus. Every new channel collapses to NNUE when its rank parameter is set to 0 (Theorem T9).

## 1. Sets and notation

- Perspectives c ∈ {w, b}. Every feature map below is defined from perspective c's point of view, with the board colour-flipped for black, as in NNUE.
- **Base features** P = {own, opp} × {P,N,B,R,Q,K} × 64 squares, so |P| = 768. These are *not* king-relative and *not* mirrored. β_c(pos) ∈ {0,1}^768, with at most 32 ones.
- **King-relative features** (HalfKAv2_hm, [Cite]): F = 32 king buckets × 704 = 22528. For bucket κ = κ_c(pos), x_c(pos) ∈ {0,1}^704 is the piece-square vector expressed relative to the mirror half that the king's file selects. Write H(pos) ∈ {e–h half, a–d half} for the mirror half, which is fixed per perspective once the king's file is known.
- **Optional explicit pair features** T (Full_Threats + PP_3Wide, [Cite]) with indicator τ_c(pos). These are kept unchanged from Stockfish master.
- A **delta** is δ ∈ ℤ^S := (S →₀ ℤ) (a Finsupp) for a feature set S. For a move m: pos → pos′, δ_m = 1_{A(pos′)} − 1_{A(pos)}. Its support is R ∪ S (removed ∪ added), with |R| + |S| = c(m) ∈ {2, 3, 3, 2, 3, 4} for quiet, capture, en passant, promotion, capture-promotion and castling respectively [Cite, nnue.md table].

## 2. Components

### C0. Base accumulator (unchanged NNUE core)

a_c = b + W_{κ} x_c ∈ ℤ^L with L = 1024, int16 storage, plus the threat accumulator a^T_c = W^T τ_c as in Stockfish master. The FT output is the pairwise product o_j = clamp(u_j,0,255)·clamp(v_j,0,255)/512 [Cite]. Nothing here changes except how a_c is produced on king moves (C3) and how pending updates are replayed (C4).

### C1. Multi-head shared-embedding FM channel (degree 2, all piece pairs)

Parameters: an embedding V ∈ ℤ^{r×768} with columns v_i ∈ ℤ^r, and a head-mixing matrix Λ ∈ ℤ^{m×r}. Typical sizes are r ∈ {16, 32}, m ∈ {16, 32}.

State per perspective:
- s_c = V β_c ∈ ℤ^r (int16), a linear accumulator;
- q_c = D β_c ∈ ℤ^m (int32), where D ∈ ℤ^{m×768} is a precomputed table with D_{h,i} = Σ_t Λ_{h,t} v_{t,i}². D is fixed at export, not learned.

Readout (once per evaluation):

  z_{c,h} = ½ ( Σ_t Λ_{h,t} s_{c,t}² − q_{c,h} ) ∈ ℤ.

By Theorem T5 this equals exactly

  z_{c,h} = Σ_{i<j} β_i β_j · Σ_t Λ_{h,t} v_{t,i} v_{t,j},

so each head h computes the sum over **all ≤ 496 unordered pairs of present pieces** of a learned pair weight Q^h_{ij} = v_iᵀ diag(Λ_h) v_j. The m heads share the r directions. The class is {Q^h = Vᵀ diag(λ_h) V}. With r = 768 and V = I it is *every* diagonal-core pair table; with a general r×r core B_h it becomes every symmetric pair table of rank ≤ r (ablation C1′, with readout cost m·r²).

The pair sum has an integer value because Σ_t λ_t s_t² − q is even (Theorem T5b), so the halving is an exact shift with no rounding.

Update for a move with delta δ (Theorem T1 applied to s and q): s += V δ and q += D δ, which costs c(m)·(r + m) lane-adds per perspective.

Readout cost: r squarings plus m·r MACs (about 1–2% of fc_0's 1024×32 MACs at r = m = 32). If evaluations are rare compared with updates, z can also be maintained incrementally instead: adding feature j gives Δz_h = Σ_t Λ_{h,t} v_{t,j} s_t, with s taken before the add (Theorem T6). That costs m·r per changed feature, and the cheaper of the two modes is chosen per engine profile [Exp].

The z_c (m values per perspective, shifted to int8 range) are concatenated to the fc_0 input. fc_0 thus reads [o ; z_stm ; z_nstm], i.e. 1024 + 2m inputs.

Why this is not redundant with what NNUE already has:
- **Pairwise multiplication** gives L/2 rank-1 quadratics per perspective, all *clipped*, built from king-relative features. Each costs 2 lanes per changed feature.
- **Explicit threats/pawn pairs** give a geometry-gated sparse subset of the 768² pair space. They cost O(#changed pairs) columns, which is 3–4× the piece churn [Cite].
- **C1** gives m unclipped quadratics of rank r over *all* pairs, at r + m lanes per changed feature, with **no refresh trigger ever**, because the base features are not king-relative.

C1 is a learned low-rank complement to explicit geometric pairs, not a replacement for them. The hybrid "explicit geometric pairs + low-rank residual pairs" is the default configuration.

### C2. Power-sum CP channel (degree 3, piece triples)

Parameters: A ∈ ℤ^{R×768} with small entries (|A| ≤ a_max, e.g. 15), and a readout ρ ∈ ℤ^{R}.

State per perspective: three int32 accumulators per component, p^{(k)}_{c,ρ} = Σ_i A_{ρ,i}^k β_i for k = 1, 2, 3. These are linear in β, so they update by Theorem T1 at 3R adds per changed feature.

Readout: 6·e_3^{(ρ)} = (p^{(1)})³ − 3 p^{(1)} p^{(2)} + 2 p^{(3)} (Theorem T7, Newton–Girard). For binary β, e_3^{(ρ)} = Σ_{i<j<k} β_iβ_jβ_k A_{ρ,i}A_{ρ,j}A_{ρ,k}. The sum is exactly divisible by 6 over ℤ.

Range: |p^{(1)}| ≤ 32·a_max = 480, so |p^{(1)}|³ ≤ 1.11·10^8 < 2^31 (Theorem T4 instance). The output is rescaled and fed to fc_0 like z. This is an optional ablation; R ∈ {0, 4, 8}.

### C3. Low-rank king transitions (Woodbury-style bucket factorization)

Constrain the king blocks inside each mirror half: W_κ = W_0 + U_κ V_K, with W_0 ∈ ℤ^{L×704}, V_K ∈ ℤ^{r_K×704} and U_κ ∈ ℤ^{L×r_K}. W_κ is still **materialized** for ordinary updates, so memory traffic on non-king moves is unchanged. An extra king-independent r_K-vector y_c = V_K x_c is maintained (r_K adds per changed feature).

On a king move κ → κ′ within the same mirror half, Theorem T8 gives

  a′ = a + (U_{κ′} − U_κ) y + W_{κ′} δ_x,

where y is taken before the move and δ_x is the move's delta (the king feature itself). The (U_{κ′} − U_κ) differences are precomputed at export for adjacent bucket pairs; otherwise the difference is formed on the fly at 2r_K columns.

King-move cost: r_K + c(m) columns, where NNUE pays a refresh of |A| ≤ 32 columns, or the Finny distance d_F = |C_κ′ △ A| [Cite]. The engine takes the minimum of {r_K + c, d_F, |A|}. This is optimal over the candidate set and never worse than Stockfish's policy (Theorems T10 and T13).

A king move that crosses the d/e mirror boundary permutes all indices and falls back to Finny/refresh. C1 and C2 are unaffected in every case.

Generalization: setting r_K = 704 and V_K = I makes every HalfKA bucket family representable (Theorem T9b), so C3 is a constraint only when r_K < 704. The design space is (number of full-rank king groups G) × (rank r_K within each group). G = 32 with r_K = 0 is exactly HalfKAv2_hm.

### C4. Delta group with cancellation (net-delta replay)

Pending updates live in the Z-set group (S →₀ ℤ). A lazy catch-up from an ancestor u to node v forms Δ_{u→v} = Σ_{plies} δ_t. Opposite terms (a piece that moves a→b→c, or a capture–recapture pair) cancel in the sum before any column is loaded. Correctness: telescoping, Δ_{p→q} + Δ_{q→r} = Δ_{p→r} (T2). Cost: |supp Δ_{u→v}| ≤ Σ_t |supp δ_t| (T2b).

The same group operation serves the base accumulator, the threat accumulator, and C1–C3, because every channel is a homomorphism of this one group. Any stored accumulator (stack, Finny entry, TT-stored state) is a valid base (T10). Base choice minimizes |A_v △ A_u| over the candidates available.

### C5. Integer semantics and certificates

All linear accumulators are specified in ZMod 2^{16} (int16 lanes) or ZMod 2^{32}, with **wrapping** adds. Theorem T3: wrapping accumulation along any path agrees with exact integer accumulation whenever the true final value lies in [−2^15, 2^15). Intermediate overflow therefore does not matter. Saturating adds are not associative (counterexample in T3c), which is why wrapping plus a certificate is the correctness notion.

The per-net **overflow certificate** (T4) is checked by an offline script: for every lane j, |b_j| + Σ_{slots σ} max_{i∈σ} |W_{j,i}| < 2^15, where a slot is one occupied piece position (≤ 32 slots). The same certificate form bounds s (int16), q, and the quadratic readouts (int32).

## 3. Update cost per move type (per perspective, lane-ops; L = 1024)

Let E := r + m + r_K + 3R be the extra lanes per changed feature (default r = m = 32, r_K = 16, R = 0, so E = 80, i.e. +7.8% of L). Threat-feature churn is unchanged and omitted.

| Move type | c(m) | NNUE | VNNUE-A |
|---|---|---|---|
| Quiet non-king | 2 | 2L | 2(L+E) |
| Capture / e.p. / capture-promo | 3 | 3L | 3(L+E) |
| Promotion | 2 | 2L | 2(L+E) |
| Castling | 4 | 4L (own side: refresh since king moves) | own side: min(r_K + 4, d_F, |A|)·L + 4E; other side: 4(L+E) |
| Own-king move, same mirror half | 1–2 | min(d_F, |A|)·L, with |A| ≤ 32 | min(r_K + c, d_F, |A|)·L + cE |
| Own-king move across d/e | 1–2 | min(d_F, |A|)·L | same + cE (C1/C2 never refresh) |
| Lazy catch-up over plies t_1..t_n | Σ c_t·L | |supp Σδ_t|·(L+E) ≤ Σ c_t·(L+E) |

Readout extra per evaluation: r + m·r (C1) + O(R) (C2) + 2m·32 MACs in fc_0 for the new inputs. At r = m = 32 this is about 3.1k MACs, against fc_0's 32.8k.

The honest cost picture:
- Non-king moves get about 8% more lane-ops. Memory bytes grow less than that, because V, D, A and V_K are tiny and stay in L1/L2 cache, whereas W columns stream from L3/DRAM [Cite, nnue.md §5 memory model].
- King moves get cheaper whenever r_K + c < min(d_F, |A|).
- Cancellation never increases cost.

Whether the net wall-clock effect is positive is **[Exp]**. A decision rule is needed: accept iff ΔElo at equal nodes > k_TC·ln(τ_new/τ_old) [UNVERIFIED constants from search notes, whose file was lost to ENOSPC].

## 4. Memory cost

Per network, int16 unless noted:

| Item | Size | Bytes (defaults) |
|---|---|---|
| W_κ (as HalfKAv2_hm) | 22528 × L | 46.1 MB (unchanged) |
| U_κ (32 buckets) | 32 × L × r_K | 1.05 MB |
| V_K | r_K × 704 × 2 halves | 45 KB |
| V (C1) | r × 768 | 49 KB |
| D (C1, int32) | m × 768 | 98 KB |
| Λ | m × r (int8) | 1 KB |
| A (C2, int8) | R × 768 | ≤ 6 KB |
| extra fc_0 weights | 32 stacks × 32 × 2m (int8) | 64 KB |

Per search-stack ply and perspective: s (64 B), q (128 B), y (32 B), p (≤ 96 B), on top of the base 2 KB accumulator, i.e. +15%. A Finny entry adds only y (C1/C2 are king-independent and need no cache).

## 5. Strict generalization of NNUE

Setting r = m = R = 0 and choosing G = 32 full king groups (r_K unused) recovers the HalfKAv2_hm + threats + pawn-pairs network exactly, bit for bit (T9a). Every NNUE parameter vector therefore embeds into VNNUE-A with an identical evaluation function, so the VNNUE-A function class contains the NNUE class at every quantization.

For C1 the containment is proper as a *first-layer* statement: the FT pre-activation (a_c, z_c) is a polynomial of degree 2 in β, while NNUE's pre-activation is affine (T9c). Downstream non-linearities in NNUE can also approximate pair terms. The claim of "more expressivity per update cost" is therefore an empirical claim [Exp], not a theorem.

## 6. Training notes (not provable; [Exp])

- Train in float with straight-through quantization (QAT is now standard in SF19 [Cite]). C1 is initialized with Λ = 0 so that training starts at the NNUE function (T9a). Weight clamps come from the T4 certificate rather than folklore.
- The C3 rank r_K can be distilled from an existing HalfKA net by truncated SVD of W_κ − W̄ per mirror half. The reconstruction error is measured, not assumed.
- Ablation grid: r ∈ {0, 8, 16, 32}, m ∈ {0, 16, 32}, R ∈ {0, 4, 8}, r_K ∈ {0, 8, 16, 32, full}, explicit threats on/off. Each setting is compared at equal measured nps.

## 7. Theorems for Lean 4 + Mathlib

All theorems are finite-dimensional, over ℤ, ZMod n or ℚ, stated with Finset sums and functions Fin n → ℤ or Finsupp. Mathlib lemma names are suggestions and are [UNVERIFIED].

**T1 (Homomorphism and path independence).** Let W : Fin L → Fin N → ℤ and acc(x) = b + W.mulVec x. For any list of deltas δ_1..δ_n with x_n = x_0 + Σ δ_t, folding `acc ← acc + W.mulVec δ_t` from acc(x_0) yields acc(x_n). Corollary: for sparse δ, W.mulVec δ = Σ_{i∈supp δ} δ_i · W_{·,i}. (Uses `Matrix.mulVec_add`, `List.foldl`, `Finset.sum_add_distrib`.) This applies verbatim to a, s, q, y and p.

**T2 (Delta telescoping and cancellation).** For Finsets S_0, S_1, S_2 of features, let Δ(S,S′) = 1_{S′} − 1_S. (a) Δ(S_0,S_1) + Δ(S_1,S_2) = Δ(S_0,S_2). (b) For any list of deltas, (Σ δ_t).support ⊆ ⋃ δ_t.support, hence #supp(Σδ_t) ≤ Σ #supp δ_t (`Finsupp.support_add`, `Finset.card_biUnion_le`). (c) #supp Δ(S,S′) = #(S △ S′).

**T3 (Wrap-around exactness).** (a) The cast ℤ → ZMod (2^16) is an additive (ring) homomorphism, so wrapped accumulation along any path equals the cast of the exact integer sum. (b) If −2^15 ≤ a < 2^15 then `((a : ZMod (2^16)).valMinAbs) = a` (`ZMod.valMinAbs_intCast` style). (c) Counterexample: sat16(sat16(30000 + 30000) + (−30000)) = 2767 ≠ 30000 = sat16(30000 + sat16(30000 + (−30000))), so saturating addition is not associative.

**T4 (Overflow certificate).** Let slots σ_1..σ_K partition the feature set, with at most one active feature per slot. If |b_j| + Σ_k max_{i∈σ_k} |W_{j,i}| ≤ B, then |acc_j(x)| ≤ B for every such x. Corollary: n active features with |W| ≤ w_max give |acc| ≤ |b| + n·w_max, e.g. 32·127 = 4064 < 2^15. Instance: |p^{(1)}| ≤ 480 ⇒ |p^{(1)}|³ + 3|p^{(1)}||p^{(2)}| + 2|p^{(3)}| < 2^31 (`norm_num`).

**T5 (FM identity with diagonal core).** For v : Fin r → Fin n → ℤ, λ : Fin r → ℤ and x : Fin n → ℤ with x_i ∈ {0,1}, s_t = Σ_i v_{t,i} x_i:
  Σ_t λ_t s_t² − Σ_i x_i Σ_t λ_t v_{t,i}² = 2 · Σ_{i<j} x_i x_j Σ_t λ_t v_{t,i} v_{t,j}.
(b) Hence the left side is even, and the readout z = (LHS)/2 is computed exactly by an arithmetic shift. (`Finset.sum_mul_sum`, `Finset.sum_comm`, diagonal/off-diagonal split via `Finset.sum_product`, `Finset.offDiag`.) Generalization T5′: the same holds for any symmetric core B : Matrix (Fin r) (Fin r) ℤ.

**T6 (FM incremental rule).** If x_j = 0 and x′ = x + e_j, then z(x′) − z(x) = Σ_t λ_t v_{t,j} s_t(x). If x_j = 1 and x′ = x − e_j, then z(x′) − z(x) = −Σ_t λ_t v_{t,j} s_t(x′). (DBSP product rule specialized: ‖s+v‖² − ‖s‖² = 2⟨s,v⟩ + ‖v‖².)

**T7 (Newton–Girard, order 3).** For any Finset ι and y : ι → ℤ (or any commutative ring), with p_k = Σ y_i^k: 6 · e_3(y) = p_1³ − 3 p_1 p_2 + 2 p_3, where e_3 = Σ over 3-element subsets of the product. Derivable from `MvPolynomial.psum`/`esymm` Newton identities (`Mathlib.RingTheory.MvPolynomial.NewtonIdentities` [UNVERIFIED names]) or by direct Finset manipulation. With y_i = A_i β_i and β binary, p_k = Σ A_i^k β_i is linear in β.

**T8 (Low-rank king transition).** For matrices W_0, U, U′, V and vector x: (W_0 + U′V).mulVec x = (W_0 + UV).mulVec x + (U′ − U).mulVec (V.mulVec x). Combined with T1 for the king feature's own delta. (`Matrix.add_mulVec`, `Matrix.mulVec_mulVec`, `Matrix.sub_mulVec`.) Cost corollary: the right side uses r_K column operations of length L.

**T9 (Strict generalization).** (a) Define eval_VNNUE(θ, ·) with extra channels; for θ with V = 0, Λ = 0, A = 0 and extra fc_0 columns = 0, eval_VNNUE(θ, pos) = eval_NNUE(θ|_NNUE, pos) for all inputs. (b) For any family {W_κ}, taking W_0 = 0, V_K = I_{704} and U_κ = W_κ gives W_0 + U_κ V_K = W_κ. (c) There exists a VNNUE-A first-layer output that is not an affine function of β: exhibit r = 1, v = e_1 + e_2, λ = 1, and z(e_1+e_2) − z(e_1) − z(e_2) + z(0) = 1 ≠ 0.

**T10 (Base selection).** (a) For any stored base u (any Finset S_u with correct accumulator acc(S_u)), acc(S_u) + W·Δ(S_u, S_v) = acc(S_v) (T1 + T2). (b) d(S,T) = #(S △ T) is a metric on Finset ι (triangle inequality via `Finset.symmDiff_subset_union`-style reasoning). (c) For candidate sets C ⊆ C′ (nonempty), min_{u∈C′} d(S_u, S_v) ≤ min_{u∈C} d(S_u, S_v) (`Finset.min'_subset`-style), so adding candidates (TT, Finny, C3 transition at cost r_K + c) never increases cost.

**T11 (Refresh-vs-walk ski rental).** Let the per-ply catch-up cost be 1 and the refresh cost be B ∈ ℕ, B ≥ 1. The deterministic policy "walk back until B−1 plies have been paid, then refresh" has cost ≤ (2 − 1/B)·OPT for every horizon n ∈ ℕ, where OPT = min(n, B). Stated as B·ALG(n) ≤ (2B − 1)·min(n, B) in ℕ.

**T12 (Clamp-delta sparsity).** For clamp_{[0,M]}: if δ_j = 0, or a_j ≤ 0 ∧ a_j + δ_j ≤ 0, or a_j ≥ M ∧ a_j + δ_j ≥ M, then clamp(a_j + δ_j) = clamp(a_j). Hence #{j | Δclamp_j ≠ 0} ≤ #supp δ, and the fc_0 delta update z += Σ_{j : Δh_j ≠ 0} W2_{·,j}·Δh_j is exact over ℤ.

**T13 (Op-count bounds).** In ℕ: for a non-king move with c changed features, cost_VNNUE = c(L+E) ≤ (1 + E/L)·cost_NNUE (as L·cost_V ≤ (L+E)·cost_N). For a king move within a mirror half, cost_VNNUE = min(r_K + c, d_F, |A|)·L + cE, and min(r_K + c, d_F, |A|) ≤ min(d_F, |A|). Lazy replay cost ≤ the sum of per-ply costs (from T2b).

**T14 (Quantization error bound).** (a) If |q_i − s·w_i| ≤ ½ for each weight and bias, then |(q_b + Σ_{i∈A} q_i)/s − (b + Σ_{i∈A} w_i)| ≤ (|A|+1)/(2s) over ℚ. (b) For the C1 readout with exact integer λ and per-coordinate accumulator error |ŝ_t − s_t| ≤ ε: |Σ λ_t ŝ_t² − Σ λ_t s_t²| ≤ Σ_t |λ_t|(2|s_t|ε + ε²).

**T15 (Colour-flip equivariance).** Let π be the colour-flip/rank-mirror permutation on the 768 base features (an involution). With tied parameters (one V shared by both perspectives, applied to the flipped board for black), s_b(pos) = s_w(flip(pos)) and z_b(pos) = z_w(flip(pos)). Formally: (V ∘ π).mulVec x = V.mulVec (x ∘ π⁻¹) (`Matrix.mulVec` with `Equiv.Perm` reindexing).

## 8. What cannot be proved, and what must be measured

- **Playing strength.** No theorem says that C1, C2 or C3 gains Elo. This needs fixed-node matches (ΔQ) and fixed-time SPRT on an engine with a trained VNNUE net.
- **Expressivity per cost.** T9c shows a strictly larger *first-layer* function class. Whether it lowers validation loss at equal nps compared with spending the same lanes on L, or on more explicit pairs, is empirical.
- **Speed.** Lane-op counts (T13) are a proxy. Real cost is memory-bound, so it must be measured with a profiler for nps, cache misses, refresh rate, the distribution of d_F versus r_K + c, and the cancellation ratio |supp Σδ| / Σ|supp δ| on real search paths.
- **C3 rank sufficiency.** The SVD reconstruction loss of the existing HalfKA blocks at r_K ∈ {8, 16, 32} is measured.
- **Second-layer delta sparsity (T12).** It is only useful if nnz(Δh)/L ≪ 1 in practice, which is unknown.
- **The T4 certificate** must be generated and checked for the actual trained net; the theorem only reduces the claim to a per-net arithmetic check.

## 9. Key references (details and verification status in the research notes)

Nasu (2018) NNUE; Stockfish nnue-pytorch docs and master source (2020–2026); Rendle (2010) Factorization Machines, IEEE ICDM; Juan et al. (2016) Field-aware FMs, RecSys [UNVERIFIED]; Blondel et al. (2016) Higher-Order FMs, NeurIPS; Hitchcock (1927); Carroll & Chang (1970); Harshman (1970); Kolda & Bader (2009) SIAM Review; Sherman & Morrison (1950) Ann. Math. Statist. 21(1):124–127; Woodbury (1950); Paige & Koenig (1982) TOPLAS; Cai et al. (2014) PLDI; Budiu et al. (2023) DBSP, PVLDB 16(7); Zobrist (1970); Karlin et al. (1988, 1994) competitive analysis / ski rental; Cohen & Welling (2016) ICML; O'Donnell (2014) Analysis of Boolean Functions.
