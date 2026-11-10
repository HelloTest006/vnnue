# VNNUE: Final Specification (v1.0)

Status: synthesis spec, 2026-11-09. It merges `design_algebraic.md` (A), `design_systems.md` (S) and `design_symmetry.md` (Sym), and applies every correction raised in the two judge reports.

Evidence tags:
- **[Lean Tn]**: a theorem in Section 4, to be machine-checked in Lean 4 + Mathlib.
- **[Exp En]**: a claim to be measured by the experiments in Section 5.
- **[V]**: a fact checked online against a primary source (this session or a cited research note).
- **[K]**: a standard reference whose bibliographic details were not re-checked this session.
- **[UNVERIFIED]**: not checked; do not state it as fact.

**No playing-strength, Elo or nps claim appears in this document.**

Note: `research/search.md` was never written because the disk was full (ENOSPC). Nothing in this spec depends on it. The Elo protocol in Section 5.6 is stated generically for that reason.

---

## 1. Name and abstract

**VNNUE: Very easily updatable Neural Network (Efficiently Updatable)**.

VNNUE keeps Stockfish's king-bucketed NNUE feature transformer (FT) at full width. It adds two kinds of things.

First, a **shared Power-Sum Interaction (PSI) channel**. This is a small, king-independent, perspective-shared accumulator over the 768 absolute piece-square features. Each of its r lanes keeps two power sums of a learned embedding:
- p1 = Σ v_i x_i, and
- p2 = Σ v_i² x_i.

Both are linear in the board, so they update exactly like NNUE columns. They are updated once per move for both perspectives and never need a refresh. Via Newton–Girard / Rendle's factorization-machine identity, they yield, per evaluation, m heads of **exact all-pairs piece×piece interactions**: Σ_{i<j} x_i x_j Σ_t Λ_{h,t} v_{t,i} v_{t,j}.

Second, VNNUE recasts accumulator maintenance as an **exact update calculus** over the delta group ℤ^F → (ℤ/2^16)^L. This calculus covers:
- wrap-around exactness with offline per-lane range certificates;
- a provably-never-worse source-selection policy for king and block changes, whose candidate set strictly contains Stockfish's {refresh, Finny cache, hybrid};
- an optional low-rank (Woodbury-style) king-transition candidate;
- symmetry-equivariant views that turn frame changes into lane permutations;
- net-delta cancellation for lazy catch-up.

Setting the PSI head weights to zero and disabling the optional modes recovers the NNUE evaluation bit for bit. All correctness, range and op-count statements are formalized as 25 Lean 4 theorems plus 2 appendix lemmas. All speed and strength questions are left to the stated experiments.

---

## 2. Architecture

### 2.0 Verified baseline (Stockfish master, Nov 2026)

Every cost and memory figure in this spec is relative to this baseline.

| Item | Value | Status |
|---|---|---|
| PSQ feature set | **K32H3**: 32 own-king buckets (horizontally mirrored, king on files e–h) × 3 material blocks × 704 piece-squares (11 planes × 64; the two kings share one plane). Dimensions = 3 × 22,528 = **67,584**; MaxActiveDimensions = 32 | [V] `src/nnue/features/k32h3.h` |
| Blocks (per perspective) | 0: no rook or queen on board; 1: some rook or queen, but opponent has no queen; 2: opponent has a queen | [V] k32h3.h |
| Other feature sets | Full_Threats (i8 weights widened to i16), PP_3Wide (pawn pairs, i8) | [V] nnue_architecture.h, research/nnue.md |
| Widths | L1 = 1024, L2 = 32, L3 = 32, LayerStacks = 32 | [V] nnue_architecture.h |
| FT output | o_j = ⌊clamp(a_j,0,255)·clamp(a_{j+512},0,255)/512⌋ per perspective, 512 u8 values each | [V] nnue_feature_transformer.h |
| PSQ FT table | 67,584 × 1024 × 2 B = **138.4 MB** (46.1 MB per block) | [DERIVED from V] |
| Accumulator adds | non-saturating `vec_add_16` = `_mm256_add_epi16` / `_mm512_add_epi16` (wrapping) | [V] simd.h, per judge report 1 |
| Finny caches | per thread, perspective, king square and block | [V] PR #5183; nnue_accumulator.cpp |
| `update_accumulator_hybrid` | used for an own-king move when: same PSQ block, previous accumulator computed, ≥ 15 pieces (`MIN_PC_COUNT_HYBRID`), king stays on the same board half (`from & 4 == to & 4`), not castling. It computes new = prev − cache[old_ksq] + cache[new_ksq] + Finny fix-ups on both entries + threat/pawn-pair deltas. | [V] nnue_accumulator.cpp (fetched 2026-11-09) |
| Lazy catch-up | `find_last_usable_accumulator`, then `forward_update_incremental` / `backward_update_incremental` (after a cache refresh), `update_accumulator_incremental_both` for both colours | [V] nnue_accumulator.cpp |
| Two-ply cancellation | PR #5988 (ces42, merged 2025-04-27 via commit fda269a): "double incremental accumulator updates" skip the intermediate accumulator when the second move captures the piece that moved in the first, and cancel add/remove pairs. The author measured that only ≈0.2% of 3-move updates involve one side moving the same piece twice. | [V] PR #5988. Whether the code is still present in current master under that name is [UNVERIFIED]. |
| Threat churn | ≈3–4× as many threat features as piece features change per move | [V] research/nnue.md |

**Credited prior art (not claimed as new):**
- wrapping int16 adds;
- Finny caches;
- the hybrid king update;
- i8 threat columns widened to i16;
- the 0..127 activation range chosen for `pmaddubsw` safety;
- two-ply add/remove cancellation (PR #5988);
- coarse king "input buckets" in non-Stockfish engines and the bullet trainer ecosystem ([UNVERIFIED which specific engines]);
- nnue-pytorch virtual-feature factorization.

### 2.1 Notation

- c ∈ {W, B}: perspective; stm/nstm: side to move / side not to move.
- **Absolute features** 𝔄 = Fin 2 (colour) × Fin 6 (type) × Fin 64 (square), |𝔄| = 768. β(pos) ∈ {0,1}^768 is White-absolute: there is no flip, mirror or king dependence, and at most 32 entries are 1.
- **PSQ features** for perspective c: x_c(pos) ∈ {0,1}^{704}, the oriented piece-square vector. The orientation is `OrientTBL[ksq] ^ (56·c)`: the colour flip for Black, plus a horizontal mirror when c's king is on files a–d.
  - Bucket: κ_c ∈ Fin 32, the own-king square in the canonical frame.
  - Block: blk_c ∈ Fin 3.
  - The full K32H3 index is (blk_c, κ_c, x-index).
- **Weight tables**: W_{blk,κ} ∈ ℤ^{L×704} (int16), b ∈ ℤ^L, L = 1024.
- **Threat / pawn-pair features** τ_c, π_c with tables W^T, W^P (i8, widened). These are unchanged from master.
- **Delta group**: for a feature universe F, Δ_F := (F → ℤ). A move with removed set R and added set A has δ = 1_A − 1_R.
  - **k** := |A| + |R| is the PSQ change count per perspective: quiet 2, capture 3, en passant 3, promotion 2, capture-promotion 3, castling 4 [V, research/nnue.md].
  - **k_a** is the same count in absolute features. It equals k for every move type, including king moves, because absolute features have no king dependence.
- **Lane rings**: ℤ (specification), ZMod 2^16 (int16 lanes), ZMod 2^32 (int32 lanes). dec16 is the two's-complement decode (Section 4, Basic.lean).
- **Cost units**:
  - **col** = one L-wide int16 column add/sub;
  - **mcol** = one L-wide multiply-add column (`vpmullw`+`vpaddw`), cost ratio μ ≥ 1 relative to col;
  - **pass** = one accumulator load+store (2·2L bytes);
  - **lane** = one scalar SIMD lane operation.

### 2.2 Components

VNNUE state per search-stack ply:

| Component | Content | Maintained | Status |
|---|---|---|---|
| **C0 FT core** | a_c = b + W_{blk_c,κ_c} x_c + W^T τ_c + W^P π_c ∈ (ℤ/2^16)^L for c ∈ {W,B} | as master | unchanged (default) |
| **C1 PSI channel** | P1 = Vβ ∈ (ℤ/2^16)^r, P2 = V^{∘2}β ∈ (ℤ/2^32)^r, **one copy per position** (shared by both perspectives) | per move, k_a·(r + r) lanes | **default, headline** |
| C1′ PSI degree 3 | P3 = A^{∘3}β ∈ (ℤ/2^32)^{R}, plus P1, P2 for those R lanes, with |A| ≤ 15 | per move, k_a·3R lanes | ablation |
| C2 King-transition state | y_c = V_K x_c ∈ ℤ^{r_K} (int16) per perspective; tables W_{blk,κ} = W⁰_{blk} + U_{blk,κ} V_K (exact in int16) | per move, k·r_K lanes | optional ("merged" mode) |
| C2′ Factored king mode | stores (W⁰_{blk} + R_{blk,ρ(κ)}) for |P| regions and U, V_K; no materialized W_κ | per move | memory experiment |
| C3 Source selection | argmin over candidates (Section 2.5) | per refresh event | default (policy) |
| C4 Net-delta catch-up | Δ = Σ δ_j with cancellation before column loads | per lazy eval | default (policy, low yield) |
| C5 Certificates | per-lane slot certificate for C0, C1, C1′, C2; checked offline | export time | default |
| C6 Mirror equivariance | isotypic threat channel or lane-permutation mirror tying | — | ablations only |

#### C1. PSI channel (definition)

Parameters:
- V ∈ ℤ^{r×768} (int8, |V| ≤ 127);
- Λ ∈ ℤ^{m×r} (int8);
- head bias b^z ∈ ℤ^m (int32);
- shifts sh_e, sh_z ∈ ℕ.

Defaults: **r = 32, m = 16**.

State (shared, White-absolute):
- P1_t = Σ_i v_{t,i} β_i, in int16 (|P1_t| ≤ 32·127 = 4064 by [Lean T9]);
- P2_t = Σ_i v_{t,i}² β_i, in int32 (|P2_t| ≤ 32·127² = 516,128).

**Perspective views without duplication.** Let σ_c be the colour flip on 𝔄: (col, type, sq) ↦ (1−col, type, sq ⊕ 56). Let P ∈ Perm(Fin r) be a fixed involutive lane permutation, e.g. swapping lane t with t ⊕ 1. Impose the tying constraint

  V_{P(t), σ_c(i)} = V_{t,i}   for all t, i.

Then the Black-perspective power sums equal the White ones, read in permuted lane order [Lean T21]:
- P1^{(B)} = P1 ∘ P;
- P2^{(B)} = P2 ∘ P.

So one state serves both perspectives. The untied alternative (two copies, 2× update cost) is ablation E7.

Readout (per evaluation, per perspective c, with P1^{(c)}, P2^{(c)} the views):

  E_t = (P1_t² − P2_t) / 2           ∈ ℤ, an exact shift because 2 ∣ P1_t² − P2_t   [Lean T12]
  ê_t = clampZ(−2^15, 2^15−1, E_t >> sh_e)                                        (int16)
  z_h = b^z_h + Σ_t Λ_{h,t} ê_t      (int32; no overflow for r ≤ 32, |Λ| ≤ 127)   [Lean T16]
  u_h = clampZ(0, 127, z_h >> sh_z)   (u8 input to fc_0)

**Meaning.** For binary β, E_t = e_2(v_{t,·} ⊙ β) = Σ_{i<j} β_iβ_j v_{t,i}v_{t,j} [Lean T12]. Hence

  Σ_t Λ_{h,t} E_t = Σ_{i<j} β_i β_j Q^h_{ij},   with Q^h = Vᵀ diag(Λ_h) V   [Lean T13].

Each head is a learned sum over **all** ≤ 496 unordered pairs of present pieces. The pair table has rank ≤ r with a diagonal core. With enough lanes, every integer pair table is reachable [Lean T17]. Note that the earlier claim that "V = I gives every diagonal-core table" was **false**: V = I gives z ≡ 0. The correct universality construction uses pair lanes v_t = e_a + e_b.

The shift and clamps act on the exact state at readout, so they never cause drift.

**fc_0 input.** For each layer stack, fc_0 reads [o_stm (512) ; o_nstm (512) ; u_stm (m) ; u_nstm (m)], i.e. 1024 + 2m inputs. Everything downstream is unchanged.

**Initialization.** Λ = 0 and fc_0's extra columns = 0, so training starts exactly at the NNUE function [Lean T18].

**Incremental alternative.** E can also be maintained directly. Adding feature j gives ΔE_t = v_{t,j}·P1_t(before); removing it gives ΔE_t = −v_{t,j}·P1_t(after) [Lean T14]. This trades r MACs per changed feature for skipping the P2 state. Which mode is used is decided by measurement [Exp E2].

**What PSI adds over NNUE, stated correctly.** NNUE's first layer is **not** affine in the absolute features β:
- K32H3 is already degree 2 in β (king one-hot × piece-square), and piecewise in material block;
- Full_Threats and PP_3Wide are explicit, geometry-gated piece-pair features;
- pairwise multiplication gives L/2 clipped rank-1 quadratics of king-relative features.

The precise statement is this. **At fixed king squares and fixed block, the K32H3 PSQ pre-activation is affine in the non-king piece indicators [Lean T18 (affine second difference = 0)]. The PSI channel is not [Lean T18 witness].** PSI therefore adds dense, low-rank, unclipped, king-independent piece×piece terms over all pairs, at r+r lanes per changed absolute feature, with no refresh trigger. It complements the sparse explicit pairs rather than replacing them. Whether this lowers loss at equal cost compared with spending the same budget on L or on explicit pairs is **[Exp E8]**, not a theorem.

#### C1′. PSI degree 3 (ablation)

Parameters: A ∈ ℤ^{R×768} with |A| ≤ 15, and R ∈ {0, 4, 8}.

State: P1, P2, P3 per lane (int32).

Readout: 6·e_3 = P1³ − 3·P1·P2 + 2·P3, which is exactly divisible by 6 [Lean T15].

Range (corrected, with all hypotheses):
- |P1| ≤ 32·15 = 480;
- |P2| ≤ 32·225 = 7200;
- |P3| ≤ 32·3375 = 108,000.

These give |P1³ − 3P1P2 + 2P3| ≤ 121,176,000 < 2^31 [Lean T16]. The old statement, which bounded only P1, was false (counterexample: P1 = 0, P3 = 2^31).

#### C2. Low-rank king transitions (optional "merged" mode)

Inside each block, constrain W_{blk,κ} = W⁰_{blk} + U_{blk,κ} V_K. The integer factors U (int16) and V_K (int8) must give **exactly** the stored int16 table. Concretely:
- the trainer quantizes U and V_K, not their product;
- the materialized W_κ is computed from the quantized factors, so the identity holds exactly;
- the certificate C5 must then admit W_κ.

The cost is a lattice restriction on the residual, whose accuracy effect is [Exp E5].

y_c = V_K x_c is maintained per perspective. Within one mirror half the oriented x_c changes only by the moved pieces, including the king's own feature, so y_c updates like any linear accumulator. On a d/e crossing, y_c is recomputed (|S|·r_K scalar ops).

Transition identity [Lean T23], for an own-king move κ → κ′ in the same block and the same mirror half:

  a′ = a + (U_{κ′} − U_κ)·y(before) + W_{κ′}·(x′ − x) + (threat/pawn-pair deltas).

Cost: r_K mcol (with ΔU formed on the fly: 2r_K mcol, or r_K if precomputed for adjacent pairs at 25 MB, which is not recommended) + k col.

This is **one candidate** in C3. It is **not** claimed to beat `update_accumulator_hybrid` in general. With r_K = 16, μ ≥ 1 and cold U loads, it usually costs more [Exp E4 measures how often it wins].

**C2′ factored mode (memory experiment).** Store T_{blk,p} = W⁰_{blk} + R_{blk,p} for |P| regions plus U, V_K. The evaluated accumulator is a = a_base + U_κ y, computed at readout. This costs **r_K·L multiply-adds per perspective per evaluation**. At r_K = 16 that is 16,384 MACs per perspective, 32,768 for both, which is about the cost of a dense fc_0. That is a real tax, not negligible. In exchange, intra-region king moves cost k col and memory drops 138.4 → 20.4 MB (Section 3.2).

#### C3. Source selection (exact, provably never worse)

When perspective c needs a = acc(S) in table T (bucket κ, block blk), any of these sources may be used:

| id | source | correctness | cost (cols, plus passes) |
|---|---|---|---|
| refresh | bias | definition | \|S\| col + τ_full (threat/PP rebuild) + 1 pass |
| finny | Finny entry (blk, κ) with board C | [Lean T3] | \|S △ C\| col + threat part as master + 1 pass |
| hybrid | SF hybrid: prev − cache_old + cache_new, with fix-ups | [Lean T24b] | \|S_prev △ C_old\| + \|S △ C_new\| col + 2 extra passes + τ |
| woodbury | C2 transition from the parent | [Lean T23] | μ·r_K (or 2μ·r_K) + k col + τ + 1 pass |
| ancestor | nearest materialized stack ancestor u, net delta | [Lean T2, T5] | \|supp Σ_{j>u} δ_j\| col + 1 pass |
| finnyMirror | Finny entry of the mirrored frame (C6 tying only) | [Lean T6, T21] | \|S △ C\| col + 1 permutation pass |

A source is **unavailable** when its preconditions fail. For example, hybrid needs ≥ 15 pieces, the same half and no castling; woodbury needs the same block and the same half. An unavailable source has cost ⊤.

The policy is argmin of cost(σ). Every cost is computable from bitboards and cache metadata before any column is loaded, so this is an **offline** choice and not an online (ski-rental) problem.

Guarantee [Lean T24]: for any cost function into a linear order, including weighted mixes of col, mcol, pass and cold-load penalties, argmin over VNNUE's set ≤ argmin over Stockfish's set {refresh, finny, hybrid}. The guarantee covers only the **FT column part**. The PSI channel adds its own k_a·2r lanes on every move, so the per-move total is ≤ SF cost + k_a·e [Lean T25].

#### C4. Net-delta catch-up

For a lazy evaluation at ply t from a materialized ancestor u:
1. Form Δ = Σ_{j=u+1}^{t} δ_j over sparse index lists (sort or hash of ≤ 4(t−u) entries).
2. Drop zeros.
3. Apply Δ in one pass.

Correctness [Lean T2]. Columns: |supp Δ| ≤ Σ|supp δ_j| [Lean T5]. Passes: 1 instead of t−u [Lean A2].

This **generalizes PR #5988**, which handles the two-ply capture case. ces42's ≈0.2% statistic implies a small expected column saving. The pass saving is the main effect, but Stockfish already fuses several updates, and fusion loses the intermediate nodes as bases for siblings. Status: **[Exp E3], low expected yield; not a headline.**

Across an own-king bucket change, the per-ply PSQ delta is ≈ 2|S| in the king-bucketed index space, so catch-up is split at the bucket change and joined via C3.

#### C5. Integer semantics and certificates

**Wrapping arithmetic.** All accumulators use wrapping arithmetic: ZMod 2^16 for int16 lanes, ZMod 2^32 for int32. The cast ℤ → ZMod n is a ring homomorphism, so after any update path the decoded value equals the exact integer value whenever the **final** value is in range [Lean T7]. Intermediate overflow is irrelevant. Saturating arithmetic is path-dependent (counterexamples in [Lean T8]), which is why wrapping plus a certificate is the correctness notion.

**Decode window.** The decode uses the custom dec16, whose window is [−32768, 32767]. Mathlib's `valMinAbs` has window (−2^15, 2^15], so it is wrong at −32768; this was the judge-flagged error in A-T3b, and the corrected `valMinAbs` variant is T7(d).

**Per-lane slot certificate** [Lean T9]:
- Partition the features by g: F → groups (e.g., (colour, type), or square for PSI lanes), with legal capacities cap_t ≥ the maximum number of simultaneously active features in group t.
- Capacities must allow promotions: per colour ≤ 16 pieces; per type ≤ 8 pawns, ≤ 10 N/B/R, ≤ 9 Q, 1 K [K: rules of chess].
- Then b_ℓ + Σ_t cap_t·M_t⁻ ≤ acc_ℓ ≤ b_ℓ + Σ_t cap_t·M_t⁺.
- The theorem also gives the alternative bound b_ℓ + K·max⁺ when |A| ≤ K, and the certificate uses the smaller of the two.
- The judge-flagged instance "≤ 32 slots with one active feature each" was ill-defined (pieces are unlabelled) and is **removed**.

Certificates are emitted per net by an offline script and checked in O(L·|F|). They cover:
- C0 PSQ lanes (per block, per bucket);
- the threat lanes (cap = maximum simultaneous threat pairs per group, computed by the script) [UNVERIFIED tight value];
- P1 (int16) and P2, P3 (int32);
- the C2 merged tables;
- the PSI readout (T16).

**On FT scale and i8 storage.** These are two separate questions:
- i8 column storage needs every |W_{ℓ,i}| ≤ 127 at the chosen scale;
- the certificate bounds the accumulator range, not individual weights.

Raising the FT scale s also moves the 0..255 clamp and the /512 product normalization, so the rounding bound (|A|+1)/(2s) [Lean T11] is not a free precision gain.

#### C6. Symmetry (framing plus ablations)

**Framing theorem** [Lean T21, T22]: if W_{σ(i)} = ρ(g)·W_i for a lane permutation ρ(g), then acc(g·x) = ρ(g)·acc(x). Under this framing:
- Stockfish's perspective sharing is the regular representation of ℤ/2;
- hm-mirror canonicalization is the regular-representation special case W′ = [W ; W∘H] with W′∘H = swap∘W′.

**Ablation E9a: isotypic threat channel** a_± = (W ± W∘H)τ. The mirror acts as +1 on a_+ and −1 on a_−, so a d/e crossing costs **one negation pass** (not "no arithmetic") and no refresh. The cost is a restricted function class. The default stays Stockfish's canonical frame for threats.

**Ablation E9b: lane-permutation mirror tying** of PSQ tables, (W_blk)_{μ(i)} = P_h·(W_blk)_i with P_h an involution and Pb = b. This holds only within a region that is closed under the mirror.

The h-symmetry is a **prior, not a law**: castling rights break it. King-translation "torus" tying from Sym is **dropped**. It aliases rank 8 with rank 1 and is only a king-dependent lane permutation.

### 2.3 Quantization summary

| Tensor | Type | Scale / range | Certificate |
|---|---|---|---|
| W_{blk,κ}, b | i16 | as master | T9 per lane |
| W^T, W^P | i8 → i16 | as master | T9 with threat caps |
| V (PSI) | i8 | \|V\| ≤ 127 | P1 ≤ 4064 (i16), P2 ≤ 516,128 (i32) |
| Λ | i8 | \|Λ\| ≤ 127 | T16: \|Σ_t Λ ê\| ≤ 32·127·2^15 = 133,169,152 < 2^31 |
| b^z | i32 | — | add to T16 bound; certified by script |
| A (C1′) | i8 | \|A\| ≤ 15 | T16 (NG3 range) |
| U, V_K (C2) | i16, i8 | exact product equals stored i16 table | T9 on materialized table |
| fc_0 extra columns | i8 | s_W = 64 as master; u_h ∈ [0,127] satisfies the `pmaddubsw` pair bound | T10 |

### 2.4 Exact update rules per move type

Write col(T, i) for column i of table T, and use Kc for the king of side c.

**PSI deltas** (shared state; absolute features f = (colour, type, square)):

| Move | removed f | added f | k_a |
|---|---|---|---|
| quiet (incl. king non-castling) | (c,p,from) | (c,p,to) | 2 |
| capture | (c,p,from), (c̄,q,to) | (c,p,to) | 3 |
| en passant | (c,P,from), (c̄,P,to∓8) | (c,P,to) | 3 |
| promotion to q | (c,P,from) | (c,q,to) | 2 |
| capture-promotion | (c,P,from), (c̄,x,to) | (c,q,to) | 3 |
| castling | (c,K,ksq), (c,R,rsq) | (c,K,ksq′), (c,R,rsq′) | 4 |

Update: P1 += Σ_{f∈A} V_{·,f} − Σ_{f∈R} V_{·,f}, and P2 += Σ_{f∈A} V_{·,f}^{∘2} − Σ_{f∈R} V_{·,f}^{∘2}. V^{∘2} is computed in-register from V (one `vpmullw` per lane group) or read from a 48 KB i16 table. **King moves are ordinary moves for PSI**; there is never a refresh.

**FT (per perspective c):**

1. **Non-king move, or opponent-king move, with block unchanged:** a_c += Σ_{A} col(W_{blk,κ}, i) − Σ_{R} col(W_{blk,κ}, i), plus threat and pawn-pair deltas as in master. If C2 is on: y_c += Σ_A V_K[:,i] − Σ_R V_K[:,i].
2. **Own-king move, same block, same half:** run C3 over {refresh, finny, hybrid, woodbury, ancestor}.
   - woodbury: a_c ← a_c + (U_{κ′} − U_κ)·y_c + Σ_A col(W_{κ′}) − Σ_R col(W_{κ′}) + threat deltas;
   - then y_c += V_K·δ_x.
3. **Own-king move across d/e (orientation change), or castling:** C3 over {refresh, finny, finnyMirror (C6 only), ancestor}; y_c is recomputed.
4. **Block change** (capture or promotion changes heavy-piece state): C3 over {refresh, finny, ancestor}.
5. **Lazy evaluation:** C4 from the nearest computed ancestor, unless C3 chooses another source.

### 2.5 Pseudocode

```text
# ---------- export time ----------
def export(net):
    assert certificate_ok(net)                # T9/T16 per lane, offline
    if net.mode == MERGED:
        for blk, kappa: net.W[blk][kappa] = net.W0[blk] + net.U[blk][kappa] @ net.VK   # exact int16
    net.V2 = net.V * net.V                     # optional i16 table

# ---------- make move ----------
def make_move(st, move):                       # st = new stack entry
    R, A = absolute_delta(move)                # Section 2.4 table
    st.P1 = st.parent.P1 + sum(V[:,f] for f in A) - sum(V[:,f] for f in R)   # wrap int16
    st.P2 = st.parent.P2 + sum(V2[:,f] for f in A) - sum(V2[:,f] for f in R) # wrap int32
    for c in (W, B):
        st.dirty[c] = psq_delta(move, c)       # sparse (R_c, A_c), threat/PP deltas as master
        st.computed[c] = False
    st.blk, st.kappa, st.half = features_meta(st.pos)

# ---------- evaluate ----------
def evaluate(st):
    for c in (W, B):
        if not st.computed[c]:
            src = argmin(candidates(st, c), key=lambda s: s.cost)    # C3; cost from bitboards
            st.acc[c] = src.build()                                  # exact by T3/T23/T24b/T2
            st.computed[c] = True
    o = [pairwise(st.acc[stm]), pairwise(st.acc[nstm])]              # as master
    u = [psi_readout(st, stm), psi_readout(st, nstm)]
    return layer_stack[bucket(st.pos)](concat(o[0], o[1], u[0], u[1]))

def psi_readout(st, c):
    P1 = st.P1 if c == W else st.P1[perm_P]                         # T21: lane permutation
    P2 = st.P2 if c == W else st.P2[perm_P]
    E  = (P1.astype(i32)**2 - P2) >> 1                               # exact (T12)
    e  = clamp(E >> sh_e, -32768, 32767)
    z  = bz + Lam @ e                                                # int32 safe (T16)
    return clamp(z >> sh_z, 0, 127).astype(u8)

def candidates(st, c):
    yield Refresh(cost=len(S) + tau_full)
    yield Finny(cost=symdiff(S, cache[c][blk][kappa].board))
    if hybrid_ok(st, c):   yield Hybrid(cost=symdiff(Sprev, Cold) + symdiff(S, Cnew) + 2*PASS + tau)
    if woodbury_ok(st, c): yield Woodbury(cost=MU*rK + k + tau)
    u = last_computed_ancestor(st, c)
    if u and no_bucket_change_between(u, st, c): yield Ancestor(cost=len(net_support(u, st, c)) + PASS)
```

---

## 3. Complexity

### 3.1 Per-move update cost (both perspectives)

Symbols:
- k: PSQ changes per perspective (= k_a);
- τ_c, π_c: changed threat and pawn-pair features for perspective c;
- d_F = |S △ C_new|; d_old = |S_prev △ C_old|;
- |S| ≤ 32: active PSQ features;
- e := 2r lanes, with byte cost 2r + 4r = 6r per changed absolute feature.

Unit: col (L = 1024 int16 lanes, 2 KB at β_w = 2; threat columns 1 KB at β_w = 1).

| Move type | NNUE (master) | VNNUE (merged/default) | VNNUE extra (PSI, shared) |
|---|---|---|---|
| Quiet non-king (k=2) | 2k + Σ_c(τ_c+π_c) | same | k·2r lanes = 128 lanes (384 B) |
| Capture / e.p. / cap-promo (k=3) | 6 + Σ_c(τ_c+π_c) | same | 192 lanes (576 B) |
| Promotion (k=2) | 4 + Σ(τ+π) | same | 128 lanes |
| Opponent-king move (our view) | k (our side) | same | — (counted once above) |
| Own-king, same block & half | min(\|S\|+τ_full, d_F, d_old+d_F+2·pass) + other side k | min(NNUE set ∪ {μr_K + k + τ, ancestor}) + other side k (+ C2: k·r_K lanes y) | k·2r lanes |
| Own-king across d/e / castling (k=4) | min(\|S\|+τ_full, d_F) + other side | min(…, finnyMirror if C6) + other side | k·2r lanes |
| Block change | refresh/Finny both sides | same (C3 superset) | k·2r lanes |
| t-ply lazy catch-up | Σ_j k_j col + ~t passes (two-ply special case per PR #5988) | \|supp Σδ_j\| col + 1 pass | Σ_j k_j·2r lanes (applied eagerly at make) |
| **Per evaluation** | FT pairwise 1024 products; fc_0 ≤ 1024·32 = 32,768 MACs (sparse) | same | 2·(r squares + r subs + r shifts + m·r MACs) + 2m·32 fc_0 MACs = **2·512 + 1024 = 2048 MACs** at r=32, m=16 (6.3% of dense fc_0; a larger share of SF's sparse fc_0) |
| Per eval, C2′ factored | — | + r_K·L MACs per perspective = 32,768 MACs for both at r_K = 16 | |

Formulas proved in Lean:
- VNNUE PSQ column cost ≤ NNUE's on every king and block event [Lean T24].
- If NNUE's per-move PSQ byte traffic is ≥ 4kL (k columns × 2 perspectives × 2L bytes), then the PSI byte overhead is ≤ e/(4L) of it [Lean T25]. With e = 6r = 192 B and 4L = 4096 B this is ≤ 4.7%, and less once threat churn (3–4× k) is counted.

**Honest summary:**
- Non-king moves get strictly more work, ≈ +3% lanes / +4.7% bytes of PSQ traffic. The extra comes from the shared PSI channel, whose 24 KB V table (+48 KB V²) is L1/L2 resident; the FT columns stream from L3/DRAM [UNVERIFIED per-CPU; Exp E4].
- King moves are never worse in column count.
- Wall-clock is [Exp].

### 3.2 Memory

| Item | NNUE master | VNNUE default | VNNUE + C2 merged (r_K=16) | VNNUE + C2′ factored (\|P\|=4, r_K=16) |
|---|---|---|---|---|
| PSQ FT table | 67,584×1024×2 B = 138.4 MB | 138.4 MB | 138.4 MB + U: 3·32·16·1024·2 B = 3.1 MB + V_K: 3·16·704 B = 34 KB | 3·4·704·1024·2 B = 17.3 MB + 3.1 MB = **20.4 MB** (6.8× smaller) |
| Threat / PP tables | as master [size UNVERIFIED] | same | same | same |
| PSI: V (i8) / V² (i16, optional) | — | 24 KB / 48 KB | same | same |
| Λ, b^z | — | 512 B + 64 B | same | same |
| fc_0 extra (32 stacks × 32 × 2m i8) | — | 32 KB | same | same |
| Per-ply state | 2 × 2 KB (PSQ acc) + threat state | + 64 B (P1) + 128 B (P2), **once per position** | + 2×32 B (y) | same |
| Finny entry | acc + board | unchanged (PSI needs no cache) | + y (32 B) | + y |

---

## 4. Theorems for Lean 4 + Mathlib

Ground rules:
- Each module imports **only** `Mathlib` and `VNNUE.Basic`.
- Definitions that a single module needs are local to it.
- Signatures use Mathlib types (`Finset`, `Fin n → ℤ`, `Matrix`, `ZMod`, `ℝ`, `Equiv.Perm`).
- Lemma names in the sketches are verified against the local Mathlib checkout (D:/lean/Mathlib, toolchain v4.35.0-rc4) where marked ✓. All others are suggestions.
- **Type-check status: NOT YET COMPILED.** The local Mathlib oleans reported "incompatible header" when used from `C:/Users/user/vnnue/lean`, and a direct `lake env lean` in D:/lean/Mathlib did not finish in 10 minutes. The next step must first fix the Lake setup (`lake update mathlib`, or rebuild/fetch the cache) and then compile Basic.lean.

### 4.0 `VNNUE/Basic.lean` (exact content)

```lean
import Mathlib

/-!
# VNNUE.Basic
Shared definitions for all VNNUE modules. Every other module imports only
`Mathlib` and this file.
-/

namespace VNNUE

open Finset

variable {R : Type*} [AddCommGroup R] {N L : ℕ}

/-- Linear part of an accumulator: `Φ W x = ∑ i, x i • W i` (column `i` is `W i`). -/
def Phi (W : Fin N → Fin L → R) (x : Fin N → ℤ) : Fin L → R :=
  ∑ i, x i • W i

/-- Accumulator over an integer (signed) feature vector. -/
def accV (b : Fin L → R) (W : Fin N → Fin L → R) (x : Fin N → ℤ) : Fin L → R :=
  b + Phi W x

/-- Accumulator over an active feature set (the "refresh" formula). -/
def accS (b : Fin L → R) (W : Fin N → Fin L → R) (S : Finset (Fin N)) : Fin L → R :=
  b + ∑ i ∈ S, W i

/-- Indicator vector of a feature set. -/
def ind (S : Finset (Fin N)) : Fin N → ℤ := fun i => if i ∈ S then 1 else 0

/-- Delta of a set change `S → T`. -/
def delta (S T : Finset (Fin N)) : Fin N → ℤ := ind T - ind S

/-- Support of a delta: the columns that must be touched. -/
def supp (δ : Fin N → ℤ) : Finset (Fin N) := univ.filter (fun i => δ i ≠ 0)

/-- Column operations needed to move an accumulator from set `S` to set `T`. -/
def colOps {α : Type*} [DecidableEq α] (S T : Finset α) : ℕ := (symmDiff S T).card

/-- Sequential incremental application of a list of deltas. -/
def foldDeltas (W : Fin N → Fin L → R) (a : Fin L → R) (l : List (Fin N → ℤ)) :
    Fin L → R :=
  l.foldl (fun acc δ => acc + Phi W δ) a

/-- Binary (0/1) feature vector. -/
def IsBinary (x : Fin N → ℤ) : Prop := ∀ i, x i = 0 ∨ x i = 1

/-- int16 range `[-32768, 32767]`. -/
def InI16 (v : ℤ) : Prop := -32768 ≤ v ∧ v ≤ 32767

/-- Two's-complement decode of a wrapped int16 lane. -/
def dec16 (y : ZMod 65536) : ℤ :=
  if y.val < 32768 then (y.val : ℤ) else (y.val : ℤ) - 65536

/-- Saturating int16 clamp (semantics of `paddsw`). -/
def sat16 (v : ℤ) : ℤ := max (-32768) (min 32767 v)

/-- Integer clamp to `[lo, hi]`. -/
def clampZ (lo hi v : ℤ) : ℤ := max lo (min hi v)

/-- Lane-wise cast of an integer lane vector to `ZMod 65536`. -/
def castLanes (b : Fin L → ℤ) : Fin L → ZMod 65536 := fun j => (b j : ZMod 65536)

/-- Entry-wise cast of an integer weight table to `ZMod 65536`. -/
def castW (W : Fin N → Fin L → ℤ) : Fin N → Fin L → ZMod 65536 :=
  fun i j => (W i j : ZMod 65536)

/-- Power sum of one PSI lane with embedding row `v`: `∑ i, x i * v i ^ k`. -/
def psum (k : ℕ) (v x : Fin N → ℤ) : ℤ := ∑ i, x i * v i ^ k

/-- Elementary symmetric sum of degree 2, over ordered index pairs `i < j`. -/
def e2 (y : Fin N → ℤ) : ℤ :=
  ∑ p ∈ univ.filter (fun p : Fin N × Fin N => p.1 < p.2), y p.1 * y p.2

end VNNUE
```

Every module starts with:

```lean
import Mathlib
import VNNUE.Basic
namespace VNNUE
open Finset
variable {R : Type*} [AddCommGroup R] {N L : ℕ}
```

### 4.1 `VNNUE/Accumulator.lean` (3 theorems)

**T1. Homomorphism and delta rule.** Φ is additive, and adding a delta to the input adds Φ(δ) to the accumulator. For sparse δ only the support columns matter.

```lean
theorem Phi_add (W : Fin N → Fin L → R) (x y : Fin N → ℤ) :
    Phi W (x + y) = Phi W x + Phi W y
theorem accV_add_delta (b : Fin L → R) (W : Fin N → Fin L → R) (x δ : Fin N → ℤ) :
    accV b W (x + δ) = accV b W x + Phi W δ
theorem Phi_eq_sum_supp (W : Fin N → Fin L → R) (δ : Fin N → ℤ) :
    Phi W δ = ∑ i ∈ supp δ, δ i • W i
```
Proof sketch:
- `Phi_add`: `simp [Phi, add_smul, Finset.sum_add_distrib]`.
- `accV_add_delta`: `simp [accV, Phi_add, add_assoc]`.
- `Phi_eq_sum_supp`: `Finset.sum_subset` (terms off the support have `δ i = 0`, so `zero_smul`).

**T2. Path independence.** Folding any list of deltas from acc(x) yields acc(x + Σ list), for any order or grouping.

```lean
theorem foldDeltas_eq (b : Fin L → R) (W : Fin N → Fin L → R) (x : Fin N → ℤ)
    (l : List (Fin N → ℤ)) :
    foldDeltas W (accV b W x) l = accV b W (x + l.sum)
theorem foldDeltas_perm (b : Fin L → R) (W : Fin N → Fin L → R) (x : Fin N → ℤ)
    {l l' : List (Fin N → ℤ)} (h : l.Perm l') :
    foldDeltas W (accV b W x) l = foldDeltas W (accV b W x) l'
```
Proof sketch:
- `foldDeltas_eq`: induction on `l` generalizing `x`. `List.foldl_cons`, then `accV_add_delta` and `add_assoc`.
- `foldDeltas_perm`: rewrite both sides with `foldDeltas_eq` and use `List.Perm.sum_eq`.

**T3. Set form and any-source correctness.** The refresh formula agrees with the vector form. Starting from **any** stored set U (Finny entry, ancestor, empty), the accumulator for S is obtained with exactly |S △ U| column operations.

```lean
theorem accS_eq_accV (b : Fin L → R) (W : Fin N → Fin L → R) (S : Finset (Fin N)) :
    accS b W S = accV b W (ind S)
theorem accS_any_source (b : Fin L → R) (W : Fin N → Fin L → R) (S U : Finset (Fin N)) :
    accS b W S = accS b W U + ∑ i ∈ S \ U, W i - ∑ i ∈ U \ S, W i
theorem colOps_eq (S U : Finset (Fin N)) :
    colOps S U = (S \ U).card + (U \ S).card
```
Proof sketch:
- `accS_eq_accV`: `Finset.sum_ite_mem` / `Finset.sum_filter` with `one_smul` and `zero_smul`.
- `accS_any_source`: `Finset.sum_sdiff` applied twice (S = (S\U) ∪ (S∩U), U = (U\S) ∪ (S∩U)), then `abel`.
- `colOps_eq`: `Finset.symmDiff_eq_union`-style unfolding (`symmDiff_def`, `sup_eq_union`), then `Finset.card_union_of_disjoint` with `Finset.disjoint_sdiff_sdiff`.

### 4.2 `VNNUE/DeltaMonoid.lean` (3 theorems)

**T4. Telescoping and delta support.** Set deltas compose along a path, and the support of a set delta is the symmetric difference.

```lean
theorem delta_telescope (S0 S1 S2 : Finset (Fin N)) :
    delta S0 S1 + delta S1 S2 = delta S0 S2
theorem ind_add_delta (S T : Finset (Fin N)) : ind S + delta S T = ind T
theorem supp_delta (S T : Finset (Fin N)) : supp (delta S T) = symmDiff S T
```
Proof sketch:
- The first two: `funext i; simp [delta]; ring` (or `abel`).
- `supp_delta`: `ext i; simp [supp, delta, ind, Finset.mem_symmDiff]; split_ifs <;> simp_all`.

**T5. Net-delta support bound (cancellation never hurts).** The number of columns touched by the summed delta is at most the sum over the per-ply deltas.

```lean
theorem supp_add_subset (δ ε : Fin N → ℤ) : supp (δ + ε) ⊆ supp δ ∪ supp ε
theorem card_supp_sum_le (l : List (Fin N → ℤ)) :
    (supp l.sum).card ≤ (l.map (fun δ => (supp δ).card)).sum
```
Proof sketch:
- `supp_add_subset`: if δ i + ε i ≠ 0 then δ i ≠ 0 ∨ ε i ≠ 0.
- `card_supp_sum_le`: induction on `l` with `List.sum_cons`, then `Finset.card_le_card` on `supp_add_subset` and `Finset.card_union_le`.

**T6. Column-op distance is a metric that is invariant under relabelling.** This gives the triangle inequality, so routing through an intermediate source never beats the direct diff. Mirrored Finny entries are valid at the same cost.

```lean
theorem colOps_triangle {α : Type*} [DecidableEq α] (A B C : Finset α) :
    colOps A C ≤ colOps A B + colOps B C
theorem colOps_comm {α : Type*} [DecidableEq α] (A B : Finset α) : colOps A B = colOps B A
theorem colOps_image {α β : Type*} [DecidableEq α] [DecidableEq β] (f : α → β)
    (hf : Function.Injective f) (S C : Finset α) :
    colOps (S.image f) (C.image f) = colOps S C
```
Proof sketch:
- `colOps_triangle`: `symmDiff_triangle` ✓ (Mathlib/Order/SymmDiff.lean:210), then `Finset.card_le_card` and `Finset.card_union_le`.
- `colOps_comm`: `symmDiff_comm`.
- `colOps_image`: `Finset.image_symmDiff` for injective f (or `Finset.image_union`/`image_sdiff` with `hf`), then `Finset.card_image_of_injective`.

### 4.3 `VNNUE/Quantization.lean` (5 theorems)

Local definitions:

```lean
def Mplus {m : ℕ} (g : Fin N → Fin m) (w : Fin N → ℤ) (t : Fin m) : ℤ :=
  (((univ.filter (fun i => g i = t)).sup (fun i => (w i).toNat) : ℕ) : ℤ)
def Mminus {m : ℕ} (g : Fin N → Fin m) (w : Fin N → ℤ) (t : Fin m) : ℤ :=
  -(((univ.filter (fun i => g i = t)).sup (fun i => (-w i).toNat) : ℕ) : ℤ)
```

**T7. Wrap-around exactness.**
- (a) The int16 cast commutes with accumulation.
- (b) dec16 is exact on [−32768, 32767].
- (c) Hence the wrapped incremental accumulator, after **any** update path, decodes to the exact integer value whenever the final value is in range. Intermediate overflow does not matter.
- (d) The corrected `valMinAbs` window.

```lean
theorem castLanes_accV (b : Fin L → ℤ) (W : Fin N → Fin L → ℤ) (x : Fin N → ℤ) :
    castLanes (accV b W x) = accV (castLanes b) (castW W) x
theorem dec16_intCast (v : ℤ) (h : InI16 v) : dec16 (v : ZMod 65536) = v
theorem wrap_exact (b : Fin L → ℤ) (W : Fin N → Fin L → ℤ) (x : Fin N → ℤ)
    (l : List (Fin N → ℤ)) (j : Fin L) (h : InI16 (accV b W (x + l.sum) j)) :
    dec16 (foldDeltas (castW W) (accV (castLanes b) (castW W) x) l j)
      = accV b W (x + l.sum) j
theorem valMinAbs_window (v : ℤ) (h1 : -32768 < v) (h2 : v ≤ 32768) :
    ((v : ZMod 65536)).valMinAbs = v
```
Proof sketch:
- `castLanes_accV`: `funext j; simp [castLanes, castW, accV, Phi, Int.cast_sum, Int.cast_mul, zsmul_eq_mul]`. Int.cast is a ring hom.
- `dec16_intCast`: `ZMod.val_intCast` ✓ (gives `↑(v : ZMod n).val = v % n`). Case split on the sign of v, then `omega`.
- `wrap_exact`: `foldDeltas_eq` (T2, instantiated at R = ZMod 65536; the T2 proof is re-stated locally or the lemma is moved to Basic if module independence is relaxed), then `castLanes_accV`, then `dec16_intCast`.
- `valMinAbs_window`: `ZMod.valMinAbs_spec` ✓ (Mathlib/Data/ZMod/ValMinAbs.lean:79), with witness `v ≡ v` and `v*2 ∈ Ioc (-65536) 65536`.

Note: the earlier statement "−2^15 ≤ a < 2^15 ⇒ valMinAbs = a" is **false** at a = −32768 (valMinAbs = +32768). It is replaced by dec16_intCast and valMinAbs_window.

**T8. Saturation is path-dependent.** Saturating adds are not associative, so they cannot replace wrapping adds in incremental updates.

```lean
theorem sat16_not_assoc :
    sat16 (sat16 (32767 + 1) + (-1)) ≠ sat16 (32767 + sat16 (1 + (-1)))
theorem sat16_path_dependent :
    sat16 (sat16 (30000 + 30000) + (-30000)) = 2767 ∧
    sat16 (30000 + sat16 (30000 + (-30000))) = 30000
```
Proof sketch: `decide` or `norm_num [sat16]`. Checked numerically: 32766 ≠ 32767; 2767 and 30000.

**T9. Per-lane slot certificate and global bound.** For a group map g with legal capacities, every admissible active set's lane sum lies between the certified bounds. If |A| ≤ K, the sum is also ≤ K·max⁺. The global bound is the coarse special case.

```lean
theorem slot_certificate {m : ℕ} (g : Fin N → Fin m) (cap : Fin m → ℕ) (w : Fin N → ℤ)
    (A : Finset (Fin N)) (hA : ∀ t, (A.filter (fun i => g i = t)).card ≤ cap t) :
    ∑ i ∈ A, w i ≤ ∑ t, (cap t : ℤ) * Mplus g w t ∧
    ∑ t, (cap t : ℤ) * Mminus g w t ≤ ∑ i ∈ A, w i
theorem card_bound (w : Fin N → ℤ) (A : Finset (Fin N)) (K : ℕ) (M : ℤ) (hM0 : 0 ≤ M)
    (hM : ∀ i, w i ≤ M) (hK : A.card ≤ K) : ∑ i ∈ A, w i ≤ K * M
theorem global_bound (b : ℤ) (w : Fin N → ℤ) (M : ℤ) (hM : ∀ i, |w i| ≤ M)
    (A : Finset (Fin N)) : |b + ∑ i ∈ A, w i| ≤ |b| + A.card * M
```
Proof sketch:
- `slot_certificate`: `Finset.sum_fiberwise` (split A by g). On each fibre, w i ≤ Mplus via `Finset.le_sup` and `Int.self_le_toNat`. Bound the fibre sum by card·Mplus with `Finset.sum_le_card_nsmul`, then by cap·Mplus using Mplus ≥ 0. Symmetric for Mminus.
- `card_bound`: `Finset.sum_le_card_nsmul`, then `mul_le_mul_of_nonneg_right`.
- `global_bound`: `abs_add`, `Finset.abs_sum_le_sum_abs`, `Finset.sum_le_card_nsmul`.

Instances (by `norm_num`):
- 32·127 = 4064 < 2^15 (P1 and i8-column lanes);
- 32·127² = 516,128 < 2^31 (P2).

The removed "≤ 32 slots" instance is replaced by a group map: square for PSI, (colour, type) for PSQ.

**T10. `pmaddubsw` pair safety.** u8 activations ≤ 127 and i8 weights cannot saturate the int16 pair sum. At 255 they can.

```lean
theorem pmaddubsw_safe (a₁ a₂ w₁ w₂ : ℤ) (ha₁ : 0 ≤ a₁ ∧ a₁ ≤ 127) (ha₂ : 0 ≤ a₂ ∧ a₂ ≤ 127)
    (hw₁ : |w₁| ≤ 127) (hw₂ : |w₂| ≤ 127) : |a₁ * w₁ + a₂ * w₂| ≤ 32258
theorem pmaddubsw_unsafe_255 : (255 : ℤ) * 127 + 255 * 127 > 32767
```
Proof sketch:
- `pmaddubsw_safe`: `abs_add`, `abs_mul`, and `mul_le_mul` with `abs_of_nonneg`; then `norm_num` (127·127·2 = 32258).
- `pmaddubsw_unsafe_255`: `norm_num` (64770 > 32767).

This covers the PSI output u_h ∈ [0,127] feeding fc_0.

**T11. Quantized-sum rounding error.** Rounding every weight and the bias to within ½ at scale s gives an accumulator error ≤ (|A|+1)/(2s).

```lean
theorem quant_sum_error {ι : Type*} (s : ℝ) (hs : 0 < s) (A : Finset ι) (w q : ι → ℝ)
    (b qb : ℝ) (hq : ∀ i ∈ A, |q i - s * w i| ≤ 1 / 2) (hb : |qb - s * b| ≤ 1 / 2) :
    |(qb + ∑ i ∈ A, q i) / s - (b + ∑ i ∈ A, w i)| ≤ (A.card + 1) / (2 * s)
```
Proof sketch: rewrite the left side as |(qb − s b) + Σ(q i − s w i)| / s (`div_sub'`, `Finset.mul_sum`). Then `abs_add`, `Finset.abs_sum_le_sum_abs`, `Finset.sum_le_card_nsmul`, and `div_le_div_of_nonneg_right`.

(The text must note that raising s also moves the clamp and the /512 normalization.)

### 4.4 `VNNUE/Interaction.lean` (7 theorems)

Local definition:

```lean
def e3ord (y : Fin N → ℤ) : ℤ :=
  ∑ i, ∑ j, ∑ k, if i ≠ j ∧ j ≠ k ∧ i ≠ k then y i * y j * y k else 0
```

**T12. Newton–Girard order 2, per lane, and exact halving.** On binary inputs, a lane's P1² − P2 is twice the pair sum. The difference is even for **every** integer input, so the right shift is exact.

```lean
theorem ng2 (y : Fin N → ℤ) :
    (∑ i, y i) ^ 2 - ∑ i, y i ^ 2 = 2 * e2 y
theorem psi_lane (v x : Fin N → ℤ) (hx : IsBinary x) :
    psum 1 v x ^ 2 - psum 2 v x = 2 * e2 (fun i => x i * v i)
theorem psi_even (v x : Fin N → ℤ) : 2 ∣ psum 1 v x ^ 2 - psum 2 v x
```
Proof sketch:
- `ng2`: `sq`, `Finset.sum_mul_sum` ✓ (Algebra/BigOperators/Ring/Finset.lean:62). Then `Finset.sum_product'` and split univ ×ˢ univ into diagonal ∪ {p.1<p.2} ∪ {p.1>p.2}. The last two are equal via `Finset.sum_bij` (swap), `mul_comm`.
- `psi_lane`: for binary x, x i ^ 2 = x i (case split), so psum 2 v x = Σ (x i v i)^2. Then `ng2`.
- `psi_even`: `Int.emod_emod_of_dvd`-style argument mod 2. (Σ a_i)² ≡ Σ a_i² and x² ≡ x (mod 2), so psum 1² ≡ Σ x_i² v_i² ≡ Σ x_i v_i² (mod 2). Alternatively, write the difference as 2·Σ_{i<j} + Σ_i (x_i² − x_i) v_i² and use 2 ∣ x(x−1) (`Int.even_mul_pred_self`).

**T13. Multi-head PSI readout = all-pairs factorization machine (Rendle 2010).**

```lean
theorem psi_heads {r m : ℕ} (V : Fin r → Fin N → ℤ) (Lam : Fin m → Fin r → ℤ)
    (x : Fin N → ℤ) (hx : IsBinary x) (h : Fin m) :
    ∑ t, Lam h t * (psum 1 (V t) x ^ 2 - psum 2 (V t) x)
      = 2 * ∑ p ∈ univ.filter (fun p : Fin N × Fin N => p.1 < p.2),
              x p.1 * x p.2 * ∑ t, Lam h t * V t p.1 * V t p.2
```
Proof sketch: `psi_lane` per t, then `Finset.mul_sum`, `Finset.sum_comm` and `ring_nf` under the binder.

**T14. Incremental rule for E (DBSP product rule).** Holds for all integer x, with no hypotheses.

```lean
theorem psum_add (k : ℕ) (v x δ : Fin N → ℤ) : psum k v (x + δ) = psum k v x + psum k v δ
theorem psi_add_feature (v x : Fin N → ℤ) (j : Fin N) :
    psum 1 v (x + Pi.single j 1) ^ 2 - psum 2 v (x + Pi.single j 1)
      = (psum 1 v x ^ 2 - psum 2 v x) + 2 * (v j * psum 1 v x)
theorem psi_remove_feature (v x : Fin N → ℤ) (j : Fin N) :
    psum 1 v (x - Pi.single j 1) ^ 2 - psum 2 v (x - Pi.single j 1)
      = (psum 1 v x ^ 2 - psum 2 v x) - 2 * (v j * psum 1 v (x - Pi.single j 1))
```
Proof sketch:
- `psum_add`: `simp [psum, add_mul, Finset.sum_add_distrib]`.
- psum k v (Pi.single j 1) = v j ^ k by `Finset.sum_pi_single'`, then `ring`.
- Check: (p+v)² − (q+v²) = p² − q + 2vp. Removal: with p′ = p − v and q′ = q − v², p′² − q′ = p² − q − 2vp′.

**T15. Newton–Girard order 3 and divisibility by 6** (degree-3 ablation C1′).

```lean
theorem ng3 (y : Fin N → ℤ) :
    e3ord y = (∑ i, y i) ^ 3 - 3 * (∑ i, y i) * (∑ i, y i ^ 2) + 2 * ∑ i, y i ^ 3
theorem ng3_six_dvd (y : Fin N → ℤ) :
    (6 : ℤ) ∣ (∑ i, y i) ^ 3 - 3 * (∑ i, y i) * (∑ i, y i ^ 2) + 2 * ∑ i, y i ^ 3
theorem ng3_binary (v x : Fin N → ℤ) (hx : IsBinary x) :
    e3ord (fun i => x i * v i)
      = psum 1 v x ^ 3 - 3 * psum 1 v x * psum 2 v x + 2 * psum 3 v x
```
Proof sketch:
- `ng3`: expand (Σy)³ as a triple sum (`Finset.sum_mul_sum` twice). Inclusion–exclusion over the diagonals {i=j}, {j=k}, {i=k}, which meet in {i=j=k}: Σ_all − Σ_{i=j} − Σ_{j=k} − Σ_{i=k} + 2Σ_{i=j=k}. Each diagonal sum is p1·p2 and the triple diagonal is p3. Use `Finset.sum_ite`, `Finset.sum_filter`.
- `ng3_six_dvd`, route 1: show e3ord y = 6·Σ_{i<j<k} y_i y_j y_k by grouping the 3! orderings.
- `ng3_six_dvd`, route 2 (easier): modular. Mod 2, y² ≡ y gives RHS ≡ p1 − p1 ≡ 0. Mod 3, (Σy)³ ≡ Σy³ (Frobenius, `ZMod.pow_card` / `frobenius` in ZMod 3) gives RHS ≡ 3p3 ≡ 0. Then `Int.dvd_of_emod_eq_zero` with 6 = 2·3 coprime (`IsCoprime.mul_dvd`).
- Alternative: Mathlib's `MvPolynomial.psum_eq_mul_esymm_sub_sum` ✓ (Mathlib/RingTheory/MvPolynomial/Symmetric/NewtonIdentities.lean:245) at k = 3, evaluated with `MvPolynomial.aeval`.
- `ng3_binary`: x^k = x for binary x, then `ng3`.
- Numerically checked on 100 random integer vectors.

**T16. PSI range certificates (int32 safety).**

```lean
theorem mix_range {r : ℕ} (Lam e : Fin r → ℤ) (hr : r ≤ 32) (hL : ∀ t, |Lam t| ≤ 127)
    (he : ∀ t, |e t| ≤ 32768) : |∑ t, Lam t * e t| < 2 ^ 31
theorem ng3_range (p1 p2 p3 : ℤ) (h1 : |p1| ≤ 480) (h2 : |p2| ≤ 7200) (h3 : |p3| ≤ 108000) :
    |p1 ^ 3 - 3 * p1 * p2 + 2 * p3| < 2 ^ 31
theorem e_lane_range (P1 P2 : ℤ) (h1 : |P1| ≤ 4064) (h2 : 0 ≤ P2) (h3 : P2 ≤ 516128) :
    |P1 ^ 2 - P2| < 2 ^ 31
```
Proof sketch:
- `mix_range`: `Finset.abs_sum_le_sum_abs`, `abs_mul`, then each term ≤ 127·32768. `Finset.sum_le_card_nsmul` with card = r ≤ 32 gives ≤ 133,169,152 < 2^31.
- `ng3_range`: `abs_sub`/`abs_add` triangle, `abs_pow`, `pow_le_pow_left` (480³ = 110,592,000), `abs_mul` (3·480·7200 = 10,368,000), 2·108,000. Total 121,176,000 < 2^31 by `norm_num`.
- `e_lane_range`: `sq_abs`, `abs_sub` ≤ 4064² + 516,128 = 17,032,224.

**T17. Universality of PSI at full width.** Every integer pair table is reproduced exactly by PSI with pair lanes v_t = e_a + e_b and λ_t = Q_ab. The low-rank restriction is therefore the only expressivity limit; it is not structural.

```lean
theorem psi_universal (Q : Fin N → Fin N → ℤ) :
    ∃ (r : ℕ) (V : Fin r → Fin N → ℤ) (lam : Fin r → ℤ), ∀ x : Fin N → ℤ, IsBinary x →
      ∑ t, lam t * (psum 1 (V t) x ^ 2 - psum 2 (V t) x)
        = 2 * ∑ p ∈ univ.filter (fun p : Fin N × Fin N => p.1 < p.2), Q p.1 p.2 * x p.1 * x p.2
```
Proof sketch:
- Let P := univ.filter (p.1 < p.2), r := P.card, and enumerate it with e := P.equivFin.
- Define V t := Pi.single (e.symm t).1.1 1 + Pi.single (e.symm t).1.2 1 and lam t := Q (e.symm t).1.1 (e.symm t).1.2.
- Apply `psi_heads` with m = 1. Show Σ_t lam t · V t i · V t j = Q i j for i < j, because V t i · V t j = 1 iff e.symm t = (i, j). Use `Finset.sum_eq_single`.
- Reindex with `Equiv.sum_comp`.

**T18. Strict generalization and corrected non-affinity.**
- (a) With Λ = 0 the PSI readout vanishes.
- (b) With zero extra fc_0 columns, fc_0 equals NNUE's.
- (c) Any affine function has zero mixed second difference.
- (d) A PSI lane has a nonzero one.

```lean
theorem psi_zero_heads {r m : ℕ} (V : Fin r → Fin N → ℤ) (x : Fin N → ℤ) :
    (fun h : Fin m => ∑ t, (0 : Fin m → Fin r → ℤ) h t * (psum 1 (V t) x ^ 2 - psum 2 (V t) x))
      = 0
theorem fc0_embed {o m : ℕ} (A : Matrix (Fin o) (Fin L) ℤ) (Bz : Matrix (Fin o) (Fin m) ℤ)
    (a : Fin L → ℤ) (z : Fin m → ℤ) (hB : Bz = 0) :
    A.mulVec a + Bz.mulVec z = A.mulVec a
theorem affine_second_diff (c : ℤ) (w : Fin 2 → ℤ) :
    let f := fun x : Fin 2 → ℤ => c + ∑ i, w i * x i
    f ![1, 1] - f ![1, 0] - f ![0, 1] + f ![0, 0] = 0
theorem psi_not_affine :
    let q := fun x : Fin 2 → ℤ => psum 1 ![1, 1] x ^ 2 - psum 2 ![1, 1] x
    q ![1, 1] - q ![1, 0] - q ![0, 1] + q ![0, 0] = 2
```
Proof sketch:
- `psi_zero_heads`: `funext; simp`.
- `fc0_embed`: `subst hB; simp [Matrix.zero_mulVec]`.
- The last two: `simp [Fin.sum_univ_two, psum]; ring` or `decide`.

Paper statement: "at fixed king squares and block, K32H3 PSQ lanes are affine in the non-king piece indicators (T18c applies coordinate-wise), while a PSI lane is not (T18d)". **Not** "NNUE is affine in β", which was false.

### 4.5 `VNNUE/Symmetry.lean` (4 theorems)

Local definitions:

```lean
def mirH (s : Fin 64) : Fin 64 := ⟨8 * (s.val / 8) + (7 - s.val % 8), by have := s.isLt; omega⟩   -- s xor 7
def flipV (s : Fin 64) : Fin 64 := ⟨8 * (7 - s.val / 8) + s.val % 8, by have := s.isLt; omega⟩   -- s xor 56
abbrev Feat := Fin 2 × Fin 6 × Fin 64
def sigH (i : Feat) : Feat := (i.1, i.2.1, mirH i.2.2)
def sigC (i : Feat) : Feat := (Fin.rev i.1, i.2.1, flipV i.2.2)
def canon (i : Feat) : Feat :=
  let j := if i.1 = 0 then i else sigC i
  if j.2.2.val % 8 < 4 then sigH j else j
```

**T19. Klein four-group on squares.** h and v are commuting, fixed-point-free involutions. Exactly 32 squares lie on files e–h, and h maps a–d onto e–h. This justifies 32 canonical king buckets.

```lean
theorem klein_squares :
    (∀ s, mirH (mirH s) = s) ∧ (∀ s, flipV (flipV s) = s) ∧
    (∀ s, mirH (flipV s) = flipV (mirH s)) ∧
    (∀ s, mirH s ≠ s) ∧ (∀ s, flipV s ≠ s) ∧ (∀ s, mirH (flipV s) ≠ s)
theorem king_half :
    (univ.filter (fun s : Fin 64 => 4 ≤ s.val % 8)).card = 32 ∧
    ∀ s : Fin 64, s.val % 8 < 4 → 4 ≤ (mirH s).val % 8
```
Proof sketch: `decide`. Alternatively `intro s; fin_cases s <;> rfl`, or `Fin.ext` with `omega` on `s.val < 64`.

**T20. Free action on the 768 absolute features; 192 orbits.**

```lean
theorem feat_free : ∀ i : Feat, sigH i ≠ i ∧ sigC i ≠ i ∧ sigH (sigC i) ≠ i
theorem canon_invariant : ∀ i : Feat, canon (sigH i) = canon i ∧ canon (sigC i) = canon i
theorem canon_image_card : (univ.image canon).card = 192
```
Proof sketch:
- `feat_free`: sigC changes the colour, and sigH has no fixed square. `decide` (768 cases; may need `set_option maxRecDepth`). Do **not** use `native_decide`, which would enlarge the trusted base.
- `canon_invariant`: `decide`, or by case analysis on colour and file.
- `canon_image_card`: the image is {i | i.1 = 0 ∧ 4 ≤ file} (`decide` or `Finset.card_filter`), with card 1·6·32 = 192.

Consequence: tying σ_c and σ_h leaves 192 free columns.

**T21. Equivariance (core lemma) and PSI perspective sharing.** If weights are tied by W_{σ i}(π j) = W_i j, the accumulator of the transformed board is the lane-permuted accumulator. For PSI, under V_{P t}(σ i) = V_t i, the Black view of the shared state is a lane permutation of the White state, for both P1 and P2.

```lean
theorem acc_equivariant {n L : ℕ} (W : Fin n → Fin L → ℤ) (σ : Equiv.Perm (Fin n))
    (π : Equiv.Perm (Fin L)) (hW : ∀ i j, W (σ i) (π j) = W i j) (x : Fin n → ℤ) (j : Fin L) :
    ∑ i, x (σ.symm i) * W i (π j) = ∑ i, x i * W i j
theorem psi_shared {n r : ℕ} (V : Fin r → Fin n → ℤ) (σ : Equiv.Perm (Fin n))
    (P : Equiv.Perm (Fin r)) (hV : ∀ t i, V (P t) (σ i) = V t i) (x : Fin n → ℤ) (t : Fin r) :
    psum 1 (V (P t)) (x ∘ σ.symm) = psum 1 (V t) x ∧
    psum 2 (V (P t)) (x ∘ σ.symm) = psum 2 (V t) x
```
Proof sketch:
- `acc_equivariant`: `(Equiv.sum_comp σ _).symm`; after substituting i = σ i′, the summand is x i′ · W (σ i′) (π j) = x i′ · W i′ j. ✓ (`Equiv.sum_comp`, to_additive of `Equiv.prod_comp`)
- `psi_shared`: unfold `psum`, apply `Equiv.sum_comp σ`, rewrite with `hV`.

**T22. Stockfish canonicalization as the regular representation, and the integer isotypic split.**

```lean
theorem regular_rep {n L : ℕ} (W : Fin n → Fin L → ℤ) (σ : Equiv.Perm (Fin n))
    (hσ : ∀ i, σ (σ i) = i) :
    let W' : Fin n → (Fin L ⊕ Fin L) → ℤ := fun i => Sum.elim (W i) (W (σ i))
    (∀ i k, W' (σ i) k = W' i (Equiv.sumComm _ _ k)) ∧
    ∀ x : Fin n → ℤ, ∀ j, (∑ i, x i * W' i (Sum.inr j)) = ∑ i, x (σ i) * W i j
theorem isotypic (u v : Fin L → ℤ) :
    (u + v) + (u - v) = 2 • u ∧ (u + v) - (u - v) = 2 • v
```
Proof sketch:
- `regular_rep` (a): `cases k` and `simp [hσ]`.
- `regular_rep` (b): `Equiv.sum_comp σ` with `hσ`.
- `isotypic`: `funext; simp; ring`.
- Remark for the paper: over ℤ, (u,v) ↦ (u+v, u−v) has determinant ±2^L, so it is injective but not unimodular, which explains the factor-2 scales.

### 4.6 `VNNUE/Refresh.lean` (2 theorems)

**T23. Low-rank king transition (Woodbury pattern); factored = merged; generalization.** For any commutative ring (instantiated at ℤ and ZMod 65536), the bucket-κ′ accumulator of x′ equals the bucket-κ accumulator of x, plus the κ′ columns of the delta, plus a rank-r correction applied to y = Vx.

```lean
theorem king_transition {S : Type*} [CommRing S] {L r n : ℕ}
    (W0 : Matrix (Fin L) (Fin n) S) (U U' : Matrix (Fin L) (Fin r) S)
    (V : Matrix (Fin r) (Fin n) S) (x x' : Fin n → S) :
    (W0 + U' * V).mulVec x'
      = (W0 + U * V).mulVec x + (W0 + U' * V).mulVec (x' - x) + (U' - U).mulVec (V.mulVec x)
theorem factored_eq_merged {S : Type*} [CommRing S] {L r n : ℕ}
    (W0 : Matrix (Fin L) (Fin n) S) (U : Matrix (Fin L) (Fin r) S)
    (V : Matrix (Fin r) (Fin n) S) (x : Fin n → S) :
    (W0 + U * V).mulVec x = W0.mulVec x + U.mulVec (V.mulVec x)
theorem lowrank_generalizes {S : Type*} [CommRing S] {K L n : ℕ}
    (Wk : Fin K → Matrix (Fin L) (Fin n) S) :
    ∃ (W0 : Matrix (Fin L) (Fin n) S) (V : Matrix (Fin n) (Fin n) S)
      (U : Fin K → Matrix (Fin L) (Fin n) S), ∀ k, W0 + U k * V = Wk k
```
Proof sketch:
- `king_transition`: `Matrix.add_mulVec` ✓, `Matrix.sub_mulVec` ✓, `Matrix.mulVec_mulVec` ✓ (Mathlib/Data/Matrix/Mul.lean:817/1094/909), `Matrix.mulVec_sub`, then `abel`.
- `factored_eq_merged`: `Matrix.add_mulVec` and `Matrix.mulVec_mulVec`.
- `lowrank_generalizes`: W0 = 0, V = 1, U = Wk, then `simp`.
- The region term R_{ρ(κ)} is absorbed into W0 within a region.

**T24. Source selection is never worse than Stockfish; hybrid is correct.**
- (a) For any cost into a linear order (unavailable = ⊤), the minimum over a superset of candidates is ≤ the minimum over the subset. In particular VNNUE's candidate set (all `Src`) is never worse than Stockfish's {refresh, finny, hybrid}.
- (b) The hybrid identity (new = prev − cache_old_corrected + cache_new_corrected + non-PSQ delta) is exact.

```lean
inductive Src | refresh | finny | hybrid | woodbury | ancestor | finnyMirror
  deriving DecidableEq, Fintype
def sfSet : Finset Src := {Src.refresh, Src.finny, Src.hybrid}

theorem inf'_superset {ι α : Type*} [LinearOrder α] (C₁ C₂ : Finset ι) (h : C₁ ⊆ C₂)
    (hne : C₁.Nonempty) (f : ι → α) :
    C₂.inf' (hne.mono h) f ≤ C₁.inf' hne f
theorem never_worse (cost : Src → WithTop ℕ) :
    (univ : Finset Src).inf' ⟨Src.refresh, mem_univ _⟩ cost
      ≤ sfSet.inf' ⟨Src.refresh, by simp [sfSet]⟩ cost
theorem hybrid_correct (b : Fin L → R) (W W' : Fin N → Fin L → R)
    (S Sprev Cold Cnew : Finset (Fin N)) (T T' : Fin L → R) :
    accS b W' S + T'
      = (accS b W Sprev + T)
        - (accS b W Cold + ∑ i ∈ Sprev \ Cold, W i - ∑ i ∈ Cold \ Sprev, W i)
        + (accS b W' Cnew + ∑ i ∈ S \ Cnew, W' i - ∑ i ∈ Cnew \ S, W' i)
        + (T' - T)
```
Proof sketch:
- `inf'_superset`: `Finset.le_inf'` together with `Finset.inf'_le` (each element of C₁ is in C₂).
- `never_worse`: `inf'_superset` with `Finset.subset_univ`.
- `hybrid_correct`: rewrite the two corrected cache terms by `accS_any_source` (re-proved locally, or via a local copy of T3) to accS b W Sprev and accS b W' S, then `abel`.
- Corollary for the paper: the woodbury source's correctness is T23, and ancestor's is T2/T5.

### 4.7 `VNNUE/Complexity.lean` (1 theorem + 2 appendix lemmas)

**T25. PSI overhead bound and king-move total.**
- (a) If NNUE's per-move PSQ byte traffic is ≥ 4kL, then adding e bytes per changed absolute feature increases it by at most a factor (1 + e/(4L)).
- (b) The VNNUE total on a king event is ≤ Stockfish's FT cost + k·e.

```lean
theorem psi_overhead (k L e costN : ℕ) (h : 4 * k * L ≤ costN) :
    4 * L * (costN + k * e) ≤ (4 * L + e) * costN
theorem king_event_total {ι : Type*} (C₁ C₂ : Finset ι) (h : C₁ ⊆ C₂) (hne : C₁.Nonempty)
    (f : ι → ℕ) (k e : ℕ) :
    C₂.inf' (hne.mono h) f + k * e ≤ C₁.inf' hne f + k * e
```
Proof sketch:
- `psi_overhead`: expand; the inequality reduces to 4L·k·e ≤ e·costN, which is `Nat.mul_le_mul_left e h` after commuting (`nlinarith`).
- `king_event_total`: `Nat.add_le_add_right` of T24's lemma (re-proved locally).
- Instance: e = 6r = 192 and 4L = 4096 give ≤ 4.7%.

**A1 (appendix). Ski-rental materialization (S-2.6).** The deterministic "materialize at the B-th use" rule is (2 − 1/B)-competitive, and no threshold does better on its worst input. It is genuinely online only for the materialization decision. The refresh-versus-walk decision is offline (A-T11 is dropped).

```lean
def alg (B n : ℕ) : ℕ := if n < B then n else 2 * B - 1
def algT (B t n : ℕ) : ℕ := if n ≤ t then n else t + B
theorem ski_upper (B n : ℕ) (hB : 1 ≤ B) : B * alg B n ≤ (2 * B - 1) * min n B
theorem ski_lower (B t : ℕ) (hB : 1 ≤ B) : (2 * B - 1) * min (t + 1) B ≤ B * algT B t (t + 1)
```
Proof sketch:
- `ski_upper`: `unfold alg; split_ifs`, then `Nat.min_def` and `nlinarith`.
- `ski_lower`: case split on t + 1 ≤ B. The first case reduces to (B−1)t ≤ (B−1)², the second to B − 1 ≤ t. Use `nlinarith`, or `zify` with `push_cast`.
- Hand-checked.

**A2 (appendix). Fused-pass traffic.**

```lean
theorem fused_traffic (ks : List ℕ) (knet c a : ℕ) (ht : 1 ≤ ks.length) (hk : knet ≤ ks.sum) :
    knet * c + 2 * a + 2 * a * (ks.length - 1) ≤ (ks.map (fun k => k * c + 2 * a)).sum
```
Proof sketch: the right side is c·Σk + 2a·t (`List.sum_map_add`, `List.sum_map_mul_right`, `List.map_const'`). The left side is ≤ c·Σk + 2a·t by `hk` and `Nat.sub_add_cancel ht`. Then `nlinarith`.

### 4.8 Theorem count and difficulty order

| Module | Theorems | Difficulty (easy → hard) |
|---|---|---|
| Basic.lean | definitions only | — |
| Accumulator.lean | T1–T3 | easy |
| DeltaMonoid.lean | T4–T6 | easy |
| Quantization.lean | T7–T11 | easy–medium (T9 fibrewise sums) |
| Interaction.lean | T12–T18 | medium (T12, T13, T17); hardest is T15 |
| Symmetry.lean | T19–T22 | easy (`decide`; watch recursion depth on 768 cases) |
| Refresh.lean | T23–T24 | easy |
| Complexity.lean | T25, A1, A2 | easy |
| **Total** | **25 main + 2 appendix** | |

Suggested proving order: T19/T20 (decide), T1–T6, T8, T10, T7, T23, T24, T25, A1, A2, T9, T11, T12–T14, T16, T18, T21, T22, T17, T15.

### 4.9 Disposition of judge-flagged statements

| Original | Problem | Disposition |
|---|---|---|
| A-T3b valMinAbs on [−2^15, 2^15) | false at −32768 | replaced by T7 dec16_intCast; valMinAbs on (−2^15, 2^15] as T7(d) |
| A-C1 "r=768, V=I gives every diagonal-core table" | false (z ≡ 0) | removed; correct universality is T17 (pair lanes) |
| A-T4 degree-3 range with no p2/p3 hypotheses | false | T16 ng3_range with \|p2\| ≤ 7200, \|p3\| ≤ 108000 |
| A-T4 / C5 "≤ 32 slots, one active each" | ill-defined | removed; T9 uses a group map with legal caps |
| A §5 / T9c "NNUE first layer affine in β" | false (K32H3 is king×piece; threats and pawn pairs are pairs) | restated in T18 at fixed kings and block |
| A-T10/T13, S-T10 "never worse than SF" with SF = {Finny, refresh} | wrong baseline | T24 includes hybrid in SF's set; applies to FT columns only; T25(b) adds PSI's k·e |
| A "readout 1–2% of fc_0" | arithmetic inconsistent | recomputed: 2048 MACs = 6.3% of dense fc_0 at r=32, m=16 |
| A-T11 ski rental on catch-up length | catch-up length is known offline | dropped; only S-2.6 materialization kept (A1, appendix) |
| A-T12 clamp-delta sparsity | vacuous | dropped |
| A-C4 / S-2.2 net-delta "new" | prior art PR #5988 (≈0.2% statistic) | credited; kept as a low-yield generalization (E3) |
| S-2.4 (ii)/(iii) conflation | i8 storage and FT scale are separate | separated in C5 |
| S factored "+r cols at eval, negligible" | r·L MACs per eval | stated as a tax in C2′ and §3 |
| Sym-T10(i) in ℕ | false with truncated subtraction | concat design dropped (it compares smaller nets); if revived, state over ℤ or add L_B ≤ L |
| Sym concat L/2 design | unequal networks | dropped; FT stays at full L |
| Sym torus translation | aliasing; not chess structure | dropped |
| Sym-S4 isotypic "no arithmetic" | it is a negation pass | stated as "one negation pass" (E9a) |
| Baseline inconsistency (46 MB vs 138 MB) | — | single verified K32H3 baseline: 138.4 MB |
| S-T12 mirror tying | needs P² = I and a mirror-closed region | hypotheses stated in C6 / E9b |

---

## 5. Experiments

Everything runs on this machine: Python 3.14.3, numpy 2.4.3, torch 2.10.0+cpu, python-chess 1.11.2 (all verified installed). No game databases or Stockfish nets are present locally. The C: drive has about 9 GB free, so every artifact must be kept under 1 GB.

Code goes in `C:/Users/user/vnnue/proto/`, results in `proto/results/*.json`. All runs are seeded.

### 5.1 Data: realistic move sequences

`proto/gen_trees.py`:
1. **Game sequences.** Generate 200 self-play games with python-chess. A small alpha-beta (depth 3, material + PST eval, MVV-LVA ordering) plays both sides, starting from 8 random opening plies; games are capped at 200 plies. Log every move with its type.
2. **Search traces.** From 2,000 positions sampled from those games, run the same alpha-beta at depth 4 with quiescence on captures. Log the exact event stream (`make m`, `unmake`, `eval`), which is what an engine accumulator stack sees. Also generate traces with a deliberately lazy evaluator that evaluates only at leaves and at 30% of interior nodes, mimicking static-eval-for-pruning.
3. **Store** traces as compact numpy arrays (< 200 MB total).

Limitation: the trees come from a toy searcher, so path statistics (catch-up length, king-move frequency) are proxies for Stockfish's. Every report says so.

### 5.2 Feature extraction

`proto/features.py`:
- Absolute 768 (White-absolute).
- K32H3 indices: orientation, 32 buckets as in `k32h3.h`, 3 blocks per the verified definitions.
- A **threat proxy**: ordered (attacker piece, square, target piece, square) pairs with target ∈ attacks(attacker), deduplicated per the nnue.md rules where feasible, and labelled "proxy".
- PP_3Wide pairs.

Unit test: incremental feature deltas equal recomputed set differences on every trace event.

### 5.3 E1 Exactness (empirical companion to T1–T3, T7, T12–T14, T21, T23)

`proto/accum.py`:
- Implement NNUE and VNNUE accumulators with **np.int16 wrapping** arithmetic (and int32 for P2).
- Use random weights scaled so that the certificate holds (|W| ≤ 127 to fit typical L1 ranges), plus a stress variant with large weights that **violates** intermediate ranges but whose final values are in range, to exhibit wrap exactness.
- At every `eval` event, assert bit-equality with a from-scratch refresh for: every C3 source; C4 net-delta; the C2 transition (with U, V_K integer and W_κ materialized); the PSI readout (identity T13 checked against a brute-force pair sum); and the Black view via lane permutation under tying.
- Also show the saturating variant diverging (T8).

Output: counts of checks and mismatches (expected 0).

### 5.4 E2–E4 Op counts, bytes and wall clock

For each trace and policy, count:
- col, mcol, passes, bytes (β_w per family), and PSI lanes;
- distributions of k, τ_proxy, |S|, d_F, d_old + d_F, catch-up length t, and the cancellation ratio |supp Σδ| / Σ|supp δ|;
- the fraction of own-king events where each C3 source is the argmin, for r_K ∈ {4, 8, 16} and μ ∈ {1, 1.5, 2}.

Policies:
- (P0) SF-like: incremental, Finny, hybrid with SF's conditions, lazy catch-up, two-ply capture cancellation;
- (P1) P0 + PSI;
- (P2) P1 + C3 full candidate set + C4;
- (P3) P2 + C2 merged;
- (P4) factored C2′, with per-eval MACs counted.

Wall clock:
- numpy kernels per update type (L = 1024, a real 138 MB int16 K32H3-sized table to reproduce cache pressure), measured with `time.perf_counter_ns`, median of 5 runs, single thread.
- Report ns per event and bytes per event.
- **Caveat stated in results**: Python/numpy dispatch overhead (~µs) dominates 2 KB column adds, so wall clock is reported only for batched kernels (many columns per call) and is a weak proxy. The op and byte counts are the primary metrics. A C/SIMD implementation in a real engine is out of scope [Exp, future].

Expected outcomes are left blank. Hypotheses:
- H1: PSI byte overhead ≤ 4.7% of PSQ traffic (T25 bound) and lower once the threat proxy is included.
- H2: woodbury wins in < 10% of own-king events at r_K = 16 [no prior; to be measured].
- H3: the C4 column saving is < 1% (consistent with PR #5988's 0.2%).

### 5.5 E5–E8 Tiny training sanity checks (torch, CPU, minutes each)

These establish only that the math trains and quantizes as specified. They say nothing about chess strength.

- **E5 (rank sufficiency, synthetic).** Use a random K32H3-shaped table with planted structure W_κ = W⁰ + U_κ V + noise. Measure SVD reconstruction error vs r_K, and the integer-factor (QAT) error vs float. If a real Stockfish `.nnue` file can be downloaded (≤ 150 MB) and parsed, repeat on real HalfKA/K32H3 blocks (optional; parser effort is significant).
- **E6 (PSI learns pairs).** Synthetic target on positions from 5.1:
  - y = Σ PST(β) + Σ_{i<j} Q_ij β_i β_j + noise, with Q in three variants: (i) low-rank, (ii) sparse "attack-geometry" (nonzero only for pairs in attack relation, via python-chess), (iii) both.
  - Models at equal per-move update lanes: (M0) 768→L linear + pairwise product + MLP head (NNUE-like, without king buckets for tractability); (M1) M0 with L reduced to pay for PSI (r=32, m=16); (M2) M0 + explicit sparse pair features.
  - Metric: validation MSE.
  - Then QAT: quantize V and Λ as in §2.3, and verify that the integer readout equals the float model within the T11/T16 bounds.
- **E7 (tying).** M1 with tied vs untied V under colour flip. Measure loss and check the T21 identity numerically.
- **E8 (real-ish labels, optional).** Labels from the toy searcher's depth-4 score on 1–2 M positions from 5.1. Compare M0 vs M1 by validation loss only. These labels are weak and noisy, and conclusions are limited to "trains / does not train".

### 5.6 What cannot be established here

The following need the full Stockfish training pipeline (billions of Leela-rescored positions, GPU weeks) and an engine-level C++ implementation:
- **Elo / playing strength** of any VNNUE variant. The protocol is: (1) validation loss at equal measured nps; (2) fixed-node matches against an identically trained K32H3 baseline; (3) fixed-time SPRT at two time controls on fishtest-like infrastructure.
- **Real nps, cache-miss and bytes-per-node** effects (perf counters on AVX2, AVX-512 and NEON).
- **Whether** the C2 integer factor constraint, mirror tying or the isotypic threat channel cost accuracy at scale.
- **Tight threat-feature capacities** for certificates on real nets.

The paper must state every speed or strength statement as a hypothesis tied to these experiments.

### 5.7 Ablation grid (for the eventual engine study)

- r ∈ {0, 8, 16, 32}
- m ∈ {0, 8, 16}
- degree ∈ {2, 3 (R ∈ {4, 8})}
- tying ∈ {tied, untied}
- C2 ∈ {off, merged r_K ∈ {8, 16}, factored |P| ∈ {4, 8}}
- C6 ∈ {off, E9a, E9b}

Each compared at equal measured nps.

---

## 6. Related work map

Status tags: [V] verified online (this session or the research notes), [K] standard reference whose details were not re-checked, [UNVERIFIED].

**NNUE and Stockfish (baseline and prior art)**
- Nasu, Y. (2018). *Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi*. WCSC28 appeal document. https://www.apply.computer-shogi.org/wcsc28/appeal/the_end_of_genesis_T.N.K.evolution_turbo_type_D/nnue.pdf ; English translation by D. Klein, https://github.com/asdfjkl/nnue [V]
- Official Stockfish. nnue-pytorch docs (`docs/nnue.md`): quantization, HalfKAv2_hm, virtual-feature factorization. https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md [V]
- Stockfish source, master (accessed 2026-11-09): `src/nnue/features/k32h3.h` [V this session], `nnue_accumulator.cpp` (update_accumulator_hybrid, forward/backward incremental) [V this session], `nnue_architecture.h`, `nnue_feature_transformer.h`, `nnue_common.h` [V notes], `simd.h` [V judge]. https://github.com/official-stockfish/Stockfish
- gab8192 (2024). Accumulator refresh table (Finny tables), PR #5183, commit 49ef4c9. [V]
- ces42 (2025). *Introduce double incremental accumulator updates*, Stockfish PR #5988, merged 2025-04-27 (commit fda269a). https://github.com/official-stockfish/Stockfish/pull/5988 [V this session]
- SFNNv10 threat inputs, commit 8e5392d [V]; nnue-pytorch PR #398 (Full_Threats v2) [V]; PR #563 (K32H3) [V]; Stockfish 18 and 19 release notes (QAT, PP_3Wide) [V].
- Viridithas author (2024). *NNUE performance improvements*. https://asteri.sm/files/2024-06-01-nnue [V content; author UNVERIFIED]
- jw1912 et al. *bullet* NNUE trainer. https://github.com/jw1912/bullet [V exists]
- Zobrist, A. L. (1970). A New Hashing Method with Application for Game Playing. Tech. Rep. 88, Univ. of Wisconsin. [K]
- Shannon, C. E. (1950). Programming a Computer for Playing Chess. Phil. Mag. 41(314). [K]

**Incremental computation (exact update calculus, C3/C4)**
- Paige, R., Koenig, S. (1982). Finite Differencing of Computable Expressions. ACM TOPLAS 4(3):402–454. [V this session; DOI 10.1145/357172.357177 UNVERIFIED]
- Cai, Y., Giarrusso, P. G., Rendel, T., Ostermann, K. (2014). A Theory of Changes for Higher-Order Languages. PLDI 2014, 145–155. DOI 10.1145/2594291.2594304 [V]
- Budiu, M., Chajed, T., McSherry, F., Ryzhyk, L., Tannen, V. (2023). DBSP: Automatic Incremental View Maintenance for Rich Query Languages. PVLDB 16(7):1601–1614. DOI 10.14778/3587136.3587137 [V]
- Acar, U. A., Blelloch, G. E., Harper, R. (2002). Adaptive Functional Programming. POPL 2002. [K]
- Neil, D., Lee, J. H., Delbruck, T., Liu, S.-C. (2017). Delta Networks for Optimized Recurrent Network Computation. ICML 2017, PMLR 70:2584–2593 [V]
- Parger, M. et al. (2022). DeltaCNN. CVPR 2022 [V]
- Pan, B. et al. (2018). Recurrent Residual Module for Fast Inference in Videos. CVPR 2018 [V]

**Interactions (PSI, C1/C1′)**
- Rendle, S. (2010). Factorization Machines. IEEE ICDM 2010, 995–1000. DOI 10.1109/ICDM.2010.127 [V this session]
- Blondel, M., Fujino, A., Ueda, N., Ishihata, M. (2016). Higher-Order Factorization Machines. NeurIPS 2016. [K]
- Newton–Girard identities: Mathlib `MvPolynomial.psum_eq_mul_esymm_sub_sum`, `Mathlib/RingTheory/MvPolynomial/Symmetric/NewtonIdentities.lean` [V locally]
- Hitchcock (1927); Carroll & Chang (1970); Harshman (1970); Kolda & Bader (2009), SIAM Review 51(3) (CP decomposition). [K]
- O'Donnell, R. (2014). *Analysis of Boolean Functions*. CUP. [K]

**Low-rank king conditioning (C2)**
- Sherman, J., Morrison, W. J. (1950). Ann. Math. Statist. 21(1):124–127. DOI 10.1214/aoms/1177729893 [V]
- Woodbury, M. A. (1950). Inverting Modified Matrices. Memo. Rep. 42, Princeton. [K]
- Hu, E. J. et al. (2021). LoRA. arXiv:2106.09685 [K] (analogy only)

**Symmetry (C6, T19–T22)**
- Cohen, T. S., Welling, M. (2016). Group Equivariant Convolutional Networks. ICML 2016, PMLR 48:2990–2999. [K]
- Ravanbakhsh, S., Schneider, J., Póczos, B. (2017). Equivariance Through Parameter-Sharing. ICML 2017. [UNVERIFIED pages]
- Zaheer, M. et al. (2017). Deep Sets. NeurIPS 2017. [K]
- Serre, J.-P. (1977). *Linear Representations of Finite Groups*. GTM 42. [K]

**Quantization and hardware (C5)**
- Jacob, B. et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018, 2704–2713. [K]
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM. [K]
- Williams, S., Waterman, A., Patterson, D. (2009). Roofline. CACM 52(4):65–76. [K]
- Intel 64 and IA-32 SDM / Intrinsics Guide (PMADDUBSW, PADDW, PADDSW). [K]

**Online algorithms (A1)**
- Karlin, A. R., Manasse, M. S., Rudolph, L., Sleator, D. D. (1988). Competitive Snoopy Caching. Algorithmica 3(1):79–119. [V this session; some sources give the first page as 77]
- Sleator, D. D., Tarjan, R. E. (1985). Amortized Efficiency of List Update and Paging Rules. CACM 28(2):202–208. [K]

**Formal methods**
- The mathlib Community (2020). The Lean Mathematical Library. CPP 2020. [K]
- de Moura, L., Ullrich, S. (2021). The Lean 4 Theorem Prover and Programming Language. CADE-28. [K]

**Contrast**
- Silver, D. et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science 362(6419):1140–1144. [K]; Leela Chess Zero https://lczero.org [K]
