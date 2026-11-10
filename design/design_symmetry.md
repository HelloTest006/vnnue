# VNNUE design, structure-first angle (VNNUE-S)

Status: design draft, 2026-11-09. Evidence tags: **[Lean]** means the claim is to be proved in Lean 4 + Mathlib (Section 7); **[Exp]** means it must be measured; **[V]** means it was verified this session against Stockfish master source or the research notes; **[Cite]** means it comes from a source in the reference list; **[UNVERIFIED]** means it has not been checked. No playing-strength claim appears anywhere in this document.

Inputs: `research/nnue.md`, `research/incremental_math.md`, `research/quant.md` and `research/efficient_nn.md`. `research/search.md` was never written (ENOSPC), so its ideas are used here only as they appear in the idea digest.

---

## 0. Thesis

NNUE already contains two symmetry tricks, though it does not describe them as group theory.

1. **Perspective sharing.** The two perspectives share one feature transformer W, and the board is colour-flipped for black (Nasu 2018; nnue-pytorch docs [Cite]).
2. **Horizontal mirroring by canonicalization.** HalfKAv2_hm and K32H3 orient the board so that the own king is on files e–h. Stockfish master computes this as `orient = OrientTBL[sq] ^ (56*c)`, and the threat features also use `OrientTBL[ksq] ^ (56*perspective)` [V, `k32h3.cpp`, `full_threats.cpp`].

Canonicalization has a cost. Whenever the frame changes (the own king moves, or crosses the d/e boundary), every feature is re-indexed and the accumulator must be refreshed. In current master, K32H3 refreshes on every own-king move: `requires_refresh` returns `diff.pc == make_piece(perspective, KING) || blockBefore != blockAfter` [V, `k32h3.cpp`]. Finny tables lower the cost of a refresh to the size of a diff against a cached board [V, `nnue_accumulator.h`, credited there to Luecx/Koivisto], but they do not remove the refresh.

The structure-first alternative is **equivariance instead of canonicalization** (Cohen & Welling 2016; Ravanbakhsh et al. 2017 [Cite]). A frame change g acts on the accumulator by a cheap linear map ρ(g), here a permutation of lanes or 16-lane blocks, instead of by re-indexing the inputs. If the weights satisfy W_{g·i} = ρ(g) W_i, then

  acc(g·x) = ρ(g) acc(x)        (Theorem T3)

and a frame change costs one permutation: an address remap or an in-register shuffle, with no column reads. VNNUE-S applies this to three groups:

- **Colour flip c** (exact symmetry of chess with the side to move swapped). One shared accumulator serves both perspectives.
- **File mirror h** (exact only without castling rights). The d/e crossing becomes free in the channels that use it.
- **King translations** t ∈ T = ℤ/4 × ℤ/8, acting regularly on the 32 mirror-canonical king squares. A king move becomes a translation of lane blocks.

This is combined with a **shared + specific factorization** of the first layer, generalizing the nnue-pytorch "virtual feature" factorizer [Cite]: a coarse-bucket king-specific table (expensive, refreshed rarely) plus a king-independent shared table (cheap, never refreshed) whose king dependence enters only through the group action. With the shared channel set to zero, VNNUE-S is exactly NNUE (T8).

## 1. Sets, groups, actions

**Squares.** Sq = Fin 64, with s = 8·rank + file. Two involutions act on squares:

- h(s) = s xor 7 (file mirror a↔h)
- v(s) = s xor 56 (rank flip 1↔8)

They commute and generate the Klein four-group K = {e, h, v, hv} ≅ ℤ/2 × ℤ/2. Every non-identity element is fixed-point-free on Sq, so the action is free, with 16 orbits of size 4. The ⟨h⟩-action has 32 orbits, which is why HalfKAv2_hm has 32 king buckets (T1).

**Pieces and features.** Pc = Col × Type, with Col = {W, B} and |Type| = 6. The absolute piece-square vector is β(pos) ∈ {0,1}^{768}, indexed by (piece, square), with at most 32 ones.

The **feature-level Klein group** K_F acts on the 768 indices:

- σ_h(p, s) = (p, h s) (mirror)
- σ_c(p, s) = (p̄, v s), where p̄ swaps the colour of p (colour flip)

For the position action, c·pos swaps colours, flips ranks and swaps the side to move. Then β(c·pos) = β(pos) ∘ σ_c⁻¹ and β(h·pos) = β(pos) ∘ σ_h⁻¹. K_F also acts freely, because σ_c changes colour and σ_h, σ_c σ_h move squares without fixed points. There are therefore exactly 768/4 = 192 orbits (T2).

**Perspective frame.** For perspective c ∈ {W, B}:

- g_c ∈ K_F is the frame transform: identity or σ_c for colour, composed with σ_h if c's (colour-normalized) king is on files a–d. This matches Stockfish's `OrientTBL[ksq] ^ 56c`.
- k̄_c ∈ Sq₃₂ is c's king square in its canonical frame (files e–h).
- t_c := ι(k̄_c) ∈ T = ℤ/4 × ℤ/8, where ι(file, rank) = (file − 4, rank).
- The king-relative features of NNUE are x_c(pos) ∈ {0,1}^{704} in the canonical frame: 11 piece planes × 64 squares, with the two kings merged into one plane as in HalfKAv2 [V].

**Lanes.** The shared channel has L_U lanes, organized as N_U = L_U/16 blocks of 16 int16 lanes (one AVX2 register). Write a block-lane index as (u, ℓ), with u ∈ T × Fin m (so N_U = 32m) and ℓ ∈ Fin 16 ≅ (ℤ/2)^4. Two families of lane permutations act on these indices:

- translation Γ_t(u, ℓ) = (u + t, ℓ) for t ∈ T, which is a block move (address arithmetic only);
- a frame representation ρ : K_F → Perm(Fin 16), applied inside every block, e.g. ρ(σ_h) = (ℓ ↦ ℓ xor 1) and ρ(σ_c) = (ℓ ↦ ℓ xor 2). This costs one `vpshufb`/`vpermw` per register.

Γ and ρ act on different coordinates, so they commute. Together they give a representation of T × K_F by permutation matrices on ℤ^{L_U}.

## 2. Components

### S1. Specific (king-bucketed) table: NNUE core with coarse buckets

a^B_c = b^B + W^{(β_blk, κ(k̄_c))} x_c ∈ ℤ^{L_B}.

Here κ : Sq₃₂ → Fin B is a king-region map, and β_blk ∈ {0,1,2} is the K32H3 material block [V: no rook/queen; rook or queen present but no opponent queen; opponent queen present]. The refresh trigger is **κ changes, or the mirror bit of g_c changes, or the block changes**. This is strictly weaker than master's "own king moved or block changed".

With B = 32, κ = id and L_B = L, S1 is exactly K32H3 (T8). The intended operating points are B ∈ {1, 2, 4, 8}, with regions that are unions of king squares. Moves inside a region then cost only the king's own column (2 column-ops).

### S2. Shared transported channel: one accumulator, both perspectives, all king squares

A(pos) = U β(pos) ∈ ℤ^{L_U}, with U ∈ ℤ^{L_U × 768}. A depends on neither king, perspective nor mirror, so **one copy per position** is kept and updated once per move. The view seen by perspective c is

  V_c(A) := Γ_{t_c} ∘ ρ(g_c) (A).

V_c is a lane permutation, so forming it does no arithmetic on lane values. The effective king-relative weight of feature i for king square k is Γ_{ι(k)} ρ(g) U_i. This is a structured 32-bucket table built from 768 columns.

**Optional symmetry tying** (the structural prior): U_{σ_g(i)} = ρ(g) U_i for g ∈ K_F. Under tying, U has only 192 free columns (T2), and

  ρ(g_c) A(pos) = A(g_c · pos)    (T3)

so V_c(A(pos)) = Γ_{t_c} A(canonical board of c). The channel then computes the same function on each perspective's canonical board, and the evaluation keeps NNUE's exact colour-flip symmetry (T6).

Without tying, the incremental-update correctness theorems still hold (T9); only the symmetry interpretation is lost. The h-tying is a prior and not a law of chess, because castling rights break the file mirror. Whether to tie σ_h is therefore experiment E3.

**King-relative interpretation.** If additionally U_{(p,s)} = Γ_{−ι(s)} u_p (convolutional tying with a per-piece kernel u_p, for s in the canonical half), then the block-u response is Σ_pieces u_p[u + t_c − ι(s)]. That is a function of the king-relative displacement s − k (modulo the torus). This recovers the classical "pieces near my king" pattern exactly as a group convolution over T, with 12·L_U parameters. In the general untied form, T only routes the 768 free columns. Torus aliasing (a displacement of +x versus x − 8 on the rank axis) is the price of the convolutional special case; in the free form it is irrelevant.

### S3. Assembly into the NNUE FT output

The pre-activation for perspective c is

  a_c = Π_B a^B_c + Π_U V_c(A) ∈ ℤ^{L},

where Π_B : ℤ^{L_B} → ℤ^L and Π_U : ℤ^{L_U} → ℤ^L are coordinate injections (0/1 matrices). The rest of the network is unchanged: FT pairwise product o_j = ⌊clamp(a_j,0,255)·clamp(a_{j+L/2},0,255)/512⌋ [V], the 32 layer stacks, and so on. There are two configurations:

- **concat** (cost-saving): Π_B and Π_U have disjoint images and L_B + L_U = L. The default is L_B = L_U = L/2, with the specific table on the u-half and the transported channel on the v-half. Each pairwise product u_j·v_j is then a learned king-bucket × king-transported bilinear interaction.
- **sum** (strict superset): Π_B = id with L_B = L, and Π_U embeds into some lanes, so the two channels are added before the clamp.

### S4. Threat and pawn-pair channels under the mirror

Stockfish orients threats by OrientTBL[ksq] [V], so a mirror change re-indexes them. Let H be the feature permutation induced by h on threat features.

- **Canonical frame (SF):** pre-activation W τ or W H τ. It costs L lane-ops per changed threat and needs a refresh on a mirror change.
- **Equivariant regular representation:** keep T' = W' τ with W' = [W ; W H] (2L lanes). Then W' H = P W', with P the half-swap. The canonical view is the first half of P^{[mirrored]} T' (T14). This removes the refresh, but doubles the per-threat cost.
- **Isotypic (integer) form of the same:** a_± = (W ± W H) τ, which gives a_+ ± a_- = 2Wτ or 2WHτ (T5). Truncating to n_+ invariant and n_- sign lanes with n_+ + n_- = L keeps the per-threat cost at L with no refresh. The function class is restricted.

Threats churn 3–4× more than piece features per move [V, nnue.md], while mirror changes happen only on a fraction of king moves. Expected cost therefore favours SF's canonical frame for threats. The cost model in Section 3 makes this precise. The isotypic variant is an ablation, not the default.

### S5. Finny cache under symmetry; base selection

Cache entries hold (board C, accumulator) per (block, region, colour). The bucket table is mirror-canonical (hm-style, P_h = I), so an entry for region κ also serves the mirrored king, via the mirrored board. The refresh base is chosen as

  argmin over candidates u ∈ {bias, Finny entries for (κ, both mirror frames), stack ancestors} of |S Δ C_u|.

Symmetric difference is invariant under the mirror bijection, and taking the min over a superset of candidates is never worse than master's single-entry policy (T12).

## 3. Update cost per move type

Units are int16 lane-ops (≈ bytes/2 of column traffic), counted for **both perspectives**, for piece-square features only; threats are unchanged and treated separately. k is the number of changed piece features (2, 3, 3, 2, 3, 4 for quiet, capture, e.p., promotion, capture-promotion, castling [V, nnue.md]). R = |S Δ C| is the Finny diff size, at most 32. Columns are compared at L = 1024 (master) against VNNUE-S concat with L_B = L_U = 512.

| Move | NNUE K32H3 (L=1024) | VNNUE-S concat (512/512) | VNNUE-S sum (L_B=1024, L_U=512) |
|---|---|---|---|
| quiet (k=2) | 2kL = 4096 | 2kL_B + kL_U = 3072 | 2kL + kL_U = 5120 |
| capture (k=3) | 6144 | 4608 | 7680 |
| own king, region and mirror unchanged | R·L + 2L (refresh, own side) | 2·2·512 + 2·512 = 3072 | 4·1024 + 1024 = 5120 |
| own king, region or mirror changes | R·L + 2L | R·512 + 2·512 + 2·512 | R·L + 2L + 2·512 |
| block change (material regime) | refresh both: (R_w+R_b)·L | (R_w+R_b)·512 + kL_U | (R_w+R_b)·L + kL_U |
| per evaluation (readout) | 0 | (L_U/16) shuffles per persp. (ρ); Γ = addressing | + L_U adds per persp. |

General formulas (T10):

- non-king move: C_S = k(2L_B + L_U), compared with C_N = 2kL. Hence **C_S ≤ C_N iff L_U ≤ 2(L − L_B)**, which always holds in concat.
- refresh cost scales with L_B instead of L.
- expected cost per move: E_S = E[k](2L_B + L_U) + p_K·q_B·E[R]·L_B + …, and E_N = 2E[k]L + p_K·E[R]·L + …, where p_K = P(own-king move) and q_B = P(region or mirror change | king move).

**Memory traffic locality.** U has 768 × 512 int16 = 768 KiB, or 384 KiB as int8 widened to int16 (as Stockfish already does for threats [V]). It is L2-resident on current desktop cores [UNVERIFIED per-CPU; Exp]. The specific table shrinks from 3 × 32 × 704 columns to 3 × B × 704 columns, a factor of 32/B.

## 4. Memory cost

| Item | NNUE K32H3, L=1024 | VNNUE-S concat, B=8, L_B=L_U=512 |
|---|---|---|
| FT specific table | 67,584 × 1024 × 2 B = 138.4 MB | 3·8·704 = 16,896 × 512 × 2 B = 17.3 MB |
| Shared table U | – | 768 × 512 × 2 B = 0.79 MB (tied: 192 cols = 0.20 MB) |
| Accumulator per ply | 2 × 1024 × 2 B = 4 KiB | 2 × 512 × 2 + 512 × 2 = 3 KiB |
| Finny entries | 3 × 64 × 2 × (2 KiB + board) | 3 × B' × 2 × (1 KiB + board), B' = cached (region, mirror) states |
| Dense layers | unchanged | unchanged |

The figure 67,584 = 3 × 22,528 is from master's K32H3 header [V]. Threat and pawn-pair tables are the same in both columns.

## 5. Strict generalization and what the structure buys

- **Generalization (T8).** In the sum configuration with U = 0, B = 32 and κ = id, VNNUE-S equals K32H3 + threats exactly, bit for bit. So VNNUE-S's best attainable loss is never worse than NNUE's, though training may not reach it.
- **Merge identity (T8).** Any VNNUE-S net is an NNUE net with a structured effective table E(β_blk, k, i) = Π_B W^{(β_blk, κ(k))}_i + Π_U Γ_{ι(k)} ρ(g) U_{abs(i)}. The factorization changes cost and parametrization, not the representable functions at a given effective table. This parallels the virtual-feature merge in nnue-pytorch [Cite]. The difference is that VNNUE-S does **not** merge at export: it keeps the factors at inference, because the factored form is the cheap form to update.
- **NNUE perspective sharing is a special case of S2's representation theory (T4).** Stacking the two perspective accumulators as one 2L vector, colour flip acts by the half-swap: the regular representation of ℤ/2. Over ℚ this splits into L invariant plus L sign lanes, the isotypic decomposition. Over ℤ the change of basis has determinant ±2^L, so it is an injective but non-unimodular embedding (T5), which is why the integer form keeps factor-2 scales.

## 6. Training notes [Exp]

- Train S2 tied and untied, initializing U from the nnue-pytorch "P" virtual factor (king-independent piece-square weights) [Cite].
- Train κ as a fixed partition, but sweep B ∈ {1, 2, 4, 8, 32}.
- Use QAT [V: adopted in SF19], with the overflow certificate (T11) as a training clamp: |b| + 32 M_B < 2^15 on specific lanes and |b| + 32 M_U < 2^15 on transported lanes. In sum mode this becomes |b| + 32(M_B + M_U) < 2^15.

## 7. Theorems for Lean 4 + Mathlib

All theorems are finite and stated over ℤ or ℚ, using `Fin n → ℤ`, `Finset.sum`, `Equiv.Perm` and `ZMod`.

- **T1 (Klein action on squares).** Let h s := s xor 7 and v s := s xor 56 on Fin 64. Then h∘h = id, v∘v = id, h∘v = v∘h, and for each g ∈ {h, v, h∘v} and every s, g s ≠ s. Also #{s | file s ≥ 4} = 32, and h maps the a–d half bijectively onto the e–h half. Proved by `decide`. This justifies 32 canonical king squares.
- **T2 (free action on the 768 features).** With σ_h, σ_c as in Section 1 on Fin 2 × Fin 6 × Fin 64, every non-identity element of the generated group has no fixed point. Hence the orbit count is 768/4 = 192, and the tied U has exactly 192 free columns. Proved by `decide` or `Fintype.card` of the quotient.
- **T3 (equivariance of the accumulator, core lemma).** Let W : Fin n → Fin L → ℤ, σ ∈ Perm(Fin n), π ∈ Perm(Fin L), and suppose ∀ i j, W (σ i) (π j) = W i j. Define acc x j := Σ_i x i * W i j. Then ∀ x j, acc (x ∘ σ.symm) (π j) = acc x j. Proof: `Equiv.sum_comp`.
- **T4 (perspective sharing = regular representation).** For W₁ : Fin n → Fin L → ℤ and an involution σ, the stacked W' : Fin n → (Fin L ⊕ Fin L) → ℤ, defined by W' i (inl j) = W₁ i j and W' i (inr j) = W₁ (σ i) j, satisfies the T3 tying with π = `Equiv.sumComm`. Moreover acc W' x = Sum.elim (acc W₁ x) (acc W₁ (x ∘ σ)).
- **T5 (integer isotypic decomposition).** For u v : Fin L → ℤ, let s = u + v and d = u − v. Then s + d = 2u, s − d = 2v, and the swap acts by (s, d) ↦ (s, −d). For matrices, if a_± = (W ± W∘H)τ then a_+ + a_- = 2·Wτ and a_+ − a_- = 2·(W∘H)τ.
- **T6 (one accumulator serves both perspectives).** Assume the tying U (σ_c i) = ρ_c (U i), with ρ_c an involutive lane permutation, and assume t_W(c·pos) = t_B(pos). Then V_W(A(c·pos)) = V_B(A(pos)). Consequently the eval of the colour-flipped position with the side to move swapped equals the original, which is the colour symmetry NNUE already has.
- **T7 (king transport is a regular group action).** Γ : T → Perm(T × Fin m × Fin 16), with Γ_t(u, r, ℓ) = (u + t, r, ℓ), is a monoid hom (Γ₀ = 1, Γ_{t+t'} = Γ_t Γ_{t'}), acts freely and transitively on T, and commutes with every within-block ρ. Hence for a king move t → t', V' = Γ_{t'−t} V. Over T = ZMod 4 × ZMod 8.
- **T8 (merge identity and strict generalization).** For all x, the factored pre-activation Π_B (W^{κ} x) + Π_U (Γ_t ρ_g (U β)) equals Σ_i x_i · E(k, i) for the effective table E defined above. If U = 0, B = 32, κ = id and Π_B = id, then it equals the NNUE accumulator b + W^{k} x.
- **T9 (path independence of the two-channel state, modulo 2^16).** For any list of feature deltas δ₁..δ_n (as `Fin N → ℤ`), folding acc ↦ acc + W δ_t on the specific channel and A ↦ A + U δ'_t on the shared channel, with a refresh at each frame change, gives (W x_final, U β_final). This holds over ℤ and after the cast to ZMod (2^16), because the cast is a ring hom. It also holds within a region: if κ k = κ k' and the mirror bit is unchanged, the specific accumulator needs only the king's own column delta.
- **T10 (op-count bounds, ℕ and ℚ).**
  - (i) k(2L_B + L_U) ≤ 2kL ↔ L_U ≤ 2(L − L_B) for k > 0.
  - (ii) R·L_B ≤ R·L when L_B ≤ L.
  - (iii) Break-even over ℚ: E_S ≤ E_N ↔ E[k](2L_B + L_U) + p_K q_B E[R] L_B ≤ 2E[k]L + p_K E[R] L.
- **T11 (overflow certificate + wrap-around exactness).** If |b_j| ≤ B₀, |W_{ij}| ≤ M_B, |U_{ij}| ≤ M_U and the active sets have card ≤ 32, then |a_j| ≤ B₀ + 32(M_B + M_U). If this bound is < 2^15, then for any path of wrapping int16 updates the `ZMod (2^16)` value lifted by `valMinAbs` equals the true integer value.
- **T12 (base-selection metric).** For an injective f, card(image f S Δ image f C) = card(S Δ C). For candidate sets X ⊆ Y, min over Y of card(S Δ C_u) ≤ min over X. So adding the mirrored Finny entry and the stack ancestors never increases refresh cost.
- **T13 (permutation commutes with elementwise activation and reindexes dense layers).** For any φ : ℤ → ℤ (clamp, pairwise-product halves) and π ∈ Perm(Fin L): (φ ∘ (a ∘ π)) = (φ ∘ a) ∘ π. Also Σ_j M i j · a (π j) = Σ_j M i (π⁻¹ j) · a j. This justifies applying views at readout time or folding them into fc_0 column order.
- **T14 (canonical frame = regular representation at 2L).** For W : Fin n → Fin L → ℤ and an involution H of features, let W' := [W ; W∘H] and P := half-swap. Then W'∘H = P∘W', and for m ∈ {0, 1}, the first half of P^m (W' τ) = W (H^m τ). In words, Stockfish's mirror canonicalization is the regular-representation special case of an equivariant channel.

## 8. What cannot be proved, and what must be measured

Lean proves correctness (exactness, path independence, equivariance, overflow freedom) and op-count relations. It cannot prove that VNNUE-S plays better. In particular:

- **E1.** Validation loss and fixed-node Elo of VNNUE-S (B ∈ {1, 2, 4, 8}, concat and sum) against K32H3 at equal training data, with an SPRT reported in normalized Elo.
- **E2.** NPS and the per-move lane-op and byte counters. Measure p_K, q_B and the distribution of E[R] on real search trees, and plug them into T10(iii).
- **E3.** The effect of the symmetry priors (σ_h-tying, σ_c-tying, convolutional tying) on loss. The h-tying is known to be inexact under castling rights.
- **E4.** Whether U is L2-resident in practice (cache counters).
- **E5.** Threat-channel ablation (canonical vs isotypic) against the cost model in Section 2, S4.

Every inequality in T10 is conditional on measured statistics. The design's claim is therefore of the form "if q_B ≪ 1 and the loss gap is small, VNNUE-S is cheaper per move at equal quality", and only E1–E2 can settle it.

## 9. References

- Nasu, Y. (2018). *Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi*. WCSC28 appeal document (Japanese). [Cite, verified in research/nnue.md]
- Stockfish developers. `src/nnue/features/k32h3.{h,cpp}`, `full_threats.{h,cpp}`, `nnue_accumulator.h`, master, accessed 2026-11-09. https://github.com/official-stockfish/Stockfish [V this session]
- nnue-pytorch documentation, *NNUE* (feature factorization / virtual features, HalfKAv2_hm). https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md [Cite, verified in notes]
- Cohen, T. S., Welling, M. (2016). Group Equivariant Convolutional Networks. ICML 2016, PMLR 48:2990–2999. [Cite]
- Ravanbakhsh, S., Schneider, J., Póczos, B. (2017). Equivariance Through Parameter-Sharing. ICML 2017, PMLR 70. [UNVERIFIED page numbers]
- Kondor, R., Trivedi, S. (2018). On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups. ICML 2018. [UNVERIFIED page numbers]
- Zaheer, M. et al. (2017). Deep Sets. NeurIPS 2017. [Cite, notes]
- Serre, J.-P. (1977). *Linear Representations of Finite Groups*. GTM 42, Springer. (Maschke and isotypic decomposition.) [UNVERIFIED edition details]
- Zobrist, A. L. (1970). A New Hashing Method with Application for Game Playing. Tech. Rep. 88, Univ. of Wisconsin. [UNVERIFIED]
- gab8192 (2024). Finny tables, Stockfish PR #5183. [V in notes]
