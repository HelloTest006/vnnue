# NNUE and Chess-Engine Evaluation: Research Notes for VNNUE

Topic owner: NNUE history, architecture, incremental-update mechanics, bottlenecks.
Status legend: **[V]** = checked this session against the cited primary or secondary source; **[UNVERIFIED]** = from memory or derived and not yet checked against a source; **[DERIVED]** = arithmetic or algebra done by us from verified definitions. These should be proved in Lean or checked by experiment before the paper relies on them.

Snapshot date: 2026-11-09. Stockfish moves fast, so architecture constants are tied to a release or commit.

---

## 1. Lineage of incremental evaluation (pre-NNUE)

1. **Linear evaluation and piece-square tables (PSTs).** Shannon (1950) proposed a material-plus-positional linear evaluation. A PST evaluation `E(pos) = sum_{(piece,sq) in pos} T[piece][sq]` is linear in the 768-dimensional one-hot "A" feature vector (12 piece types x 64 squares). Engines have long updated it incrementally on make/unmake: `E <- E - T[p][from] + T[p][to]`. NNUE is exactly this idea with a vector-valued table: L1 numbers per (piece, square, king-square) instead of one. [UNVERIFIED as a historical claim; the algebra is standard]
2. **Zobrist hashing (1970)** is the hashing analogue. The key `h(pos) = XOR of R[f]` over active features f is a group homomorphism from (feature sets, symmetric difference) to (Z/2)^64, so it updates incrementally by XOR. The NNUE accumulator is the same construction with the target group (Z/2^16)^L1 under addition. **This shared structure is the central mathematical observation for VNNUE (Section 8).**
3. **Samuel (1959)** learned linear evaluation weights for checkers. **Tesauro (TD-Gammon, 1995)** used a shallow NN evaluator. **Giraffe (Lai, 2015)** used a learned NN evaluation in chess but was far slower than handcrafted evaluation. [UNVERIFIED details; the citations themselves are well known]
4. **Deep Blue (Campbell, Hoane, Hsu 2002)** computed evaluation in custom VLSI chess chips. It had a fast evaluation and a slow evaluation, with on the order of 8,000 feature patterns whose weights were set partly by hand and partly by tuning. [UNVERIFIED: the figure of about 8000 should be checked against the AIJ paper]
5. **Texel tuning (Peter Osterlund, 2014).** This fits handcrafted evaluation weights by minimizing `sum_i (R_i - sigmoid(K * E(pos_i)))^2` over game results R_i in {0, 0.5, 1}. It is the ancestor of the WDL-space loss used in NNUE training. [UNVERIFIED exact form; CPW page "Texel's Tuning Method"]

## 2. Origin: Nasu 2018 (shogi)

- Yu Nasu, "NNUE: Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi", April 2018. It appeared as an appeal document for the 28th World Computer Shogi Championship (Ziosoft Computer Shogi Club), in Japanese with an English abstract. Dominik Klein's English translation is at github.com/asdfjkl/nnue. [V]
- Main idea: a shallow network whose very large, sparse, binary input layer feeds a first affine layer. That first layer's output (the **accumulator**) is updated incrementally on make/unmake rather than recomputed, so the expensive layer costs O(#changed features x width) per move. [V]
- First deployed in YaneuraOu-family shogi engines and then ported to chess as "Stockfish NNUE" by nodchip (2019). [V, CPW]

## 3. Stockfish adoption timeline (verified milestones)

| Date | Event | Architecture |
|---|---|---|
| 2019 | nodchip ports NNUE to Stockfish 10 (proof of concept) [V, CPW] | HalfKP |
| Aug 2020 | NNUE merged to master. Fishtest measured it at least 80 Elo above classical. Hybrid eval used classical evaluation in imbalanced positions. [V, CPW] | HalfKP-41024-256x2-32-32-1 |
| 2 Sep 2020 | **Stockfish 12**, first NNUE release [V, CPW] | same |
| Nov 2020 | Gary Linscott starts nnue-pytorch [V, CPW] | |
| 2021 | Sobczyk et al.: HalfKA, then HalfKAv2. PSQT side outputs and 8 layer stacks. **SF14**. [V, CPW] | HalfKAv2 45056 -> 2x520 (incl. PSQT) -> 8 x (512x2->16->32->1) |
| Aug 2021 | HalfKAv2_hm (horizontal mirroring) in use [V, nnue.md] | 22528 inputs [DERIVED] |
| 2023 | **SF16** removes handcrafted eval from the default path [V, CPW] | |
| 7 Jan 2024 | Dual NNUE: L1-128 small net trained on abs(simple_eval) > 1000 positions (PR #4915, linrock; code by mstembera, #4898) [V] | big + small |
| Feb 2024 | **SF16.1**: architecture "SFNNv8", handcrafted eval fully removed, secondary net added [V, blog] | big L1 = 2560 [UNVERIFIED] |
| 24 Apr 2024 | **Finny tables** (accumulator refresh cache), commit 49ef4c9, PR #5183, gab8192 [V] | |
| 2024 | SFNNv9, L1 = 3072 (PR #5149, Viren6) [V]; shipped in SF17 [UNVERIFIED] | 3072 |
| ~Nov 2025 | **SFNNv10: Full Threat inputs** added on top of HalfKAv2_hm (commit 8e5392d, nn-49c1193b131c). ARM LTC +13.9 +/- 1.9 Elo. Credits prior threat-input work in Monty, Yukari, PlentyChess. [V] | HalfKAv2_hm + Full_Threats |
| 31 Jan 2026 | **SF18**: SFNNv10, up to +46 Elo vs SF17 [V, blog] | |
| Feb 2026 | nnue-pytorch PR #398: compressed threat indexing ("Full_Threatsv2"), `--l1=1024` [V] | |
| 5 Sep 2026 | **SF19**: SFNNv16. Pawn-pair features (PP_3Wide), redundant threat features removed, **small net retired**, QAT used, up to +44 Elo vs SF18 [V] | |
| 2026 | SFNNv18: K32Q2 (32 own-king buckets x opponent-queen present/absent) [V] | |
| ~Oct 2026 | SFNNv19: **K32H3** replaces K32Q2/HalfKAv2_hm, adding a third block for "no heavy pieces left"; block-major layout (nnue-pytorch PR #563) [V] | |
| Master, Nov 2026 | `L1 = 1024, L2 = 32, L3 = 32, LayerStacks = 32`. Feature sets: `FullThreats`, `PP_3Wide`, `K32H3` [V, nnue_architecture.h] | see Section 6 |

Takeaway: the trend reversed from **wider accumulators** (256 to 520 to 1024 to 2560 to 3072) to **narrower accumulators (1024) with richer inputs** (threats, pawn pairs, material-conditioned buckets). Input expressiveness was worth more than width once memory bandwidth bound the width.

## 4. Feature sets: exact definitions and counts

Notation: perspective c in {white, black}; ksq = that perspective's king square; for each perspective the board is flipped so "our" pieces are "stm" pieces.

**A (768).** 64 sq x 6 types x 2 colors = 768. At most 32 active. A move changes at most 4 features (castling), fewer than 3 on average. [V, nnue.md]

**HalfKP (40960 + 64 = 41024 per perspective).** Feature = (ksq, psq, ptype, pcolor) for non-king pieces: 64 x 64 x 5 x 2 = 40960, plus 64 unused shogi leftovers (BONA_PIECE_ZERO). Index: `p_idx = 2*ptype + pcolor; idx = psq + (p_idx + 10*ksq)*64`. At most 30 active per perspective. The original color flip was a 180-degree rotation (sq XOR 63), another shogi leftover. First-layer parameters: 41024 x 256 = 10,502,144 per perspective (shared weights), about 20 MB quantized. [V, nnue.md and CPW]

**HalfKA.** Same as HalfKP but kings are included as pieces: 64 x 64 x 12 = 49152. The CPW page lists 45056 for HalfKA, which is an arithmetic slip; 45056 is HalfKAv2. [V/DERIVED]

**HalfKAv2 (45056).** Our king and the enemy king can share one piece bucket, because for a fixed king square their square sets do not overlap. That gives 11 piece buckets: 64 x 11 x 64 = 45056 ("8% less space"). [V]

**HalfKAv2_hm (22528).** Each perspective mirrors horizontally so its own king is on files e-h. This halves the king squares to 32: 32 x 704 = 22528. In this set, **every own-king move changes ksq and therefore forces a refresh of that perspective**. [V definition; DERIVED count, matches Stockfish's `SQUARE_NB * PS_NB / 2`]

**Full_Threats (SFNNv10+).** An add-on, not a replacement. Active features are ordered piece pairs (attacker on s1, target on s2) where s2 is in attack(s1), encoded relative to the perspective ("stm queen d1 attacks nstm knight h5"). Deduplication [V, nnue.md]:
- drop pairs where the attacker's attack set is a subset of the target's: pb, pq, pk, bq, rq, kq;
- drop king-king pairs;
- for same-type pairs, drop the orientation with the lower attacker square.

Threats were mirrored a-d (a legacy choice) until K32H3 aligned them with HalfKA. They have a **separate accumulator**, refreshed only when the king crosses the d/e file boundary. Total feature count in SFNNv13 (all input sets) is **82,672** [DeepWiki, secondary; UNVERIFIED against source].

**PP_3Wide (SFNNv16+).** Unordered pairs of pawns on the same or adjacent files. It generalizes pawn-pawn attacks, which are removed from Full_Threats when it is active. Index: `id = 48*color + sq - 8`, pair index `hi*(hi-1)/2 + lo`. Stored as i8. Pairs more than one file apart are never active. [V]

**K32Q2 / K32H3.** Weight blocks selected by own-king bucket (32) x material state: opponent queen present/absent for Q2, plus "no queens or rooks" for H3. The H3 states are inferred from the PR text [UNVERIFIED exact semantics]. The input becomes piecewise-linear in material regime, and a refresh is needed whenever the block changes, for example on the capture of the last queen. [V for existence]

## 5. The accumulator: exact identities

Let x in {0,1}^N be the active-feature indicator for one perspective, W in Z^{L1 x N}, and b in Z^{L1}.

- **Definition:** `a(x) = b + W x = b + sum_{i : x_i = 1} W[:, i]`.
- **Incremental update:** for a move with added set A and removed set R (A and R disjoint, R a subset of supp(x), A disjoint from supp(x)):
  `a(x') = a(x) + sum_{i in A} W[:,i] - sum_{i in R} W[:,i]`. [V, nnue.md]
- **Refresh:** `a(x) = b + sum_{i in supp x} W[:,i]`. [V]
- **Path independence (the correctness invariant):** for any make/unmake sequence, the incrementally maintained accumulator equals the refresh of the current position, because x -> W x is a homomorphism (Z^N, +) -> (Z^{L1}, +). [DERIVED; Lean target]
- **Wrap-around safety:** Stockfish adds int16 accumulators with non-saturating SIMD adds (`vec_add_16`). Addition in Z/2^16 is still a homomorphism, so intermediate overflow is harmless whenever the true final value lies in [-32768, 32767]. Only the final value has to fit. [DERIVED; Lean target; that SF uses wrapping adds is UNVERIFIED in current source]
- **Two perspectives:** separate accumulators. The transformed output concatenates [stm half, nstm half], which lets the net learn tempo. [V]
- **Float drift avoidance:** accumulators live on a search stack (copy-on-make) and are not "un-updated", so error is bounded by O(depth). In integer arithmetic there is no drift at all. [V]

### Update cost per move (per perspective, piece features only)

| Move type | Removed columns | Added columns | Total column ops |
|---|---|---|---|
| Quiet (non-king) | 1 | 1 | 2 |
| Capture | 2 | 1 | 3 |
| En passant | 2 | 1 | 3 [DERIVED] |
| Promotion | 1 (pawn) | 1 (new piece) | 2 |
| Capture-promotion | 2 | 1 | 3 [DERIVED] |
| Castling | 2 | 2 | 4 |
| Own-king move (HalfKA_hm) | refresh: up to 32 columns, or, with Finny cache, the symmetric difference to the cached board | | |

Source: [V, asteri.sm "NNUE performance improvements", 2024]. Both perspectives must be updated, so a quiet move costs 4 column operations in total.

**Memory traffic model [DERIVED]:** one int16 column of width L1 is 2*L1 bytes (2 KB at L1=1024, 6 KB at L1=3072). A fused update (`add1_sub1`, `add2_sub1`, ...) reads k columns, reads the previous accumulator once and writes the new one once: about (k + 2) * 2*L1 bytes per perspective. A 22528 x 3072 x 2 B feature transformer is about 138 MB, far bigger than L2 cache, so column fetches are DRAM/L3-bound. This is why width stopped growing and why threat weights moved to i8.

**Threat-feature churn:** in a typical middlegame, **3-4x as many threat features change per move as piece features** [V, nnue.md]. One quiet move changes the moved piece's outgoing attacks and the attacks of sliders it blocks or unblocks. This is why threat weights are stored as **i8 and widened to i16** during accumulation (about +10% speed on ARM, +5% on x86, with negligible loss from weight clipping). [V]

## 6. Rest of the network (current Stockfish)

- **Feature transformer (FT) output with pairwise multiplication** [V, nnue_feature_transformer.h]: each perspective's L1-wide accumulator is split into halves u = a[0:L1/2], v = a[L1/2:L1]:
  `out_j = (clamp(u_j, 0, 255) * clamp(v_j, 0, 255)) / 512`, giving L1/2 uint8 values per perspective and L1 in total.
  On SIMD this is computed as `mulhi16(clamp(u) << 7, min(v, 255))`: (<<7, >>16) gives a net >>9, and `packus` zeroes negative products, which saves a `max`. **Mathematically this is a rank-structured quadratic form in the binary inputs:** `out_j = (b_u + W_u x)_j * (b_v + W_v x)_j` when nothing is clipped. That is a degree-2 polynomial whose pairwise coefficient matrix for x_i x_k is `W_u[j,i]*W_v[j,k]` (rank 1 per output). [DERIVED]
- **"Product pooling"** in nnue.md: multiply the two halves and divide by 128, giving range [0,126]. [V]
- **Hidden stack** (per layer-stack bucket) [V, nnue_architecture.h]: `fc_0: 1024 -> 32` (sparse affine). Its output goes through both SqrClippedReLU and ClippedReLU, concatenated to 64. Then `fc_1: 64 -> 32`, again both activations. `fc_2` reads all four activation blocks and outputs 1. A skip term `fc_0_out[30] - fc_0_out[31]` is added, and the result is rescaled by 600*OutputScale.
- **Layer stacks / output buckets:** historically 8, selected by `(piece_count - 1) / 4` [V, nnue.md]; master now has `LayerStacks = 32` (selection rule not in that header; UNVERIFIED).
- **PSQT side output:** 8 buckets x 2 FT outputs routed directly to the final output, so the net can learn material imbalance linearly [V, CPW]. This is not visible in the current transform code, which may have changed in SFNNv18/19 [UNVERIFIED].
- **Activations:** ClippedReLU `min(max(x,0),1)`. SCReLU (squared clipped ReLU) `clamp(x,0,1)^2`. With quantization factor Q = 255, v^2 * w overflows i16, so the "Lizard trick" computes (v*w) in i16 (since 255*126 = 32130 < 32767) and then multiplies by v while widening to i32. [V, asteri.sm]
- **Sparsity:** after the FT, the input to the first dense layer is at most 15% dense for common sizes. Stockfish's sparse affine layer skips zero 4-byte blocks, and reordering FT neurons to group likely-nonzeros gives about 2%. [V, nnue.md]

## 7. Quantization (Stockfish scheme)

[V, nnue.md unless noted]
- FT weights and biases are int16, originally scale 127 (CReLU 0..127 to int8). With pairwise multiplication the clamp is 0..255 (`FtMaxVal == 255`, uint8 output). [V, source]
- Hidden weights are int8 with scale s_W = 64: maximum magnitude 127/64 = 1.984375, step 1/64 = 0.015625. Biases are int32 at scale s_A*s_W. Accumulation is in int32 and the >>6 is a shift. Output-layer weight clamp is +/-127*127/9600.
- Original HalfKP: final output divided by FV_SCALE = 16. [V, CPW]
- Training loss is in WDL space with `sigmoid(cp/410)`, eval and result are blended by lambda, and the MSE exponent was 2.6 for SF. [V]
- SF19 introduced **quantization-aware training (QAT)**. [V, release notes]
- **Saturation bound [DERIVED, Lean target]:** if every FT weight satisfies |W_ij| <= w_max, |b_j| <= b_max and at most n features are active, then |a_j| <= b_max + n*w_max. With n = 32 and i8 threat weights (w_max = 127), the threat-part magnitude is at most 4064. Threat features, however, can number far more than 32 active (each piece attacks up to 27 squares), so n must be bounded by the maximum number of attack pairs. Saturation is a real risk that is currently handled by training-time clipping.

## 8. Bottlenecks and open problems

1. **Memory bandwidth.** Each update streams k columns of 2*L1 bytes. Width growth stalled at 3072 and SF went back to 1024 plus richer inputs. Threat churn (3-4x) made bandwidth the main concern, which led to i8 storage. [V]
2. **Refresh cost on king moves.** With per-square king buckets, every own-king move refreshes one perspective. Finny tables reduce the cost to |S_cache XOR S_now|, which is typically small because the rest of the board changes slowly. The cache is per thread, per bucket, per color, and is cleared like histories rather than at every search. [V] Material-conditioned blocks (K32Q2/H3) add refresh triggers on queen or rook captures. Threats use a separate accumulator so they refresh only on d/e-file crossings. [V]
3. **Linear first layer cannot represent piece-pair interactions.** a(x) is linear in x, so any interaction must be built downstream. Two fixes have appeared: (a) pairwise multiplication, an implicit rank-1-per-neuron quadratic, and (b) explicit pair features (Full_Threats, PP_3Wide), a sparse hand-chosen subset of the C(768,2) pair space whose activity depends on attack geometry. Each gained Elo, which is evidence that cheap second-order structure is the frontier. [V Elo; interpretation ours]
4. **Quantization saturation and precision.** int16 accumulators, int8 hidden weights limited to |w| < 2, and clip ranges must be enforced in training. QAT has only recently been adopted. [V]
5. **Lazy-update path length.** Lazy updates walk back to the last computed accumulator and replay deltas forward. [V, asteri.sm] Deltas along the path are not cancelled: a feature added at ply t and removed at ply t+2 is applied twice. [Observation; UNVERIFIED whether any engine cancels]
6. **Bucket boundaries are hand-designed** (king square, d/e mirror, piece-count output buckets, queen/heavy presence). There is no principled objective trading refresh probability against expressiveness.

## 9. Contrast: Leela Chess Zero

AlphaZero-style (Silver et al., Science 2018): a deep residual, and now transformer, network with policy and value heads, evaluated on GPU with no incremental update. Throughput is orders of magnitude lower than NNUE. Lc0 searches with PUCT MCTS instead of alpha-beta. Lc0 evaluation data now trains Stockfish nets: 100+ billion positions for SF18, and hundreds of billions rescored with a strong Leela net for SF19. [V, release notes] So the two lineages are complementary: a deep, slow teacher distilled into a shallow, incrementally updatable student.

## 10. Implications for VNNUE: concrete design ideas

Each idea comes with the formal statement we would prove in Lean 4/Mathlib and the experiment that would test it.

1. **Formalize the update algebra as a group homomorphism.** Define features as `Finset ι` (or `ι ->0 Z`) and `acc : (ι ->0 Z) ->+ (Fin L -> Z)`. Lean theorems: (i) `acc (x + d) = acc x + acc d`; (ii) path independence: folding any list of deltas equals refresh; (iii) for indicator sets, `delta(S, S') = 1_{S' \ S} - 1_{S \ S'}`, and composing deltas along a path telescopes. This is the rigorous core that NNUE practice relies on but has never stated formally.
2. **Modular (wrap-around) arithmetic theorem.** Prove that `ZMod (2^16)` accumulation, followed by a final cast, equals exact integer accumulation whenever the true value lies in [-2^15, 2^15). This justifies non-saturating SIMD adds and allows reordering or fusing any number of updates. It is a Mathlib-level `ZMod.intCast` lemma.
3. **Delta cancellation for lazy updates ("net-delta replay").** Instead of replaying each ply's dirty pieces, accumulate a signed multiset of deltas and cancel opposite terms before touching memory. Theorem: the replayed result is invariant under cancellation, and the number of column operations is the support size of the net delta, which is at most the sum of per-ply sizes. Experiment: count column loads saved in real search trees.
4. **Optimal refresh policy (generalized Finny).** Given cached feature set C and current S, the cost is min(|S|, |C XOR S|) column operations. Lean: a cache-based update is correct for any cache content. Optionally, prove that keeping k caches (one per bucket) and picking the argmin of |C_k XOR S| is optimal under the column-operation cost model. Experiment: measure the distribution of |C XOR S| on refreshes.
5. **Factored second-order accumulator (the main "very easily updatable" idea).** Pairwise multiplication gives `out_j = u_j(x) * v_j(x)`, a rank-1 quadratic per neuron that is free to update because u and v are linear. VNNUE could make this explicit and principled: choose the interaction model `x^T Q_j x` with Q_j of low rank r, maintaining r pairs of linear accumulators. Update cost stays O(k * r * L) per move. Pair features (threats) need O(#changed pairs * L), and that is 3-4x the piece churn. Lean: prove the identity `(b_u + W_u x)(b_v + W_v x) = b_u b_v + sum_i (b_u W_v[i] + b_v W_u[i]) x_i + sum_{i,k} W_u[i] W_v[k] x_i x_k` and that incremental maintenance of u and v gives the quadratic exactly. Experiment: compare a low-rank-quadratic FT with explicit threat features at equal bandwidth.
6. **Hybrid: sparse explicit pairs plus low-rank residual.** Keep explicit features only for pairs whose interaction is geometric (attacks, pawn structure), which low rank cannot capture cheaply, and model the rest with the factored quadratic. The design question is the bandwidth budget split.
7. **Provable no-overflow quantization bounds.** Lean theorem: `|b_j| + n_max * w_max < 2^15` implies the int16 accumulator never needs saturation. Derive the tight n_max for each feature set (maximum simultaneous attack pairs for Full_Threats). Training then enforces the clip that the theorem requires, which replaces empirical clip choices with certified ones.
8. **Bucket design as an optimization problem.** Model the expected per-move cost as `E[k] + P(bucket change) * E[refresh cost]` and expressiveness as the number of blocks. Choose boundaries (king zones, material regimes) to minimize the cost at fixed accuracy. Measurable directly on self-play move statistics.
9. **Precision split by churn.** Store high-churn feature columns (threats, pairs) as i8 and low-churn ones as i16, generalizing SF's threat-only i8. Lean: an error bound on accumulated quantization error, with at most n*Delta/2 per neuron for rounding step Delta.
10. **Sparse-output bookkeeping.** Since FT output density is at most 15%, maintain the nonzero index set incrementally alongside the accumulator. Proof obligation: the set of j with u_j > 0 and v_j > 0 changes only at coordinates touched by the update. This only holds coordinate-wise if columns are sparse, so it is an open question and should be framed as an experiment.

**Honesty constraints for the paper:** do not claim Elo for VNNUE without our own experiments. Stockfish Elo numbers above are per-patch fishtest results or release-level claims and must be quoted as such.

---

## References

Format: Author(s) (Year). *Title*. Venue/URL. [status]

1. Nasu, Y. (2018). *NNUE: Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi*. 28th World Computer Shogi Championship appeal document, Ziosoft Computer Shogi Club (Japanese). https://www.apply.computer-shogi.org/wcsc28/appeal/the_end_of_genesis_T.N.K.evolution_turbo_type_D/nnue.pdf ; English translation by D. Klein: https://github.com/asdfjkl/nnue [V]
2. Official Stockfish / nnue-pytorch contributors (Sobczyk, T. et al.). *NNUE* (docs/nnue.md). https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md ; mirrored at https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html [V]
3. Stockfish developers. *src/nnue/nnue_architecture.h* and *nnue_feature_transformer.h*, master branch (accessed 2026-11-09). https://github.com/official-stockfish/Stockfish/tree/master/src/nnue [V]
4. Stockfish developers (2025). *Update NNUE architecture to SFNNv10 with Threat Inputs and net nn-49c1193b131c.nnue*, commit 8e5392d. https://github.com/official-stockfish/Stockfish/commit/8e5392d79a36aba5b997cf6fb590937e3e624e80 [V]
5. gab8192 (2024). *Implement accumulator refresh table* (Finny tables), commit 49ef4c9 / PR #5183. https://github.com/official-stockfish/Stockfish/commit/49ef4c935a5cb0e4d94096e6354caa06b36b3e3c [V]
6. Xu, L. (linrock) (2024). *Dual NNUE with L1-128 smallnet*, PR #4915. https://github.com/official-stockfish/Stockfish/pull/4915 [V]
7. Viren6 (2024). *Update NNUE architecture to SFNNv9 with larger L1 size of 3072*, PR #5149. https://github.com/official-stockfish/Stockfish/pull/5149 [V]
8. sscg13 (2026). *Compress threat indexing, update feature set name to Full Threats v2*, nnue-pytorch PR #398. https://github.com/official-stockfish/nnue-pytorch/pull/398 [V]
9. Xu, S. (xu-shawn) (2026). *K32H3 & Refactors*, nnue-pytorch PR #563. https://github.com/official-stockfish/nnue-pytorch/pull/563 [V]
10. Stockfish team (2024). *Stockfish 16.1* release blog. https://stockfishchess.org/blog/2024/stockfish-16-1/ [V]
11. Stockfish team (2026). *Stockfish 18* release blog. https://stockfishchess.org/blog/2026/stockfish-18/ [V]
12. Stockfish team (2026). *Stockfish releases (incl. Stockfish 19)*. https://github.com/official-stockfish/Stockfish/releases [V]
13. Chessprogramming Wiki. *Stockfish NNUE*. https://www.chessprogramming.org/Stockfish_NNUE ; *NNUE*. https://www.chessprogramming.org/NNUE [V]
14. Viridithas author (cosmobobak) (2024). *NNUE performance improvements*. https://asteri.sm/files/2024-06-01-nnue [V content; author attribution UNVERIFIED]
15. jw1912 et al. *bullet: specialised ML library for NNUE training* (Rust). https://github.com/jw1912/bullet [V]
16. DeepWiki. *NNUE Architecture Reference* (nnue-pytorch). https://deepwiki.com/official-stockfish/nnue-pytorch/9-nnue-architecture-reference [V exists; secondary, auto-generated, cross-check before citing]
17. Shannon, C. E. (1950). *Programming a Computer for Playing Chess*. Philosophical Magazine, Ser. 7, 41(314), 256-275. [UNVERIFIED page numbers]
18. Zobrist, A. L. (1970). *A New Hashing Method with Application for Game Playing*. Tech. Rep. 88, Computer Sciences Dept., Univ. of Wisconsin. [UNVERIFIED]
19. Samuel, A. L. (1959). *Some Studies in Machine Learning Using the Game of Checkers*. IBM J. Res. Dev. 3(3), 210-229. [UNVERIFIED]
20. Tesauro, G. (1995). *Temporal Difference Learning and TD-Gammon*. Communications of the ACM 38(3), 58-68. [UNVERIFIED]
21. Campbell, M., Hoane, A. J., Hsu, F.-h. (2002). *Deep Blue*. Artificial Intelligence 134(1-2), 57-83. [UNVERIFIED]
22. Lai, M. (2015). *Giraffe: Using Deep Reinforcement Learning to Play Chess*. arXiv:1509.01549. [UNVERIFIED]
23. Silver, D. et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*. Science 362(6419), 1140-1144. [UNVERIFIED]
24. Osterlund, P. (2014). *Texel's Tuning Method*. Chessprogramming Wiki / TalkChess. https://www.chessprogramming.org/Texel%27s_Tuning_Method [UNVERIFIED]
25. Leela Chess Zero project. https://lczero.org [UNVERIFIED for specific architecture claims]
