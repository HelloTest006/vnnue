# VNNUE prototype experiments: results

All numbers below come from runs of the scripts in `experiments/`. The tables are generated mechanically from `results/*.json` by `make_report.py`; the Key findings section was written by hand from those tables (text in `findings_text.py`). Seeds are fixed in every script. Machine: 4-core Windows laptop CPU, Python 3.14, numpy 2.4, torch 2.10 (CPU), python-chess 1.11.

**Scope warning.** This is a Python prototype. Op and byte counts are exact counts under an explicit cost model; wall-clock numbers are numpy/torch proxies and are not representative of SIMD C++. The search traces come from a toy alpha-beta searcher, not Stockfish. No statement here is about playing strength, Elo or nps.


## Key findings (numbers copied from the tables below)

**E1 Exactness: confirmed, 0 mismatches.**
- 6,670,640 accumulator-source checks ran on real game sequences and search trees, with normal and stress weights: refresh, Finny, SF hybrid, SF chain catch-up with two-ply cancellation, VNNUE ancestor net-delta, and the C2 Woodbury transition on a materialised `W0 + U V_K` table taken mod 2^16. Every incremental path matched the from-scratch int16 refresh bit for bit. That includes 58,084,584 (candidate, lane) results whose path had a partial sum outside the int16 range, which still decoded exactly because the final value was in range.
- 7,803,876 PSI, y-state and feature-delta checks ran with 0 mismatches. These covered:
  - P1 and P2 incremental vs full recomputation;
  - P1^2 - P2 always even;
  - incrementally maintained E (T14);
  - the multi-head readout vs an explicit brute-force sum over all piece pairs (T13);
  - the Black view as a lane permutation under tying (T21).
- Saturating adds are path-dependent. With stress weights, 25,280 of 25,680 incrementally updated accumulators differed from the saturating from-scratch value. In 456,904 of the differing lanes the exact value was in range, so saturation gives a wrong answer that wrapping would have got right. Wrapping arithmetic had 0 mismatches. With normal weights nothing saturated, so both agreed.

**E2 PSI overhead: within the T25 bound (H1 holds under the byte model).**
- On a quiet move PSI adds exactly 4.69% of the PSQ column bytes. That is the T25 bound, which is tight when k = 2.
- Over all Lichess moves it adds 2.68% of PSQ column bytes, 1.75% once accumulator passes are counted, and 1.14% once the threat proxy is counted. The threat-proxy figures are computed on the first 600 Lichess games (36,701 moves). The proxy changes 2.98 features for every changed piece-square feature, slightly below the "3-4x in a typical midgame" stated in the nnue-pytorch documentation.
- The PSI byte model charges PSI only for its weight columns (192 B per changed absolute feature); it does not charge PSI its own state pass (load + store of P1, P2: 384 B per make), while NNUE is charged its accumulator passes. Ratios "incl. passes" therefore favour PSI.
- Own-king moves (quiet, captures and castling) are 13.1% of Lichess moves (26,231 of 199,984). 8.94% of accumulator builds in games need a table change (king bucket, mirror or block).

**E3/E4 source selection: the "never worse" property holds per event, but the superset adds almost nothing, and greedy C4 hurts in a stateful search.**
- At every build event, the minimum over the full VNNUE candidate set was <= the minimum over SF's set (T24 sanity check).
- Almost all of the gain of VN over the SF *rule* comes from taking the argmin over SF's *own* set (SFmin). SFmin and VN cost within 0.1% of each other in the eager runs.
  - Under this cost model, SF's hybrid rule is often more expensive than a plain Finny refresh: on Lichess games hybrid averages 36.7 columns and Finny 16.3 at table changes.
  - This is a property of our simplified cache/cost model and is **not** evidence about Stockfish itself. In sections 2.2-2.5 the Finny cache persists across games and trees (stale entries from unrelated games); section 2.7 reruns with the cache cleared per root.
  - SFmin means: chain catch-up whenever available (as SF), otherwise the argmin over {refresh, Finny, hybrid}. VN_noC4 differs from SFmin only by being allowed to skip an available chain. Woodbury is not in the E3 candidate sets.
- **Cache-reset rerun (section 2.7).** With the Finny cache cleared at every root (as Stockfish does on `ucinewgame`; warmed by the run itself), the gain of SFmin over the SF rule on Lichess games shrinks from 0.8211 to 0.9387 in columns (FT bytes 0.9466); self-play 0.9234, eager trees 0.9780, lazy trees 0.9812. VN again adds at most 0.11% per event over SFmin in eager runs. The remaining gain comes from our model of the SF rule never choosing a refresh: at table changes on Lichess games, hybrid averages 30.4 columns and Finny 18.8. On lazy trees VN still costs 1.2169x SF's columns (1.0992x FT bytes). The king-move savings of the main run (king_quiet 25.332 vs 14.822 cols) become 19.48 vs 14.874 with per-game reset; they come from SFmin's own candidate set, not from the VNNUE superset, and are a property of our simulator.
- Even without C4, VN_noC4 is worse than SFmin on lazy trees
 (5.805 vs 5.665 cols/eval, 20,228 vs 20,022 FT bytes/eval) despite a 9.96% per-event counterfactual saving, so per-event greedy argmin over the larger set does not compose.
- **C4 net-delta catch-up, applied greedily, makes whole-trace cost worse** on the lazy search traces: +20.3% columns and +9.2% FT bytes per evaluation vs SF, with 6% fewer passes. Two things combine here:
  - It skips materialising intermediate accumulators that sibling nodes would later reuse.
  - The per-event counterfactual "saving" (18.73%) is therefore misleading. Policies diverge in state, and per-event optimality does not compose.
- Within catch-ups of length >= 2, cancellation saved 10.85% of columns. PR #5988 gives no figure for this saving (its author's informal "~0.2%" is the share of three-move updates where one side moves the same piece twice, a different statistic). We attribute our saving to frequent same-square quiescence recaptures in our toy trees.
- H3 ("C4 column saving < 1%") is therefore not supported as stated. The whole-trace effect of greedy C4 is negative, though, so C4 needs a reuse-aware policy before it can help.
- Caveat: our "lazy" mode removed only 4% of evaluations, because most nodes in these trees are quiescence nodes.

**E4 low-rank king transition (C2): rarely worth it at the spec defaults.**
- Woodbury is the argmin at own-king events only when mu*r_K*wb_mult <= 24 column-equivalents.
- At the spec default r_K = 16 with dU formed on the fly, it won 0% of own-king events for every mu in {1, 1.5, 2}, on every swept dataset (Lichess games, eager and lazy trees). This is consistent with H2.
- With precomputed dU at mu = 1 it wins 30.7% of own-king events on Lichess games and uses 0.8734 of VN's plain columns, but adds 129,056 mcols: in col-equivalents the ratio is 0.9722 and in FT bytes 0.9828 (21,005.8 vs 21,372.8 B per evaluation). It also costs about 91 extra y-maintenance lanes per make event and a precomputed dU table (about 18.5 MB for same-half adjacent bucket pairs).
- On search trees Woodbury lowers plain columns slightly (0.9749 at r_K = 4, mu = 1, eager) but increases FT bytes once its mcols are counted (1.0250 eager, 1.0596 lazy).

- E5 shows the integer-factor constraint is expensive without QAT. On a planted rank-8 table, rank 8 recovers the residual in float (relative error 0.0098). Rounding U and V_K so that their product is exact in table units raises the error to 0.218, against 2e-5 for a plain int16 table at the same scale.

**E6-E8 training: PSI learns pair interactions on synthetic teachers; no benefit on real-ish labels.**
- On all three synthetic pair teachers, the cost-matched M1 (L = 192 + PSI r = 32, m = 16; 256 update lanes per changed feature) had lower held-out MSE than M0 (L = 256) and M0s (L = 192). Each result is over 3 seeds; s.d.s are in section 3.2.
  - Low-rank teacher: 0.0389 vs 0.0666.
  - Sparse attack teacher: 0.1487 vs 0.1607.
  - Both: 0.1336 vs 0.1612.
- The non-cost-matched explicit-pair reference M2 (1 seed) is best on the sparse attack teacher (0.0424) and on "both" (0.1222), and worst on the low-rank teacher (0.1476).
- These teachers were built to contain pair structure. The result shows that PSI can learn such structure; it shows nothing about chess evaluation.
- Integer PSI pipeline (int8 V and Lam): all range bounds held, the integer readout equalled a float evaluation of the quantized weights, and quantization gave a relative RMS error of 0.0161 in z.
- E7: tying V under the colour flip hurt on a non-symmetric teacher (0.1498 vs 0.1337) and helped on a colour-symmetrised one (0.1209 vs 0.1362). T21 held exactly on the trained model.
- E8:
  - On game results, every model was *worse* than predicting the training mean (0.2412), so this data/label combination is too small and noisy to compare models on. PSI was the worst (0.2920 vs 0.2807).
  - On a linear material+PST label, all models tie (0.0155-0.0158), as expected.
  - **No real-label benefit of PSI was observed.**

**Wall-clock (rough proxy only).** In batched torch kernels a PSI quiet-move update costs 0.066x an NNUE quiet-move update (4 columns), against a byte ratio of 384 / 8192 = 0.047. Unbatched numpy timings measure Python dispatch, not memory traffic. None of this predicts engine nps.

### Limitations
- Simplified K32H3. No threat or pawn-pair columns inside the accumulator; threat bytes are a proxy that is only counted. No PSQT outputs and no layer stacks.
- Search traces come from a toy alpha-beta (depth 3 + quiescence, 2,500-node cap per tree, 116 of 200 trees truncated), not from Stockfish. Catch-up lengths, king-move frequencies and cache warmth are proxies.
- The Stockfish policy is modelled from the spec text. The backward catch-up fill after a Finny refresh is not modelled. Pass costs are a uniform 4L bytes.
- Training uses a single White-absolute perspective with no king buckets, 6 epochs on 140k positions, and synthetic teachers. M2 used 1 seed. E5 uses post-training quantization of the factors, not QAT.
- Wall-clock comes from Python on a 4 GB laptop.

### Reproduce
```
cd experiments
python gen_data.py          # ~8 min; needs data/lichess/lichess_db_standard_rated_2013-01.pgn.zst (17.8 MB, database.lichess.org)
python e1_exactness.py      # ~35 min
python e2_e4_opcounts.py    # ~53 min
python e2_threat.py         # ~1 min
python e3_cache_reset.py     # ~11 min (E3/E4 rerun, Finny cache cleared per root)
python train.py             # ~60 min
python bench.py             # ~3 min
python make_report.py       # tables, figures, RESULTS.md
```

### Figures
- `figures/e2_bytes_per_move_type.png`: per-move FT bytes, NNUE vs VNNUE, plus PSI bytes, by move type.
- `figures/e3_policy_totals.png`: whole-trace FT bytes per policy, relative to SF.
- `figures/e4_woodbury_win_rate.png`: Woodbury win rate vs its mcol cost.
- `figures/e5_rank.png`: low-rank reconstruction error, float vs integer factors.
- `figures/e6_training.png`: E6 held-out MSE per teacher and model.


---
# Full tables

## 0. Data

| Source | Content | Size |
|---|---|---|
| Lichess standard rated games, 2013-01 (CC0, `database.lichess.org`), first 3,000 standard games | real human move sequences (+ game results) | 202,984 positions; 175,984 training positions (ply >= 8) |
| Self-play | depth-2 alpha-beta + quiescence, material+PST eval, 8 random opening plies, 30 games | 2,440 positions |
| Search traces | 200 roots (160 from Lichess games, 40 from self-play); alpha-beta depth 3 + capture quiescence (max 4 plies), MVV-LVA ordering; each tree capped at 2,500 nodes (116 trees hit the cap) | 389,664 make events |

Trees are stored as DFS pre-order node lists with parent pointers, which is exactly the make/unmake event stream of an accumulator stack. Evaluation modes: **eager** = every node evaluated (like Stockfish's static eval at almost every node); **lazy** = roots, quiescence/leaf nodes always, interior nodes with probability 0.3 (seeded).

**Feature model (simplified K32H3).** 32 own-king buckets with horizontal mirroring (king on files e-h), 3 material blocks (0: no rook/queen on board; 1: some R/Q but opponent has no queen; 2: opponent has a queen), 11 planes x 64 (both kings share one plane) = 704 local features, 67,584 per perspective. **Not modelled inside the accumulator:** Full_Threats and PP_3Wide columns, PSQT outputs, layer stacks. A threat *proxy* (attacker-target piece pairs with Full_Threats-style dedup, absolute frame) is counted for byte accounting on game sequences only. Stockfish's backward catch-up fill after a Finny refresh is not modelled.

## 1. E1 Exactness

Prototype with L = 64 int16 lanes, PSI r = 32, m = 16, C2 rank r_K = 8. The K32H3 table is the *materialised merged* C2 table W[blk,kappa] = W0[blk] + U[blk,kappa] V_K computed modulo 2^16, so the Woodbury identity is tested on the same table every other source uses. At every accumulator build, **every available source** (refresh, Finny, hybrid, SF chain catch-up with two-ply cancellation, VNNUE ancestor net-delta, Woodbury transition) is computed with numpy int16 wrapping adds and compared bit-for-bit with the from-scratch refresh. The chosen source's value is then stored and reused downstream (Finny entries, chain intermediates, stack ancestors), so errors would propagate. PSI checks run at every make event: P1 (int16) and P2 (int32) vs full recomputation; P1^2 - P2 even; incrementally maintained E (T14) vs (P1^2 - P2)/2; the multi-head readout sum_t Lam_ht E_t vs an explicit brute-force sum over all present piece pairs (T13); the Black view from the colour-flipped board vs the lane-permuted White state under tying (T21); the C2 state y = V_K x maintained incrementally vs recomputed. Also, incremental absolute deltas vs set differences.

Two weight regimes: **normal** (small weights, accumulators mostly in range) and **stress** (|W0| <= 6000, |U| <= 300, |V_K| <= 127, |b| <= 20000), where partial sums regularly leave the int16 range.

| Total | Value |
|---|---|
| accumulator candidate checks (source value == refresh, bit-exact) | 6,670,640 |
| mismatches | **0** |
| builds where some lane decoded != exact integer although the exact final value was in int16 range | **0** |
| (candidate, lane) results whose update path had a partial sum outside int16 but whose exact final value was in range, all decoded exactly | 58,084,584 |
| PSI / y / feature-delta checks | 7,803,876 |
| PSI / y / feature-delta mismatches | **0** |

Per run (policy = which source is actually *used* downstream; `VN mu=0` makes Woodbury artificially cheap so that its outputs are reused often):

**Subsets.** E1 runs on subsets (constants in `e1_exactness.py`): the first 300 Lichess games, all 30 self-play games, the first 15 search roots (eager) and the first 40 search roots (lazy) of the 200; `nodes` below gives the resulting size. Under stress weights most *final* lane values are themselves outside int16 (`final lanes out of range`); for those lanes bit-exactness means agreement modulo 2^16 with the refresh, not a correct int16 decode.

| dataset | eval | weights | policy | nodes | builds | sources used | checks | mismatches | (candidate, lane) results exact despite partial sums outside int16 | final lanes in int16 range | final lanes out of range |
|---|---|---|---|---|---|---|---|---|---|---|---|
| games_lichess | eager | normal | SF mu=1 | 19,559 | 39,118 | chain 35625, finny 2667, hybrid 826 | 153,205 | 0 | 0 | 2,503,552 | 0 |
| games_lichess | eager | normal | VN mu=1 | 19,559 | 39,118 | chain 35607, refresh 1793, finny 884, woodbury 823, ancestor 11 | 155,658 | 0 | 0 | 2,503,552 | 0 |
| games_lichess | eager | normal | VN mu=0 | 19,559 | 39,118 | chain 35616, woodbury 2135, finny 695, refresh 661, ancestor 11 | 155,658 | 0 | 0 | 2,503,552 | 0 |
| games_selfplay | eager | normal | SF mu=1 | 2,440 | 4,880 | chain 4361, finny 394, hybrid 125 | 19,066 | 0 | 0 | 312,320 | 0 |
| games_selfplay | eager | normal | VN mu=1 | 2,440 | 4,880 | chain 4357, refresh 282, finny 169, woodbury 71, ancestor 1 | 19,468 | 0 | 0 | 312,320 | 0 |
| games_selfplay | eager | normal | VN mu=0 | 2,440 | 4,880 | chain 4361, woodbury 350, refresh 103, finny 65, ancestor 1 | 19,468 | 0 | 0 | 312,320 | 0 |
| trees | eager | normal | SF mu=1 | 34,534 | 69,068 | chain 64713, finny 2630, hybrid 1725 | 273,612 | 0 | 0 | 4,420,352 | 0 |
| trees | eager | normal | VN mu=1 | 34,534 | 69,068 | chain 64462, finny 3387, woodbury 1072, refresh 139, ancestor 8 | 275,609 | 0 | 0 | 4,420,352 | 0 |
| trees | eager | normal | VN mu=0 | 34,534 | 69,068 | chain 64599, finny 2598, woodbury 1751, refresh 112, ancestor 8 | 275,609 | 0 | 0 | 4,420,352 | 0 |
| trees | lazy | normal | SF mu=1 | 88,026 | 168,696 | chain 146946, finny 18633, hybrid 3117 | 656,071 | 0 | 0 | 10,796,544 | 0 |
| trees | lazy | normal | VN mu=1 | 88,026 | 168,696 | chain 95918, ancestor 36864, finny 32345, woodbury 3081, refresh 488 | 665,948 | 0 | 0 | 10,796,544 | 0 |
| trees | lazy | normal | VN mu=0 | 88,026 | 168,696 | chain 96198, ancestor 37128, finny 25083, woodbury 9955, refresh 332 | 665,948 | 0 | 0 | 10,796,544 | 0 |
| games_lichess | eager | stress | SF mu=1 | 19,559 | 39,118 | chain 35625, finny 2667, hybrid 826 | 153,205 | 0 | 2,824,741 | 740,964 | 1,762,588 |
| games_lichess | eager | stress | VN mu=1 | 19,559 | 39,118 | chain 35607, refresh 1793, finny 884, woodbury 823, ancestor 11 | 155,658 | 0 | 2,754,658 | 740,964 | 1,762,588 |
| games_lichess | eager | stress | VN mu=0 | 19,559 | 39,118 | chain 35616, woodbury 2135, finny 695, refresh 661, ancestor 11 | 155,658 | 0 | 2,681,726 | 740,964 | 1,762,588 |
| games_selfplay | eager | stress | SF mu=1 | 2,440 | 4,880 | chain 4361, finny 394, hybrid 125 | 19,066 | 0 | 364,975 | 96,709 | 215,611 |
| games_selfplay | eager | stress | VN mu=1 | 2,440 | 4,880 | chain 4357, refresh 282, finny 169, woodbury 71, ancestor 1 | 19,468 | 0 | 358,846 | 96,709 | 215,611 |
| games_selfplay | eager | stress | VN mu=0 | 2,440 | 4,880 | chain 4361, woodbury 350, refresh 103, finny 65, ancestor 1 | 19,468 | 0 | 344,355 | 96,709 | 215,611 |
| trees | eager | stress | SF mu=1 | 34,534 | 69,068 | chain 64713, finny 2630, hybrid 1725 | 273,612 | 0 | 4,878,069 | 1,238,217 | 3,182,135 |
| trees | eager | stress | VN mu=1 | 34,534 | 69,068 | chain 64462, finny 3387, woodbury 1072, refresh 139, ancestor 8 | 275,609 | 0 | 4,776,872 | 1,238,217 | 3,182,135 |
| trees | eager | stress | VN mu=0 | 34,534 | 69,068 | chain 64599, finny 2598, woodbury 1751, refresh 112, ancestor 8 | 275,609 | 0 | 4,734,806 | 1,238,217 | 3,182,135 |
| trees | lazy | stress | SF mu=1 | 88,026 | 168,696 | chain 146946, finny 18633, hybrid 3117 | 656,071 | 0 | 11,569,113 | 3,003,185 | 7,793,359 |
| trees | lazy | stress | VN mu=1 | 88,026 | 168,696 | chain 95918, ancestor 36864, finny 32345, woodbury 3081, refresh 488 | 665,948 | 0 | 11,467,537 | 3,003,185 | 7,793,359 |
| trees | lazy | stress | VN mu=0 | 88,026 | 168,696 | chain 96198, ancestor 37128, finny 25083, woodbury 9955, refresh 332 | 665,948 | 0 | 11,328,886 | 3,003,185 | 7,793,359 |

**Saturation is path-dependent (T8), empirically.** Along the first 200 Lichess games, each perspective's accumulator was updated incrementally with (a) wrapping int16 adds and (b) saturating adds (`paddsw` semantics), and compared with the same arithmetic applied from scratch (bias + columns in sorted order):

| weights | accumulators compared | wrapping mismatches | saturating mismatches (accumulators) | saturating mismatching lanes | of which the exact final value was in range |
|---|---|---|---|---|---|
| normal | 25,680 | 0 | 0 | 0 | 0 |
| stress | 25,680 | 0 | 25,280 | 1,518,777 | 456,904 |

## 2. E2-E4 Op counts, bytes and source selection

Cost model (SPEC 3.1): L = 1024; col = one int16 column (2048 B); pass = accumulator load+store (4096 B); PSI = 192 B and 2r = 64 lanes per changed absolute feature (r = 32), applied eagerly at every make event; threat proxy columns are i8 (1 KB). Argmin scalar = `cols + mu*mcols + 2*passes`. Woodbury cost = `wb_mult*r_K mcol (wb_mult=2: dU formed on the fly) + |x_t ^ x_u| col + 1 pass`.

Policies: **SF** = Stockfish-like rule (chain catch-up if no table change, else hybrid if its conditions hold, else Finny); **SFmin** = argmin over SF's set {refresh, Finny, hybrid}; **VN_noC4** = argmin over {refresh, Finny, hybrid, chain}; **VN** = VN_noC4 + ancestor net-delta (C4). The C2 Woodbury candidate is evaluated separately (sweep below).

### 2.1 Move statistics

| dataset | positions | quiet | capture | king quiet | king capture | castling | en passant | promotion | capture-promotion | block-change moves | k_a mean | |S| mean / p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| games_lichess/eager | 202,984 | 64.5% | 22.1% | 9.7% | 1.2% | 2.3% | 0.0% | 0.3% | 0.0% | 1.98% | 2.279 | 23.048 / 25 / 32 |
| games_selfplay/eager | 2,440 | 60.9% | 21.4% | 15.6% | 1.4% | 0.0% | 0.0% | 0.7% | 0.0% | 2.16% | 2.229 | 21.48 / 22 / 32 |
| trees/eager | 389,664 | 43.2% | 48.8% | 4.9% | 2.5% | 0.1% | 0.0% | 0.3% | 0.1% | 6.49% | 2.518 | 23.793 / 25 / 31 |

Accumulator build events (one per evaluated node and perspective not already computed), SF policy run:

| dataset | builds | builds needing a table change (king bucket / mirror / block) | of which own-king moves | SF Finny cols at table change (mean / p50 / p95) | SF hybrid cols (mean / p50 / p95) | hybrid admissible at own-king events |
|---|---|---|---|---|---|---|
| games_lichess/eager | 405,968 | 36,312 (8.94%) | 26,231 | 16.289 / 16 / 35 | 36.683 / 36 / 65 | 30.7% |
| games_selfplay/eager | 4,880 | 519 (10.64%) | 409 | 13.81 / 12 / 34.35 | 32.224 / 32 / 60.8 | 30.6% |
| trees/eager | 779,328 | 52,504 (6.74%) | 29,468 | 5.877 / 4 / 21 | 8.48 / 6 / 30 | 57.9% |
| trees/lazy | 746,760 | 86,616 (11.60%) | 27,815 | 4.635 / 4 / 10 | 8.436 / 7 / 28 | 49.0% |

### 2.2 Per move type, Lichess games (eager evaluation): NNUE vs VNNUE per-move cost

Columns and bytes are per move, both perspectives, mean / p50 / p95. The PSI overhead ratios are totals over the class.

| move type | count | SF cols | VN cols | SF bytes | PSI lanes | PSI bytes | PSI / SF PSQ column bytes | PSI / SF PSQ bytes incl. passes |
|---|---|---|---|---|---|---|---|---|
| capture | 44,132 | 7.724 / 6 / 28 | 7.241 / 6 / 21 | 24,010 / 20,480 / 65,536 | 192 / 192 / 192 | 576 / 576 / 576 | 3.64% | 2.40% | 
| capture_promotion | 30 | 17.133 / 12.5 / 35.2 | 11.933 / 10 / 23.55 | 43,281 / 33,792 / 80,282 | 192 / 192 / 192 | 576 / 576 / 576 | 1.64% | 1.33% | 
| castling | 4,519 | 33.317 / 34 / 44 | 28.883 / 30 / 36 | 76,425 / 77,824 / 98,304 | 256 / 256 / 256 | 768 / 768 / 768 | 1.13% | 1.00% | 
| en_passant | 90 | 6 / 6 / 6 | 6 / 6 / 6 | 20,480 / 20,480 / 20,480 | 192 / 192 / 192 | 576 / 576 / 576 | 4.69% | 2.81% | 
| king_capture | 2,407 | 27.734 / 25 / 55.7 | 17.363 / 16 / 30 | 67,551 / 59,392 / 129,024 | 192 / 192 / 192 | 576 / 576 / 576 | 1.01% | 0.85% | 
| king_quiet | 19,305 | 25.332 / 22 / 57 | 14.822 / 14 / 28 | 63,173 / 53,248 / 133,120 | 128 / 128 / 128 | 384 / 384 / 384 | 0.74% | 0.61% | 
| promotion | 502 | 20.45 / 21 / 41 | 10.765 / 10.5 / 20 | 50,074 / 51,200 / 92,160 | 128 / 128 / 128 | 384 / 384 / 384 | 0.92% | 0.77% | 
| quiet | 128,999 | 4 / 4 / 4 | 4 / 4 / 4 | 16,384 / 16,384 / 16,384 | 128 / 128 / 128 | 384 / 384 / 384 | 4.69% | 2.34% | 

Threat proxy (first 600 Lichess games, 37,301 positions, where it was computed): changed threat-proxy features per move (absolute frame, per perspective) mean / p50 / p95 = 6.798 / 6 / 13, i.e. 2.98x the number of changed piece-square features.

| move type | threat-proxy changes (mean / p50 / p95) | PSI / SF PSQ column bytes | PSI / (SF PSQ incl. passes + threat proxy, both perspectives) bytes |
|---|---|---|---|
| capture | 8.594 / 8 / 15 | 3.68% | 1.39% |
| capture_promotion | 6.571 / 7 / 10.7 | 1.48% | 0.95% |
| castling | 13.454 / 14 / 16 | 1.14% | 0.75% |
| en_passant | 9.75 / 10 / 13 | 4.69% | 1.42% |
| king_capture | 5.362 / 5 / 10.45 | 1.02% | 0.73% |
| king_quiet | 4.474 / 4 / 10 | 0.72% | 0.52% |
| promotion | 2.664 / 2 / 6 | 0.99% | 0.73% |
| quiet | 6.294 / 6 / 11 | 4.69% | 1.31% |

Overall PSI byte overhead (H1; the SPEC T25 bound is 4.7% of PSQ column bytes when every move costs >= 4kL bytes):

| dataset | PSI / PSQ column bytes | PSI / PSQ bytes incl. passes | PSI / (PSQ + threat proxy) bytes |
|---|---|---|---|
| games_lichess/eager | 2.68% | 1.75% | 1.14% |
| games_selfplay/eager | 2.62% | 1.71% | 1.19% |

### 2.3 Whole-trace totals per policy (stateful simulation)

Each policy is simulated independently over the whole event stream (its own Finny caches and materialised accumulators). `PSI/FT bytes` divides the PSI bytes (eager, every make event) by that policy's FT bytes.

| dataset | policy | cols / eval | passes / eval | FT bytes / eval | total cols vs SF | total passes vs SF | PSI / FT bytes | sources used |
|---|---|---|---|---|---|---|---|---|
| games_lichess/eager | SF | 7.841 | 2.079 | 24,576 | 1.0000 | 1.0000 | 1.75% | chain 369656, finny 28250, hybrid 8062 |
| games_lichess/eager | SFmin | 6.438 | 2 | 21,377 | 0.8211 | 0.9618 | 2.02% | chain 369656, refresh 24599, finny 11713 |
| games_lichess/eager | VN_noC4 | 6.436 | 2 | 21,373 | 0.8208 | 0.9618 | 2.02% | chain 369343, refresh 24592, finny 12033 |
| games_lichess/eager | VN | 6.436 | 2 | 21,373 | 0.8208 | 0.9618 | 2.02% | chain 369343, refresh 24560, finny 12013, ancestor 52 |
| games_selfplay/eager | SF | 7.877 | 2.102 | 24,744 | 1.0000 | 1.0000 | 1.71% | chain 4361, finny 394, hybrid 125 |
| games_selfplay/eager | SFmin | 6.382 | 2 | 21,263 | 0.8102 | 0.9513 | 1.99% | chain 4361, refresh 336, finny 183 |
| games_selfplay/eager | VN_noC4 | 6.38 | 2 | 21,257 | 0.8099 | 0.9513 | 1.99% | chain 4356, refresh 336, finny 188 |
| games_selfplay/eager | VN | 6.379 | 2 | 21,256 | 0.8098 | 0.9513 | 1.99% | chain 4357, refresh 335, finny 187, ancestor 1 |
| trees/eager | SF | 5.586 | 2.088 | 19,990 | 1.0000 | 1.0000 | 2.42% | chain 726824, finny 35438, hybrid 17066 |
| trees/eager | SFmin | 5.399 | 2 | 19,249 | 0.9666 | 0.9580 | 2.51% | chain 726824, finny 49261, refresh 3243 |
| trees/eager | VN_noC4 | 5.394 | 2 | 19,239 | 0.9656 | 0.9580 | 2.51% | chain 724034, finny 52049, refresh 3245 |
| trees/eager | VN | 5.398 | 2 | 19,248 | 0.9665 | 0.9580 | 2.51% | chain 724036, finny 51946, refresh 3233, ancestor 113 |
| trees/lazy | SF | 5.842 | 2.129 | 20,683 | 1.0000 | 1.0000 | 2.44% | chain 660144, finny 72998, hybrid 13618 |
| trees/lazy | SFmin | 5.665 | 2.056 | 20,022 | 0.9697 | 0.9657 | 2.52% | chain 660144, finny 83335, refresh 3281 |
| trees/lazy | VN_noC4 | 5.805 | 2.036 | 20,228 | 0.9938 | 0.9564 | 2.49% | chain 588299, finny 155116, refresh 3345 |
| trees/lazy | VN | 7.026 | 2 | 22,581 | 1.2027 | 0.9396 | 2.23% | chain 431199, ancestor 157884, finny 154373, refresh 3304 |

Per-event counterfactual (T24 sanity check): at every build event of each run, the minimum over the full VNNUE candidate set was <= the minimum over SF's set:

| dataset | policy | full-set min <= SF-set min at every event | summed per-event saving (col-equivalents) | as a fraction of SF-set cost |
|---|---|---|---|---|
| games_lichess/eager | VN_noC4 | True | 347 | 0.02% |
| games_lichess/eager | VN | True | 534 | 0.03% |
| games_selfplay/eager | VN_noC4 | True | 7 | 0.03% |
| games_selfplay/eager | VN | True | 9 | 0.04% |
| trees/eager | VN_noC4 | True | 3,335 | 0.09% |
| trees/eager | VN | True | 3,760 | 0.10% |
| trees/lazy | VN_noC4 | True | 407,985 | 9.96% |
| trees/lazy | VN | True | 948,711 | 18.73% |

### 2.4 Net-delta catch-up (C4, E3)

| dataset | catch-up events with chain length >= 2 (VN run) | net support / sum of per-ply supports (mean / p50 / p95) | column saving from cancellation | SF chain length (mean / p50 / p95) |
|---|---|---|---|---|
| games_lichess/eager | 0 | n/a | n/a | 1 / 1 / 1 |
| games_selfplay/eager | 0 | n/a | n/a | 1 / 1 / 1 |
| trees/eager | 0 | n/a | n/a | 1 / 1 / 1 |
| trees/lazy | 227,886 | 0.905 / 1 / 1 | 10.85% | 1.045 / 1 / 1 |

### 2.5 Low-rank king transition (C2 Woodbury) as a C3 candidate (E4)

Fraction of own-king table-change events where Woodbury is the argmin, VN policy + Woodbury candidate. `wb_mult = 2`: (U_k' - U_k) y formed on the fly (2 r_K mcol); `wb_mult = 1`: precomputed dU. Total cols/mcols are relative to the VN run without Woodbury.

`col-equiv vs VN` = (cols + mu*mcols) / VN cols; `FT bytes vs VN` counts cols, mcols (2 KB each) and passes. The Woodbury sweep was run on games_lichess/eager, trees/eager and trees/lazy only (not on self-play).

| dataset | r_K | mu | wb_mult | Woodbury chosen at own-king events | total cols vs VN | total mcols | col-equiv vs VN | FT bytes / eval | FT bytes vs VN | C2 y-maintenance lanes per make |
|---|---|---|---|---|---|---|---|---|---|---|
| games_lichess/eager | 4 | 1.0 | 2 | 59.7% | 0.8052 | 125,208 | 0.9011 | 20,069.0 | 0.9390 | 22.706 |
| games_lichess/eager | 4 | 1.5 | 2 | 44.4% | 0.8357 | 93,224 | 0.9427 | 20,147.6 | 0.9427 | 22.706 |
| games_lichess/eager | 4 | 2.0 | 2 | 30.7% | 0.8734 | 64,528 | 0.9722 | 20,354.7 | 0.9524 | 22.706 |
| games_lichess/eager | 8 | 1.0 | 2 | 30.7% | 0.8734 | 129,056 | 0.9722 | 21,005.8 | 0.9828 | 45.411 |
| games_lichess/eager | 8 | 1.5 | 2 | 6.9% | 0.9667 | 28,768 | 0.9997 | 21,223.5 | 0.9930 | 45.411 |
| games_lichess/eager | 8 | 2.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 21,372.8 | 1.0000 | 45.411 |
| games_lichess/eager | 16 | 1.0 | 1 | 30.7% | 0.8734 | 129,056 | 0.9722 | 21,005.8 | 0.9828 | 90.823 |
| games_lichess/eager | 16 | 1.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 21,372.8 | 1.0000 | 90.823 |
| games_lichess/eager | 16 | 1.5 | 1 | 6.9% | 0.9667 | 28,768 | 0.9997 | 21,223.5 | 0.9930 | 90.823 |
| games_lichess/eager | 16 | 1.5 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 21,372.8 | 1.0000 | 90.823 |
| games_lichess/eager | 16 | 2.0 | 1 | 0.0% | 1.0000 | 0 | 1.0000 | 21,372.8 | 1.0000 | 90.823 |
| games_lichess/eager | 16 | 2.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 21,372.8 | 1.0000 | 90.823 |
| trees/eager | 4 | 1.0 | 2 | 61.2% | 0.9749 | 144,264 | 1.0435 | 19,729.2 | 1.0250 | 21.354 |
| trees/eager | 4 | 1.5 | 2 | 50.8% | 0.9776 | 119,792 | 1.0631 | 19,630.5 | 1.0199 | 21.354 |
| trees/eager | 4 | 2.0 | 2 | 40.5% | 0.9811 | 95,392 | 1.0718 | 19,540.5 | 1.0152 | 21.354 |
| trees/eager | 8 | 1.0 | 2 | 40.5% | 0.9811 | 190,784 | 1.0718 | 20,041.9 | 1.0412 | 42.707 |
| trees/eager | 8 | 1.5 | 2 | 9.9% | 0.9942 | 46,720 | 1.0275 | 19,429.7 | 1.0094 | 42.707 |
| trees/eager | 8 | 2.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 19,248.1 | 1.0000 | 42.707 |
| trees/eager | 16 | 1.0 | 1 | 40.5% | 0.9811 | 190,784 | 1.0718 | 20,041.9 | 1.0412 | 85.414 |
| trees/eager | 16 | 1.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 19,248.1 | 1.0000 | 85.414 |
| trees/eager | 16 | 1.5 | 1 | 9.9% | 0.9942 | 46,720 | 1.0275 | 19,429.7 | 1.0094 | 85.414 |
| trees/eager | 16 | 1.5 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 19,248.1 | 1.0000 | 85.414 |
| trees/eager | 16 | 2.0 | 1 | 0.0% | 1.0000 | 0 | 1.0000 | 19,248.1 | 1.0000 | 85.414 |
| trees/eager | 16 | 2.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 19,248.1 | 1.0000 | 85.414 |
| trees/lazy | 4 | 1.0 | 2 | 53.2% | 1.0057 | 230,344 | 1.0935 | 23,927.1 | 1.0596 | 21.354 |
| trees/lazy | 4 | 1.5 | 2 | 43.6% | 1.0034 | 186,336 | 1.1099 | 23,652.1 | 1.0474 | 21.354 |
| trees/lazy | 4 | 2.0 | 2 | 26.5% | 1.0001 | 110,960 | 1.0847 | 23,191.5 | 1.0270 | 21.354 |
| trees/lazy | 8 | 1.0 | 2 | 26.5% | 1.0001 | 221,920 | 1.0847 | 23,800.2 | 1.0540 | 42.707 |
| trees/lazy | 8 | 1.5 | 2 | 2.4% | 0.9992 | 14,160 | 1.0073 | 22,648.0 | 1.0030 | 42.707 |
| trees/lazy | 8 | 2.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 22,581.3 | 1.0000 | 42.707 |
| trees/lazy | 16 | 1.0 | 1 | 26.5% | 1.0001 | 221,920 | 1.0847 | 23,800.2 | 1.0540 | 85.414 |
| trees/lazy | 16 | 1.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 22,581.3 | 1.0000 | 85.414 |
| trees/lazy | 16 | 1.5 | 1 | 2.4% | 0.9992 | 14,160 | 1.0073 | 22,648.0 | 1.0030 | 85.414 |
| trees/lazy | 16 | 1.5 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 22,581.3 | 1.0000 | 85.414 |
| trees/lazy | 16 | 2.0 | 1 | 0.0% | 1.0000 | 0 | 1.0000 | 22,581.3 | 1.0000 | 85.414 |
| trees/lazy | 16 | 2.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 22,581.3 | 1.0000 | 85.414 |

### 2.7 Robustness: Finny cache cleared at every root (E3/E4 rerun)

In sections 2.2-2.5 (`e2_e4_opcounts.py`) one Finny cache is kept for the whole dataset, so entries carry over between unrelated games and trees and are *stale* (they are updated only when a Finny/refresh/hybrid source is used). Stockfish clears its refresh table on `ucinewgame` and warms it during search. `e3_cache_reset.py` reruns the policies with the cache cleared at every root (each game, each search tree), so it starts empty (= refresh cost) and is warmed by the run itself. Everything else is identical.

| dataset | policy | cols / eval | passes / eval | FT bytes / eval | total cols vs SF | FT bytes vs SF | per-event saving vs SF-set min | sources used |
|---|---|---|---|---|---|---|---|---|
| games_lichess/eager | SF | 7.939 | 2.079 | 24,777 | 1.0000 | 1.0000 |  | chain 369656, finny 28250, hybrid 8062 |
| games_lichess/eager | SFmin | 7.452 | 2 | 23,455 | 0.9387 | 0.9466 |  | chain 369656, refresh 33583, finny 2729 |
| games_lichess/eager | VN_noC4 | 7.45 | 2 | 23,451 | 0.9384 | 0.9465 | 0.01% | chain 369343, refresh 33582, finny 3043 |
| games_lichess/eager | VN | 7.45 | 2 | 23,450 | 0.9384 | 0.9464 | 0.02% | chain 369343, refresh 33531, finny 3033, ancestor 61 |
| games_selfplay/eager | SF | 7.682 | 2.102 | 24,344 | 1.0000 | 1.0000 |  | chain 4361, finny 394, hybrid 125 |
| games_selfplay/eager | SFmin | 7.094 | 2 | 22,720 | 0.9234 | 0.9333 |  | chain 4361, refresh 393, finny 126 |
| games_selfplay/eager | VN_noC4 | 7.091 | 2 | 22,714 | 0.9231 | 0.9330 | 0.03% | chain 4356, refresh 393, finny 131 |
| games_selfplay/eager | VN | 7.09 | 2 | 22,713 | 0.9230 | 0.9330 | 0.03% | chain 4357, refresh 392, finny 130, ancestor 1 |
| trees/eager | SF | 5.527 | 2.088 | 19,870 | 1.0000 | 1.0000 |  | chain 726824, finny 35438, hybrid 17066 |
| trees/eager | SFmin | 5.405 | 2 | 19,262 | 0.9780 | 0.9694 |  | chain 726824, finny 48814, refresh 3690 |
| trees/eager | VN_noC4 | 5.4 | 2 | 19,251 | 0.9770 | 0.9689 | 0.09% | chain 724034, finny 51604, refresh 3690 |
| trees/eager | VN | 5.406 | 2 | 19,263 | 0.9781 | 0.9695 | 0.11% | chain 724037, finny 51489, refresh 3667, ancestor 135 |
| trees/lazy | SF | 5.78 | 2.129 | 20,555 | 1.0000 | 1.0000 |  | chain 660144, finny 72998, hybrid 13618 |
| trees/lazy | SFmin | 5.671 | 2.056 | 20,034 | 0.9812 | 0.9746 |  | chain 660144, finny 82898, refresh 3718 |
| trees/lazy | VN_noC4 | 5.811 | 2.036 | 20,240 | 1.0055 | 0.9847 | 9.96% | chain 588299, finny 154701, refresh 3760 |
| trees/lazy | VN | 7.033 | 2 | 22,595 | 1.2169 | 1.0992 | 18.72% | chain 431200, ancestor 157910, finny 153939, refresh 3711 |

| dataset | SF Finny cols at table change (mean / p50 / p95) | SF hybrid cols (mean / p50 / p95) |
|---|---|---|
| games_lichess/eager | 18.788 / 17 / 32 | 30.398 / 29 / 55 |
| games_selfplay/eager | 13.954 / 11 / 32 | 27.96 / 26 / 54 |
| trees/eager | 5.451 / 4 / 13 | 8.017 / 6 / 24 |
| trees/lazy | 4.39 / 4 / 10 | 8.041 / 7 / 23 |

Per move type, Lichess games, cache cleared per game (mean cols and FT bytes per move, both perspectives):

| move type | count | SF cols | VN cols | SF bytes | VN bytes |
|---|---|---|---|---|---|
| capture | 44,132 | 7.308 | 7.298 | 23,158 | 23,139 |
| capture_promotion | 30 | 13.867 | 11.933 | 36,591 | 32,631 |
| castling | 4,519 | 32.414 | 32.414 | 74,575 | 74,575 |
| en_passant | 90 | 6 | 6 | 20,480 | 20,480 |
| king_capture | 2,407 | 21.408 | 17.558 | 54,594 | 44,152 |
| king_quiet | 19,305 | 19.48 | 14.874 | 51,188 | 38,655 |
| promotion | 502 | 12.048 | 10.777 | 32,866 | 30,263 |
| quiet | 128,999 | 4 | 4 | 16,384 | 16,383 |

Woodbury subset with the cache cleared per root (relative to the VN run of this section):

| dataset | r_K | mu | wb_mult | Woodbury chosen at own-king events | total cols vs VN | total mcols | col-equiv vs VN | FT bytes vs VN |
|---|---|---|---|---|---|---|---|---|
| games_lichess/eager | 4 | 1.0 | 2 | 59.8% | 0.8236 | 125,416 | 0.9066 | 0.9392 |
| games_lichess/eager | 16 | 1.0 | 1 | 31.0% | 0.8807 | 130,224 | 0.9668 | 0.9784 |
| games_lichess/eager | 16 | 1.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 1.0000 |
| trees/eager | 4 | 1.0 | 2 | 61.4% | 0.9747 | 144,840 | 1.0434 | 1.0250 |
| trees/eager | 16 | 1.0 | 1 | 41.4% | 0.9804 | 195,232 | 1.0731 | 1.0420 |
| trees/eager | 16 | 1.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 1.0000 |
| trees/lazy | 4 | 1.0 | 2 | 53.6% | 1.0056 | 231,456 | 1.0937 | 1.0598 |
| trees/lazy | 16 | 1.0 | 1 | 28.3% | 0.9999 | 235,184 | 1.0894 | 1.0570 |
| trees/lazy | 16 | 1.0 | 2 | 0.0% | 1.0000 | 0 | 1.0000 | 1.0000 |

### 2.6 Wall-clock microbenchmark (rough proxy only)

> Python/numpy/torch wall clock; dispatch overhead dominates; weak proxy for SIMD C++. Single thread, median of 5 repetitions. Table = 138.4 MB int16 (67,584 x 1024). These timings were taken with no other experiment running on a 4-core, 4 GB RAM laptop and include Python interpreter overhead; they must not be read as engine speed.

| kernel (unbatched, one numpy call sequence per event) | ns / event |
|---|---|
| nnue_quiet_move_both_persp_4cols | 9,572 |
| psi_quiet_move_P1_P2 | 5,360 |
| refresh_32cols_one_persp | 49,841 |
| woodbury_rK16_plus_2cols_one_persp | 39,960 |
| psi_readout_one_persp_m16 | 27,140 |

| kernel (batched torch gather+add) | ns / event | column bytes / event | effective GB/s |
|---|---|---|---|
| nnue_quiet_4cols | 7,578 | 8,192 | 1.081 |
| psi_quiet_P1_P2 | 503.502 | 384 | 0.763 |
| refresh_32cols | 46,933 | 65,536 | 1.396 |

Ratio PSI-update / NNUE-quiet-update: unbatched 0.560, batched 0.066. The unbatched ratio mostly measures Python dispatch (four numpy calls each), not memory traffic.

## 3. E5-E8 Tiny training sanity checks

Config: 6 epochs, batch 1024, Adam onecycle max_lr 2e-3 (SparseAdam lr 2e-3 const for M2 pair table), seeds [0, 1, 2]; single White-absolute perspective, no king buckets, no layer stacks. These runs check that the math trains and quantizes; they say nothing about chess strength.

### 3.1 E5 Rank sufficiency (synthetic planted table)

One block, 32 buckets, 704 features, 256 lanes. Planted W_k = W0 + U_k V (true rank 8) + Gaussian noise, plus an unstructured control. Shared V_K is the top-r_K right singular subspace of the stacked bucket residuals. `int factors` quantizes V_K to int8 (max |V_K| = 127) and U to int16 so that U V_K is in table units at scale 64, then materialises the integer product (the C2 exactness requirement). Error = relative squared Frobenius error of the per-bucket residual.

| table | r_K | rel. err (float SVD) | rel. err (int factors, materialised) | rel. err (plain int16 table, same scale) |
|---|---|---|---|---|
| planted_rank8_noise0.1 | 1 | 0.86 | 0.895 | 2.08e-05 |
| planted_rank8_noise0.1 | 2 | 0.723 | 0.784 | 2.08e-05 |
| planted_rank8_noise0.1 | 4 | 0.466 | 0.57 | 2.08e-05 |
| planted_rank8_noise0.1 | 8 | 0.00978 | 0.218 | 2.08e-05 |
| planted_rank8_noise0.1 | 16 | 0.0096 | 0.18 | 2.08e-05 |
| planted_rank8_noise0.1 | 32 | 0.00924 | 0.164 | 2.08e-05 |
| planted_rank8_noise0.1 | 64 | 0.00856 | 0.164 | 2.08e-05 |
| planted_rank8_noise0.5 | 1 | 0.88 | 0.901 | 1.65e-05 |
| planted_rank8_noise0.5 | 2 | 0.766 | 0.796 | 1.65e-05 |
| planted_rank8_noise0.5 | 4 | 0.558 | 0.618 | 1.65e-05 |
| planted_rank8_noise0.5 | 8 | 0.195 | 0.315 | 1.65e-05 |
| planted_rank8_noise0.5 | 16 | 0.191 | 0.315 | 1.65e-05 |
| planted_rank8_noise0.5 | 32 | 0.184 | 0.284 | 1.65e-05 |
| planted_rank8_noise0.5 | 64 | 0.17 | 0.284 | 1.65e-05 |
| unstructured | 1 | 0.998 | 1 | 2.1e-05 |
| unstructured | 2 | 0.995 | 1 | 2.1e-05 |
| unstructured | 4 | 0.991 | 1 | 2.1e-05 |
| unstructured | 8 | 0.981 | 1 | 2.1e-05 |
| unstructured | 16 | 0.963 | 1 | 2.1e-05 |
| unstructured | 32 | 0.927 | 1 | 2.1e-05 |
| unstructured | 64 | 0.86 | 1 | 2.1e-05 |

### 3.2 E6 Does PSI learn piece-pair interactions? (synthetic teachers on real positions)

Positions: 175,984 Lichess positions (ply >= 8), split by game (20% games val): 140,307 train / 35,677 val. Teacher y = z(PST . beta) + z(pair term) + noise, standardised to unit variance (so predicting the mean gives MSE ~ 1). Pair terms: (i) low-rank Q (rank 8, dense over all 768 features); (ii) sparse 'attack geometry' Q with 72,590 nonzero unordered pairs (piece i's empty-board attack set contains j's square); (iii) both.

| model | params | update lanes per changed absolute feature |
|---|---|---|
| M0_L256 | 201,281 | 256 |
| M0s_L192 | 150,977 | 192 |
| M1_L192_PSI_r32_m16 | 176,593 | 256 |
| M2_L192_explicit_pairs | 14,088,449 | 192 + M2 additionally updates L lanes per changed active pair |

M2's explicit pair table has 72,590 rows; on average 40.044 pairs are active per position (each one is an extra L-wide column when it changes), so M2 is **not** cost-matched; it is a reference with oracle knowledge of teacher (ii)'s support. M2 was run with 1 seed (its 14 M-parameter pair table is slow on CPU).

| teacher | M0_L256 | M0s_L192 | M1_L192_PSI_r32_m16 | M2_L192_explicit_pairs |
|---|---|---|---|---|
| i_lowrank | 0.0666 +- 0.0025 | 0.0706 +- 0.0022 | 0.0389 +- 0.0031 | 0.1476 (1 seed) |
| ii_sparse_attack | 0.1607 +- 0.0003 | 0.1691 +- 0.0015 | 0.1487 +- 0.0035 | 0.0424 (1 seed) |
| iii_both | 0.1612 +- 0.0022 | 0.1672 +- 0.0049 | 0.1336 +- 0.0020 | 0.1222 (1 seed) |

Values: held-out MSE, mean +- sample s.d. over seeds.

**Quantized PSI readout check** (M1 trained on teacher iii, seed 0; V and Lam quantized to int8, 4,000 val positions): max |P1| = 571 (bound 4064), max P2 = 43,527 (bound 516,128), P1^2 - P2 even everywhere: True; max |z_int| = 21,117,603 < 2^31: True; integer pipeline equals float evaluation of the quantized weights: True; relative RMS error of z from quantization: 0.0161.

### 3.3 E7 Tied vs untied PSI embedding (colour flip)

M1 (L = 192, r = 32, m = 16). Tied: V[t+1, j] = V[t, sigma(j)], so only r/2 rows are free and one shared state serves both perspectives (T21).

| teacher | untied MSE | tied MSE | untied params | tied params |
|---|---|---|---|---|
| iii_both | 0.1337 +- 0.0020 | 0.1498 +- 0.0062 | 176,593 | 164,305 |
| iii_both_colour_symmetrised | 0.1362 +- 0.0029 | 0.1209 +- 0.0013 | 176,593 | 164,305 |

T21 numerically on the trained tied model (2,000 val positions, float32): iii_both: max |P1(flipped) - P1[perm]| = 0.00e+00, max |P2(flipped) - P2[perm]| = 0.00e+00; iii_both_colour_symmetrised: max |P1(flipped) - P1[perm]| = 0.00e+00, max |P2(flipped) - P2[perm]| = 0.00e+00.

### 3.4 E8 Real-ish labels

| label | predict-train-mean MSE | M0_L256 | M0s_L192 | M1_L192_PSI_r32_m16 |
|---|---|---|---|---|
| game_result_white_pov | 0.2412 | 0.2807 +- 0.0030 | 0.2809 +- 0.0037 | 0.2920 +- 0.0022 |
| material_pst_eval_tanh | 0.3025 | 0.0158 +- 0.0008 | 0.0155 +- 0.0015 | 0.0158 +- 0.0023 |

Game result: sigmoid output, MSE against {0, 0.5, 1}. Material+PST: tanh(eval/400) of the toy evaluator, which is linear in beta, so PSI has nothing to add there by construction.
