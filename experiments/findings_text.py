"""Interpretation text for RESULTS.md. Every number here was copied from the generated
tables (results/*.json via make_report.py); re-check them if the experiments are re-run."""
FINDINGS = r"""
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
"""
