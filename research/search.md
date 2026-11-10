# Evaluation and Search in Chess Engines: A Century of Interplay

Research notes for the VNNUE paper. Claims were checked against web sources cited inline. **[UNVERIFIED]** marks unconfirmed claims, and *(derived)* marks my own calculations.

---

## 1. Why this matters for VNNUE

Evaluation and search compete for one budget: time per move. A more accurate evaluation costs more per node, so the search goes shallower. NNUE (Nasu 2018) changed this trade by making a large network cheap per node through incremental first-layer updates. VNNUE has to be judged in the same currency, **Elo at fixed time**, which combines accuracy per node with throughput. Sections 2–4 give the history, and Section 5 gives a model that converts speed into depth and then into Elo.

---

## 2. Foundations: game-theoretic value and minimax

### 2.1 Zermelo (1913)
- Ernst Zermelo, "Über eine Anwendung der Mengenlehre auf die Theorie des Schachspiels," *Proceedings of the Fifth International Congress of Mathematicians* (Cambridge 1912), vol. 2, pp. 501–504, Cambridge University Press, 1913.
- It is generally regarded as the first formal theorem of game theory. Modern restatements of what Zermelo proved vary. He showed, roughly, that in chess either one side can force a win or both sides can force at least a draw, and he studied how long a forced win can take. Schwalbe & Walker, "Zermelo and the Early History of Game Theory" (*Games and Economic Behavior* 34, 2001), includes an English translation and points out that the usual "Zermelo's theorem" (backward induction) is not exactly what the 1913 paper proves. Source: https://www.rasmusen.org/GI/reader/04a.zermelo.pdf
- Relevance: every evaluation function estimates this game value (or a win probability) where it cannot be computed.

### 2.2 von Neumann (1928)
- John von Neumann, "Zur Theorie der Gesellschaftsspiele," *Mathematische Annalen* 100 (1928), 295–320, doi:10.1007/BF01448847 (https://link.springer.com/article/10.1007/BF01448847). It proved the minimax theorem for two-person zero-sum games.
- Engine relevance: the minimax backup rule, applied to heuristic leaf values.

---

## 3. The classical era: evaluation functions plus brute-force search

### 3.1 Shannon (1950)
- Claude E. Shannon, "Programming a Computer for Playing Chess," *Philosophical Magazine*, Ser. 7, Vol. 41, No. 314, pp. 256–275, March 1950, doi:10.1080/14786445008521796. PDF: https://vision.unipv.it/IA1/ProgrammingaComputerforPlayingChess.pdf
- The first technical paper on computer chess. It was entirely theoretical.
- **Evaluation function.** Shannon proposed an approximate evaluation f(P) applied at the leaves of a minimax tree. His example combined material, pawn structure (doubled, isolated and backward pawns) and mobility.
- **Game-tree size.** He estimated about 10^3 possibilities for a pair of moves (one White, one Black) and a typical game of 40 such pairs, which gives about 10^120 variations. This is the "Shannon number."
- **Type A vs Type B.** Type A is full-width brute-force minimax to a fixed depth. Shannon judged it too slow: at one evaluation per microsecond, three moves deep needs about 10^9 evaluations, more than 16 minutes per move. Type B adds two refinements: (i) evaluate only at "quiescent" (stable) positions, which anticipates quiescence search, and (ii) select only plausible moves for deeper examination, which anticipates forward pruning.
- Relevance: Shannon already treated evaluation quality and search selectivity as coupled choices.

### 3.2 Turing and Champernowne's "paper machine" (Turochamp, 1948–1952)
- Alan Turing and David Champernowne wrote Turochamp in 1948. No computer could run it, so Turing executed the rules by hand. The program's game against Alick Glennie in 1952, with Turing computing each move, took up to about 30 minutes per move. Turochamp (White) lost in 29 moves. Turing described the rules in his chapter on digital computers applied to games in B. V. Bowden (ed.), *Faster Than Thought* (1953). Sources: https://en.wikipedia.org/wiki/Turochamp , https://www.chessprogramming.org/Turochamp
- The evaluation combined material with a positional score covering mobility, piece safety, king safety, castling and pawn advancement. Search was a shallow selective lookahead that continued "considerable" moves (captures, recaptures, checks) beyond the nominal depth, an early quiescence-style extension. (Feature details follow chessprogramming.org's summary of the rules.)
- Relevance: the earliest hand-crafted evaluation plus selective tactical search, at roughly one move per half hour.

### 3.3 Knuth & Moore (1975): alpha-beta analysis
- Donald E. Knuth and Ronald W. Moore, "An analysis of alpha-beta pruning," *Artificial Intelligence* 6(4):293–326, 1975, doi:10.1016/0004-3702(75)90019-3 (https://www.sciencedirect.com/science/article/abs/pii/0004370275900193).
- Alpha-beta predates the paper (McCarthy, Newell–Shaw–Simon, Brudno and others). Knuth & Moore gave the definitive analysis: correctness, an optimality argument, node types (later PV/CUT/ALL), and bounds under random data.
- **Best case.** With perfect move ordering, a uniform tree of branching factor b and depth d has a minimal tree of b^⌈d/2⌉ + b^⌊d/2⌋ − 1 leaves, i.e. **O(b^{d/2})**. Alpha-beta can therefore search about twice as deep as minimax in the same time, so the effective branching factor drops from b to about √b. (Formula also given at https://chessprogramming.org/Alpha-Beta .)
- Relevance: alpha-beta's savings depend on move ordering, which modern engines partly drive with static eval, so a better eval can also shrink the tree.

### 3.4 Iterative deepening
- Korf, "Depth-first iterative-deepening: An optimal admissible tree search," *Artificial Intelligence* 27(1):97–109, 1985, names Slate & Atkin's Chess 4.5 as "the first use of the algorithm that is documented in the literature": D. Slate and L. Atkin, "CHESS 4.5 – The Northwestern University Chess Program," in P. W. Frey (ed.), *Chess Skill in Man and Machine*, Springer, 1977. Others trace it to Scott (1969) and the idea of progressive deepening to de Groot (*Thought and Choice in Chess*). Source: https://chessprogramming.org/Iterative_Deepening
- In chess the purpose is not memory (as with IDA*). It gives time control (stop at any iteration) and, more importantly, move ordering: the principal variation and hash moves from iteration d−1 order iteration d, bringing alpha-beta close to its best case. The overhead of re-searching shallower depths is small when b_eff > 1. The geometric series gives a total cost factor of about b_eff/(b_eff−1) *(derived)*.

### 3.5 Transposition tables and Zobrist hashing (1970)
- Albert L. Zobrist, "A New Hashing Method With Application for Game Playing," Technical Report #88, Computer Sciences Dept., University of Wisconsin, Madison, April 1970 (scan at https://minds.wisconsin.edu/handle/1793/57624). Reprinted in *ICCA Journal* 13(2):69–73, 1990, doi:10.3233/ICG-1990-13203.
- Method: assign a random bitstring to every (piece, square) pair and XOR the strings for the occupied squares. A move updates the hash *incrementally* with h' = h ⊕ z[p, from] ⊕ z[p, to] (plus captures, castling, en passant, side to move). The report also treats detecting retrieval errors and controlling the error rate. In the hashing literature it is recognised as the origin of simple tabulation hashing. Source: https://chessprogramming.org/Zobrist_Hashing
- Transposition tables were used before Zobrist's paper (Greenblatt's Mac Hack, 1967, per van Rijswijck's Alberta thesis). Slate & Atkin (1977) showed they combine especially well with iterative deepening.
- **Parallel to NNUE.** Zobrist hashing is the first efficiently updatable position function. It is linear over GF(2) in sparse one-hot (piece, square) features and updated by adding and removing columns. NNUE's first layer does the same over the integers with (king-bucket, piece, square) features, and VNNUE extends this lineage.

### 3.6 Null-move pruning
- Early sources: J. Schaeffer, "Speculative Computing," *ICCA Journal* (1987), describes Beal's null move used non-recursively. D. F. Beal, "Experiments with the null move," *Advances in Computer Chess 5* (1989), pp. 65–79, later generalised as "A Generalized Quiescence Search Algorithm" (*Artificial Intelligence* 43, 1990). G. Goetsch and M. Campbell, "Experiments with the null-move heuristic" (AAAI Spring Symposium 1988, published 1990), already consider recursive use. C. Donninger, "Null move and deep search: Selective search heuristics for obtuse chess programs," *ICCA Journal* 16(3):137–143, 1993, popularised recursive null-move with an adaptive reduction R, after Frans Morsch's use in Fritz. Source: https://chessprogramming.org/Null_Move_Pruning
- Idea: give the opponent a free move and search to reduced depth d−1−R. If the result still fails high, prune. It rests on the null-move observation (in most non-zugzwang positions, moving is better than passing).
- David-Tabibi & Netanyahu ("Verified Null-Move Pruning," *ICGA Journal* 25(3):153–161, 2002, arXiv:0808.1125) state that null-move pruning in the early 1990s "ended the era in which brute-force programs dominated computer chess."
- Evaluation coupling: modern engines (including Stockfish) try a null move only when the static evaluation is at or above beta, among other conditions. Evaluation accuracy therefore directly controls how much of the tree null-move removes.

### 3.7 Late move reductions (LMR)
- LMR searches moves late in the ordering at reduced depth and re-searches at full depth only if they beat alpha. Reduction ideas are older, but chessprogramming.org states LMR "became very popular in 2005 after Fruit and Glaurung used open source implementations," based on history-heuristic ordering. Fabien Letouzey (Fruit) used history-based "history pruning", and Tord Romstad (Glaurung, the ancestor of Stockfish) wrote "An Introduction to Late Move Reductions." Sources: https://chessprogramming.org/Late_Move_Reductions , https://chessprogramming.org/Tord_Romstad
- A claim that LMR descends from the 1989 Levy–Broughton–Taylor "SEX" (Search EXtension) algorithm appears in secondary sources only. **[UNVERIFIED]**
- Combined selectivity pushes b_eff far below the alpha-beta optimum √35 ≈ 6 *(derived)*, to about 1.5–1.7 for Stockfish at depth 49 (Section 5.1).

### 3.8 Thompson's Belle depth-vs-strength experiments (1982/83)
- K. Thompson, "Computer Chess Strength," *Advances in Computer Chess 3* (ed. M. R. B. Clarke), Pergamon, 1982 (conference held 1981). J. H. Condon and K. Thompson, "BELLE," in *Chess Skill in Man and Machine* (2nd ed., 1983). Summarised by Ernst Heinz: https://people.csail.mit.edu/heinz/dt/node49.html and https://chessprogramming.org/Depth
- Self-play of Belle at depth n+1 against depth n. In the 1982 experiment (100 games, 20-game matches, depths about 3–8) the gain averaged **about 246 rating points per ply**, with no clear diminishing returns. The 1983 repeat gave **about 217 points per ply**. Heinz notes these results are not statistically significant. Comparable self-play figures: HITECH about 195/ply, LOTECH about 232/ply, THE TURK (Junghanns) about 200/ply.
- Later work found diminishing returns at higher depths. D. R. Ferreira, "The Impact of the Search Depth on Chess Playing Strength," *ICGA Journal* 36(2):67–80, 2013 (https://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf), using Houdini, reports about 66 Elo per additional ply near depth 20.
- Stockfish's documentation ("Elo gain from depth increase," https://official-stockfish.github.io/docs/stockfish-wiki/Useful-data.html) shows the same curve for a modern engine: about 170 Elo (depth 2 vs 1), a peak of about 191 (8 vs 7), about 162 (10 vs 9), about 89 (12 vs 11), about 67 (15 vs 14), about 54 (16 vs 15) and about 52 ± 8 (18 vs 17).
- Relevance: a ply was once worth 200+ Elo and is now worth about 50 at moderate depth, which favours spending time on evaluation.

### 3.9 Deep Blue (Campbell, Hoane, Hsu 2002)
- M. Campbell, A. J. Hoane Jr., F.-h. Hsu, "Deep Blue," *Artificial Intelligence* 134(1–2):57–83, 2002, doi:10.1016/S0004-3702(01)00129-1.
- 1997 system: a 30-node IBM RS/6000 SP. Each node controlled 16 custom chess chips, 480 chips in all. Each chip searched about 2–2.5 million positions per second. In the 1997 Kasparov match the system averaged **about 100 million positions/s**, with about 200M/s typical in quiet positions and a peak around 330M/s. Software search ran the top of the tree, and the chips searched the last ~4 plies plus quiescence. Nominal depth was about 12 plies, with selective extensions (e.g. singular extensions) along forcing lines reaching much deeper (chessprogramming.org cites about 40 plies). Source: https://chessprogramming.org/Deep_Blue
- **Evaluation in hardware.** The chip evaluation was a weighted sum of about **8000 features**, up from about 6400 in the 1996 chip. It was split into a "fast evaluation" and a "slow evaluation" because the full evaluation was too large to compute fully in parallel. The authors say they "spent the vast majority of our time between the two matches designing, testing, and tuning the new evaluation function." Weights were hand-tuned with grandmaster input.
- Relevance: even this brute-force machine invested heavily in evaluation, and fast/slow evaluation is an early form of lazy evaluation.

---

## 4. The NNUE era: making a big evaluation cheap

### 4.1 NNUE origin
- Yu Nasu, "Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi," Ziosoft Computer Shogi Club, appeal document for the 28th World Computer Shogi Championship, April 2018 (Japanese; English translation by Dominik Klein at https://github.com/asdfjkl/nnue). First used in YaneuraOu (shogi), then ported to Stockfish. Source: https://chessprogramming.org/NNUE
- Core principles: (i) sparse, overparameterised inputs (king-relative piece-square features) where a move changes only a few features, (ii) a first layer updated incrementally by adding and subtracting weight columns, (iii) a small, quantised integer network after the first layer, evaluated with SIMD.

### 4.2 Elo gain from introducing NNUE into Stockfish (verified, with ranges)
- The "Introducing NNUE Evaluation" post (7 Aug 2020) reported **>80 Elo on fishtest**, with individual tests of about **+93 Elo** (fast games, 1 thread) and **+90 Elo** (8 threads). (As reported by Chess.com, https://www.chess.com/news/view/stockfish-12-released-130-elo-points-stronger .)
- Stockfish 12 (2 Sep 2020), the first release with NNUE, was estimated at **about 130 Elo** stronger than Stockfish 11 (Chess.com, citing the official blog). The official blog put it as "Stockfish 12 will typically win at least 10 times more game pairs than it loses" against SF11. The ~130 figure includes non-NNUE improvements made between SF11 and SF12. Lichess's blog says ">100 Elo."
- **Suggested wording:** NNUE gained about 80–95 Elo on fishtest at merge (Aug 2020), and SF12 overall was about 130 Elo above SF11.

### 4.3 Why incremental updates matter: Stockfish's own measurement
- The Stockfish wiki (Useful data, section "The impact of efficient (incremental) updates (NNUE)") measured **1,287,575 nps with incremental updates vs 696,064 nps without**, i.e. disabling them costs about 46% of speed, worth **−67.55 ± 9.5 Elo at LTC on fishtest** (UHO book). Source: https://official-stockfish.github.io/docs/stockfish-wiki/Useful-data.html
- This is the single most relevant number for VNNUE. *The "efficiently updatable" part of NNUE alone is worth about 70 Elo at LTC*, comparable to the whole classical→NNUE gain. Any successor architecture has to keep or improve this per-node cost.
- The last classical Stockfish (July 2020) ran at 2.30 Mnps on the same reference machine. That early NNUE was slower is plausible but not read from the graph **[UNVERIFIED]**.

### 4.4 Accumulator stack, lazy updates and refresh caches in Stockfish
- **Per-position accumulator on the search stack.** The nnue-pytorch documentation (https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html) describes keeping the first-layer output (the "accumulator") as part of the position state. When a feature is removed, the matching weight column is subtracted, and when one is added, it is added. The accumulator is not updated on unmake, because the parent's copy is still on the stack. White's and Black's perspectives are separate halves.
- **Lazy vs eager.** The same document: "The accumulator can either be updated lazily on evaluation, or on each move," and "whether it's better to update lazily or eagerly depends on the number of evaluations done during search." In the 2020 code, `do_move()` cleared a `computed_accumulation` flag and the update ran only when evaluation was requested. It walked back to the nearest computed ancestor, or the last position before a king move that forced a refresh (commit 9b49670 "Remove EvalList"; PR #3156 "Use incremental updates more," which extends the look-back). Current code has an `AccumulatorStack` with a comment saying the update will "Find the earliest usable accumulator, this can either be a computed accumulator or the accumulator state just before a change that requires full refresh." The exact date the `AccumulatorStack` refactor was merged **[UNVERIFIED]**.
- **King moves force refreshes.** All features are king-relative, so a king move (in that perspective) changes every feature: "This makes king moves more costly but on average it still keeps the number of updates per evaluation low" (nnue-pytorch docs).
- **Finny tables (refresh cache).** Stockfish commit 49ef4c9, "Implement accumulator refresh table" (gab8192, 24 Apr 2024), keeps a per-thread cache with one entry per king square/perspective. On refresh it diffs the cached bitboards against the current position and applies only the difference. The commit credits the idea to Luecx (Koivisto), "commonly referred to as 'Finny Tables'." Source: https://github.com/official-stockfish/Stockfish/commit/49ef4c935a5cb0e4d94096e6354caa06b36b3e3c . Fishtest Elo of that commit: not retrieved **[UNVERIFIED]**. For comparison, a third-party engine (sleepmind2, PR #30) reported +10–35% nps and +26.5 Elo STC / +11.6 Elo LTC from adding Finny tables.
- Implication: per-eval cost ≈ (changed features) × (accumulator width) plus refresh frequency. VNNUE updates must compose across plies (lazy look-back) and support cheap refresh from a cache.

### 4.5 What fraction of nodes are evaluated?
- No authoritative published statistic for Stockfish was found. **[UNVERIFIED]** for any precise figure.
- Qualitative facts: (i) quiescence-search nodes need a stand-pat static evaluation, and (ii) Stockfish's main search computes static evaluation at most interior nodes (unless in check or a TT entry supplies it), because pruning decisions (razoring, futility, null-move conditions, the "improving" flag) depend on it. So a large fraction of visited nodes trigger an evaluation, though transposition-table hits and in-check nodes avoid some.
- Share of nodes in quiescence search: Feng-hsiung Hsu (rec.games.chess.computer) said that simple capture quiescence roughly doubles the node count, i.e. about half of nodes are in qsearch. Forum and hobby-engine reports range from about 30–50% (well-pruned) to 85%+ (poorly pruned or naive engines). These are weakly sourced; treat as **[UNVERIFIED]** ranges.
- Recommendation: instrument Stockfish to count `evaluate()` calls, accumulator updates and refreshes per node at STC/LTC.

---

## 5. Quantitative model: speed → depth → Elo

### 5.1 Speed to depth
Let the search tree to nominal depth d cost N(d) ≈ K · b_eff^d nodes, where b_eff is the effective branching factor (the ratio N(d+1)/N(d) under iterative deepening). With a fixed time budget T and node rate v, the reachable depth is d = log_{b_eff}(vT/K). A speedup by factor c (nps × c, or equivalently time × c) gives

  **Δd = log_{b_eff}(c) = ln c / ln b_eff.**

Worked values *(derived)*:

| b_eff | c = 2 (doubling) | c = 1.1 (10% speedup) | c = 0.54 (no incremental update) |
|---|---|---|---|
| 6 (alpha-beta best case, b ≈ 35) | 0.39 ply | 0.053 ply | −0.34 ply |
| 2.0 | 1.00 ply | 0.14 ply | −0.89 ply |
| 1.7 (≈ SF10 at depth 49) | 1.31 ply | 0.18 ply | −1.16 ply |
| 1.5 (≈ SF17 at depth 49) | 1.71 ply | 0.24 ply | −1.52 ply |

Stockfish's branching factor (Useful data, "Branching factor of Stockfish," defined by nodes = B_f^rootDepth from startpos): SF10 needed 1.206 × 10^11 nodes to reach depth 49, and SF17 3.567 × 10^8. These correspond to B_f ≈ 1.68 and ≈ 1.49 respectively *(derived)*. B_f falls with depth and with newer versions. Caveat: nominal "depth" in a heavily reduced search is not comparable across engines or versions, so Δd is meaningful only within one engine.

Then **ΔElo ≈ Δd × (Elo per ply at the operating depth)**. From Section 3.8, Elo per ply ranges from about 200+ at shallow depth to about 50 at depth 16–18 for Stockfish. Combining b_eff ≈ 1.5–1.7 with about 50–70 Elo/ply at depth ~16–18 gives about 65–120 Elo per doubling *(derived)*. That is roughly consistent with the direct measurements below, but the composition is only a heuristic. Both factors depend on depth and time control.

### 5.2 Elo per doubling of speed: measured data, ranges and caveats
Direct measurements are better than the composed model:

1. **Stockfish wiki, LTC 8× time.** 1 thread at 480+4.8 vs 60+0.6: **+217.9 ± 15.8 Elo**, about **73 Elo per doubling** in that TC range *(derived: 217.9/3)*.
2. **Stockfish wiki, small speedups (linear model, valid for speedups below ~5%).** Elo_STC(x) = 2.10·x and Elo_LTC(x) = 1.43·x, with x the speedup in percent. Raw data: 16% speedup → +32.4 Elo at 10+0.1 and +20.9 Elo at 60+0.6. "A 1% speedup has nearly 85% passing chance at LTC." *(Extrapolating the linear model to 100% is invalid. Use ln-scaling: 16% = 0.214 doublings, so about 150 Elo/doubling-equivalent at STC and about 97 at LTC locally (derived). These local slopes are higher than the 8× figure, which reflects diminishing returns and opening-book effects.)*
3. **Stockfish wiki, 20% time odds.** About 28–31 Elo for SF15–SF17.1 (e.g. sf17.1: 28.95 ± 3.95). The TC is listed on the page, and at 5+0.05 UHO SF15 shows 24.5. 20% = 0.263 doublings, so about 95–120 Elo per doubling-equivalent locally *(derived)*.
4. **Houdini 3 self-play** (chessprogramming.org, Match Statistics): Elo per doubling fell steadily from **144** (20+0.2 vs 10+0.1) through 133, 112, 101, 93, 73 and 59 to **51** (2560+25.6 vs 1280+12.8).
5. **Older rule of thumb:** about **50–70 Elo per doubling** for engines circa 2010–2016 at long TCs. One SF8 test (40/4′ vs 40/2′) gave about 69 Elo (TalkChess/CCC).
6. **Extreme draw saturation:** a May 2026 TalkChess thread ("Time doubling gives 3 elo for Stockfish now") reported about +2.8 Elo per doubling for a 1000:1 handicap from the balanced start position, and about +6 per doubling (16:1, 8-move book) in a reply. These are forum self-play results with unclear statistics. **[UNVERIFIED as rigorous]**, but they show how strongly the measured value depends on the opening book and draw rate.

**Recommended statement for the paper:** *For modern Stockfish, a doubling of effective speed is worth roughly 70–150 Elo in fishtest-style self-play with an unbalanced (UHO) book, higher at short TCs and lower at long TCs. In balanced-opening self-play at long time controls the gain falls much further (Houdini 3: 144 → 51 over eight doublings; recent forum reports of under 10 Elo at extreme TCs) because of draw saturation. Small speedups (≤5%) are worth about 2.1 Elo/% at STC and about 1.4 Elo/% at LTC (Stockfish wiki).*

Caveats to state explicitly:
- **Diminishing returns.** Elo per ply and Elo per doubling both fall with depth and TC (Thompson's lack of diminishing returns was a low-depth artefact and statistically weak).
- **Draw saturation and book dependence.** Elo differences are compressed when most games are drawn. Fishtest uses unbalanced (UHO) books partly to counter this, and "normalized game-pair Elo" is an alternative. Note that the Stockfish wiki also states "the common wisdom that increased TC causes elo compression is not always true" (SF7→8, SF8→9 gained *more* Elo at LTC).
- **Self-play inflation.** Self-play Elo differences typically exceed those measured against diverse opponents (e.g. CCRL lists) **[UNVERIFIED magnitude]**.
- **Hardware dependence.** The speedup model is hardware-dependent (SF wiki).
- **Eval-cost vs nps.** For VNNUE, a speedup must be measured as end-to-end nps in real search (including lazy-update savings and cache effects), not as isolated `evaluate()` microbenchmarks.

### 5.3 Evaluation accuracy vs speed: the break-even
Combining the pieces: a new evaluation that costs a factor c_eval more time per node (c = 1/c_eval) pays off only if its accuracy gain per node exceeds the speed penalty, roughly **ΔElo_accuracy > S · log2(c_eval)**, where S is Elo per doubling at the target TC (S ≈ 70–150 from §5.2) *(derived)*. Examples:
- NNUE in 2020 was slower in nps than classical SF (2.30 Mnps classical; early NNUE lower, exact ratio on the same machine not retrieved **[UNVERIFIED]**), yet gained about 80–95 Elo net. The accuracy gain was therefore larger than 80–95 Elo plus the speed loss.
- Removing incremental updates (c ≈ 0.54, i.e. 0.89 doublings lost) costs about 68 Elo at LTC. Applying S ≈ 76 Elo/doubling is consistent with the 8×-time figure (~73) *(derived)*, a useful internal consistency check of the model.
- For VNNUE: if eval/update is a share r of node time and VNNUE cuts it by a fraction f, then c = 1/(1−f·r) and ΔElo ≈ S·log2(c) at equal accuracy. Measure r on target hardware; no authoritative published profile was found **[UNVERIFIED]**.

---

## References

Full bibliographic details (venue, volume, pages, DOI or URL) appear inline at first citation in Sections 2–5. Primary sources: Zermelo 1913; von Neumann 1928; Shannon 1950; Turing 1953 (in Bowden, *Faster Than Thought*); Zobrist 1970 (UW TR 88; ICCA J. 13(2), 1990); Knuth & Moore 1975; Slate & Atkin 1977; Thompson 1982; Condon & Thompson 1983; Korf 1985; Beal 1989; Goetsch & Campbell 1990; Donninger 1993; David-Tabibi & Netanyahu 2002; Campbell, Hoane & Hsu 2002; Ferreira 2013; Nasu 2018. Engine and data sources: Stockfish wiki "Useful data"; nnue-pytorch NNUE docs; Stockfish commits 9b49670 and 49ef4c9, PR #3156; chessprogramming.org (Alpha-Beta, Depth, Deep Blue, Iterative Deepening, LMR, Match Statistics, NNUE, Null Move Pruning, Turochamp, Zobrist Hashing); Heinz's depth-strength summary (people.csail.mit.edu/heinz/dt/node49.html); Chess.com and Lichess Stockfish 12 announcements.
