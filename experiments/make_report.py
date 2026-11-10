"""Build figures/*.png and RESULTS.md from results/*.json (no numbers typed by hand)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = BASE + "/results", BASE + "/figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def load(n):
    p = f"{RES}/{n}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, d=3):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{d}g}" if abs(x) < 1 else f"{x:.{d}f}".rstrip("0").rstrip(".")


def pct(x, d=2):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def dstr(d, k=("mean", "p50", "p95")):
    if not d or d.get("n", 0) == 0:
        return "n/a"
    return " / ".join(f(d[x]) for x in k)


md = []
W = md.append

e1, e2, tr, be = load("e1_exactness"), load("e2_e4_opcounts"), load("e5_e8_training"), load("bench")
meta = json.load(open(BASE + "/data/gen_meta.json"))

W("# VNNUE prototype experiments: results\n")
W("All numbers below come from runs of the scripts in `experiments/`. The tables are generated mechanically from "
  "`results/*.json` by `make_report.py`; the Key findings section was written by hand from those tables (text in `findings_text.py`). Seeds are fixed in every script. Machine: 4-core Windows laptop CPU, "
  "Python 3.14, numpy 2.4, torch 2.10 (CPU), python-chess 1.11.\n")
W("**Scope warning.** This is a Python prototype. Op and byte counts are exact counts under an explicit cost "
  "model; wall-clock numbers are numpy/torch proxies and are not representative of SIMD C++. The search traces "
  "come from a toy alpha-beta searcher, not Stockfish. No statement here is about playing strength, Elo or nps.\n")

W("## 0. Data\n")
W("| Source | Content | Size |\n|---|---|---|")
W(f"| Lichess standard rated games, 2013-01 (CC0, `database.lichess.org`), first {meta['lichess_games']:,} standard games | real human move sequences (+ game results) | {meta['lichess_nodes']:,} positions; {meta['positions']:,} training positions (ply >= 8) |")
W(f"| Self-play | depth-2 alpha-beta + quiescence, material+PST eval, 8 random opening plies, 30 games | {meta['selfplay_nodes']:,} positions |")
W(f"| Search traces | {meta['tree_roots']} roots (160 from Lichess games, 40 from self-play); alpha-beta depth {meta['search']['depth']} + capture quiescence (max {meta['search']['qs_max']} plies), MVV-LVA ordering; each tree capped at {meta['search']['node_cap']:,} nodes ({meta['tree_aborted_at_cap']} trees hit the cap) | {meta['tree_nodes']:,} make events |")
W("\nTrees are stored as DFS pre-order node lists with parent pointers, which is exactly the make/unmake event stream "
  "of an accumulator stack. Evaluation modes: **eager** = every node evaluated (like Stockfish's static eval at almost "
  "every node); **lazy** = roots, quiescence/leaf nodes always, interior nodes with probability 0.3 (seeded).\n")
W("**Feature model (simplified K32H3).** 32 own-king buckets with horizontal mirroring (king on files e-h), "
  "3 material blocks (0: no rook/queen on board; 1: some R/Q but opponent has no queen; 2: opponent has a queen), "
  "11 planes x 64 (both kings share one plane) = 704 local features, 67,584 per perspective. "
  "**Not modelled inside the accumulator:** Full_Threats and PP_3Wide columns, PSQT outputs, layer stacks. "
  "A threat *proxy* (attacker-target piece pairs with Full_Threats-style dedup, absolute frame) is counted for byte "
  "accounting on game sequences only. Stockfish's backward catch-up fill after a Finny refresh is not modelled.\n")

# ---------------------------------------------------------------- E1
if e1:
    s = e1["summary"]
    W("## 1. E1 Exactness\n")
    W(f"Prototype with L = {e1['config']['L']} int16 lanes, PSI r = {e1['config']['r']}, m = {e1['config']['m']}, "
      f"C2 rank r_K = {e1['config']['rK']}. The K32H3 table is the *materialised merged* C2 table "
      "W[blk,kappa] = W0[blk] + U[blk,kappa] V_K computed modulo 2^16, so the Woodbury identity is tested on the same table "
      "every other source uses. At every accumulator build, **every available source** (refresh, Finny, hybrid, SF chain "
      "catch-up with two-ply cancellation, VNNUE ancestor net-delta, Woodbury transition) is computed with numpy int16 "
      "wrapping adds and compared bit-for-bit with the from-scratch refresh. The chosen source's value is then stored and "
      "reused downstream (Finny entries, chain intermediates, stack ancestors), so errors would propagate. PSI checks run at "
      "every make event: P1 (int16) and P2 (int32) vs full recomputation; P1^2 - P2 even; incrementally maintained E (T14) vs "
      "(P1^2 - P2)/2; the multi-head readout sum_t Lam_ht E_t vs an explicit brute-force sum over all present piece pairs (T13); "
      "the Black view from the colour-flipped board vs the lane-permuted White state under tying (T21); the C2 state y = V_K x "
      "maintained incrementally vs recomputed. Also, incremental absolute deltas vs set differences.\n")
    W("Two weight regimes: **normal** (small weights, accumulators mostly in range) and **stress** (|W0| <= 6000, "
      "|U| <= 300, |V_K| <= 127, |b| <= 20000), where partial sums regularly leave the int16 range.\n")
    W("| Total | Value |\n|---|---|")
    W(f"| accumulator candidate checks (source value == refresh, bit-exact) | {f(s['candidate_checks'])} |")
    W(f"| mismatches | **{f(s['candidate_mismatches'])}** |")
    W(f"| builds where some lane decoded != exact integer although the exact final value was in int16 range | **{f(s['decode_mismatch_events'])}** |")
    W(f"| (candidate, lane) results whose update path had a partial sum outside int16 but whose exact final value was in range, all decoded exactly | {f(s['lanes_exact_despite_intermediate_overflow'])} |")
    W(f"| PSI / y / feature-delta checks | {f(s['psi_and_feature_checks'])} |")
    W(f"| PSI / y / feature-delta mismatches | **{f(s['psi_mismatches'])}** |")
    W("\nPer run (policy = which source is actually *used* downstream; `VN mu=0` makes Woodbury artificially cheap so that "
      "its outputs are reused often):\n")
    W("**Subsets.** E1 runs on subsets (constants in `e1_exactness.py`): the first 300 Lichess games, all 30 self-play games, "
      "the first 15 search roots (eager) and the first 40 search roots (lazy) of the 200; `nodes` below gives the resulting size. "
      "Under stress weights most *final* lane values are themselves outside int16 (`final lanes out of range`); for those lanes "
      "bit-exactness means agreement modulo 2^16 with the refresh, not a correct int16 decode.\n")
    W("| dataset | eval | weights | policy | nodes | builds | sources used | checks | mismatches | (candidate, lane) results exact despite partial sums outside int16 | final lanes in int16 range | final lanes out of range |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in e1["runs"]:
        ch = sum(v["checks"] for v in r["candidate_checks"].values())
        mm = sum(v["mismatches"] for v in r["candidate_checks"].values())
        srcs = ", ".join(f"{k} {v}" for k, v in sorted(r["chosen"].items(), key=lambda kv: -kv[1]))
        W(f"| {r['dataset']} | {r['eval_mode']} | {'stress' if r['stress_weights'] else 'normal'} | {r['policy']} mu={r['mu']:g} | "
          f"{f(r['nodes'])} | {f(r['builds'])} | {srcs} | {f(ch)} | {mm} | {f(r['lane_results_exact_despite_intermediate_overflow'])} | "
          f"{f(r['lanes_final_in_range'])} | {f(r['lanes_final_out_of_range'])} |")
    W("\n**Saturation is path-dependent (T8), empirically.** Along the first 200 Lichess games, each perspective's "
      "accumulator was updated incrementally with (a) wrapping int16 adds and (b) saturating adds (`paddsw` semantics), and "
      "compared with the same arithmetic applied from scratch (bias + columns in sorted order):\n")
    W("| weights | accumulators compared | wrapping mismatches | saturating mismatches (accumulators) | saturating mismatching lanes | of which the exact final value was in range |")
    W("|---|---|---|---|---|---|")
    for k, v in e1["saturation"].items():
        W(f"| {k} | {f(v['positions'])} | {v['wrap_mismatch_positions']} | {f(v['sat_mismatch_positions'])} | {f(v['sat_mismatch_lanes'])} | {f(v['sat_mismatch_lanes_final_in_range'])} |")
    W("")

# ---------------------------------------------------------------- E2-E4
if e2:
    cfg = e2["config"]
    W("## 2. E2-E4 Op counts, bytes and source selection\n")
    W(f"Cost model (SPEC 3.1): L = {cfg['L']}; col = one int16 column ({cfg['bytes_col']} B); pass = accumulator load+store "
      f"({cfg['bytes_pass']} B); PSI = {cfg['psi_bytes_per_changed_abs_feature']} B and 2r = 64 lanes per changed absolute feature "
      "(r = 32), applied eagerly at every make event; threat proxy columns are i8 (1 KB). Argmin scalar = "
      f"`{cfg['cost_scalar']}`. Woodbury cost = `{cfg['woodbury_cost']}`.\n")
    W("Policies: **SF** = Stockfish-like rule (chain catch-up if no table change, else hybrid if its conditions hold, else Finny); "
      "**SFmin** = argmin over SF's set {refresh, Finny, hybrid}; **VN_noC4** = argmin over {refresh, Finny, hybrid, chain}; "
      "**VN** = VN_noC4 + ancestor net-delta (C4). The C2 Woodbury candidate is evaluated separately (sweep below).\n")
    D = e2["datasets"]
    W("### 2.1 Move statistics\n")
    W("| dataset | positions | quiet | capture | king quiet | king capture | castling | en passant | promotion | capture-promotion | block-change moves | k_a mean | |S| mean / p50 / p95 |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, R in D.items():
        if k.endswith("lazy"):
            continue
        mf = R["move_type_fraction"]
        g = lambda x: pct(mf.get(x, 0.0), 1)
        W(f"| {k} | {f(R['nodes'])} | {g('quiet')} | {g('capture')} | {g('king_quiet')} | {g('king_capture')} | {g('castling')} | "
          f"{g('en_passant')} | {g('promotion')} | {g('capture_promotion')} | {pct(R['block_change_fraction_of_moves'], 2)} | "
          f"{f(R['k_abs']['mean'])} | {dstr(R['S_size'])} |")
    W("\nAccumulator build events (one per evaluated node and perspective not already computed), SF policy run:\n")
    W("| dataset | builds | builds needing a table change (king bucket / mirror / block) | of which own-king moves | SF Finny cols at table change (mean / p50 / p95) | SF hybrid cols (mean / p50 / p95) | hybrid admissible at own-king events |")
    W("|---|---|---|---|---|---|---|")
    for k, R in D.items():
        b = R["build_events"]
        W(f"| {k} | {f(b['total'])} | {f(b['table_change'])} ({pct(b['table_change_fraction'])}) | {f(b['own_king_table_change'])} | "
          f"{dstr(R['SF_finny_cost_at_table_change'])} | {dstr(R['SF_hybrid_cost'])} | {pct(R['SF_hybrid_available_fraction_of_own_king_events'], 1)} |")
    W("\n### 2.2 Per move type, Lichess games (eager evaluation): NNUE vs VNNUE per-move cost\n")
    W("Columns and bytes are per move, both perspectives, mean / p50 / p95. The PSI overhead ratios are totals over the class.\n")
    W("| move type | count | SF cols | VN cols | SF bytes | PSI lanes | PSI bytes | PSI / SF PSQ column bytes | PSI / SF PSQ bytes incl. passes |")
    W("|---|---|---|---|---|---|---|---|---|")
    mt = D["games_lichess/eager"]["per_move_type"]
    for k, r in mt.items():
        W(f"| {k} | {f(r['count'])} | {dstr(r['SF_cols'])} | {dstr(r['VN_cols'])} | {dstr(r['SF_bytes'])} | {dstr(r['psi_lanes'])} | "
          f"{dstr(r['psi_bytes'])} | {pct(r['psi_bytes_over_SF_psq_col_bytes_mean_ratio'])} | {pct(r['psi_bytes_over_SF_psq_bytes_mean_ratio'])} | "
          "")
    tp = load("e2_threat_proxy")
    if tp:
        W(f"\nThreat proxy (first {tp['games']} Lichess games, {f(tp['positions'])} positions, where it was computed): changed threat-proxy "
          f"features per move (absolute frame, per perspective) mean / p50 / p95 = {dstr(tp['overall']['threat_proxy_changes_abs'])}, i.e. "
          f"{tp['overall']['threat_over_psi_feature_changes']:.2f}x the number of changed piece-square features.\n")
        W("| move type | threat-proxy changes (mean / p50 / p95) | PSI / SF PSQ column bytes | PSI / (SF PSQ incl. passes + threat proxy, both perspectives) bytes |")
        W("|---|---|---|---|")
        for k, r in tp["per_move_type"].items():
            W(f"| {k} | {dstr(r['threat_proxy_changes_abs'])} | {pct(r['psi_over_psq_col_bytes'])} | {pct(r['psi_over_psq_plus_threat_bytes'])} |")
    W("\nOverall PSI byte overhead (H1; the SPEC T25 bound is 4.7% of PSQ column bytes when every move costs >= 4kL bytes):\n")
    W("| dataset | PSI / PSQ column bytes | PSI / PSQ bytes incl. passes | PSI / (PSQ + threat proxy) bytes |")
    W("|---|---|---|---|")
    for k, R in D.items():
        o = R.get("psi_overhead_overall")
        if o:
            W(f"| {k} | {pct(o['psi_over_psq_col_bytes'])} | {pct(o['psi_over_psq_bytes_incl_passes'])} | {pct(o['psi_over_psq_plus_threat_bytes_threat_subset'])} |")
    W("\n### 2.3 Whole-trace totals per policy (stateful simulation)\n")
    W("Each policy is simulated independently over the whole event stream (its own Finny caches and materialised "
      "accumulators). `PSI/FT bytes` divides the PSI bytes (eager, every make event) by that policy's FT bytes.\n")
    W("| dataset | policy | cols / eval | passes / eval | FT bytes / eval | total cols vs SF | total passes vs SF | PSI / FT bytes | sources used |")
    W("|---|---|---|---|---|---|---|---|---|")
    for k, R in D.items():
        sf = R["policies"]["SF"]
        for p, d in R["policies"].items():
            srcs = ", ".join(f"{a} {b}" for a, b in sorted(d["sources"].items(), key=lambda kv: -kv[1]))
            W(f"| {k} | {p} | {f(d['cols_per_eval'])} | {f(d['passes_per_eval'])} | {f(d['ft_bytes_per_eval'])} | "
              f"{d['cols'] / sf['cols']:.4f} | {d['passes'] / sf['passes']:.4f} | {pct(d['psi_bytes_over_ft_bytes'])} | {srcs} |")
    W("\nPer-event counterfactual (T24 sanity check): at every build event of each run, the minimum over the full VNNUE "
      "candidate set was <= the minimum over SF's set:\n")
    W("| dataset | policy | full-set min <= SF-set min at every event | summed per-event saving (col-equivalents) | as a fraction of SF-set cost |")
    W("|---|---|---|---|---|")
    for k, R in D.items():
        for p in ("VN_noC4", "VN"):
            d = R["policies"][p]
            W(f"| {k} | {p} | {d['counterfactual_full_le_sf_all']} | {f(d['counterfactual_saving_colequiv'])} | "
              f"{pct(d['counterfactual_saving_colequiv'] / d['counterfactual_sf_colequiv'])} |")
    W("\n### 2.4 Net-delta catch-up (C4, E3)\n")
    W("| dataset | catch-up events with chain length >= 2 (VN run) | net support / sum of per-ply supports (mean / p50 / p95) | column saving from cancellation | SF chain length (mean / p50 / p95) |")
    W("|---|---|---|---|---|")
    for k, R in D.items():
        W(f"| {k} | {f(R['catchup_events_len_ge2'])} | {dstr(R.get('cancellation_ratio_len_ge2'))} | "
          f"{pct(R.get('cancellation_col_saving_fraction_len_ge2'))} | {dstr(R['SF_chain_len'])} |")
    W("\n### 2.5 Low-rank king transition (C2 Woodbury) as a C3 candidate (E4)\n")
    W("Fraction of own-king table-change events where Woodbury is the argmin, VN policy + Woodbury candidate. "
      "`wb_mult = 2`: (U_k' - U_k) y formed on the fly (2 r_K mcol); `wb_mult = 1`: precomputed dU. Total cols/mcols are "
      "relative to the VN run without Woodbury.\n")
    W("`col-equiv vs VN` = (cols + mu*mcols) / VN cols; `FT bytes vs VN` counts cols, mcols (2 KB each) and passes. "
      "The Woodbury sweep was run on games_lichess/eager, trees/eager and trees/lazy only (not on self-play).\n")
    W("| dataset | r_K | mu | wb_mult | Woodbury chosen at own-king events | total cols vs VN | total mcols | col-equiv vs VN | FT bytes / eval | FT bytes vs VN | C2 y-maintenance lanes per make |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, R in D.items():
        if "woodbury_sweep" not in R:
            continue
        base = R["policies"]["VN"]["cols"]
        for kk, d in R["woodbury_sweep"].items():
            rk, mu, wbm = [x.split("=")[1] for x in kk.split(",")]
            yl = d.get("c2_y_lanes_total")
            vn = R["policies"]["VN"]
            W(f"| {k} | {rk} | {mu} | {wbm} | {pct(d['woodbury_chosen_fraction_of_own_king_events'], 1)} | {d['cols'] / base:.4f} | "
              f"{f(d['mcols'])} | {(d['cols'] + float(mu) * d['mcols']) / base:.4f} | {d['ft_bytes_per_eval']:,.1f} | "
              f"{d['ft_bytes'] / vn['ft_bytes']:.4f} | {f(yl / R['nodes']) if yl else 'n/a'} |")
    W("")
    cr = load("e3_cache_reset")
    if cr:
        W("### 2.7 Robustness: Finny cache cleared at every root (E3/E4 rerun)\n")
        W("In sections 2.2-2.5 (`e2_e4_opcounts.py`) one Finny cache is kept for the whole dataset, so entries carry over "
          "between unrelated games and trees and are *stale* (they are updated only when a Finny/refresh/hybrid source is "
          "used). Stockfish clears its refresh table on `ucinewgame` and warms it during search. `e3_cache_reset.py` reruns the "
          "policies with the cache cleared at every root (each game, each search tree), so it starts empty (= refresh cost) "
          "and is warmed by the run itself. Everything else is identical.\n")
        W("| dataset | policy | cols / eval | passes / eval | FT bytes / eval | total cols vs SF | FT bytes vs SF | per-event saving vs SF-set min | sources used |")
        W("|---|---|---|---|---|---|---|---|---|")
        for k, R in cr["datasets"].items():
            sf = R["policies"]["SF"]
            for p, d in R["policies"].items():
                srcs = ", ".join(f"{a} {b}" for a, b in sorted(d["sources"].items(), key=lambda kv: -kv[1]))
                cf = pct(d["counterfactual_saving_colequiv"] / d["counterfactual_sf_colequiv"]) if p.startswith("VN") else ""
                W(f"| {k} | {p} | {f(d['cols_per_eval'])} | {f(d['passes_per_eval'])} | {f(d['ft_bytes_per_eval'])} | "
                  f"{d['cols'] / sf['cols']:.4f} | {d['ft_bytes'] / sf['ft_bytes']:.4f} | {cf} | {srcs} |")
        W("\n| dataset | SF Finny cols at table change (mean / p50 / p95) | SF hybrid cols (mean / p50 / p95) |\n|---|---|---|")
        for k, R in cr["datasets"].items():
            W(f"| {k} | {dstr(R['SF_finny_cost_at_table_change'])} | {dstr(R['SF_hybrid_cost'])} |")
        mtc = cr["datasets"]["games_lichess/eager"].get("per_move_type")
        if mtc:
            W("\nPer move type, Lichess games, cache cleared per game (mean cols and FT bytes per move, both perspectives):\n")
            W("| move type | count | SF cols | VN cols | SF bytes | VN bytes |\n|---|---|---|---|---|---|")
            for k, r in mtc.items():
                W(f"| {k} | {f(r['count'])} | {f(r['SF_cols_mean'])} | {f(r['VN_cols_mean'])} | {f(r['SF_bytes_mean'])} | {f(r['VN_bytes_mean'])} |")
        W("\nWoodbury subset with the cache cleared per root (relative to the VN run of this section):\n")
        W("| dataset | r_K | mu | wb_mult | Woodbury chosen at own-king events | total cols vs VN | total mcols | col-equiv vs VN | FT bytes vs VN |")
        W("|---|---|---|---|---|---|---|---|---|")
        for k, R in cr["datasets"].items():
            vn = R["policies"]["VN"]
            for kk, d in R.get("woodbury_subset", {}).items():
                rk, mu, wbm = [x.split("=")[1] for x in kk.split(",")]
                W(f"| {k} | {rk} | {mu} | {wbm} | {pct(d['woodbury_chosen_fraction_of_own_king_events'], 1)} | "
                  f"{d['cols'] / vn['cols']:.4f} | {f(d['mcols'])} | {(d['cols'] + float(mu) * d['mcols']) / vn['cols']:.4f} | "
                  f"{d['ft_bytes'] / vn['ft_bytes']:.4f} |")
        W("")

# ---------------------------------------------------------------- bench
if be:
    W("### 2.6 Wall-clock microbenchmark (rough proxy only)\n")
    W(f"> {be['caveat']} Table = {be['table_MB']:.1f} MB int16 (67,584 x 1024). These timings were taken with no other experiment running on a "
      "4-core, 4 GB RAM laptop and include Python interpreter overhead; they must not be read as engine speed.\n")
    W("| kernel (unbatched, one numpy call sequence per event) | ns / event |\n|---|---|")
    for k, v in be["unbatched"].items():
        W(f"| {k} | {f(v['ns_per_call'])} |")
    W("\n| kernel (batched torch gather+add) | ns / event | column bytes / event | effective GB/s |\n|---|---|---|---|")
    for k, v in be["batched"].items():
        W(f"| {k} | {f(v['ns_per_event'])} | {f(v['column_bytes_per_event'])} | {f(v['effective_GBps'])} |")
    W(f"\nRatio PSI-update / NNUE-quiet-update: unbatched {be['ratios']['unbatched_psi_over_nnue_quiet']:.3f}, "
      f"batched {be['ratios']['batched_psi_over_nnue_quiet']:.3f}. The unbatched ratio mostly measures Python dispatch "
      "(four numpy calls each), not memory traffic.\n")

# ---------------------------------------------------------------- training
if tr:
    W("## 3. E5-E8 Tiny training sanity checks\n")
    W(f"Config: {tr['config']['epochs']} epochs, batch {tr['config']['batch']}, {tr['config']['optimizer']}, seeds "
      f"{tr['config']['seeds']}; {tr['config']['note']}. These runs check that the math trains and quantizes; they say "
      "nothing about chess strength.\n")
    W("### 3.1 E5 Rank sufficiency (synthetic planted table)\n")
    W("One block, 32 buckets, 704 features, 256 lanes. Planted W_k = W0 + U_k V (true rank 8) + Gaussian noise, plus an "
      "unstructured control. Shared V_K is the top-r_K right singular subspace of the stacked bucket residuals. "
      "`int factors` quantizes V_K to int8 (max |V_K| = 127) and U to int16 so that U V_K is in table units at scale 64, "
      "then materialises the integer product (the C2 exactness requirement). Error = relative squared Frobenius error of the "
      "per-bucket residual.\n")
    W("| table | r_K | rel. err (float SVD) | rel. err (int factors, materialised) | rel. err (plain int16 table, same scale) |")
    W("|---|---|---|---|---|")
    for k, rows in tr["E5"].items():
        for r in rows:
            W(f"| {k} | {r['rK']} | {f(r['rel_err_float'])} | {f(r['rel_err_int_factors'])} | {f(r['rel_err_plain_int16_table'])} |")
    d = tr["data"]
    W("\n### 3.2 E6 Does PSI learn piece-pair interactions? (synthetic teachers on real positions)\n")
    W(f"Positions: {f(d['positions'])} Lichess positions (ply >= 8), split {d['split']}: {f(d['train'])} train / {f(d['val'])} val. "
      "Teacher y = z(PST . beta) + z(pair term) + noise, standardised to unit variance (so predicting the mean gives MSE ~ 1). "
      "Pair terms: (i) low-rank Q (rank 8, dense over all 768 features); (ii) sparse 'attack geometry' Q with "
      f"{f(d['attack_pairs'])} nonzero unordered pairs (piece i's empty-board attack set contains j's square); (iii) both.\n")
    W("| model | params | update lanes per changed absolute feature |\n|---|---|---|")
    for k, m in tr["models"].items():
        W(f"| {k} | {f(m['params'])} | {m['update_lanes_per_changed_abs_feature']}{' + ' + m['note'] if m['note'] else ''} |")
    W(f"\nM2's explicit pair table has {f(d['explicit_pair_vocab'])} rows; on average {f(d['mean_active_pairs'])} pairs are active per position "
      "(each one is an extra L-wide column when it changes), so M2 is **not** cost-matched; it is a reference with oracle "
      "knowledge of teacher (ii)'s support. M2 was run with 1 seed (its 14 M-parameter pair table is slow on CPU).\n")
    W("| teacher | " + " | ".join(tr["E6"][next(iter(tr["E6"]))].keys()) + " |")
    W("|---|" + "---|" * len(tr["E6"][next(iter(tr["E6"]))]))
    for t, res in tr["E6"].items():
        cells = []
        for m, r in res.items():
            cells.append(f"{r['mean']:.4f}" + (f" +- {r['sd']:.4f}" if r.get("sd") else " (1 seed)"))
        W(f"| {t} | " + " | ".join(cells) + " |")
    W("\nValues: held-out MSE, mean +- sample s.d. over seeds.\n")
    if "E6_qat_check" in tr:
        q = tr["E6_qat_check"]
        W("**Quantized PSI readout check** (M1 trained on teacher iii, seed 0; V and Lam quantized to int8, 4,000 val positions): "
          f"max |P1| = {q['P1_max_abs']} (bound {q['P1_bound_T9']}), max P2 = {f(q['P2_max'])} (bound {f(q['P2_bound'])}), "
          f"P1^2 - P2 even everywhere: {q['E_even_all']}; max |z_int| = {f(q['z_int_max_abs'])} < 2^31: {q['z_int_lt_2p31']}; "
          f"integer pipeline equals float evaluation of the quantized weights: {q['int_pipeline_equals_float_of_quantised']}; "
          f"relative RMS error of z from quantization: {q['quant_rel_err_z']:.4f}.\n")
    tr.setdefault("E7", {})
    W("### 3.3 E7 Tied vs untied PSI embedding (colour flip)\n")
    W("M1 (L = 192, r = 32, m = 16). Tied: V[t+1, j] = V[t, sigma(j)], so only r/2 rows are free and one shared state serves both "
      "perspectives (T21).\n")
    W("| teacher | untied MSE | tied MSE | untied params | tied params |\n|---|---|---|---|---|")
    for t, r in tr["E7"].items():
        W(f"| {t} | {r['untied']['mean']:.4f} +- {r['untied']['sd']:.4f} | {r['tied']['mean']:.4f} +- {r['tied']['sd']:.4f} | "
          f"{f(r['untied']['params'])} | {f(r['tied']['params'])} |")
    if "E7_T21_check" in tr:
        W("\nT21 numerically on the trained tied model (2,000 val positions, float32): " + "; ".join(
            f"{k}: max |P1(flipped) - P1[perm]| = {v['max_abs_diff_P1']:.2e}, max |P2(flipped) - P2[perm]| = {v['max_abs_diff_P2']:.2e}"
            for k, v in tr["E7_T21_check"].items()) + ".\n")
    if "E8" in tr:
        W("### 3.4 E8 Real-ish labels\n")
        W("| label | predict-train-mean MSE | M0_L256 | M0s_L192 | M1_L192_PSI_r32_m16 |\n|---|---|---|---|---|")
        for l, r in tr["E8"].items():
            W(f"| {l} | {r['predict_train_mean_mse']:.4f} | " + " | ".join(
                f"{r[m]['mean']:.4f} +- {r[m]['sd']:.4f}" for m in ("M0_L256", "M0s_L192", "M1_L192_PSI_r32_m16")) + " |")
        W("\nGame result: sigmoid output, MSE against {0, 0.5, 1}. Material+PST: tanh(eval/400) of the toy evaluator, which is "
          "linear in beta, so PSI has nothing to add there by construction.\n")

from findings_text import FINDINGS
i0 = next(i for i, x in enumerate(md) if x.startswith("## 0. Data"))
open(BASE + "/RESULTS.md", "w", encoding="utf-8").write("\n".join(md[:i0] + [FINDINGS, "\n---\n# Full tables\n"] + md[i0:]))

# ---------------------------------------------------------------- figures
if e2:
    D = e2["datasets"]
    mt = D["games_lichess/eager"]["per_move_type"]
    ks = list(mt.keys())
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(ks)); w = 0.38
    sfb = [mt[k]["SF_bytes"]["mean"] / 1024 for k in ks]
    psb = [mt[k]["psi_bytes"]["mean"] / 1024 for k in ks]
    vnb = [mt[k]["VN_bytes"]["mean"] / 1024 for k in ks]
    ax.bar(x - w / 2, sfb, w - 0.02, color=C[0], label="NNUE (SF policy) FT bytes")
    ax.bar(x + w / 2, vnb, w - 0.02, color=C[1], label="VNNUE (VN policy) FT bytes")
    ax.bar(x + w / 2, psb, w - 0.02, bottom=vnb, color=C[2], label="+ PSI bytes")
    ax.set_xticks(x, [k.replace("_", "\n") for k in ks]); ax.set_ylabel("KiB per move (both perspectives), mean")
    ax.set_title("Per-move accumulator traffic by move type, Lichess games (eager eval)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG + "/e2_bytes_per_move_type.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    keys = list(D.keys())
    pols = ["SF", "SFmin", "VN_noC4", "VN"]
    w = 0.2
    for j, p in enumerate(pols):
        vals = [D[k]["policies"][p]["ft_bytes"] / D[k]["policies"]["SF"]["ft_bytes"] for k in keys]
        ax.bar(np.arange(len(keys)) + (j - 1.5) * w, vals, w - 0.02, color=C[j], label=p)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(len(keys)), keys); ax.set_ylabel("FT bytes relative to SF")
    ax.set_ylim(0.8, max(1.3, ax.get_ylim()[1]))
    ax.set_title("Whole-trace FT traffic per policy (stateful simulation)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(FIG + "/e3_policy_totals.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    j = 0
    for k, R in D.items():
        if "woodbury_sweep" not in R:
            continue
        pts = {}
        for kk, d in R["woodbury_sweep"].items():
            rk, mu, wbm = [float(x.split("=")[1]) for x in kk.split(",")]
            pts.setdefault(rk * mu * wbm, []).append(d["woodbury_chosen_fraction_of_own_king_events"])
        xs = sorted(pts)
        ax.plot(xs, [100 * np.mean(pts[v]) for v in xs], "-o", color=C[j], lw=2, ms=5, label=k)
        j += 1
    ax.set_xlabel("Woodbury mcol cost in col-equivalents (mu * r_K * wb_mult)")
    ax.set_ylabel("% of own-king events where Woodbury is argmin")
    ax.set_title("C2 low-rank king transition: how often it wins", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG + "/e4_woodbury_win_rate.png", dpi=150); plt.close(fig)

if tr:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ts = list(tr["E6"].keys()); ms = list(tr["E6"][ts[0]].keys())
    w = 0.2
    for j, m in enumerate(ms):
        for i, t in enumerate(ts):
            r = tr["E6"][t][m]
            xpos = i + (j - 1.5) * w
            ax.bar(xpos, r["mean"], w - 0.02, color=C[j], label=m if i == 0 else None)
            ax.scatter([xpos] * len(r["val_mse"]), r["val_mse"], s=10, color=INK, zorder=3)
    ax.set_xticks(range(len(ts)), ts); ax.set_ylabel("held-out MSE (target variance = 1)")
    ax.set_title("E6: synthetic pair-interaction teachers (bars = mean, dots = seeds)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG + "/e6_training.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    j = 0
    for k, rows in tr["E5"].items():
        rk = [r["rK"] for r in rows]
        ax.plot(rk, [r["rel_err_float"] for r in rows], "-o", color=C[j], lw=2, ms=5, label=f"{k} (float)")
        ax.plot(rk, [r["rel_err_int_factors"] for r in rows], "--s", color=C[j], lw=1.5, ms=5, label=f"{k} (int factors)")
        j += 1
    ax.set_xscale("log", base=2); ax.set_xlabel("r_K"); ax.set_ylabel("relative residual error")
    ax.set_title("E5: low-rank king-transition reconstruction", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG + "/e5_rank.png", dpi=150); plt.close(fig)
print("tables + figures written")
