"""E2-E4: op / lane / byte counts for NNUE (Stockfish-like) vs VNNUE policies on real game
sequences and alpha-beta search traces (eager and lazy evaluation).

Writes results/e2_e4_opcounts.json
"""
import json, os, time, gc
import numpy as np
from common import *
from sim import Dataset, Sim

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
L = 1024             # Stockfish master L1
R_PSI = 32
BYTES_COL = 2 * L    # one int16 column load
BYTES_PASS = 4 * L   # accumulator load + store
BYTES_THREAT_COL = L # i8 threat column
PSI_BYTES_PER_FEAT = 6 * R_PSI   # SPEC 3.1: 2r (P1 int16) + 4r (P2 int32)
PSI_LANES_PER_FEAT = 2 * R_PSI
import os
QUICK = int(os.environ["QUICK"]) if os.environ.get("QUICK") else None


def dist(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), p50=float(np.percentile(x, 50)),
                p95=float(np.percentile(x, 95)), max=float(x.max()))


EV_FIELDS = ["t", "c", "src", "cols", "mcols", "passes", "sf_min", "full_min", "chain_len",
             "table_change", "sum_k", "net_supp", "own_king", "wb_avail", "hyb_avail"]


def ev_arrays(sim):
    if not sim.events:
        return {}
    cols = list(zip(*sim.events))
    out = {}
    for k, v in zip(EV_FIELDS, cols):
        out[k] = np.array(v) if k != "src" else np.array(v, dtype=object)
    return out


def move_class(ds, t):
    if ds.parent[t] < 0:
        return None
    mt = ds.mtype[t]
    if mt == CASTLE:
        return "castling"
    if ds.is_king[t]:
        return "king_capture" if mt == CAPTURE else "king_quiet"
    return MTYPE_NAMES[mt]


def k_abs(ds, t):
    return int((ds.rem[t] >= 0).sum() + (ds.add[t] >= 0).sum())


def run_policy(ds, policy, rK=0, mu=1.0, wb_mult=2):
    t0 = time.time()
    sim = Sim(ds, policy=policy, rK=rK, mu=mu, wb_mult=wb_mult).run()
    ev = ev_arrays(sim)
    sim.events = None
    return sim, ev, time.time() - t0


def summarize(ds, sim, ev, psi_lanes_total, psi_bytes_total, c2_lanes=0):
    tot = sim.tot
    ft_bytes = tot["cols"] * BYTES_COL + tot["mcols"] * BYTES_COL + tot["passes"] * BYTES_PASS
    d = dict(evals=tot["evals"], builds=tot["builds"], cols=tot["cols"], mcols=tot["mcols"],
             passes=tot["passes"], ft_bytes=ft_bytes, sources=sim.src_count,
             cols_per_eval=tot["cols"] / max(1, tot["evals"]),
             passes_per_eval=tot["passes"] / max(1, tot["evals"]),
             ft_bytes_per_eval=ft_bytes / max(1, tot["evals"]))
    d["psi_lanes_total"] = psi_lanes_total
    d["psi_bytes_total"] = psi_bytes_total
    d["psi_bytes_over_ft_bytes"] = psi_bytes_total / ft_bytes
    if c2_lanes:
        d["c2_y_lanes_total"] = c2_lanes
    if ev:
        d["counterfactual_full_le_sf_all"] = bool(np.all(ev["full_min"] <= ev["sf_min"] + 1e-9))
        d["counterfactual_saving_colequiv"] = float((ev["sf_min"] - ev["full_min"]).sum())
        d["counterfactual_sf_colequiv"] = float(ev["sf_min"].sum())
    return d


def node_level(ds, ev, policy_name, extra):
    """Per-node costs (both perspectives) for game sequences (eager eval => one build per node/persp)."""
    n = ds.n
    cols = np.zeros(n); passes = np.zeros(n); mcols = np.zeros(n)
    np.add.at(cols, ev["t"], ev["cols"]); np.add.at(passes, ev["t"], ev["passes"]); np.add.at(mcols, ev["t"], ev["mcols"])
    extra[policy_name] = (cols, passes, mcols)


def c2_y_lanes(ds, rK):
    tot = 0
    for t in range(ds.n):
        p = ds.parent[t]
        for c in (0, 1):
            if p < 0 or ds.ori[p, c] != ds.ori[t, c]:
                tot += int(ds.valid[t].sum()) * rK
            else:
                tot += len(ds.X(t, c) ^ ds.X(p, c)) * rK
    return tot


def main():
    t0 = time.time()
    out = dict(config=dict(L=L, r=R_PSI, bytes_col=BYTES_COL, bytes_pass=BYTES_PASS,
                           psi_bytes_per_changed_abs_feature=PSI_BYTES_PER_FEAT,
                           cost_scalar="cols + mu*mcols + 2*passes",
                           woodbury_cost="wb_mult*r_K mcol (wb_mult=2: dU formed on the fly) + |x_t ^ x_u| col + 1 pass"),
               datasets={})
    specs = [("games_lichess", "eager"), ("games_selfplay", "eager"), ("trees", "eager"), ("trees", "lazy")]
    for name, mode in specs:
        key = f"{name}/{mode}"
        ds = Dataset(name, eval_mode=mode, seed=7, roots=QUICK)
        R = dict(nodes=ds.n, evals=int(ds.evalm.sum()), roots=int((ds.parent < 0).sum()))
        # ---- move statistics ----
        cls = [move_class(ds, t) for t in range(ds.n)]
        moves = [c for c in cls if c is not None]
        R["move_type_fraction"] = {k: moves.count(k) / len(moves) for k in sorted(set(moves))}
        kab = np.array([k_abs(ds, t) if ds.parent[t] >= 0 else 0 for t in range(ds.n)])
        R["k_abs"] = dist(kab[ds.parent >= 0])
        blkchg = np.array([ds.parent[t] >= 0 and bool(np.any(ds.blk[t] != ds.blk[ds.parent[t]])) for t in range(ds.n)])
        R["block_change_fraction_of_moves"] = float(blkchg[ds.parent >= 0].mean())
        R["S_size"] = dist(ds.valid.sum(1))
        psi_lanes = int(kab.sum()) * PSI_LANES_PER_FEAT
        psi_bytes = int(kab.sum()) * PSI_BYTES_PER_FEAT
        thr = ds.threat[ds.parent >= 0]
        if np.all(thr >= 0) or (thr >= 0).mean() > 0.5:
            thr_ok = thr[thr >= 0]
            R["threat_proxy_churn_per_move_abs"] = dist(thr_ok)
        # ---- policies ----
        R["policies"] = {}
        pernode = {}
        evs = {}
        for pol in ("SF", "SFmin", "VN_noC4", "VN"):
            sim, ev, sec = run_policy(ds, pol)
            R["policies"][pol] = summarize(ds, sim, ev, psi_lanes, psi_bytes)
            R["policies"][pol]["seconds"] = sec
            if mode == "eager" and name.startswith("games"):
                node_level(ds, ev, pol, pernode)
            evs[pol] = ev
            print(key, pol, sim.tot, sim.src_count, round(sec, 1), flush=True)
        # ---- event-level distributions (from SF and VN runs) ----
        ev = evs["SF"]
        tc = ev["table_change"].astype(bool)
        ok = ev["own_king"].astype(bool) & tc
        R["build_events"] = dict(total=int(len(ev["t"])), table_change=int(tc.sum()),
                                 own_king_table_change=int(ok.sum()),
                                 table_change_fraction=float(tc.mean()))
        R["SF_finny_cost_at_table_change"] = dist(ev["cols"][tc & (ev["src"] == "finny")])
        R["SF_hybrid_cost"] = dist(ev["cols"][ev["src"] == "hybrid"])
        R["SF_hybrid_available_fraction_of_own_king_events"] = float(ev["hyb_avail"][ok].mean()) if ok.any() else None
        evv = evs["VN"]
        ch = (~evv["table_change"].astype(bool)) & (evv["chain_len"] >= 1) & (evv["sum_k"] > 0)
        R["catchup_chain_len_VNrun"] = dist(evv["chain_len"][evv["chain_len"] > 0])
        ch2 = ch & (evv["chain_len"] >= 2)
        R["catchup_events_len_ge2"] = int(ch2.sum())
        if ch2.any():
            R["cancellation_ratio_len_ge2"] = dist(evv["net_supp"][ch2] / evv["sum_k"][ch2])
            R["cancellation_col_saving_fraction_len_ge2"] = float(1 - evv["net_supp"][ch2].sum() / evv["sum_k"][ch2].sum())
        evs_sf_chain = evs["SF"]
        chs = (evs_sf_chain["src"] == "chain")
        R["SF_chain_len"] = dist(evs_sf_chain["chain_len"][chs])
        R["SF_chain_sum_k"] = dist(evs_sf_chain["sum_k"][chs])
        # ---- per move type tables for games ----
        if pernode:
            mt = {}
            for k in sorted(set(moves)):
                idx = np.array([i for i, c in enumerate(cls) if c == k])
                row = dict(count=int(len(idx)))
                for pol in ("SF", "VN"):
                    cols, passes, mcols = pernode[pol]
                    b = cols[idx] * BYTES_COL + mcols[idx] * BYTES_COL + passes[idx] * BYTES_PASS
                    row[pol + "_cols"] = dist(cols[idx])
                    row[pol + "_passes"] = dist(passes[idx])
                    row[pol + "_bytes"] = dist(b)
                row["psi_lanes"] = dist(kab[idx] * PSI_LANES_PER_FEAT)
                row["psi_bytes"] = dist(kab[idx] * PSI_BYTES_PER_FEAT)
                cols, passes, mcols = pernode["SF"]
                b = cols[idx] * BYTES_COL + passes[idx] * BYTES_PASS
                row["psi_bytes_over_SF_psq_bytes_mean_ratio"] = float((kab[idx] * PSI_BYTES_PER_FEAT).sum() / b.sum())
                row["psi_bytes_over_SF_psq_col_bytes_mean_ratio"] = float((kab[idx] * PSI_BYTES_PER_FEAT).sum() / (cols[idx] * BYTES_COL).sum())
                if "threat_proxy_churn_per_move_abs" in R:
                    th = ds.threat[idx]; okk = th >= 0
                    if okk.any():
                        thb = 2 * th[okk] * BYTES_THREAT_COL   # both perspectives, i8 columns
                        row["threat_proxy_cols_both_persp"] = dist(2 * th[okk])
                        row["psi_bytes_over_SF_psq_plus_threat_bytes"] = float(
                            (kab[idx][okk] * PSI_BYTES_PER_FEAT).sum() / (b[okk].sum() + thb.sum()))
                mt[k] = row
            R["per_move_type"] = mt
            # overall incl. threat proxy
            cols, passes, mcols = pernode["SF"]
            th = ds.threat; okk = (th >= 0) & (ds.parent >= 0)
            b = cols * BYTES_COL + passes * BYTES_PASS
            R["psi_overhead_overall"] = dict(
                psi_over_psq_col_bytes=float(psi_bytes / (cols.sum() * BYTES_COL)),
                psi_over_psq_bytes_incl_passes=float(psi_bytes / b.sum()),
                psi_over_psq_plus_threat_bytes_threat_subset=float(
                    (kab[okk] * PSI_BYTES_PER_FEAT).sum() / (b[okk].sum() + (2 * th[okk] * BYTES_THREAT_COL).sum()))
                if okk.any() else None)
        del evs, pernode
        gc.collect()
        # ---- woodbury (C2) sweep ----
        if name != "games_selfplay":
            sw = {}
            for rK in (4, 8, 16):
                c2l = c2_y_lanes(ds, rK) if mode == "eager" or name == "trees" else 0
                for mu in (1.0, 1.5, 2.0):
                    for wbm in ((1, 2) if rK == 16 else (2,)):
                        sim, ev, sec = run_policy(ds, "VN", rK=rK, mu=mu, wb_mult=wbm)
                        tc = ev["table_change"].astype(bool)
                        ok = ev["own_king"].astype(bool) & tc
                        d = summarize(ds, sim, ev, psi_lanes, psi_bytes, c2l)
                        d["woodbury_chosen_fraction_of_own_king_events"] = float((ev["src"][ok] == "woodbury").mean()) if ok.any() else None
                        d["woodbury_chosen_fraction_of_table_change_events"] = float((ev["src"][tc] == "woodbury").mean()) if tc.any() else None
                        d["woodbury_available_fraction_of_own_king_events"] = float(ev["wb_avail"][ok].mean()) if ok.any() else None
                        d["sources_at_own_king_events"] = {s: int((ev["src"][ok] == s).sum()) for s in set(ev["src"][ok])}
                        sw[f"rK={rK},mu={mu},wb_mult={wbm}"] = d
                        print(key, "C2", rK, mu, wbm, d["woodbury_chosen_fraction_of_own_king_events"], sim.tot, round(sec, 1), flush=True)
                        del ev
            R["woodbury_sweep"] = sw
        out["datasets"][key] = R
        json.dump(out, open(RES + ("/e2_e4_opcounts.json" if QUICK is None else "/quick_e2.json"), "w"), indent=1)
        del ds
        gc.collect()
    out["seconds"] = time.time() - t0
    json.dump(out, open(RES + ("/e2_e4_opcounts.json" if QUICK is None else "/quick_e2.json"), "w"), indent=1)
    print("done", out["seconds"])


if __name__ == "__main__":
    main()
