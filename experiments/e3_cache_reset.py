"""E3/E4 robustness rerun: Finny cache cleared at every root (new game / new search tree).

The original E2-E4 runs (e2_e4_opcounts.py) keep one Finny cache for the whole dataset, so
entries carry over between unrelated games and trees (stale caches).  Stockfish clears its
refresh table on ucinewgame and warms it during search.  This script reruns the four E3
policies, a small Woodbury subset and the Lichess per-move-type table with the cache reset per
root.  Writes results/e3_cache_reset.json.
"""
import json, time, gc
import numpy as np
from common import *
from sim import Dataset, Sim
from e2_e4_opcounts import (ev_arrays, summarize, node_level, move_class, k_abs, dist,
                            BYTES_COL, BYTES_PASS, PSI_BYTES_PER_FEAT, PSI_LANES_PER_FEAT, RES)


def run(ds, pol, **kw):
    t0 = time.time()
    sim = Sim(ds, policy=pol, reset_cache_per_root=True, **kw).run()
    ev = ev_arrays(sim)
    sim.events = None
    return sim, ev, time.time() - t0


def main():
    t0 = time.time()
    out = dict(config=dict(reset_cache_per_root=True, cost_scalar="cols + mu*mcols + 2*passes"), datasets={})
    for name, mode in [("games_lichess", "eager"), ("games_selfplay", "eager"), ("trees", "eager"), ("trees", "lazy")]:
        key = f"{name}/{mode}"
        ds = Dataset(name, eval_mode=mode, seed=7)
        kab = np.array([k_abs(ds, t) if ds.parent[t] >= 0 else 0 for t in range(ds.n)])
        psi_lanes = int(kab.sum()) * PSI_LANES_PER_FEAT
        psi_bytes = int(kab.sum()) * PSI_BYTES_PER_FEAT
        R = dict(nodes=ds.n, roots=int((ds.parent < 0).sum()), policies={})
        pernode, evs = {}, {}
        for pol in ("SF", "SFmin", "VN_noC4", "VN"):
            sim, ev, sec = run(ds, pol)
            R["policies"][pol] = summarize(ds, sim, ev, psi_lanes, psi_bytes)
            if mode == "eager" and name == "games_lichess":
                node_level(ds, ev, pol, pernode)
            evs[pol] = ev
            print(key, pol, sim.tot, sim.src_count, round(sec, 1), flush=True)
        ev = evs["SF"]
        tc = ev["table_change"].astype(bool)
        R["SF_finny_cost_at_table_change"] = dist(ev["cols"][tc & (ev["src"] == "finny")])
        R["SF_hybrid_cost"] = dist(ev["cols"][ev["src"] == "hybrid"])
        if pernode:
            cls = [move_class(ds, t) for t in range(ds.n)]
            mt = {}
            for k in sorted(set(c for c in cls if c is not None)):
                idx = np.array([i for i, c in enumerate(cls) if c == k])
                row = dict(count=int(len(idx)))
                for pol in ("SF", "VN"):
                    cols, passes, mcols = pernode[pol]
                    row[pol + "_cols_mean"] = float(cols[idx].mean())
                    row[pol + "_bytes_mean"] = float((cols[idx] * BYTES_COL + passes[idx] * BYTES_PASS).mean())
                mt[k] = row
            R["per_move_type"] = mt
        del evs, pernode
        if name != "games_selfplay":
            sw = {}
            for rK, mu, wbm in ((4, 1.0, 2), (16, 1.0, 1), (16, 1.0, 2)):
                sim, ev, sec = run(ds, "VN", rK=rK, mu=mu, wb_mult=wbm)
                tcc = ev["table_change"].astype(bool)
                ok = ev["own_king"].astype(bool) & tcc
                d = summarize(ds, sim, ev, psi_lanes, psi_bytes)
                d["woodbury_chosen_fraction_of_own_king_events"] = float((ev["src"][ok] == "woodbury").mean()) if ok.any() else None
                sw[f"rK={rK},mu={mu},wb_mult={wbm}"] = d
                print(key, "C2", rK, mu, wbm, d["woodbury_chosen_fraction_of_own_king_events"], sim.tot, round(sec, 1), flush=True)
                del ev
            R["woodbury_subset"] = sw
        out["datasets"][key] = R
        json.dump(out, open(RES + "/e3_cache_reset.json", "w"), indent=1)
        del ds
        gc.collect()
    out["seconds"] = time.time() - t0
    json.dump(out, open(RES + "/e3_cache_reset.json", "w"), indent=1)
    print("done", out["seconds"])


if __name__ == "__main__":
    main()
