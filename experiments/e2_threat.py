"""Supplement to E2: threat-proxy churn per move type on the Lichess games where the proxy was
computed (first 600 games), and PSI overhead relative to PSQ + threat-proxy bytes.
Writes results/e2_threat_proxy.json"""
import os
import json
import numpy as np
from sim import Dataset, Sim
from e2_e4_opcounts import (move_class, k_abs, dist, BYTES_COL, BYTES_PASS, BYTES_THREAT_COL,
                            PSI_BYTES_PER_FEAT, ev_arrays)

ds = Dataset("games_lichess", roots=600)
assert np.all(ds.threat[ds.parent >= 0] >= 0)
sim = Sim(ds, policy="SF").run()
ev = ev_arrays(sim)
cols = np.zeros(ds.n); passes = np.zeros(ds.n)
np.add.at(cols, ev["t"], ev["cols"]); np.add.at(passes, ev["t"], ev["passes"])
cls = np.array([move_class(ds, t) or "root" for t in range(ds.n)])
kab = np.array([k_abs(ds, t) if ds.parent[t] >= 0 else 0 for t in range(ds.n)])
out = dict(games=600, positions=int(ds.n), per_move_type={})
for k in sorted(set(cls) - {"root"}):
    i = cls == k
    psq = cols[i] * BYTES_COL + passes[i] * BYTES_PASS
    thr = 2 * ds.threat[i] * BYTES_THREAT_COL       # both perspectives
    psi = kab[i] * PSI_BYTES_PER_FEAT
    out["per_move_type"][k] = dict(count=int(i.sum()), threat_proxy_changes_abs=dist(ds.threat[i]),
                                   psq_cols_SF=dist(cols[i]),
                                   psi_over_psq_col_bytes=float(psi.sum() / (cols[i] * BYTES_COL).sum()),
                                   psi_over_psq_plus_threat_bytes=float(psi.sum() / (psq.sum() + thr.sum())))
i = cls != "root"
out["overall"] = dict(threat_proxy_changes_abs=dist(ds.threat[i]),
                      threat_over_psi_feature_changes=float(ds.threat[i].sum() / kab[i].sum()),
                      psi_over_psq_plus_threat_bytes=float((kab[i] * PSI_BYTES_PER_FEAT).sum() /
                                                           ((cols[i] * BYTES_COL + passes[i] * BYTES_PASS).sum() + (2 * ds.threat[i] * BYTES_THREAT_COL).sum())))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e2_threat_proxy.json"), "w"), indent=1)
print(json.dumps(out["overall"], indent=1))
