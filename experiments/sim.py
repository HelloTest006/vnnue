"""Accumulator simulator shared by E1 (exactness) and E2-E4 (op counts).

A dataset is a forest of nodes in DFS pre-order (parent pointers); this is exactly the
make/unmake event stream an engine accumulator stack sees.  At every node marked for
evaluation, each perspective whose accumulator is not computed is built by a *source*
chosen by a policy (SPEC 2.2 C3/C4):

  refresh   : bias + |S| columns, 1 pass
  finny     : Finny cache entry (c, ksq, blk) with board C: |S xor C| cols, 1 pass
  hybrid    : SF update_accumulator_hybrid (own-king move, parent computed, same block,
              >= 15 pieces, same board half, not castling): |Sprev^Cold|+|S^Cnew| cols, 3 passes
  chain     : SF forward incremental catch-up from nearest computed ancestor u when no
              table change on the path; one pass per ply; intermediate nodes materialised;
              two-ply cancellation (PR #5988) when ply j+1 captures on the to-square of ply j
  ancestor  : VNNUE C4 net delta from u: |S_t xor S_u| cols, 1 pass (only target materialised)
  woodbury  : VNNUE C2 transition from u (same block, different king square):
              a_t = a_u + (U_kt - U_ku) y_u + W_kt (x_t - x_u):  mu*mult*r_K mcol + |x_t^x_u| cols, 1 pass

Policies:
  SF     : chain if possible; else hybrid if its conditions hold; else finny (Stockfish rule)
  SFmin  : chain if possible; else argmin{refresh, finny, hybrid}
  VN     : argmin over {refresh, finny, hybrid, chain, ancestor} (+ woodbury if r_K > 0)
  VN_noC4: like VN without 'ancestor'
Scalar cost used by argmin: cols + mu*mcols + 2*passes  (a pass = load+store of 2L bytes each
= 2 column-equivalents).
"""
import os
import numpy as np
from common import *

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COST_PASS = 2.0


class Dataset:
    def __init__(self, name, eval_mode="eager", lazy_p=0.3, seed=0, max_nodes=None, roots=None):
        d = np.load(f"{DATA}/{name}.npz")
        n = len(d["parent"]) if max_nodes is None else max_nodes
        if roots is not None:  # keep only the first `roots` trees / games
            rt = np.flatnonzero(d["parent"] == -1)
            if roots < len(rt):
                n = min(n, int(rt[roots]))
        self.name = name
        self.n = n
        self.parent = d["parent"][:n].astype(np.int64)
        self.absset = d["absset"][:n].astype(np.int64)
        self.ksq = d["ksq"][:n].astype(np.int64)
        self.blk = d["blk"][:n].astype(np.int64)
        self.npieces = d["npieces"][:n].astype(np.int64)
        self.mtype = d["mtype"][:n].astype(np.int64)
        self.mover = d["mover"][:n].astype(np.int64)
        self.is_king = d["is_king"][:n].astype(bool)
        self.mfrom = d["mfrom"][:n].astype(np.int64)
        self.mto = d["mto"][:n].astype(np.int64)
        self.rem = d["rem"][:n].astype(np.int64)
        self.add = d["add"][:n].astype(np.int64)
        self.kind = d["kind"][:n].astype(np.int64)
        self.threat = d["threat"][:n].astype(np.int64)
        valid = self.absset >= 0
        self.valid = valid
        G = np.full((n, 2, 32), -1, np.int64)
        ORI = np.zeros((n, 2), np.int64)
        a = np.where(valid, self.absset, 0)
        col, t, sq = a // 384, (a // 64) % 6, a % 64
        for c in (0, 1):
            k = self.ksq[:, c]
            ori = (56 if c else 0) ^ np.where((k & 7) < 4, 7, 0)
            ORI[:, c] = ori
            s = k ^ ori
            buck = (s >> 3) * 4 + (s & 7) - 4
            plane = np.where(t == 5, 10, np.where(col == c, t, 5 + t))
            loc = plane * 64 + (sq ^ ori[:, None])
            G[:, c] = np.where(valid, (self.blk[:, c] * 32 + buck)[:, None] * 704 + loc, -1)
        self.G = G
        self.ori = ORI
        self.bucket = ((self.ksq ^ ORI) >> 3) * 4 + ((self.ksq ^ ORI) & 7) - 4
        rng = np.random.default_rng(seed)
        if eval_mode == "eager":
            self.evalm = np.ones(n, bool)
        else:  # lazy: roots, game nodes and qsearch/leaf nodes always, interior w.p. lazy_p
            self.evalm = (self.kind != 1) | (rng.random(n) < lazy_p)
        self.eval_mode = eval_mode

    def S(self, i, c):
        g = self.G[i, c]
        return set(g[g >= 0].tolist())

    def X(self, i, c):
        g = self.G[i, c]
        return set((g[g >= 0] % 704).tolist())

    def absS(self, i):
        a = self.absset[i]
        return a[a >= 0]

    def key(self, i, c):
        return (int(self.blk[i, c]), int(self.ksq[i, c]))


class Cost:
    __slots__ = ("cols", "mcols", "passes")

    def __init__(self, cols=0, mcols=0, passes=0):
        self.cols, self.mcols, self.passes = cols, mcols, passes

    def scalar(self, mu):
        return self.cols + mu * self.mcols + COST_PASS * self.passes


class Sim:
    """Policy simulation.  If `exact` is an ExactState, every candidate source is also
    computed numerically and checked against the from-scratch refresh."""

    def __init__(self, ds, policy="VN", rK=0, mu=1.0, wb_mult=2, exact=None, record=True, reset_cache_per_root=False):
        self.ds, self.policy, self.rK, self.mu, self.wb_mult = ds, policy, rK, mu, wb_mult
        # False (default, used by E1-E4): Finny entries persist across games/trees (stale caches).
        # True: Finny cache cleared at every root (ucinewgame / new search), warmed by the run itself.
        self.reset_cache_per_root = reset_cache_per_root
        self.exact = exact
        self.record = record
        self.finny = {}
        self.events = []      # per (eval node, perspective) records
        self.tot = dict(cols=0, mcols=0, passes=0, evals=0, builds=0)
        self.src_count = {}

    # ---- candidate generation ----
    def candidates(self, stack, i, c):
        ds = self.ds
        t = stack[i]["node"]
        St = stack[i]["S"][c]
        j = i
        while j >= 0 and not stack[j]["computed"][c]:
            j -= 1
        cands = {}
        cands["refresh"] = Cost(len(St), 0, 1)
        kt = ds.key(t, c)
        C = self.finny.get((c,) + kt, set())
        cands["finny"] = Cost(len(St ^ C), 0, 1)
        info = dict(u_idx=j, chain_len=i - j if j >= 0 else None)
        if j >= 0:
            u = stack[j]["node"]
            Su = stack[j]["S"][c]
            chain_nodes = [stack[m]["node"] for m in range(j, i + 1)]
            tc = any(ds.key(chain_nodes[m], c) != ds.key(chain_nodes[m - 1], c) for m in range(1, len(chain_nodes)))
            info["table_change"] = tc
            # hybrid (parent computed, own king move)
            if (j == i - 1 and ds.mover[t] == c and ds.is_king[t] and ds.mtype[t] != CASTLE
                    and ds.blk[u, c] == ds.blk[t, c] and ds.npieces[t] >= 15
                    and (ds.mfrom[t] & 4) == (ds.mto[t] & 4)):
                Cold = self.finny.get((c,) + ds.key(u, c), set())
                cands["hybrid"] = Cost(len(Su ^ Cold) + len(St ^ C), 0, 3)
            if not tc:
                cols, passes, steps, sumk = 0, 0, [], 0
                m = j + 1
                while m <= i:
                    n1 = stack[m]["node"]
                    Sprev = stack[m - 1]["S"][c]
                    if (m + 1 <= i and ds.mto[stack[m + 1]["node"]] == ds.mto[n1]
                            and ds.mtype[stack[m + 1]["node"]] in (CAPTURE, CAPPROMO)):
                        S2 = stack[m + 1]["S"][c]
                        cols += len(S2 ^ Sprev); passes += 1; steps.append((m + 1, True))
                        sumk += len(stack[m]["S"][c] ^ Sprev) + len(S2 ^ stack[m]["S"][c])
                        m += 2
                    else:
                        d = len(stack[m]["S"][c] ^ Sprev)
                        cols += d; passes += 1; steps.append((m, False)); sumk += d
                        m += 1
                cands["chain"] = Cost(cols, 0, passes)
                info["chain_steps"] = steps
                info["sum_k"] = sumk
            if self.policy in ("VN",):
                cands["ancestor"] = Cost(len(St ^ Su), 0, 1)
            info["net_supp"] = len(St ^ Su)
            if self.policy.startswith("VN") and self.rK > 0 and ds.blk[u, c] == ds.blk[t, c] \
                    and ds.ksq[u, c] != ds.ksq[t, c]:
                Xu, Xt = stack[j]["X"][c], stack[i]["X"][c]
                cands["woodbury"] = Cost(len(Xt ^ Xu), self.wb_mult * self.rK, 1)
        return cands, info

    def choose(self, cands, info):
        p = self.policy
        if p == "SF":
            if "chain" in cands:
                return "chain"
            if "hybrid" in cands:
                return "hybrid"
            return "finny"
        if p == "SFmin":
            if "chain" in cands:
                return "chain"
            keys = [k for k in ("refresh", "finny", "hybrid") if k in cands]
        else:
            keys = list(cands)
        return min(keys, key=lambda k: (cands[k].scalar(self.mu), k != "chain"))

    # ---- main loop ----
    def run(self):
        ds = self.ds
        stack = []
        ex = self.exact
        for t in range(ds.n):
            p = ds.parent[t]
            if p < 0:
                stack = []
                if self.reset_cache_per_root:
                    self.finny = {}
            else:
                while stack[-1]["node"] != p:
                    stack.pop()
            ent = dict(node=t, computed=[False, False], S=[ds.S(t, 0), ds.S(t, 1)],
                       X=[ds.X(t, 0), ds.X(t, 1)], acc=[None, None])
            if ex is not None:
                ex.on_make(stack[-1] if stack else None, ent)
            stack.append(ent)
            if not ds.evalm[t]:
                continue
            self.tot["evals"] += 1
            i = len(stack) - 1
            for c in (0, 1):
                if ent["computed"][c]:
                    continue
                cands, info = self.candidates(stack, i, c)
                src = self.choose(cands, info)
                cost = cands[src]
                self.tot["cols"] += cost.cols; self.tot["mcols"] += cost.mcols; self.tot["passes"] += cost.passes
                self.tot["builds"] += 1
                self.src_count[src] = self.src_count.get(src, 0) + 1
                if self.record:
                    sfk = [k for k in ("chain",) if k in cands] or [k for k in ("refresh", "finny", "hybrid") if k in cands]
                    sf_min = min(cands[k].scalar(self.mu) for k in sfk)
                    full_min = min(v.scalar(self.mu) for v in cands.values())
                    own_king = bool(ds.is_king[t] and ds.mover[t] == c)
                    self.events.append((t, c, src, cost.cols, cost.mcols, cost.passes, sf_min, full_min,
                                        info.get("chain_len") or 0, info.get("table_change", True),
                                        info.get("sum_k", -1), info.get("net_supp", -1), own_king,
                                        "woodbury" in cands, "hybrid" in cands))
                if ex is not None:
                    ex.check_eval(self, stack, i, c, cands, info)
                # apply source side effects
                kt = (c,) + ds.key(t, c)
                if src in ("finny", "refresh"):
                    self.finny[kt] = ent["S"][c]
                    if ex is not None:
                        ex.finny_store(kt, ex.cand_vals[src])
                elif src == "hybrid":
                    u = stack[i - 1]["node"]
                    self.finny[(c,) + ds.key(u, c)] = stack[i - 1]["S"][c]
                    self.finny[kt] = ent["S"][c]
                    if ex is not None:
                        ex.finny_store((c,) + ds.key(u, c), stack[i - 1]["acc"][c])
                        ex.finny_store(kt, ex.cand_vals[src])
                elif src == "chain":
                    for (m, merged) in info["chain_steps"]:
                        stack[m]["computed"][c] = True
                        if ex is not None:
                            stack[m]["acc"][c] = ex.chain_vals[m]
                if ex is not None:
                    ent["acc"][c] = ex.cand_vals[src]
                ent["computed"][c] = True
        return self
