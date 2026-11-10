"""E1 Exactness: every incremental path equals full recomputation, bit for bit, with int16
wrap-around; PSI channel identities; C2 low-rank king transition; saturation path dependence.

Writes results/e1_exactness.json
"""
import os
import json, time
import numpy as np
from common import *
from sim import Dataset, Sim

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 2026
L = 64          # lanes in the E1 prototype (exactness does not depend on L)
R_PSI, M_HEADS = 32, 16
R_K = 8


def wrap16(x):
    return np.asarray(x).astype(np.int64).astype(np.int16)  # two's-complement wrap


def make_weights(rng, stress):
    """Merged C2 table W[blk,kappa] = W0[blk] + U[blk,kappa] @ VK  (exact in int16 / wrapping)."""
    if stress:
        W0 = rng.integers(-6000, 6001, (3, 704, L))
        U = rng.integers(-300, 301, (3, 32, L, R_K))
        VK = rng.integers(-127, 128, (R_K, 704))
        b = rng.integers(-20000, 20001, L)
    else:
        W0 = rng.integers(-60, 61, (3, 704, L))
        U = rng.integers(-6, 7, (3, 32, L, R_K))
        VK = rng.integers(-8, 9, (R_K, 704))
        b = rng.integers(-500, 501, L)
    U16 = wrap16(U)
    VK8 = VK.astype(np.int8)
    T = np.zeros((3, 32, 704, L), np.int16)
    for blk in range(3):
        for k in range(32):
            prod = (U16[blk, k].astype(np.int64) @ VK8.astype(np.int64)).T   # (704, L)
            T[blk, k] = wrap16(W0[blk] + prod)
    T = T.reshape(-1, L)
    return dict(T=T, T64=T.astype(np.int64), b=wrap16(b), b64=wrap16(b).astype(np.int64),
                U=U16, VK=VK8)


def make_psi(rng):
    half = rng.integers(-127, 128, (R_PSI // 2, 768))
    V = np.zeros((R_PSI, 768), np.int64)
    V[0::2] = half
    V[1::2] = half[:, SIGMA_C]          # V[t+1, j] = V[t, sigma(j)]  => V_{P t, sigma i} = V_{t,i}
    Lam = rng.integers(-127, 128, (M_HEADS, R_PSI))
    perm = np.arange(R_PSI) ^ 1
    return dict(V=V.astype(np.int8), V64=V, V2=(V * V).astype(np.int32), Lam=Lam, perm=perm)


class ExactState:
    def __init__(self, w, psi, ds):
        self.w, self.psi, self.ds = w, psi, ds
        self.cache = {}
        self.stats = {}
        self.cand_vals = {}
        self.chain_vals = {}
        self.wrap_lanes_exact = 0     # lanes whose path left int16 range but final value in range
        self.lanes_checked_inrange = 0
        self.lanes_out_of_range_final = 0
        self.dec_mismatch = 0
        self.psi_checks = dict(P1=0, P2=0, E_inc=0, readout_vs_bruteforce=0, black_view=0, even=0,
                               mismatches=0, y_recompute=0, y_checks=0, feature_delta=0)

    def bump(self, k, ok):
        s = self.stats.setdefault(k, [0, 0])
        s[0] += 1
        s[1] += 0 if ok else 1

    # --- column application with int64 shadow and intermediate-overflow tracking ---
    def apply(self, acc, add, rem):
        a16, a64, ovf = acc
        add = sorted(add); rem = sorted(rem)
        T, T64 = self.w["T"], self.w["T64"]
        if add or rem:
            rows = np.concatenate(([T64[add]] if add else []) + ([-T64[rem]] if rem else []))
            part = a64[None, :] + np.cumsum(rows, axis=0)
            ovf = ovf | np.any((part > 32767) | (part < -32768), axis=0)
            a64 = part[-1]
            a16 = a16.copy()
            for r_ in add:
                a16 += T[r_]          # int16 + int16 wraps (non-saturating, like vpaddw)
            for r_ in rem:
                a16 -= T[r_]
        return (a16, a64, ovf)

    def bias(self):
        return (self.w["b"].copy(), self.w["b64"].copy(), np.zeros(L, bool))

    def finny_store(self, key, acc):
        self.cache[key] = acc

    # --- PSI / y maintenance at make ---
    def on_make(self, pent, ent):
        ds, P = self.ds, self.psi
        t = ent["node"]
        absS = ds.absS(t)
        V, V2 = P["V"], P["V2"]
        full1 = V[:, absS].astype(np.int64).sum(1)
        full2 = P["V2"][:, absS].astype(np.int64).sum(1)
        ent["y"] = [None, None]
        if pent is None:
            ent["P1"] = wrap16(full1); ent["P2"] = full2.astype(np.int32)
            ent["E"] = (full1 ** 2 - full2) // 2
        else:
            rem = [f for f in ds.rem[t] if f >= 0]; add = [f for f in ds.add[t] if f >= 0]
            pabs = set(ds.absS(pent["node"]).tolist())
            ok = (pabs - set(rem)) | set(add) == set(absS.tolist()) and set(rem) <= pabs
            self.psi_checks["feature_delta"] += 1
            self.psi_checks["mismatches"] += 0 if ok else 1
            P1 = pent["P1"].copy(); P2 = pent["P2"].copy()
            E = pent["E"].copy(); p1 = pent["P1"].astype(np.int64)
            for f in rem:                       # T14 removal: dE = -v_j * P1(after)
                P1 -= V[:, f].astype(np.int16); P2 -= V2[:, f]
                p1 = p1 - V[:, f]
                E -= V[:, f].astype(np.int64) * p1
            for f in add:                       # T14 addition: dE = v_j * P1(before)
                E += V[:, f].astype(np.int64) * p1
                P1 += V[:, f].astype(np.int16); P2 += V2[:, f]
                p1 = p1 + V[:, f]
            ent["P1"], ent["P2"], ent["E"] = P1, P2, E
        c = self.psi_checks
        P1d = ent["P1"].astype(np.int64); P2d = ent["P2"].astype(np.int64)
        bad = 0
        bad += int(not np.array_equal(P1d, full1)); c["P1"] += 1
        bad += int(not np.array_equal(P2d, full2)); c["P2"] += 1
        diff = P1d ** 2 - P2d
        bad += int(np.any(diff % 2 != 0)); c["even"] += 1
        Ecl = diff // 2
        bad += int(not np.array_equal(Ecl, ent["E"])); c["E_inc"] += 1
        # readout identity (T13) vs brute-force explicit pair sum
        z = P["Lam"] @ Ecl
        n = len(absS)
        iu, ju = np.triu_indices(n, 1)
        Vp = P["V64"][:, absS]
        pair_prod = Vp[:, iu] * Vp[:, ju]                  # (r, npairs)
        brute = (P["Lam"] @ pair_prod).sum(1)
        bad += int(not np.array_equal(z, brute)); c["readout_vs_bruteforce"] += 1
        # black view (T21): power sums of the colour-flipped board == lane-permuted state
        fl = SIGMA_C[absS]
        b1 = V[:, fl].astype(np.int64).sum(1); b2 = P["V2"][:, fl].astype(np.int64).sum(1)
        bad += int(not (np.array_equal(b1, P1d[P["perm"]]) and np.array_equal(b2, P2d[P["perm"]])))
        c["black_view"] += 1
        c["mismatches"] += bad
        # C2 y-state per perspective
        VK = self.w["VK"].astype(np.int64)
        for cc in (0, 1):
            x = sorted(ent["X"][cc])
            full = wrap16(VK[:, x].sum(1))
            if pent is None or ds.ori[pent["node"], cc] != ds.ori[t, cc]:
                y = full; c["y_recompute"] += 1
            else:
                xa = sorted(ent["X"][cc] - pent["X"][cc]); xr = sorted(pent["X"][cc] - ent["X"][cc])
                y = pent["y"][cc].copy()
                for f in xa:
                    y += self.w["VK"][:, f].astype(np.int16)
                for f in xr:
                    y -= self.w["VK"][:, f].astype(np.int16)
            ent["y"][cc] = y
            c["y_checks"] += 1
            c["mismatches"] += int(not np.array_equal(y, full))

    # --- at each accumulator build: compute every candidate and compare with refresh ---
    def check_eval(self, sim, stack, i, c, cands, info):
        ds = self.ds
        t = stack[i]["node"]
        St = stack[i]["S"][c]
        vals = {}
        ref = self.apply(self.bias(), St, [])
        vals["refresh"] = ref
        kt = (c,) + ds.key(t, c)
        C = sim.finny.get(kt, set())
        Cacc = self.cache.get(kt, self.bias())
        vals["finny"] = self.apply(Cacc, St - C, C - St)
        j = info["u_idx"]
        if "hybrid" in cands:
            u = stack[i - 1]["node"]
            ku = (c,) + ds.key(u, c)
            Cold = sim.finny.get(ku, set()); Coldacc = self.cache.get(ku, self.bias())
            Sprev = stack[i - 1]["S"][c]
            old_corr = self.apply(Coldacc, Sprev - Cold, Cold - Sprev)
            new_corr = vals["finny"]
            prev = stack[i - 1]["acc"][c]
            a16 = prev[0] - old_corr[0] + new_corr[0]
            a64 = prev[1] - old_corr[1] + new_corr[1]
            vals["hybrid"] = (a16, a64, prev[2] | old_corr[2] | new_corr[2])
        self.chain_vals = {}
        if "chain" in cands:
            acc = stack[j]["acc"][c]
            prevS = stack[j]["S"][c]
            for (m, merged) in info["chain_steps"]:
                Sm = stack[m]["S"][c]
                acc = self.apply(acc, Sm - prevS, prevS - Sm)
                prevS = Sm
                self.chain_vals[m] = acc
            vals["chain"] = acc
        if j >= 0:
            Su = stack[j]["S"][c]
            vals["ancestor"] = self.apply(stack[j]["acc"][c], St - Su, Su - St)
        if "woodbury" in cands:
            u = stack[j]["node"]
            au = stack[j]["acc"][c]
            Ut = self.w["U"][ds.blk[t, c], ds.bucket[t, c]].astype(np.int64)
            Uu = self.w["U"][ds.blk[u, c], ds.bucket[u, c]].astype(np.int64)
            yu = stack[j]["y"][c].astype(np.int64)
            corr = wrap16((Ut - Uu) @ yu)
            a16 = au[0] + corr
            off = (ds.blk[t, c] * 32 + ds.bucket[t, c]) * 704
            Xt, Xu = stack[i]["X"][c], stack[j]["X"][c]
            for x in sorted(Xt - Xu):
                a16 = a16 + self.w["T"][off + x]
            for x in sorted(Xu - Xt):
                a16 = a16 - self.w["T"][off + x]
            vals["woodbury"] = (a16, ref[1].copy(), np.zeros(L, bool))
        for k, v in vals.items():
            self.bump(k, np.array_equal(v[0], ref[0]))
        # decode check: dec16(wrapped) == exact integer when the final value is in range
        exact = ref[1]
        inr = (exact >= -32768) & (exact <= 32767)
        self.lanes_checked_inrange += int(inr.sum())
        self.lanes_out_of_range_final += int((~inr).sum())
        for k, v in vals.items():
            dec = v[0].astype(np.int64)
            self.dec_mismatch += int(np.any(dec[inr] != exact[inr]))
            self.wrap_lanes_exact += int((v[2] & inr).sum())
        self.cand_vals = vals


def saturation_experiment(ds, w, n_games=200):
    """Sequential incremental updates along game sequences with (a) wrapping int16 adds and
    (b) saturating adds (paddsw semantics), compared with the corresponding from-scratch
    computation (adds in sorted feature order from the bias)."""
    T64, b64 = w["T64"], w["b64"]
    sat = lambda v: np.clip(v, -32768, 32767)
    res = dict(positions=0, wrap_mismatch_positions=0, sat_mismatch_positions=0,
               sat_mismatch_lanes=0, lanes=0, sat_mismatch_lanes_final_in_range=0)
    roots = np.flatnonzero(ds.parent == -1)
    games = 0
    for t in range(ds.n):
        if ds.parent[t] == -1:
            games += 1
            if games > n_games:
                break
        for c in (0, 1):
            S = sorted(ds.S(t, c))
            # from scratch
            sacc = b64.copy()
            for f in S:
                sacc = sat(sacc + T64[f])
            exact = b64 + T64[S].sum(0)
            if ds.parent[t] == -1:
                st_sat = sacc.copy(); st_wrap = wrap16(exact)
            else:
                pS = prevS[c]
                st_sat = prev_sat[c].copy(); st_wrap = prev_wrap[c].copy()
                for f in sorted(set(S) - pS):
                    st_sat = sat(st_sat + T64[f]); st_wrap = st_wrap + w["T"][f]
                for f in sorted(pS - set(S)):
                    st_sat = sat(st_sat - T64[f]); st_wrap = st_wrap - w["T"][f]
            if c == 0:
                prev_sat_new, prev_wrap_new, prevS_new = [None, None], [None, None], [None, None]
            prev_sat_new[c], prev_wrap_new[c], prevS_new[c] = st_sat, st_wrap, set(S)
            res["positions"] += 1
            res["lanes"] += L
            res["wrap_mismatch_positions"] += int(not np.array_equal(st_wrap, wrap16(exact)))
            mm = st_sat != sacc
            res["sat_mismatch_positions"] += int(mm.any())
            res["sat_mismatch_lanes"] += int(mm.sum())
            inr = (exact >= -32768) & (exact <= 32767)
            res["sat_mismatch_lanes_final_in_range"] += int((mm & inr).sum())
        prev_sat, prev_wrap, prevS = prev_sat_new, prev_wrap_new, prevS_new
    return res


def main():
    t0 = time.time()
    out = dict(config=dict(L=L, r=R_PSI, m=M_HEADS, rK=R_K, seed=SEED,
                           note="Simplified K32H3 (no threat/pawn-pair columns); random integer weights; "
                                "merged C2 table W[blk,kappa]=W0[blk]+U[blk,kappa]VK materialised mod 2^16"),
               runs=[])
    rng = np.random.default_rng(SEED)
    psi = make_psi(rng)
    datasets = [("games_lichess", "eager", dict(roots=300)),
                ("games_selfplay", "eager", dict()),
                ("trees", "eager", dict(roots=15)),
                ("trees", "lazy", dict(roots=40))]
    for stress in (False, True):
        w = make_weights(np.random.default_rng(SEED + int(stress)), stress)
        for (name, mode, kw) in datasets:
            ds = Dataset(name, eval_mode=mode, seed=SEED, **kw)
            for policy, mu in (("SF", 1.0), ("VN", 1.0), ("VN", 0.0)):
                ex = ExactState(w, psi, ds)
                sim = Sim(ds, policy=policy, rK=R_K, mu=mu, exact=ex, record=False).run()
                r = dict(dataset=name, eval_mode=mode, nodes=ds.n, stress_weights=stress, policy=policy,
                         mu=mu, builds=sim.tot["builds"], evals=sim.tot["evals"], chosen=sim.src_count,
                         candidate_checks={k: dict(checks=v[0], mismatches=v[1]) for k, v in ex.stats.items()},
                         decode_mismatch_events=ex.dec_mismatch,
                         lanes_final_in_range=ex.lanes_checked_inrange,
                         lanes_final_out_of_range=ex.lanes_out_of_range_final,
                         lane_results_exact_despite_intermediate_overflow=ex.wrap_lanes_exact,
                         psi_checks=ex.psi_checks)
                out["runs"].append(r)
                print(name, mode, stress, policy, mu, sim.tot["builds"], sim.src_count,
                      {k: v for k, v in ex.stats.items()}, ex.dec_mismatch, ex.wrap_lanes_exact,
                      ex.psi_checks["mismatches"], round(time.time() - t0, 1), flush=True)
        ds = Dataset("games_lichess", roots=200)
        out.setdefault("saturation", {})["stress" if stress else "normal"] = saturation_experiment(ds, w)
        print("saturation", stress, out["saturation"], flush=True)
    tot_checks = sum(v["checks"] for r in out["runs"] for v in r["candidate_checks"].values())
    tot_mm = sum(v["mismatches"] for r in out["runs"] for v in r["candidate_checks"].values())
    psi_tot = sum(sum(v for k, v in r["psi_checks"].items() if k not in ("mismatches", "y_recompute"))
                  for r in out["runs"])
    psi_mm = sum(r["psi_checks"]["mismatches"] for r in out["runs"])
    out["summary"] = dict(candidate_checks=tot_checks, candidate_mismatches=tot_mm,
                          decode_mismatch_events=sum(r["decode_mismatch_events"] for r in out["runs"]),
                          psi_and_feature_checks=psi_tot, psi_mismatches=psi_mm,
                          lanes_exact_despite_intermediate_overflow=sum(
                              r["lane_results_exact_despite_intermediate_overflow"] for r in out["runs"]),
                          seconds=time.time() - t0)
    print(out["summary"])
    json.dump(out, open(RES + "/e1_exactness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
