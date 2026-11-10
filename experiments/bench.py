"""Wall-clock microbenchmark of numpy / torch-CPU update kernels (rough proxy only).

CAVEAT: Python/numpy dispatch overhead (~1 us per call) dominates 2 KB column adds; these
numbers are NOT representative of SIMD C++ in an engine.  Batched kernels amortise dispatch
and are a (weak) proxy for memory traffic.  Single thread.
Writes results/bench.json
"""
import json, time, os
import numpy as np
import torch

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
torch.set_num_threads(1)
rng = np.random.default_rng(99)
L, NF, R, M, RK = 1024, 67584, 32, 16, 16
W = rng.integers(-100, 101, (NF, L), dtype=np.int16)       # 138.4 MB K32H3-sized table
V = rng.integers(-127, 128, (768, R)).astype(np.int16)      # feature-major PSI table
V2 = (V.astype(np.int32) ** 2)
Lam = rng.integers(-127, 128, (M, R)).astype(np.int32)
dU = rng.integers(-50, 51, (L, RK)).astype(np.int16)
b = rng.integers(-100, 101, L).astype(np.int16)


def timeit(fn, n, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for i in range(n):
            fn(i)
        ts.append((time.perf_counter_ns() - t0) / n)
    return float(np.median(ts))


def main():
    out = dict(caveat="Python/numpy/torch wall clock; dispatch overhead dominates; weak proxy for SIMD C++. "
                      "Single thread, median of 5 repetitions.", L=L, table_MB=W.nbytes / 1e6, unbatched={}, batched={})
    N = 20000
    ia = rng.integers(0, NF, (N, 4)); ir = rng.integers(0, NF, (N, 4))
    fa = rng.integers(0, 768, (N, 2)); fr = rng.integers(0, 768, (N, 2))
    S = rng.integers(0, NF, (N, 32))
    acc = [b.copy(), b.copy()]
    P1 = np.zeros(R, np.int16); P2 = np.zeros(R, np.int32)
    y = rng.integers(-300, 300, RK).astype(np.int16)

    def nnue_quiet(i):  # 2 perspectives x (1 add + 1 sub) columns
        for c in (0, 1):
            a = acc[c]
            a += W[ia[i, 2 * c]]
            a -= W[ir[i, 2 * c]]

    def psi_quiet(i):
        np.add(P1, V[fa[i, 0]], out=P1); np.subtract(P1, V[fr[i, 0]], out=P1)
        np.add(P2, V2[fa[i, 0]], out=P2); np.subtract(P2, V2[fr[i, 0]], out=P2)

    def refresh(i):
        acc[0][:] = b + W[S[i]].sum(0, dtype=np.int16)

    def woodbury(i):
        a = acc[0]
        a += (dU.astype(np.int32) @ y.astype(np.int32)).astype(np.int16)
        a += W[ia[i, 0]]; a -= W[ir[i, 0]]

    P1r = rng.integers(-4000, 4000, R).astype(np.int64); P2r = rng.integers(0, 500000, R).astype(np.int64)

    def psi_readout(i):
        E = (P1r * P1r - P2r) >> 1
        e = np.clip(E >> 6, -32768, 32767)
        z = Lam @ e
        return np.clip(z >> 8, 0, 127)

    for name, fn in [("nnue_quiet_move_both_persp_4cols", nnue_quiet), ("psi_quiet_move_P1_P2", psi_quiet),
                     ("refresh_32cols_one_persp", refresh), ("woodbury_rK16_plus_2cols_one_persp", woodbury),
                     ("psi_readout_one_persp_m16", psi_readout)]:
        out["unbatched"][name] = dict(ns_per_call=timeit(fn, N if "refresh" not in name else N // 4))
        print(name, out["unbatched"][name], flush=True)

    # batched: B independent updates per call (gather + add); bytes = columns touched
    B = 4096
    Wt = torch.from_numpy(W)
    Vt = torch.from_numpy(V); V2t = torch.from_numpy(V2)
    accB = torch.zeros(B, L, dtype=torch.int16)
    P1B = torch.zeros(B, R, dtype=torch.int16); P2B = torch.zeros(B, R, dtype=torch.int32)
    ia_t = [torch.from_numpy(rng.integers(0, NF, B)) for _ in range(8)]
    fa_t = [torch.from_numpy(rng.integers(0, 768, B)) for _ in range(8)]
    BR = 256
    S_t = torch.from_numpy(rng.integers(0, NF, (BR, 32)))
    accR = torch.zeros(BR, L, dtype=torch.int16)

    def b_nnue(i):  # 4 column ops per event (quiet move, both perspectives folded into one acc)
        accB.add_(Wt[ia_t[i % 8]]); accB.sub_(Wt[ia_t[(i + 1) % 8]])
        accB.add_(Wt[ia_t[(i + 2) % 8]]); accB.sub_(Wt[ia_t[(i + 3) % 8]])

    def b_psi(i):
        P1B.add_(Vt[fa_t[i % 8]]); P1B.sub_(Vt[fa_t[(i + 1) % 8]])
        P2B.add_(V2t[fa_t[i % 8]]); P2B.sub_(V2t[fa_t[(i + 1) % 8]])

    def b_refresh(i):
        accR.copy_(Wt[S_t].sum(1, dtype=torch.int16))

    for name, fn, cols_bytes in [("nnue_quiet_4cols", b_nnue, 4 * 2 * L), ("psi_quiet_P1_P2", b_psi, 2 * (2 * R + 4 * R)),
                                 ("refresh_32cols", b_refresh, 32 * 2 * L)]:
        n = 20 if "refresh" not in name else 3
        ns = timeit(fn, n, reps=5) / (BR if 'refresh' in name else B)
        out["batched"][name] = dict(ns_per_event=ns, column_bytes_per_event=cols_bytes,
                                    effective_GBps=cols_bytes / ns)
        print(name, out["batched"][name], flush=True)
    out["ratios"] = dict(
        unbatched_psi_over_nnue_quiet=out["unbatched"]["psi_quiet_move_P1_P2"]["ns_per_call"] /
        out["unbatched"]["nnue_quiet_move_both_persp_4cols"]["ns_per_call"],
        batched_psi_over_nnue_quiet=out["batched"]["psi_quiet_P1_P2"]["ns_per_event"] /
        out["batched"]["nnue_quiet_4cols"]["ns_per_event"])
    json.dump(out, open(RES + "/bench.json", "w"), indent=1)
    print(out["ratios"])


if __name__ == "__main__":
    main()
