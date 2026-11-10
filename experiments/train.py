"""E5-E8 tiny training sanity checks (CPU).  Establishes only that the math trains /
quantizes as specified; says nothing about chess strength.

E5  rank sufficiency of low-rank king transitions on a planted synthetic K32H3-shaped table
E6  does a PSI channel learn piece-pair interactions? synthetic teachers on real positions
E7  tied vs untied PSI embedding under colour flip
E8  real-ish labels (game result; material+PST) -- weak and noisy
Writes results/e5_e8_training.json
"""
import json, time, os, math
import numpy as np
import torch, torch.nn as nn
import chess
from common import SIGMA_C

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
SEEDS = [0, 1, 2]
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH = 1024
PAD = 768


# ------------------------------------------------------------------ E5
def e5():
    rng = np.random.default_rng(5)
    Lw, F, K, rtrue = 256, 704, 32, 8
    out = {}
    for name, noise, planted in (("planted_rank8_noise0.1", 0.1, True), ("planted_rank8_noise0.5", 0.5, True),
                                 ("unstructured", 1.0, False)):
        W0 = rng.normal(0, 1, (Lw, F))
        if planted:
            U = rng.normal(0, 1, (K, Lw, rtrue)); V = rng.normal(0, 1 / math.sqrt(rtrue), (rtrue, F))
            Wk = np.stack([W0 + U[k] @ V for k in range(K)]) + rng.normal(0, noise, (K, Lw, F))
        else:
            Wk = np.stack([W0 + rng.normal(0, 1, (Lw, F)) for _ in range(K)])
        Wbar = Wk.mean(0)
        Rm = (Wk - Wbar).reshape(K * Lw, F)            # residual stacked over buckets
        # Shared V_K across buckets: best rank-r row space of the stacked residual
        Uu, s, Vt = np.linalg.svd(Rm, full_matrices=False)
        tot = (Rm ** 2).sum()
        rows = []
        for rK in (1, 2, 4, 8, 16, 32, 64):
            Vk = Vt[:rK]                                  # (rK, F)
            Uk = Rm @ Vk.T                                # (K*Lw, rK)
            rec = Uk @ Vk
            err = ((Rm - rec) ** 2).sum() / tot
            # integer factors: V_K int8, U int16 at table scale s_tab (stored table int16 = round(s_tab*W))
            s_tab = 64.0
            sv = 127 / np.abs(Vk).max()
            Vq = np.round(Vk * sv)                        # int8
            Uq = np.round(Uk * s_tab / sv)                # int16
            assert np.abs(Vq).max() <= 127 and np.abs(Uq).max() < 32768
            recq = (Uq @ Vq) / s_tab                      # materialised integer product, back in float units
            errq = ((Rm - recq) ** 2).sum() / tot
            # relative to plain int16 quantisation of the full table at the same scale
            errfull = ((Rm - np.round(Rm * s_tab) / s_tab) ** 2).sum() / tot
            rows.append(dict(rK=rK, rel_err_float=float(err), rel_err_int_factors=float(errq),
                             rel_err_plain_int16_table=float(errfull)))
        out[name] = rows
        print("E5", name, [(r["rK"], round(r["rel_err_float"], 4), round(r["rel_err_int_factors"], 4)) for r in rows], flush=True)
    return out


# ------------------------------------------------------------------ data
def load_positions():
    d = np.load(DATA + "/positions.npz")
    a = d["absset"].astype(np.int64)
    a[a < 0] = PAD
    return a, d["result"].astype(np.float32), d["game"]


def attack_pairs():
    """Sparse 'attack geometry' pair set: (i, j) with piece i's empty-board attack set
    containing j's square (i != j, any colours)."""
    A = np.zeros((768, 768), bool)
    for f in range(768):
        col, t, sq = f // 384, (f // 64) % 6 + 1, f % 64
        if t == 1:
            bb = chess.BB_PAWN_ATTACKS[col == 0][sq]
        elif t == 2:
            bb = chess.BB_KNIGHT_ATTACKS[sq]
        elif t == 6:
            bb = chess.BB_KING_ATTACKS[sq]
        else:
            bb = 0
            if t in (3, 5):
                bb |= chess.BB_DIAG_ATTACKS[sq][0]
            if t in (4, 5):
                bb |= chess.BB_RANK_ATTACKS[sq][0] | chess.BB_FILE_ATTACKS[sq][0]
        for s2 in chess.SquareSet(bb):
            A[f, np.arange(768) % 64 == s2] = True
    A = A | A.T
    np.fill_diagonal(A, False)
    return A


def pair_term(idx, Q, chunk=4096):
    out = np.zeros(len(idx), np.float64)
    Qp = np.zeros((769, 769)); Qp[:768, :768] = Q
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        blk = Qp[ii[:, :, None], ii[:, None, :]]
        out[s:s + chunk] = 0.5 * blk.sum((1, 2))            # Q symmetric, zero diagonal
    return out


def make_teachers(idx, rng):
    pst = rng.normal(0, 1, 768)
    lin = np.append(pst, 0)[idx].sum(1)
    std = lambda v: (v - v.mean()) / v.std()
    A = rng.normal(0, 1, (8, 768)); s = rng.normal(0, 1, 8)
    Qlr = (A.T * s) @ A; np.fill_diagonal(Qlr, 0)
    Ag = attack_pairs()
    G = rng.normal(0, 1, (768, 768)); G = np.triu(G, 1); G = G + G.T
    Qsp = np.where(Ag, G, 0.0)
    plr, psp = std(pair_term(idx, Qlr)), std(pair_term(idx, Qsp))
    linz = std(lin)
    noise = rng.normal(0, 0.1, len(idx))
    T = {"i_lowrank": linz + plr, "ii_sparse_attack": linz + psp, "iii_both": linz + plr + psp}
    T = {k: std(v + noise).astype(np.float32) for k, v in T.items()}
    meta = dict(attack_pairs=int(Ag.sum() // 2), noise_sd_before_std=0.1)
    return T, Ag, meta


def pair_ids(idx, Ag, maxp=None):
    """Active attack-geometry pair ids per position (for model M2)."""
    pid = -np.ones((769, 769), np.int64)
    iu = np.argwhere(np.triu(Ag, 1))
    pid[iu[:, 0], iu[:, 1]] = np.arange(len(iu)); pid[iu[:, 1], iu[:, 0]] = np.arange(len(iu))
    lists = []
    for s in range(0, len(idx), 4096):
        ii = idx[s:s + 4096]
        blk = pid[ii[:, :, None], ii[:, None, :]]
        tri = np.triu(np.ones((32, 32), bool), 1)
        for b in blk:
            v = b[tri]; lists.append(v[v >= 0])
    mp = max(len(v) for v in lists)
    out = np.full((len(idx), mp), len(iu), np.int64)
    for k, v in enumerate(lists):
        out[k, :len(v)] = v
    return out, len(iu)


# ------------------------------------------------------------------ models
class Net(nn.Module):
    """NNUE-like: EmbeddingBag FT (768 abs features, single White-absolute perspective) ->
    clipped ReLU -> SF-style pairwise product -> 32 -> 1.  Optional PSI channel (r lanes,
    m heads, u = crelu(Lam E + bz) appended to fc_0 input).  Optional explicit pair features
    (M2) added into the FT accumulator."""

    def __init__(self, L, r=0, m=0, tied=False, npairs=0):
        super().__init__()
        self.L, self.r, self.m, self.tied = L, r, m, tied
        self.ft = nn.EmbeddingBag(769, L, mode="sum", padding_idx=PAD)
        nn.init.normal_(self.ft.weight, 0, 0.1)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()
        self.ftb = nn.Parameter(torch.full((L,), 0.5))
        self.pairs = None
        if npairs:
            self.pairs = nn.EmbeddingBag(npairs + 1, L, mode="sum", padding_idx=npairs, sparse=True)
            nn.init.normal_(self.pairs.weight, 0, 0.02)
            with torch.no_grad():
                self.pairs.weight[npairs].zero_()
        if r:
            rows = r // 2 if tied else r
            self.V = nn.Parameter(torch.randn(rows, 768) * 0.1)
            self.Lam = nn.Parameter(torch.randn(m, r) * 0.1)
            self.bz = nn.Parameter(torch.full((m,), 0.5))
            self.register_buffer("sig", torch.from_numpy(SIGMA_C))
        self.fc0 = nn.Linear(L // 2 + m, 32)
        self.fc1 = nn.Linear(32, 1)

    def fullV(self):
        if not self.tied:
            return self.V
        H = self.V
        out = torch.stack([H, H[:, self.sig]], 1).reshape(self.r, 768)   # V[t+1, j] = V[t, sigma(j)]
        return out

    def psi(self, idx):
        V = torch.cat([self.fullV(), torch.zeros(self.r, 1)], 1)   # pad column
        g = V.t()[idx]                       # (B, 32, r)
        P1, P2 = g.sum(1), (g * g).sum(1)
        E = 0.5 * (P1 * P1 - P2)
        z = E @ self.Lam.t() + self.bz
        return torch.clamp(z, 0, 1), P1, P2

    def forward(self, idx, pidx=None):
        a = self.ft(idx) + self.ftb
        if self.pairs is not None:
            a = a + self.pairs(pidx)
        a = torch.clamp(a, 0, 1)
        h = a[:, :self.L // 2] * a[:, self.L // 2:]
        if self.r:
            u, _, _ = self.psi(idx)
            h = torch.cat([h, u], 1)
        return self.fc1(torch.clamp(self.fc0(h), 0, 1)).squeeze(1)


def nparams(model):
    return sum(p.numel() for p in model.parameters())


def train_eval(model, Xtr, ytr, Xva, yva, seed, Ptr=None, Pva=None, sigmoid=False):
    g = torch.Generator().manual_seed(seed)
    sp = [p for n, p in model.named_parameters() if n.startswith("pairs.")]
    dense = [p for n, p in model.named_parameters() if not n.startswith("pairs.")]
    opt = torch.optim.Adam(dense, lr=2e-3)
    opt_sp = torch.optim.SparseAdam(sp, lr=2e-3) if sp else None
    steps = EPOCHS * math.ceil(len(Xtr) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    out_fn = (lambda o: torch.sigmoid(o)) if sigmoid else (lambda o: o)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr), generator=g)
        model.train()
        for s in range(0, len(Xtr), BATCH):
            b = perm[s:s + BATCH]
            pi = torch.from_numpy(Ptr[b.numpy()]) if Ptr is not None else None
            loss = ((out_fn(model(Xtr_t[b], pi)) - ytr_t[b]) ** 2).mean()
            opt.zero_grad()
            if opt_sp: opt_sp.zero_grad()
            loss.backward(); opt.step(); sched.step()
            if opt_sp: opt_sp.step()
    model.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, len(Xva), 8192):
            pi = torch.from_numpy(Pva[s:s + 8192]) if Pva is not None else None
            preds.append(out_fn(model(torch.from_numpy(Xva[s:s + 8192]), pi)))
        p = torch.cat(preds).numpy()
    return float(((p - yva) ** 2).mean())


def model_specs(npairs):
    return {
        "M0_L256": dict(L=256),
        "M0s_L192": dict(L=192),
        "M1_L192_PSI_r32_m16": dict(L=192, r=32, m=16),
        "M2_L192_explicit_pairs": dict(L=192, npairs=npairs),
    }


def qat_check(model, Xva, n=4000):
    """Quantise V (int8) and Lam (int8); run the integer PSI pipeline (SPEC C1) and compare with
    float.  Checks: P1^2-P2 even, ranges (T9/T16), integer z equals float z of the quantised
    weights (exact identity), and the relative error introduced by quantisation."""
    with torch.no_grad():
        V = model.fullV().numpy().astype(np.float64); Lam = model.Lam.numpy().astype(np.float64)
        sv = 127 / np.abs(V).max(); sl = 127 / np.abs(Lam).max()
        Vq = np.round(V * sv).astype(np.int64); Lq = np.round(Lam * sl).astype(np.int64)
        Vp = np.concatenate([Vq, np.zeros((Vq.shape[0], 1), np.int64)], 1)
        idx = Xva[:n]
        g = Vp.T[idx]
        P1, P2 = g.sum(1), (g * g).sum(1)
        diff = P1 * P1 - P2
        even = bool(np.all(diff % 2 == 0))
        E = diff // 2
        z_int = E @ Lq.T
        # float model z (no bias) and float evaluation of the quantised weights
        gf = np.concatenate([V, np.zeros((V.shape[0], 1))], 1).T[idx]
        Ef = 0.5 * (gf.sum(1) ** 2 - (gf * gf).sum(1))
        zf = Ef @ Lam.T
        zq_float = z_int / (sv * sv * sl)
        gq = (Vp.T[idx]).astype(np.float64)
        zq_check = (0.5 * (gq.sum(1) ** 2 - (gq * gq).sum(1))) @ Lq.T.astype(np.float64)
        return dict(P1_max_abs=int(np.abs(P1).max()), P1_bound_T9=32 * 127, P2_max=int(P2.max()),
                    P2_bound=32 * 127 ** 2, E_even_all=even, z_int_max_abs=int(np.abs(z_int).max()),
                    z_int_lt_2p31=bool(np.abs(z_int).max() < 2 ** 31),
                    int_pipeline_equals_float_of_quantised=bool(np.array_equal(z_int.astype(np.float64), zq_check)),
                    quant_rel_err_z=float(np.sqrt(((zq_float - zf) ** 2).mean() / max(1e-12, (zf ** 2).mean()))))


def main():
    t0 = time.time()
    out = dict(config=dict(epochs=EPOCHS, batch=BATCH, seeds=SEEDS, optimizer="Adam onecycle max_lr 2e-3 (SparseAdam lr 2e-3 const for M2 pair table)",
                           note="single White-absolute perspective, no king buckets, no layer stacks"))
    out["E5"] = e5()
    idx, result, game = load_positions()
    rng = np.random.default_rng(6)
    ug = np.unique(game); rng.shuffle(ug)
    va_g = set(ug[: len(ug) // 5].tolist())
    is_va = np.array([g in va_g for g in game])
    T, Ag, tmeta = make_teachers(idx, np.random.default_rng(7))
    P, npairs = pair_ids(idx, Ag)
    out["data"] = dict(positions=len(idx), train=int((~is_va).sum()), val=int(is_va.sum()), split="by game (20% games val)",
                       explicit_pair_vocab=npairs, max_active_pairs=int(P.shape[1]),
                       mean_active_pairs=float((P < npairs).sum(1).mean()), **tmeta)
    print(out["data"], flush=True)
    Xtr, Xva = idx[~is_va], idx[is_va]
    Ptr, Pva = P[~is_va], P[is_va]
    specs = model_specs(npairs)
    out["models"] = {}
    for k, s in specs.items():
        m = Net(**s)
        upd = s["L"] + 2 * s.get("r", 0)
        out["models"][k] = dict(params=nparams(m), update_lanes_per_changed_abs_feature=upd,
                                note="M2 additionally updates L lanes per changed active pair" if "npairs" in s else "")
    # ---------------- E6
    out["E6"] = {}
    for tname, y in T.items():
        res = {}
        for mname, s in specs.items():
            mses = []
            for seed in (SEEDS if "npairs" not in s else SEEDS[:1]):   # M2 is slow (72k-pair table): 1 seed
                torch.manual_seed(seed)
                m = Net(**s)
                usep = "npairs" in s
                mse = train_eval(m, Xtr, y[~is_va], Xva, y[is_va], seed, Ptr if usep else None, Pva if usep else None)
                mses.append(mse)
                if mname.startswith("M1") and tname == "iii_both" and seed == 0:
                    out["E6_qat_check"] = qat_check(m, Xva)
            res[mname] = dict(val_mse=mses, mean=float(np.mean(mses)), sd=float(np.std(mses, ddof=1)) if len(mses) > 1 else None)
            print("E6", tname, mname, res[mname], round(time.time() - t0), flush=True)
        out["E6"][tname] = res
        json.dump(out, open(RES + "/e5_e8_training.json", "w"), indent=1)
    # ---------------- E7 tied vs untied
    out["E7"] = {}
    sym = (T["iii_both"] + 0)  # same teacher (not colour-symmetric) plus a colour-symmetrised variant
    idx_fl = np.where(idx == PAD, PAD, SIGMA_C[np.minimum(idx, 767)])
    # symmetrised teacher: y_sym(x) = (y(x) + y(sigma x)) / 2 requires y on flipped boards: recompute
    T2, _, _ = make_teachers(np.concatenate([idx, idx_fl]), np.random.default_rng(7))
    ysym = 0.5 * (T2["iii_both"][:len(idx)] + T2["iii_both"][len(idx):])
    ysym = ((ysym - ysym.mean()) / ysym.std()).astype(np.float32)
    for tname, y in (("iii_both", sym), ("iii_both_colour_symmetrised", ysym)):
        res = {}
        for tied in (False, True):
            mses = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                m = Net(L=192, r=32, m=16, tied=tied)
                mses.append(train_eval(m, Xtr, y[~is_va], Xva, y[is_va], seed))
                if tied and seed == 0:
                    with torch.no_grad():
                        _, P1, P2 = m.psi(torch.from_numpy(Xva[:2000]))
                        _, Q1, Q2 = m.psi(torch.from_numpy(idx_fl[is_va][:2000]))
                        perm = torch.arange(32) ^ 1
                        out.setdefault("E7_T21_check", {})[tname] = dict(
                            max_abs_diff_P1=float((Q1 - P1[:, perm]).abs().max()),
                            max_abs_diff_P2=float((Q2 - P2[:, perm]).abs().max()))
            k = "tied" if tied else "untied"
            res[k] = dict(val_mse=mses, mean=float(np.mean(mses)), sd=float(np.std(mses, ddof=1)),
                          params=nparams(Net(L=192, r=32, m=16, tied=tied)))
            print("E7", tname, k, res[k], round(time.time() - t0), flush=True)
        out["E7"][tname] = res
    json.dump(out, open(RES + "/e5_e8_training.json", "w"), indent=1)
    # ---------------- E8 real-ish labels
    out["E8"] = {}
    from gen_data import VAL, CENTER
    vals = np.zeros(769)
    for f in range(768):
        col, t, sq = f // 384, (f // 64) % 6 + 1, f % 64
        v = VAL[t] + (4 * CENTER[sq] if t in (2, 3) else 0)
        if t == 1:
            v += 5 * ((sq >> 3) if col == 0 else (7 - (sq >> 3)))
        vals[f] = v if col == 0 else -v
    mat = vals[idx].sum(1)
    labels = {"game_result_white_pov": (result, True),
              "material_pst_eval_tanh": (np.tanh(mat / 400).astype(np.float32), False)}
    for lname, (y, sig) in labels.items():
        res = {}
        base = float(((y[is_va] - y[~is_va].mean()) ** 2).mean())
        for mname in ("M0_L256", "M0s_L192", "M1_L192_PSI_r32_m16"):
            mses = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                m = Net(**specs[mname])
                mses.append(train_eval(m, Xtr, y[~is_va], Xva, y[is_va], seed, sigmoid=sig))
            res[mname] = dict(val_mse=mses, mean=float(np.mean(mses)), sd=float(np.std(mses, ddof=1)))
            print("E8", lname, mname, res[mname], round(time.time() - t0), flush=True)
        res["predict_train_mean_mse"] = base
        out["E8"][lname] = res
    out["seconds"] = time.time() - t0
    json.dump(out, open(RES + "/e5_e8_training.json", "w"), indent=1)
    print("done", out["seconds"])


if __name__ == "__main__":
    main()
