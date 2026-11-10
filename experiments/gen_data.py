"""Data generation: real game sequences (lichess 2013-01, CC0), self-play games and
alpha-beta search traces.  All seeded.  Output: data/*.npz

Usage: python gen_data.py  (writes data/games_lichess.npz, data/games_selfplay.npz,
                            data/trees.npz, data/positions.npz)
"""
import os
import io, sys, time, json, random
import numpy as np
import chess, chess.pgn
from compression import zstd
from common import *

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PGN_ZST = DATA + "/lichess/lichess_db_standard_rated_2013-01.pgn.zst"
SEED = 12345

N_LICHESS_GAMES = 3000          # move sequences + training positions
N_THREAT_GAMES = 600            # threat-proxy churn computed on this many lichess games
N_SELFPLAY = 30
N_ROOTS_LICHESS = 160
N_ROOTS_SELF = 40
SEARCH_DEPTH = 3
QS_MAX = 4
NODE_CAP = 2500


class NodeStore:
    def __init__(self):
        self.cols = {k: [] for k in ["parent", "absset", "ksq", "blk", "npieces", "mtype", "mover",
                                     "is_king", "mfrom", "mto", "rem", "add", "kind", "threat", "seq"]}

    def add(self, parent, board, minfo=None, kind=0, threat=-1, seq=0):
        a = np.array(abs_set(board), dtype=np.int64)
        pad = np.full(32, -1, np.int16); pad[:len(a)] = a
        c = self.cols
        c["parent"].append(parent); c["absset"].append(pad)
        c["ksq"].append([king_sq(a, 0), king_sq(a, 1)])
        c["blk"].append([block_of(a, 0), block_of(a, 1)])
        c["npieces"].append(len(a))
        if minfo is None:
            c["mtype"].append(-1); c["mover"].append(-1); c["is_king"].append(0)
            c["mfrom"].append(-1); c["mto"].append(-1)
            c["rem"].append([-1, -1]); c["add"].append([-1, -1])
        else:
            mt, rem, add, isk, mover, mv = minfo
            c["mtype"].append(mt); c["mover"].append(mover); c["is_king"].append(int(isk))
            c["mfrom"].append(mv.from_square); c["mto"].append(mv.to_square)
            c["rem"].append(rem + [-1] * (2 - len(rem))); c["add"].append(add + [-1] * (2 - len(add)))
        c["kind"].append(kind); c["threat"].append(threat); c["seq"].append(seq)
        return len(c["parent"]) - 1

    def save(self, path):
        dt = dict(parent=np.int32, absset=np.int16, ksq=np.int8, blk=np.int8, npieces=np.int8,
                  mtype=np.int8, mover=np.int8, is_king=np.int8, mfrom=np.int8, mto=np.int8,
                  rem=np.int16, add=np.int16, kind=np.int8, threat=np.int16, seq=np.int32)
        np.savez_compressed(path, **{k: np.array(v, dtype=dt[k]) for k, v in self.cols.items()})
        return len(self.cols["parent"])


def minfo_of(board, mv):
    mt, rem, add, isk = move_info(board, mv)
    return (mt, rem, add, isk, 0 if board.turn == chess.WHITE else 1, mv)


# ---------------- simple material + PST evaluator (for self-play / search) ----------------
VAL = piece_values()
CENTER = np.array([[min(f, 7 - f) + min(r, 7 - r) for f in range(8)] for r in range(8)]).ravel()


def evaluate(board):
    s = 0
    for sq, p in board.piece_map().items():
        v = VAL[p.piece_type] + (4 * CENTER[sq] if p.piece_type in (2, 3) else 0)
        if p.piece_type == 1:
            v += 5 * ((sq >> 3) if p.color else (7 - (sq >> 3)))
        s += v if p.color else -v
    return s if board.turn else -s


def ordered(board, captures_only=False):
    mv = list(board.generate_legal_captures() if captures_only else board.legal_moves)
    if captures_only:
        mv += [m for m in board.legal_moves if m.promotion and not board.is_capture(m)]
    def key(m):
        if board.is_capture(m):
            v = board.piece_at(m.to_square)
            vv = VAL[v.piece_type] if v else 100
            return -(10 * vv - VAL[board.piece_at(m.from_square).piece_type] // 10) - 100000
        return -(VAL[m.promotion] if m.promotion else 0)
    mv.sort(key=key)
    return mv


class Abort(Exception):
    pass


class Searcher:
    def __init__(self, store=None, cap=NODE_CAP):
        self.store, self.cap, self.n = store, cap, 0

    def rec(self, parent, board, mv_info, kind):
        self.n += 1
        if self.store is None:
            return -1
        if self.n > self.cap:
            raise Abort
        return self.store.add(parent, board, mv_info, kind)

    def qs(self, board, alpha, beta, node, qd):
        stand = evaluate(board)
        if stand >= beta or qd >= QS_MAX:
            return stand
        alpha = max(alpha, stand)
        for m in ordered(board, True):
            mi = minfo_of(board, m) if self.store is not None else None
            board.push(m)
            child = self.rec(node, board, mi, 2)
            sc = -self.qs(board, -beta, -alpha, child, qd + 1)
            board.pop()
            if sc >= beta:
                return sc
            alpha = max(alpha, sc)
        return alpha

    def ab(self, board, depth, alpha, beta, node):
        if board.is_game_over():
            return -30000 if board.is_checkmate() else 0
        if depth == 0:
            return self.qs(board, alpha, beta, node, 0)
        best = -10 ** 9
        for m in ordered(board):
            mi = minfo_of(board, m) if self.store is not None else None
            board.push(m)
            child = self.rec(node, board, mi, 2 if depth == 1 else 1)
            sc = -self.ab(board, depth - 1, -beta, -alpha, child)
            board.pop()
            if sc > best:
                best = sc
            alpha = max(alpha, sc)
            if alpha >= beta:
                break
        return best

    def bestmove(self, board, depth):
        best, bm = -10 ** 9, None
        for m in ordered(board):
            board.push(m)
            sc = -self.ab(board, depth - 1, -10 ** 9, 10 ** 9, -1)
            board.pop()
            if sc > best:
                best, bm = sc, m
        return bm


def read_lichess(n):
    games = []
    with zstd.open(PGN_ZST, "rt", encoding="utf-8") as fh:
        while len(games) < n:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            if g.headers.get("Variant", "Standard") != "Standard" or "FEN" in g.headers:
                continue
            mv = list(g.mainline_moves())
            if len(mv) < 10:
                continue
            res = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(g.headers.get("Result"), None)
            if res is None:
                continue
            games.append((mv, res))
    return games


def main():
    rng = random.Random(SEED)
    t0 = time.time()
    meta = {}
    games = read_lichess(N_LICHESS_GAMES)
    print("lichess games", len(games), time.time() - t0, flush=True)

    # ---- lichess game sequences + training positions ----
    st = NodeStore()
    pos_abs, pos_res, pos_stm, pos_ply, pos_game = [], [], [], [], []
    for gi, (mv, res) in enumerate(games):
        b = chess.Board()
        node = st.add(-1, b, None, 3, -1 if gi >= N_THREAT_GAMES else 0, gi)
        th = threat_set(b) if gi < N_THREAT_GAMES else None
        for ply, m in enumerate(mv):
            mi = minfo_of(b, m)
            b.push(m)
            tc = -1
            if th is not None:
                th2 = threat_set(b); tc = len(th ^ th2); th = th2
            node = st.add(node, b, mi, 3, tc, gi)
            if ply >= 8:
                a = np.full(32, -1, np.int16); s = abs_set(b); a[:len(s)] = s
                pos_abs.append(a); pos_res.append(res); pos_stm.append(int(b.turn)); pos_ply.append(ply)
                pos_game.append(gi)
    n = st.save(DATA + "/games_lichess.npz")
    meta["lichess_nodes"] = n
    meta["lichess_games"] = len(games)
    np.savez_compressed(DATA + "/positions.npz", absset=np.array(pos_abs), result=np.array(pos_res, np.float32),
                        stm=np.array(pos_stm, np.int8), ply=np.array(pos_ply, np.int16),
                        game=np.array(pos_game, np.int32))
    meta["positions"] = len(pos_abs)
    print("lichess nodes", n, "positions", len(pos_abs), time.time() - t0, flush=True)

    # ---- self-play games (depth-2 alpha-beta + qsearch, 8 random opening plies) ----
    st2 = NodeStore()
    sp_boards = []
    for gi in range(N_SELFPLAY):
        b = chess.Board()
        node = st2.add(-1, b, None, 3, 0, gi)
        th = threat_set(b)
        srch = Searcher(None)
        for ply in range(200):
            if b.is_game_over(claim_draw=True):
                break
            if ply < 8:
                m = rng.choice(list(b.legal_moves))
            else:
                m = srch.bestmove(b, 2)
            mi = minfo_of(b, m)
            b.push(m)
            th2 = threat_set(b); tc = len(th ^ th2); th = th2
            node = st2.add(node, b, mi, 3, tc, gi)
            if ply >= 8 and rng.random() < 0.05:
                sp_boards.append(b.copy(stack=False))
    meta["selfplay_nodes"] = st2.save(DATA + "/games_selfplay.npz")
    print("selfplay nodes", meta["selfplay_nodes"], time.time() - t0, flush=True)

    # ---- search traces ----
    roots = []
    lg = [g for g in games if len(g[0]) > 20]
    while len(roots) < N_ROOTS_LICHESS:
        mv, _ = rng.choice(lg)
        k = rng.randrange(6, len(mv) - 1)
        b = chess.Board()
        for m in mv[:k]:
            b.push(m)
        if not b.is_game_over():
            roots.append(b)
    rng.shuffle(sp_boards)
    roots += [b for b in sp_boards if not b.is_game_over()][:N_ROOTS_SELF]
    st3 = NodeStore()
    aborted = 0
    for ri, b in enumerate(roots):
        b = b.copy(stack=False)
        st3.cols  # noqa
        root = st3.add(-1, b, None, 0, -1, ri)
        s = Searcher(st3)
        s.n = 1
        try:
            s.ab(b, SEARCH_DEPTH, -10 ** 9, 10 ** 9, root)
        except Abort:
            aborted += 1
        # fill seq for nodes of this tree
        L = len(st3.cols["seq"])
        for j in range(root, L):
            st3.cols["seq"][j] = ri
    meta["tree_nodes"] = st3.save(DATA + "/trees.npz")
    meta["tree_roots"] = len(roots)
    meta["tree_aborted_at_cap"] = aborted
    meta["search"] = dict(depth=SEARCH_DEPTH, qs_max=QS_MAX, node_cap=NODE_CAP)
    meta["seconds"] = time.time() - t0
    print(meta, flush=True)
    json.dump(meta, open(DATA + "/gen_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
