"""Shared feature definitions for the VNNUE experiments.

Feature spaces
--------------
* Absolute 768 (White-absolute, SPEC 2.1):  f = colour*384 + (ptype-1)*64 + sq
  colour 0 = White, 1 = Black; ptype 1..6 = P,N,B,R,Q,K (python-chess numbering).
* K32H3-like PSQ features (SPEC 2.0), per perspective c:
    orient(c, ksq) = (56 if c == Black else 0) ^ (7 if file(ksq) < 4 else 0)
    bucket         = canonical own-king square (files e-h) -> 0..31
    plane          = own P..Q -> 0..4, opp P..Q -> 5..9, both kings -> 10   (11 planes)
    local index    = plane*64 + (sq ^ orient)                               (0..703)
    block          = 0: no rook/queen on board; 1: some R/Q but opponent has no queen;
                     2: opponent has a queen
    global index   = (block*32 + bucket)*704 + local                        (0..67583)
  Simplifications vs Stockfish master (documented in RESULTS.md): no Full_Threats / PP_3Wide
  columns inside the accumulator (a threat *proxy* is only counted for op/byte accounting),
  no PSQT side outputs, no layer stacks.
"""
import numpy as np
import chess

N_ABS = 768
N_LOCAL = 704
N_BLOCKS = 3
N_BUCKETS = 32
N_GLOBAL = N_BLOCKS * N_BUCKETS * N_LOCAL  # 67,584

# move type codes
QUIET, CAPTURE, EP, PROMO, CAPPROMO, CASTLE = range(6)
MTYPE_NAMES = ["quiet", "capture", "en_passant", "promotion", "capture_promotion", "castling"]


def abs_feat(colour_white: bool, ptype: int, sq: int) -> int:
    return (0 if colour_white else 1) * 384 + (ptype - 1) * 64 + sq


def sigma_c_array():
    """Colour flip on absolute features: (col,type,sq) -> (1-col,type,sq^56)."""
    f = np.arange(N_ABS)
    col, t, sq = f // 384, (f // 64) % 6, f % 64
    return ((1 - col) * 384 + t * 64 + (sq ^ 56)).astype(np.int64)


SIGMA_C = sigma_c_array()


def abs_set(board: chess.Board):
    return sorted(abs_feat(p.color, p.piece_type, sq) for sq, p in board.piece_map().items())


def orient(persp: int, ksq: int) -> int:
    """persp 0 = White, 1 = Black."""
    return (56 if persp == 1 else 0) ^ (7 if (ksq & 7) < 4 else 0)


def bucket(persp: int, ksq: int) -> int:
    s = ksq ^ orient(persp, ksq)
    return (s >> 3) * 4 + ((s & 7) - 4)


def block_of(absarr: np.ndarray, persp: int) -> int:
    """absarr: 1-d array of active absolute features."""
    t = (absarr // 64) % 6  # 0..5
    col = absarr // 384
    heavy = np.any((t == 3) | (t == 4))
    if not heavy:
        return 0
    opp = 1 - persp
    return 2 if np.any((t == 4) & (col == opp)) else 1


def local_features(absarr: np.ndarray, persp: int, ksq: int) -> np.ndarray:
    """Vectorised K32H3 local (704) indices of an absolute feature array."""
    col = absarr // 384
    t = (absarr // 64) % 6
    sq = absarr % 64
    own = col == persp
    plane = np.where(t == 5, 10, np.where(own, t, 5 + t))
    return (plane * 64 + (sq ^ orient(persp, ksq))).astype(np.int64)


def table_offset(persp: int, ksq: int, blk: int) -> int:
    return (blk * N_BUCKETS + bucket(persp, ksq)) * N_LOCAL


def global_features(absarr, persp, ksq, blk):
    return local_features(absarr, persp, ksq) + table_offset(persp, ksq, blk)


def king_sq(absarr: np.ndarray, persp: int) -> int:
    k = absarr[(absarr // 64) % 6 == 5]
    for f in k:
        if f // 384 == persp:
            return int(f % 64)
    raise ValueError("no king")


def move_info(board: chess.Board, move: chess.Move):
    """Classify a move and return its absolute removed / added feature lists (SPEC 2.4)."""
    mover = board.turn
    piece = board.piece_at(move.from_square)
    rem, add = [abs_feat(mover, piece.piece_type, move.from_square)], []
    if board.is_castling(move):
        mtype = CASTLE
        # python-chess encodes castling as king-takes-own-rook or king two squares
        rank = move.from_square & 56
        kside = board.is_kingside_castling(move)
        rfrom = chess.square(7 if kside else 0, rank >> 3)
        # chess960 not used: standard rook squares
        kto = rank + (6 if kside else 2)
        rto = rank + (5 if kside else 3)
        rem.append(abs_feat(mover, chess.ROOK, rfrom))
        add += [abs_feat(mover, chess.KING, kto), abs_feat(mover, chess.ROOK, rto)]
        return mtype, rem, add, True
    ptype_new = move.promotion if move.promotion else piece.piece_type
    add.append(abs_feat(mover, ptype_new, move.to_square))
    if board.is_en_passant(move):
        mtype = EP
        capsq = move.to_square + (-8 if mover == chess.WHITE else 8)
        rem.append(abs_feat(not mover, chess.PAWN, capsq))
    elif board.is_capture(move):
        cap = board.piece_at(move.to_square)
        rem.append(abs_feat(not mover, cap.piece_type, move.to_square))
        mtype = CAPPROMO if move.promotion else CAPTURE
    else:
        mtype = PROMO if move.promotion else QUIET
    return mtype, rem, add, piece.piece_type == chess.KING


def threat_set(board: chess.Board):
    """Threat *proxy*: ordered (attacker feature, target feature) pairs with the target square
    in the attacker's attack set; Full_Threats-style dedup: drop king-king pairs, drop
    pb/pq/pk/bq/rq/kq (attacker->target type) pairs, and for same-type pairs keep only the
    orientation with the higher attacker square. Absolute frame (count per perspective is the
    same outside table changes)."""
    pm = board.piece_map()
    out = set()
    drop = {(1, 3), (1, 5), (1, 6), (3, 5), (4, 5), (6, 5)}
    for sq, p in pm.items():
        att = board.attacks_mask(sq) & board.occupied
        while att:
            t = (att & -att).bit_length() - 1
            att &= att - 1
            q = pm[t]
            if p.piece_type == 6 and q.piece_type == 6:
                continue
            if (p.piece_type, q.piece_type) in drop:
                continue
            if p.piece_type == q.piece_type and sq < t:
                # same type: keep only one orientation (higher attacker square), when mutual
                if board.attacks_mask(t) & (1 << sq):
                    continue
            out.add((abs_feat(p.color, p.piece_type, sq), abs_feat(q.color, q.piece_type, t)))
    return out


def piece_values():
    return {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}
