import Mathlib
import VNNUE.Basic

/-!
# VNNUE.Complexity
Spec section 4.7: the PSI overhead bound and the king-move total (T25), plus the two
appendix lemmas: ski-rental materialization (A1) and fused-pass traffic (A2).

All statements are pure arithmetic over `ℕ` (or a linear order), so they do not use
the `R`-valued definitions of `VNNUE.Basic`. Helper lemmas carry the suffix `_cx` so
that they do not clash with same-purpose lemmas in other modules (e.g. `Refresh.lean`)
if all modules are imported together.
-/

namespace VNNUE

open Finset

/-! ## T25. PSI overhead bound and king-move total -/

/-- **T25(a) (PSI overhead bound).** If NNUE's per-move PSQ byte traffic `costN` is at
least `4kL` (k columns, 2 perspectives, 2L bytes each), then adding `e` bytes for each of
the `k` changed absolute features increases the traffic by at most a factor
`1 + e/(4L)`, i.e. `4L·(costN + k·e) ≤ (4L + e)·costN`. -/
theorem psi_overhead (k L e costN : ℕ) (h : 4 * k * L ≤ costN) :
    4 * L * (costN + k * e) ≤ (4 * L + e) * costN := by
  have h' : e * (4 * k * L) ≤ e * costN := Nat.mul_le_mul_left e h
  nlinarith [h']

/-- **T25(a), real-ratio form.** Under the same hypothesis, and `L > 0`, the relative
PSI overhead `(k·e)/costN` is at most `e/(4L)` (as real numbers; when `costN = 0` the
left side is `0` by Lean's division convention). -/
theorem psi_overhead_ratio (k L e costN : ℕ) (hL : 0 < L) (h : 4 * k * L ≤ costN) :
    ((k * e : ℕ) : ℝ) / costN ≤ (e : ℝ) / (4 * L) := by
  rcases Nat.eq_zero_or_pos costN with h0 | hpos
  · subst h0; simp; positivity
  · have hc : (0 : ℝ) < costN := by exact_mod_cast hpos
    have hL' : (0 : ℝ) < 4 * L := by positivity
    rw [div_le_div_iff₀ hc hL']
    have h' : ((e * (4 * k * L) : ℕ) : ℝ) ≤ ((e * costN : ℕ) : ℝ) := by
      exact_mod_cast Nat.mul_le_mul_left e h
    push_cast at h' ⊢
    nlinarith [h']

/-- **T25(a), spec instance.** With `e = 6r = 192` bytes (r = 32) and `L = 1024`
(so `4L = 4096`), the PSI byte overhead is at most `4.7%` of NNUE's PSQ traffic:
`1000·(costN + 192k) ≤ 1047·costN` whenever `costN ≥ 4096k`. (Exactly, the bound is
`192/4096 = 4.6875%`.) -/
theorem psi_overhead_instance (k costN : ℕ) (h : 4 * k * 1024 ≤ costN) :
    1000 * (costN + k * 192) ≤ 1047 * costN := by
  omega

/-- **T25(a), exact spec instance.** With `e = 192` and `L = 1024` the factor-form bound of
T25(a) reads `4096·(costN + 192k) ≤ 4288·costN`. -/
theorem psi_overhead_instance_exact (k costN : ℕ) (h : 4 * k * 1024 ≤ costN) :
    4 * 1024 * (costN + k * 192) ≤ (4 * 1024 + 192) * costN :=
  psi_overhead k 1024 192 costN h

/-- Helper for T25(b) (local copy of T24(a)): over a linear order, the minimum of `f`
over a superset `C₂ ⊇ C₁` is at most the minimum over `C₁`. -/
theorem inf'_superset_cx {ι α : Type*} [LinearOrder α] (C₁ C₂ : Finset ι) (h : C₁ ⊆ C₂)
    (hne : C₁.Nonempty) (f : ι → α) :
    C₂.inf' (hne.mono h) f ≤ C₁.inf' hne f :=
  Finset.le_inf' hne _ (fun i hi => Finset.inf'_le f (h hi))

/-- **T25(b) (king-move total).** On a king event, VNNUE's total per-move cost (FT column
part = minimum over its candidate set `C₂`, plus the PSI channel's `k·e`) is at most
Stockfish's FT cost (minimum over its subset `C₁`) plus `k·e`. -/
theorem king_event_total {ι : Type*} (C₁ C₂ : Finset ι) (h : C₁ ⊆ C₂) (hne : C₁.Nonempty)
    (f : ι → ℕ) (k e : ℕ) :
    C₂.inf' (hne.mono h) f + k * e ≤ C₁.inf' hne f + k * e :=
  Nat.add_le_add_right (inf'_superset_cx C₁ C₂ h hne f) _

/-! ## A1 (appendix). Ski-rental materialization -/

/-- Cost of the deterministic "materialize at the B-th use" rule on an input of `n` uses:
walk (cost 1 each) while `n < B`; otherwise `B - 1` walks plus materialization cost `B`. -/
def alg (B n : ℕ) : ℕ := if n < B then n else 2 * B - 1

/-- Cost of the general threshold rule "walk `t` times, then materialize (cost `B`)" on an
input of `n` uses. -/
def algT (B t n : ℕ) : ℕ := if n ≤ t then n else t + B

/-- **A1 (ski-rental upper bound).** For `B ≥ 1`, the rule "materialize at the B-th use" is
`(2 − 1/B)`-competitive against the offline optimum `min n B`:
`B · alg B n ≤ (2B − 1) · min n B`. -/
theorem ski_upper (B n : ℕ) (hB : 1 ≤ B) : B * alg B n ≤ (2 * B - 1) * min n B := by
  unfold alg
  split_ifs with h
  · rw [min_eq_left h.le]
    have : B ≤ 2 * B - 1 := by omega
    exact Nat.mul_le_mul_right n this
  · rw [min_eq_right (by omega)]
    rw [Nat.mul_comm]

/-- **A1 (ski-rental lower bound).** No threshold rule does better on its worst input:
for any threshold `t`, on the input of `t + 1` uses (the first use after the walks
stop), `(2B − 1) · OPT ≤ B · algT B t (t + 1)`, so its competitive ratio is at least
`2 − 1/B`. -/
theorem ski_lower (B t : ℕ) (hB : 1 ≤ B) :
    (2 * B - 1) * min (t + 1) B ≤ B * algT B t (t + 1) := by
  unfold algT
  rw [if_neg (by omega)]
  obtain ⟨b, rfl⟩ : ∃ b, B = b + 1 := ⟨B - 1, by omega⟩
  have h2 : 2 * (b + 1) - 1 = 2 * b + 1 := by omega
  rw [h2]
  rcases le_total (t + 1) (b + 1) with hc | hc
  · rw [min_eq_left hc]
    have ht : t ≤ b := by omega
    -- (2b+1)(t+1) ≤ (b+1)(t+b+1)  ⟺  b·t ≤ b·b
    have : b * t ≤ b * b := Nat.mul_le_mul_left b ht
    nlinarith [this]
  · rw [min_eq_right hc]
    have ht : b ≤ t := by omega
    have : (b + 1) * b ≤ (b + 1) * t := Nat.mul_le_mul_left (b + 1) ht
    nlinarith [this]

/-! ## A2 (appendix). Fused-pass traffic -/

/-- Helper for A2: the sequential traffic `∑_j (k_j·c + 2a)` equals `c·∑k + 2a·t`. -/
theorem sum_map_traffic_cx (ks : List ℕ) (c a : ℕ) :
    (ks.map (fun k => k * c + 2 * a)).sum = ks.sum * c + 2 * a * ks.length := by
  induction ks with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, ih]
    ring

/-- **A2 (fused-pass traffic).** For a catch-up over `t = ks.length ≥ 1` plies with
per-ply column counts `ks`, column cost `c` and pass cost `2a`, the fused traffic
(`knet` net columns with `knet ≤ ∑ k_j`, plus one pass `2a`, plus `2a·(t − 1)` for the
remaining pass budget) is at most the sequential traffic `∑_j (k_j·c + 2a)`. -/
theorem fused_traffic (ks : List ℕ) (knet c a : ℕ) (ht : 1 ≤ ks.length) (hk : knet ≤ ks.sum) :
    knet * c + 2 * a + 2 * a * (ks.length - 1) ≤ (ks.map (fun k => k * c + 2 * a)).sum := by
  rw [sum_map_traffic_cx]
  have h1 : 2 * a + 2 * a * (ks.length - 1) = 2 * a * ks.length := by
    obtain ⟨m, hm⟩ : ∃ m, ks.length = m + 1 := ⟨ks.length - 1, by omega⟩
    rw [hm, Nat.add_sub_cancel]
    ring
  have h2 : knet * c ≤ ks.sum * c := Nat.mul_le_mul_right c hk
  omega

/-- **A2 (one pass instead of `t`).** Stronger form: the fused traffic with a single pass,
`knet·c + 2a`, is at most the sequential traffic `∑_j (k_j·c + 2a)`; and the saving is at
least `2a·(t − 1)` (this is `fused_traffic`). -/
theorem fused_one_pass (ks : List ℕ) (knet c a : ℕ) (ht : 1 ≤ ks.length)
    (hk : knet ≤ ks.sum) :
    knet * c + 2 * a ≤ (ks.map (fun k => k * c + 2 * a)).sum :=
  le_trans (Nat.le_add_right _ _) (fused_traffic ks knet c a ht hk)

end VNNUE
