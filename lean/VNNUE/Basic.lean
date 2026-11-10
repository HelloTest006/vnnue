import Mathlib

/-!
# VNNUE.Basic
Shared definitions for all VNNUE modules. Every other module imports only
`Mathlib` and this file.
-/

namespace VNNUE

open Finset

variable {R : Type*} [AddCommGroup R] {N L : ℕ}

/-- Linear part of an accumulator: `Φ W x = ∑ i, x i • W i` (column `i` is `W i`). -/
def Phi (W : Fin N → Fin L → R) (x : Fin N → ℤ) : Fin L → R :=
  ∑ i, x i • W i

/-- Accumulator over an integer (signed) feature vector. -/
def accV (b : Fin L → R) (W : Fin N → Fin L → R) (x : Fin N → ℤ) : Fin L → R :=
  b + Phi W x

/-- Accumulator over an active feature set (the "refresh" formula). -/
def accS (b : Fin L → R) (W : Fin N → Fin L → R) (S : Finset (Fin N)) : Fin L → R :=
  b + ∑ i ∈ S, W i

/-- Indicator vector of a feature set. -/
def ind (S : Finset (Fin N)) : Fin N → ℤ := fun i => if i ∈ S then 1 else 0

/-- Delta of a set change `S → T`. -/
def delta (S T : Finset (Fin N)) : Fin N → ℤ := ind T - ind S

/-- Support of a delta: the columns that must be touched. -/
def supp (δ : Fin N → ℤ) : Finset (Fin N) := univ.filter (fun i => δ i ≠ 0)

/-- Column operations needed to move an accumulator from set `S` to set `T`. -/
def colOps {α : Type*} [DecidableEq α] (S T : Finset α) : ℕ := (symmDiff S T).card

/-- Sequential incremental application of a list of deltas. -/
def foldDeltas (W : Fin N → Fin L → R) (a : Fin L → R) (l : List (Fin N → ℤ)) :
    Fin L → R :=
  l.foldl (fun acc δ => acc + Phi W δ) a

/-- Binary (0/1) feature vector. -/
def IsBinary (x : Fin N → ℤ) : Prop := ∀ i, x i = 0 ∨ x i = 1

/-- int16 range `[-32768, 32767]`. -/
def InI16 (v : ℤ) : Prop := -32768 ≤ v ∧ v ≤ 32767

/-- Two's-complement decode of a wrapped int16 lane. -/
def dec16 (y : ZMod 65536) : ℤ :=
  if y.val < 32768 then (y.val : ℤ) else (y.val : ℤ) - 65536

/-- Saturating int16 clamp (semantics of `paddsw`). -/
def sat16 (v : ℤ) : ℤ := max (-32768) (min 32767 v)

/-- Integer clamp to `[lo, hi]`. -/
def clampZ (lo hi v : ℤ) : ℤ := max lo (min hi v)

/-- Lane-wise cast of an integer lane vector to `ZMod 65536`. -/
def castLanes (b : Fin L → ℤ) : Fin L → ZMod 65536 := fun j => (b j : ZMod 65536)

/-- Entry-wise cast of an integer weight table to `ZMod 65536`. -/
def castW (W : Fin N → Fin L → ℤ) : Fin N → Fin L → ZMod 65536 :=
  fun i j => (W i j : ZMod 65536)

/-- Power sum of one PSI lane with embedding row `v`: `∑ i, x i * v i ^ k`. -/
def psum (k : ℕ) (v x : Fin N → ℤ) : ℤ := ∑ i, x i * v i ^ k

/-- Elementary symmetric sum of degree 2, over ordered index pairs `i < j`. -/
def e2 (y : Fin N → ℤ) : ℤ :=
  ∑ p ∈ univ.filter (fun p : Fin N × Fin N => p.1 < p.2), y p.1 * y p.2

/-! ## Sanity lemmas (not part of the spec's theorem list T1..T25) -/

/-- Basic sanity lemma (supports all spec IDs): the indicator vector evaluates to
`1` on members of `S` and `0` elsewhere. -/
@[simp] theorem ind_apply (S : Finset (Fin N)) (i : Fin N) :
    ind S i = if i ∈ S then 1 else 0 := rfl

/-- Basic sanity lemma: the indicator is `1` on a member. -/
theorem ind_of_mem {S : Finset (Fin N)} {i : Fin N} (h : i ∈ S) : ind S i = 1 := by
  simp [h]

/-- Basic sanity lemma: the indicator is `0` off the set. -/
theorem ind_of_not_mem {S : Finset (Fin N)} {i : Fin N} (h : i ∉ S) : ind S i = 0 := by
  simp [h]

/-- Basic sanity lemma: an indicator vector is binary (0/1), as used by T12/T13. -/
theorem ind_isBinary (S : Finset (Fin N)) : IsBinary (ind S) := by
  intro i
  by_cases h : i ∈ S <;> simp [h]

/-- Basic sanity lemma (supports T4): pointwise form of the set delta. -/
@[simp] theorem delta_apply (S T : Finset (Fin N)) (i : Fin N) :
    delta S T i = ind T i - ind S i := rfl

/-- Basic sanity lemma (supports T1, T5): membership in the support of a delta
means the coordinate is nonzero. -/
@[simp] theorem mem_supp (δ : Fin N → ℤ) (i : Fin N) : i ∈ supp δ ↔ δ i ≠ 0 := by
  simp [supp]

/-- Basic sanity lemma (supports T2): folding the empty list of deltas is the identity. -/
@[simp] theorem foldDeltas_nil (W : Fin N → Fin L → R) (a : Fin L → R) :
    foldDeltas W a [] = a := rfl

/-- Basic sanity lemma (supports T2): one step of the incremental fold. -/
@[simp] theorem foldDeltas_cons (W : Fin N → Fin L → R) (a : Fin L → R)
    (δ : Fin N → ℤ) (l : List (Fin N → ℤ)) :
    foldDeltas W a (δ :: l) = foldDeltas W (a + Phi W δ) l := rfl

/-- Basic sanity lemma (preview of T7(b)): decoding the `ZMod 65536` cast of an
integer in the int16 range `[-32768, 32767]` returns that integer. Named
`dec16_intCast_of_inI16` so as not to clash with the spec's `dec16_intCast` in
`VNNUE/Quantization.lean`. -/
theorem dec16_intCast_of_inI16 (v : ℤ) (h : InI16 v) : dec16 (v : ZMod 65536) = v := by
  have hv := ZMod.val_intCast (n := 65536) v
  unfold InI16 at h
  unfold dec16
  split_ifs with hc <;> omega

/-- Basic sanity lemma (supports T8): `sat16` is the identity on the int16 range. -/
theorem sat16_of_inI16 (v : ℤ) (h : InI16 v) : sat16 v = v := by
  unfold InI16 at h
  unfold sat16
  omega

/-- Basic sanity lemma: `sat16` is `clampZ` at the int16 bounds. -/
theorem sat16_eq_clampZ (v : ℤ) : sat16 v = clampZ (-32768) 32767 v := rfl

end VNNUE
