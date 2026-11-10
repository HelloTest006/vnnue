import Mathlib
import VNNUE.Basic

/-!
# VNNUE.Accumulator
Spec section 4.1: theorems T1 (homomorphism / delta rule), T2 (path independence)
and T3 (set form and any-source correctness), stated over an arbitrary additive
commutative group `R` of lane values, plus instantiations at `ℤ` (exact
specification semantics) and `ZMod 65536` (wrapping int16 lanes).
-/

namespace VNNUE

open Finset

variable {R : Type*} [AddCommGroup R] {N L : ℕ}

/-! ## T1. Homomorphism and delta rule -/

/-- **T1 (a).** The linear part of the accumulator is additive:
`Φ(x + y) = Φ(x) + Φ(y)`. -/
theorem Phi_add (W : Fin N → Fin L → R) (x y : Fin N → ℤ) :
    Phi W (x + y) = Phi W x + Phi W y := by
  simp [Phi, add_smul, Finset.sum_add_distrib]

/-- **T1 (b).** Delta rule: adding a delta `δ` to the input vector adds `Φ(δ)` to
the accumulator, `acc(x + δ) = acc(x) + Φ(δ)`. -/
theorem accV_add_delta (b : Fin L → R) (W : Fin N → Fin L → R) (x δ : Fin N → ℤ) :
    accV b W (x + δ) = accV b W x + Phi W δ := by
  simp [accV, Phi_add, add_assoc]

/-- **T1 (c).** Sparsity: `Φ(δ)` only depends on the support columns,
`Φ(δ) = ∑_{i ∈ supp δ} δ_i • W_i`. -/
theorem Phi_eq_sum_supp (W : Fin N → Fin L → R) (δ : Fin N → ℤ) :
    Phi W δ = ∑ i ∈ supp δ, δ i • W i := by
  unfold Phi supp
  rw [Finset.sum_filter]
  refine Finset.sum_congr rfl ?_
  intro i _
  split_ifs with h
  · rfl
  · have h' : δ i = 0 := not_not.mp h
    simp [h']

/-! ## T2. Path independence -/

/-- **T2 (a).** Folding any list of deltas incrementally, starting from `acc(x)`,
yields `acc(x + ∑ list)`. -/
theorem foldDeltas_eq (b : Fin L → R) (W : Fin N → Fin L → R) (x : Fin N → ℤ)
    (l : List (Fin N → ℤ)) :
    foldDeltas W (accV b W x) l = accV b W (x + l.sum) := by
  induction l generalizing x with
  | nil => simp
  | cons δ l ih =>
    rw [foldDeltas_cons, ← accV_add_delta, ih, List.sum_cons, add_assoc]

/-- **T2 (b).** Order independence: permuting the list of deltas does not change
the resulting accumulator. -/
theorem foldDeltas_perm (b : Fin L → R) (W : Fin N → Fin L → R) (x : Fin N → ℤ)
    {l l' : List (Fin N → ℤ)} (h : l.Perm l') :
    foldDeltas W (accV b W x) l = foldDeltas W (accV b W x) l' := by
  rw [foldDeltas_eq, foldDeltas_eq, h.sum_eq]

/-- Helper for **T2** (grouping independence): folding a concatenation is folding
the two parts in sequence. -/
theorem foldDeltas_append (W : Fin N → Fin L → R) (a : Fin L → R)
    (l₁ l₂ : List (Fin N → ℤ)) :
    foldDeltas W a (l₁ ++ l₂) = foldDeltas W (foldDeltas W a l₁) l₂ := by
  simp [foldDeltas, List.foldl_append]

/-- Helper for **T2** (grouping independence): replacing a run of deltas by its
single net delta (as in net-delta catch-up) gives the same accumulator,
`fold(acc x, l₁ ++ l₂) = fold(acc x, [∑ l₁] ++ l₂)`. -/
theorem foldDeltas_group (b : Fin L → R) (W : Fin N → Fin L → R) (x : Fin N → ℤ)
    (l₁ l₂ : List (Fin N → ℤ)) :
    foldDeltas W (accV b W x) (l₁ ++ l₂) =
      foldDeltas W (accV b W x) (l₁.sum :: l₂) := by
  rw [foldDeltas_eq, foldDeltas_eq, List.sum_append, List.sum_cons]

/-! ## T3. Set form and any-source correctness -/

/-- **T3 (a).** The refresh formula over a feature set agrees with the vector form
applied to its indicator: `accS b W S = accV b W (1_S)`. -/
theorem accS_eq_accV (b : Fin L → R) (W : Fin N → Fin L → R) (S : Finset (Fin N)) :
    accS b W S = accV b W (ind S) := by
  unfold accS accV Phi
  congr 1
  rw [← Finset.sum_filter_add_sum_filter_not univ (fun i => i ∈ S)]
  have h1 : ∑ i ∈ univ.filter (fun i => i ∈ S), ind S i • W i
      = ∑ i ∈ univ.filter (fun i => i ∈ S), W i := by
    refine Finset.sum_congr rfl ?_
    intro i hi
    rw [Finset.mem_filter] at hi
    simp [hi.2]
  have h2 : ∑ i ∈ univ.filter (fun i => ¬ i ∈ S), ind S i • W i = 0 := by
    refine Finset.sum_eq_zero ?_
    intro i hi
    rw [Finset.mem_filter] at hi
    simp [hi.2]
  rw [h1, h2, add_zero, Finset.filter_mem_eq_inter, Finset.univ_inter]

/-- **T3 (b).** Any-source correctness: from the accumulator of **any** stored set
`U`, the accumulator of `S` is obtained by adding the columns of `S \ U` and
subtracting the columns of `U \ S`. -/
theorem accS_any_source (b : Fin L → R) (W : Fin N → Fin L → R) (S U : Finset (Fin N)) :
    accS b W S = accS b W U + ∑ i ∈ S \ U, W i - ∑ i ∈ U \ S, W i := by
  unfold accS
  have split : ∀ A B : Finset (Fin N),
      ∑ i ∈ A, W i = ∑ i ∈ A ∩ B, W i + ∑ i ∈ A \ B, W i := by
    intro A B
    rw [← Finset.sum_filter_add_sum_filter_not A (fun i => i ∈ B)]
    have e1 : A.filter (fun i => i ∈ B) = A ∩ B := by
      ext i; simp [Finset.mem_filter, Finset.mem_inter]
    have e2 : A.filter (fun i => ¬ i ∈ B) = A \ B := by
      ext i; simp [Finset.mem_filter, Finset.mem_sdiff]
    rw [e1, e2]
  rw [split S U, split U S, Finset.inter_comm U S]
  abel

/-- **T3 (c).** The number of column operations from `S` to `U` is
`|S △ U| = |S \ U| + |U \ S|`. -/
theorem colOps_eq (S U : Finset (Fin N)) :
    colOps S U = (S \ U).card + (U \ S).card := by
  unfold colOps
  rw [symmDiff_def, Finset.sup_eq_union]
  exact Finset.card_union_of_disjoint disjoint_sdiff_sdiff

/-- Helper for **T3**: the delta between two sets, viewed through `Φ`, is exactly
the any-source correction `∑_{S \ U} W - ∑_{U \ S} W`. -/
theorem Phi_delta (W : Fin N → Fin L → R) (U S : Finset (Fin N)) :
    Phi W (delta U S) = ∑ i ∈ S \ U, W i - ∑ i ∈ U \ S, W i := by
  have h := accS_any_source (0 : Fin L → R) W S U
  have e1 := accS_eq_accV (0 : Fin L → R) W S
  have e2 := accS_eq_accV (0 : Fin L → R) W U
  have hd : ind S = ind U + delta U S := by
    unfold delta; abel
  rw [e1, e2, hd, accV_add_delta] at h
  have : Phi W (delta U S) = (accV 0 W (ind U) + ∑ i ∈ S \ U, W i - ∑ i ∈ U \ S, W i)
      - accV 0 W (ind U) := by
    rw [← h]; abel
  rw [this]; abel

/-! ## Instantiations at `ℤ` (exact semantics) and `ZMod 65536` (int16 lanes) -/

/-- **T1 instantiated at ℤ** (delta rule for exact integer lanes). -/
theorem accV_add_delta_int (b : Fin L → ℤ) (W : Fin N → Fin L → ℤ) (x δ : Fin N → ℤ) :
    accV b W (x + δ) = accV b W x + Phi W δ :=
  accV_add_delta b W x δ

/-- **T1 instantiated at ZMod 65536** (delta rule for wrapping int16 lanes). -/
theorem accV_add_delta_zmod (b : Fin L → ZMod 65536) (W : Fin N → Fin L → ZMod 65536)
    (x δ : Fin N → ℤ) :
    accV b W (x + δ) = accV b W x + Phi W δ :=
  accV_add_delta b W x δ

/-- **T2 instantiated at ℤ** (path independence for exact integer lanes). -/
theorem foldDeltas_eq_int (b : Fin L → ℤ) (W : Fin N → Fin L → ℤ) (x : Fin N → ℤ)
    (l : List (Fin N → ℤ)) :
    foldDeltas W (accV b W x) l = accV b W (x + l.sum) :=
  foldDeltas_eq b W x l

/-- **T2 instantiated at ZMod 65536** (path independence for wrapping int16 lanes). -/
theorem foldDeltas_eq_zmod (b : Fin L → ZMod 65536) (W : Fin N → Fin L → ZMod 65536)
    (x : Fin N → ℤ) (l : List (Fin N → ℤ)) :
    foldDeltas W (accV b W x) l = accV b W (x + l.sum) :=
  foldDeltas_eq b W x l

/-- **T3 instantiated at ℤ** (any-source correctness for exact integer lanes). -/
theorem accS_any_source_int (b : Fin L → ℤ) (W : Fin N → Fin L → ℤ) (S U : Finset (Fin N)) :
    accS b W S = accS b W U + ∑ i ∈ S \ U, W i - ∑ i ∈ U \ S, W i :=
  accS_any_source b W S U

/-- **T3 instantiated at ZMod 65536** (any-source correctness for wrapping int16 lanes). -/
theorem accS_any_source_zmod (b : Fin L → ZMod 65536) (W : Fin N → Fin L → ZMod 65536)
    (S U : Finset (Fin N)) :
    accS b W S = accS b W U + ∑ i ∈ S \ U, W i - ∑ i ∈ U \ S, W i :=
  accS_any_source b W S U

/-- Helper linking the two instantiations (supports T1–T3 with T7): casting the
exact integer accumulator lane-wise into `ZMod 65536` gives the wrapping
accumulator computed from the cast bias and cast weights. -/
theorem castLanes_accV_int (b : Fin L → ℤ) (W : Fin N → Fin L → ℤ) (x : Fin N → ℤ) :
    castLanes (accV b W x) = accV (castLanes b) (castW W) x := by
  funext j
  simp [castLanes, castW, accV, Phi, Finset.sum_apply]

end VNNUE
