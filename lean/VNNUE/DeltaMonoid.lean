import Mathlib
import VNNUE.Basic

/-!
# VNNUE.DeltaMonoid
Spec section 4.2: set deltas telescope (T4), net-delta support bound (T5), and the
column-op distance `colOps` is a relabelling-invariant metric (T6).
-/

namespace VNNUE

open Finset

variable {R : Type*} [AddCommGroup R] {N L : ℕ}

/-! ## T4. Telescoping and delta support -/

/-- **T4 (a).** Set deltas compose along a path `S0 → S1 → S2`:
`(1_{S1} - 1_{S0}) + (1_{S2} - 1_{S1}) = 1_{S2} - 1_{S0}`. -/
theorem delta_telescope (S0 S1 S2 : Finset (Fin N)) :
    delta S0 S1 + delta S1 S2 = delta S0 S2 := by
  funext i
  simp only [Pi.add_apply, delta_apply]
  ring

/-- **T4 (b).** Adding the delta `S → T` to the indicator of `S` gives the indicator
of `T`: `1_S + (1_T - 1_S) = 1_T`. -/
theorem ind_add_delta (S T : Finset (Fin N)) : ind S + delta S T = ind T := by
  funext i
  simp only [Pi.add_apply, delta_apply]
  ring

/-- **T4 (c).** The support of the set delta `1_T - 1_S` is exactly the symmetric
difference `S ∆ T`. -/
theorem supp_delta (S T : Finset (Fin N)) : supp (delta S T) = symmDiff S T := by
  ext i
  rw [mem_supp, Finset.mem_symmDiff, delta_apply, ind_apply, ind_apply]
  by_cases hS : i ∈ S <;> by_cases hT : i ∈ T <;> simp [hS, hT]

/-! ## T5. Net-delta support bound -/

/-- **T5 (a).** The support of a sum of deltas is contained in the union of the
supports: if `δ i + ε i ≠ 0` then `δ i ≠ 0` or `ε i ≠ 0`. -/
theorem supp_add_subset (δ ε : Fin N → ℤ) : supp (δ + ε) ⊆ supp δ ∪ supp ε := by
  intro i hi
  rw [mem_supp, Pi.add_apply] at hi
  rw [Finset.mem_union, mem_supp, mem_supp]
  by_contra h
  push_neg at h
  rw [h.1, h.2] at hi
  exact hi (by norm_num)

/-- **T5 (b).** Cancellation never hurts: the number of columns touched by the net
delta `∑ δ_j` is at most `∑ |supp δ_j|`. -/
theorem card_supp_sum_le (l : List (Fin N → ℤ)) :
    (supp l.sum).card ≤ (l.map (fun δ => (supp δ).card)).sum := by
  induction l with
  | nil =>
    simp only [List.sum_nil, List.map_nil, nonpos_iff_eq_zero, Finset.card_eq_zero]
    ext i
    simp
  | cons δ l ih =>
    rw [List.sum_cons, List.map_cons, List.sum_cons]
    calc (supp (δ + l.sum)).card ≤ (supp δ ∪ supp l.sum).card :=
          Finset.card_le_card (supp_add_subset δ l.sum)
      _ ≤ (supp δ).card + (supp l.sum).card := Finset.card_union_le _ _
      _ ≤ (supp δ).card + (l.map (fun δ => (supp δ).card)).sum := by omega

/-! ## T6. `colOps` is a relabelling-invariant metric -/

/-- **T6 (a).** Triangle inequality for column-op distance:
`|A ∆ C| ≤ |A ∆ B| + |B ∆ C|`, so routing through an intermediate source never beats
the direct diff. -/
theorem colOps_triangle {α : Type*} [DecidableEq α] (A B C : Finset α) :
    colOps A C ≤ colOps A B + colOps B C := by
  unfold colOps
  calc (symmDiff A C).card ≤ (symmDiff A B ∪ symmDiff B C).card :=
        Finset.card_le_card (symmDiff_triangle A B C)
    _ ≤ (symmDiff A B).card + (symmDiff B C).card := Finset.card_union_le _ _

/-- **T6 (b).** Column-op distance is symmetric: `|A ∆ B| = |B ∆ A|`. -/
theorem colOps_comm {α : Type*} [DecidableEq α] (A B : Finset α) :
    colOps A B = colOps B A := by
  unfold colOps
  rw [symmDiff_comm]

/-- **T6 (c).** Column-op distance is invariant under an injective relabelling `f`
(e.g. board mirroring): `|f(S) ∆ f(C)| = |S ∆ C|`. Mirrored Finny entries are valid
at the same cost. -/
theorem colOps_image {α β : Type*} [DecidableEq α] [DecidableEq β] (f : α → β)
    (hf : Function.Injective f) (S C : Finset α) :
    colOps (S.image f) (C.image f) = colOps S C := by
  unfold colOps
  rw [← Finset.image_symmDiff _ _ hf, Finset.card_image_of_injective _ hf]

end VNNUE
