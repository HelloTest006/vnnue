import Mathlib
import VNNUE.Basic

/-!
# VNNUE.Quantization  (spec section 4.3: T7, T8, T9, T10, T11)

* T7  wrap-around exactness of int16 (`ZMod 65536`) accumulators;
* T8  saturating int16 adds are path dependent;
* T9  per-lane slot certificate, cardinality bound and global bound;
* T10 `pmaddubsw` pair safety (and failure at 255);
* T11 quantized-sum rounding error `(|A|+1)/(2s)`.

Note (spec C5 / T11): raising the FT scale `s` shrinks the rounding bound
`(|A|+1)/(2s)`, but it also moves the `0..255` clamp and the `/512` product
normalization, so it is not a free precision gain.
-/

namespace VNNUE

open Finset

variable {N L : ℕ}

/-! ## Local definitions -/

/-- Largest positive part of a weight in group `t` (0 if none is positive). -/
def Mplus {m : ℕ} (g : Fin N → Fin m) (w : Fin N → ℤ) (t : Fin m) : ℤ :=
  (((univ.filter (fun i => g i = t)).sup (fun i => (w i).toNat) : ℕ) : ℤ)

/-- Most negative part of a weight in group `t` (0 if none is negative). -/
def Mminus {m : ℕ} (g : Fin N → Fin m) (w : Fin N → ℤ) (t : Fin m) : ℤ :=
  -(((univ.filter (fun i => g i = t)).sup (fun i => (-w i).toNat) : ℕ) : ℤ)

/-! ## Helper lemmas -/

/-- Helper: `Phi` is additive in the feature vector (local copy of the T1/T2 fact). -/
theorem quant_Phi_add {R : Type*} [AddCommGroup R] (W : Fin N → Fin L → R)
    (x y : Fin N → ℤ) : Phi W (x + y) = Phi W x + Phi W y := by
  unfold Phi
  simp [add_smul, Finset.sum_add_distrib]

/-- Helper: the incremental fold equals one application of the summed delta
(local copy of T2, any additive group `R`). -/
theorem quant_foldDeltas_eq {R : Type*} [AddCommGroup R] (W : Fin N → Fin L → R)
    (a : Fin L → R) (l : List (Fin N → ℤ)) :
    foldDeltas W a l = a + Phi W l.sum := by
  induction l generalizing a with
  | nil =>
    have : Phi W (0 : Fin N → ℤ) = 0 := by simp [Phi]
    simp [this]
  | cons δ l ih =>
    rw [foldDeltas_cons, ih, List.sum_cons, quant_Phi_add, add_assoc]

/-- Helper: `accV` absorbs an extra delta: `accV b W x + Φ W δ = accV b W (x + δ)`. -/
theorem quant_accV_add {R : Type*} [AddCommGroup R] (b : Fin L → R)
    (W : Fin N → Fin L → R) (x δ : Fin N → ℤ) :
    accV b W x + Phi W δ = accV b W (x + δ) := by
  unfold accV
  rw [quant_Phi_add, add_assoc]

/-- Helper: every weight of group `t` is `≤ Mplus g w t`. -/
theorem le_Mplus {m : ℕ} (g : Fin N → Fin m) (w : Fin N → ℤ) (i : Fin N) :
    w i ≤ Mplus g w (g i) := by
  unfold Mplus
  have h1 : (w i).toNat ≤ (univ.filter (fun k => g k = g i)).sup (fun k => (w k).toNat) :=
    Finset.le_sup (f := fun k => (w k).toNat) (by simp)
  have h2 : w i ≤ ((w i).toNat : ℤ) := Int.self_le_toNat (w i)
  exact h2.trans (by exact_mod_cast h1)

/-- Helper: every weight of group `t` is `≥ Mminus g w t`. -/
theorem Mminus_le {m : ℕ} (g : Fin N → Fin m) (w : Fin N → ℤ) (i : Fin N) :
    Mminus g w (g i) ≤ w i := by
  unfold Mminus
  have h1 : (-w i).toNat ≤ (univ.filter (fun k => g k = g i)).sup (fun k => (-w k).toNat) :=
    Finset.le_sup (f := fun k => (-w k).toNat) (by simp)
  have h2 : -w i ≤ ((-w i).toNat : ℤ) := Int.self_le_toNat (-w i)
  have h3 : (((-w i).toNat : ℕ) : ℤ) ≤
      (((univ.filter (fun k => g k = g i)).sup (fun k => (-w k).toNat) : ℕ) : ℤ) := by
    exact_mod_cast h1
  linarith

/-- Helper: `Mplus` is nonnegative. -/
theorem Mplus_nonneg {m : ℕ} (g : Fin N → Fin m) (w : Fin N → ℤ) (t : Fin m) :
    0 ≤ Mplus g w t := by
  unfold Mplus; positivity

/-- Helper: `Mminus` is nonpositive. -/
theorem Mminus_nonpos {m : ℕ} (g : Fin N → Fin m) (w : Fin N → ℤ) (t : Fin m) :
    Mminus g w t ≤ 0 := by
  unfold Mminus
  have : (0 : ℤ) ≤ (((univ.filter (fun i => g i = t)).sup (fun i => (-w i).toNat) : ℕ) : ℤ) :=
    by positivity
  linarith

/-! ## T7. Wrap-around exactness -/

/-- **T7(a).** Casting to `ZMod 65536` lane-wise commutes with accumulation:
`cast (b + Σ xᵢ Wᵢ) = cast b + Σ xᵢ (cast Wᵢ)`, since `ℤ → ZMod 65536` is a ring hom. -/
theorem castLanes_accV (b : Fin L → ℤ) (W : Fin N → Fin L → ℤ) (x : Fin N → ℤ) :
    castLanes (accV b W x) = accV (castLanes b) (castW W) x := by
  funext j
  simp [castLanes, castW, accV, Phi, Finset.sum_apply, Int.cast_sum, zsmul_eq_mul,
    Int.cast_mul]

/-- **T7(b).** The two's-complement decode `dec16` is exact on the int16 range
`[-32768, 32767]`. -/
theorem dec16_intCast (v : ℤ) (h : InI16 v) : dec16 (v : ZMod 65536) = v :=
  dec16_intCast_of_inI16 v h

/-- **T7(c).** The wrapped (`ZMod 65536`) incremental accumulator, started at the
wrapped accumulator of `x` and updated along any list of deltas `l`, decodes to the
exact integer accumulator of `x + Σ l` at lane `j`, provided only that this *final*
value lies in the int16 range. Intermediate overflow is irrelevant. -/
theorem wrap_exact (b : Fin L → ℤ) (W : Fin N → Fin L → ℤ) (x : Fin N → ℤ)
    (l : List (Fin N → ℤ)) (j : Fin L) (h : InI16 (accV b W (x + l.sum) j)) :
    dec16 (foldDeltas (castW W) (accV (castLanes b) (castW W) x) l j)
      = accV b W (x + l.sum) j := by
  rw [quant_foldDeltas_eq, quant_accV_add, ← castLanes_accV]
  exact dec16_intCast _ h

/-- **T7(d).** Mathlib's `valMinAbs` on `ZMod 65536` recovers `v` exactly on the
window `(-32768, 32768]` (note: this window differs from int16's at both ends). -/
theorem valMinAbs_window (v : ℤ) (h1 : -32768 < v) (h2 : v ≤ 32768) :
    ((v : ZMod 65536)).valMinAbs = v := by
  rw [ZMod.valMinAbs_spec]
  refine ⟨rfl, ?_, ?_⟩ <;> omega

/-- **T7(d), negative companion.** The earlier claim "`valMinAbs` is exact on
`[-2^15, 2^15)`" is false at `-32768`: there `valMinAbs` returns `+32768`. -/
theorem valMinAbs_neg32768 : ((-32768 : ℤ) : ZMod 65536).valMinAbs = 32768 := by
  rw [ZMod.valMinAbs_spec]
  refine ⟨?_, ?_, ?_⟩
  · push_cast
    decide
  · norm_num
  · norm_num

/-! ## T8. Saturation is path dependent -/

/-- **T8.** Saturating int16 addition is not associative:
`sat(sat(32767+1) + (-1)) = 32766 ≠ 32767 = sat(32767 + sat(1 + (-1)))`. -/
theorem sat16_not_assoc :
    sat16 (sat16 (32767 + 1) + (-1)) ≠ sat16 (32767 + sat16 (1 + (-1))) := by
  unfold sat16; norm_num

/-- **T8.** Path dependence of saturating updates: adding `+30000` then `-30000`
to `30000` gives `2767` when the first add saturates, but `30000` if the deltas
are combined first. -/
theorem sat16_path_dependent :
    sat16 (sat16 (30000 + 30000) + (-30000)) = 2767 ∧
    sat16 (30000 + sat16 (30000 + (-30000))) = 30000 := by
  unfold sat16; norm_num

/-! ## T9. Slot certificate and global bounds -/

/-- **T9 (slot certificate).** If the active set `A` has at most `cap t` features in
each group `t` of the group map `g`, then the lane sum `Σ_{i∈A} wᵢ` lies between
`Σ_t cap t · Mminus t` and `Σ_t cap t · Mplus t`. -/
theorem slot_certificate {m : ℕ} (g : Fin N → Fin m) (cap : Fin m → ℕ) (w : Fin N → ℤ)
    (A : Finset (Fin N)) (hA : ∀ t, (A.filter (fun i => g i = t)).card ≤ cap t) :
    ∑ i ∈ A, w i ≤ ∑ t, (cap t : ℤ) * Mplus g w t ∧
    ∑ t, (cap t : ℤ) * Mminus g w t ≤ ∑ i ∈ A, w i := by
  rw [← Finset.sum_fiberwise A g w]
  constructor
  · apply Finset.sum_le_sum
    intro t _
    have hle : ∑ i ∈ A.filter (fun i => g i = t), w i ≤
        (A.filter (fun i => g i = t)).card • Mplus g w t := by
      apply Finset.sum_le_card_nsmul
      intro i hi
      rw [Finset.mem_filter] at hi
      rw [← hi.2]; exact le_Mplus g w i
    rw [nsmul_eq_mul] at hle
    have hc : ((A.filter (fun i => g i = t)).card : ℤ) ≤ (cap t : ℤ) := by
      exact_mod_cast hA t
    have := mul_le_mul_of_nonneg_right hc (Mplus_nonneg g w t)
    linarith
  · apply Finset.sum_le_sum
    intro t _
    have hle : (A.filter (fun i => g i = t)).card • Mminus g w t ≤
        ∑ i ∈ A.filter (fun i => g i = t), w i := by
      apply Finset.card_nsmul_le_sum
      intro i hi
      rw [Finset.mem_filter] at hi
      rw [← hi.2]; exact Mminus_le g w i
    rw [nsmul_eq_mul] at hle
    have hc : ((A.filter (fun i => g i = t)).card : ℤ) ≤ (cap t : ℤ) := by
      exact_mod_cast hA t
    have := mul_le_mul_of_nonpos_right hc (Mminus_nonpos g w t)
    linarith

/-- **T9 (cardinality bound).** If `|A| ≤ K` and every weight is `≤ M` with `M ≥ 0`,
then `Σ_{i∈A} wᵢ ≤ K · M`. -/
theorem card_bound (w : Fin N → ℤ) (A : Finset (Fin N)) (K : ℕ) (M : ℤ) (hM0 : 0 ≤ M)
    (hM : ∀ i, w i ≤ M) (hK : A.card ≤ K) : ∑ i ∈ A, w i ≤ K * M := by
  have h1 : ∑ i ∈ A, w i ≤ A.card • M := Finset.sum_le_card_nsmul _ _ _ (fun i _ => hM i)
  rw [nsmul_eq_mul] at h1
  have hc : (A.card : ℤ) ≤ (K : ℤ) := by exact_mod_cast hK
  have := mul_le_mul_of_nonneg_right hc hM0
  linarith

/-- Helper (supports T9): `|Σ_{i∈A} wᵢ| ≤ |A| · M` when every `|wᵢ| ≤ M`. -/
theorem abs_sum_le_card_mul (w : Fin N → ℤ) (M : ℤ) (hM : ∀ i, |w i| ≤ M)
    (A : Finset (Fin N)) : |∑ i ∈ A, w i| ≤ A.card * M := by
  have h1 := Finset.abs_sum_le_sum_abs w A
  have h2 : ∑ i ∈ A, |w i| ≤ A.card • M := Finset.sum_le_card_nsmul _ _ _ (fun i _ => hM i)
  rw [nsmul_eq_mul] at h2
  linarith

/-- **T9 (global bound).** If every `|wᵢ| ≤ M`, then
`|b + Σ_{i∈A} wᵢ| ≤ |b| + |A| · M` (the coarse special case of the certificate). -/
theorem global_bound (b : ℤ) (w : Fin N → ℤ) (M : ℤ) (hM : ∀ i, |w i| ≤ M)
    (A : Finset (Fin N)) : |b + ∑ i ∈ A, w i| ≤ |b| + A.card * M := by
  have h1 := abs_add_le b (∑ i ∈ A, w i)
  have h2 := abs_sum_le_card_mul w M hM A
  linarith

/-- **T9 instance (numeric).** `32 · 127 = 4064 < 2^15` (P1 and i8-column lanes). -/
theorem P1_numeric : (32 : ℤ) * 127 = 4064 ∧ (4064 : ℤ) < 2 ^ 15 := by norm_num

/-- **T9 instance (numeric).** `32 · 127² = 516128 < 2^31` (P2). -/
theorem P2_numeric : (32 : ℤ) * 127 ^ 2 = 516128 ∧ (516128 : ℤ) < 2 ^ 31 := by norm_num

/-- **T9 instance (P1).** With at most 32 active features and i8 weights
`|wᵢ| ≤ 127`, the lane sum satisfies `|Σ wᵢ| ≤ 4064`, hence fits in int16. -/
theorem P1_bound (w : Fin N → ℤ) (hw : ∀ i, |w i| ≤ 127) (A : Finset (Fin N))
    (hA : A.card ≤ 32) : |∑ i ∈ A, w i| ≤ 4064 ∧ InI16 (∑ i ∈ A, w i) := by
  have h := abs_sum_le_card_mul w 127 hw A
  have hc : (A.card : ℤ) ≤ 32 := by exact_mod_cast hA
  have h' : |∑ i ∈ A, w i| ≤ 4064 := by linarith
  refine ⟨h', ?_⟩
  rw [abs_le] at h'
  unfold InI16; constructor <;> linarith

/-- **T9 instance (P2).** With at most 32 active features and weights
`|wᵢ| ≤ 127²`, `|Σ wᵢ| ≤ 516128 < 2^31`, so the sum fits in int32. -/
theorem P2_bound (w : Fin N → ℤ) (hw : ∀ i, |w i| ≤ 127 ^ 2) (A : Finset (Fin N))
    (hA : A.card ≤ 32) : |∑ i ∈ A, w i| ≤ 516128 ∧ |∑ i ∈ A, w i| < 2 ^ 31 := by
  have h := abs_sum_le_card_mul w (127 ^ 2) hw A
  have hc : (A.card : ℤ) ≤ 32 := by exact_mod_cast hA
  have h' : |∑ i ∈ A, w i| ≤ 516128 := by nlinarith
  exact ⟨h', by linarith⟩

/-! ## T10. `pmaddubsw` pair safety -/

/-- **T10.** With u8 activations in `[0,127]` and i8 weights `|w| ≤ 127`, a
`pmaddubsw` pair sum satisfies `|a₁w₁ + a₂w₂| ≤ 2·127² = 32258`, so the int16
saturation is never hit. -/
theorem pmaddubsw_safe (a₁ a₂ w₁ w₂ : ℤ) (ha₁ : 0 ≤ a₁ ∧ a₁ ≤ 127) (ha₂ : 0 ≤ a₂ ∧ a₂ ≤ 127)
    (hw₁ : |w₁| ≤ 127) (hw₂ : |w₂| ≤ 127) : |a₁ * w₁ + a₂ * w₂| ≤ 32258 := by
  have e1 : |a₁ * w₁| ≤ 16129 := by
    rw [abs_mul, abs_of_nonneg ha₁.1]
    nlinarith [abs_nonneg w₁]
  have e2' : |a₂ * w₂| ≤ 16129 := by
    rw [abs_mul, abs_of_nonneg ha₂.1]
    nlinarith [abs_nonneg w₂]
  have := abs_add_le (a₁ * w₁) (a₂ * w₂)
  linarith

/-- **T10 (failure at 255).** With full-range u8 activations `255`, the pair sum
`255·127 + 255·127 = 64770` exceeds the int16 maximum `32767`. -/
theorem pmaddubsw_unsafe_255 : (255 : ℤ) * 127 + 255 * 127 > 32767 := by norm_num

/-! ## T11. Quantized-sum rounding error -/

/-- **T11.** If every quantized weight `qᵢ` and the quantized bias `qb` are within `1/2`
of `s·wᵢ` and `s·b` respectively, then the dequantized accumulator differs from the
real one by at most `(|A| + 1)/(2s)`. (Raising `s` also moves the 0..255 clamp and
the /512 normalization, so this is not a free precision gain.) -/
theorem quant_sum_error {ι : Type*} (s : ℝ) (hs : 0 < s) (A : Finset ι) (w q : ι → ℝ)
    (b qb : ℝ) (hq : ∀ i ∈ A, |q i - s * w i| ≤ 1 / 2) (hb : |qb - s * b| ≤ 1 / 2) :
    |(qb + ∑ i ∈ A, q i) / s - (b + ∑ i ∈ A, w i)| ≤ (A.card + 1) / (2 * s) := by
  have key : (qb + ∑ i ∈ A, q i) / s - (b + ∑ i ∈ A, w i)
      = ((qb - s * b) + ∑ i ∈ A, (q i - s * w i)) / s := by
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum]
    field_simp
    ring
  rw [key, abs_div, abs_of_pos hs]
  have h1 := abs_add_le (qb - s * b) (∑ i ∈ A, (q i - s * w i))
  have h2 := Finset.abs_sum_le_sum_abs (fun i => q i - s * w i) A
  have h3 : ∑ i ∈ A, |q i - s * w i| ≤ A.card • (1 / 2 : ℝ) :=
    Finset.sum_le_card_nsmul _ _ _ hq
  rw [nsmul_eq_mul] at h3
  have hX : |(qb - s * b) + ∑ i ∈ A, (q i - s * w i)| ≤ ((A.card : ℝ) + 1) / 2 := by
    linarith
  rw [div_le_div_iff₀ hs (by positivity)]
  nlinarith [abs_nonneg ((qb - s * b) + ∑ i ∈ A, (q i - s * w i))]

end VNNUE
