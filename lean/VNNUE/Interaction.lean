import Mathlib
import VNNUE.Basic

/-!
# VNNUE.Interaction

Section 4.4 of the VNNUE specification: the PSI (power-sum interaction) channel.

* T12: Newton–Girard order 2 per lane, and exact halving (`ng2`, `psi_lane`, `psi_even`).
* T13: multi-head PSI readout equals an all-pairs factorization machine (`psi_heads`).
* T14: incremental rule for `E` (`psum_add`, `psi_add_feature`, `psi_remove_feature`).
* T15: Newton–Girard order 3 and divisibility by 6 (`ng3`, `ng3_six_dvd`, `ng3_binary`).
* T16: int32 range certificates (`mix_range`, `ng3_range`, `e_lane_range`).
* T17: universality of PSI at full width (`psi_universal`).
* T18: strict generalization and corrected non-affinity (`psi_zero_heads`, `fc0_embed`,
  `affine_second_diff`, `psi_not_affine`).
-/

namespace VNNUE

open Finset

variable {R : Type*} [AddCommGroup R] {N L : ℕ}

/-- Ordered elementary symmetric sum of degree 3: the sum of `y i * y j * y k` over all
ordered triples of pairwise distinct indices (this is `6 * e₃ y`). -/
def e3ord (y : Fin N → ℤ) : ℤ :=
  ∑ i, ∑ j, ∑ k, if i ≠ j ∧ j ≠ k ∧ i ≠ k then y i * y j * y k else 0

/-! ### Helper lemmas -/

/-- Helper (T12, T15): a double sum of a product of separated factors is the product of the
sums, `∑ i, ∑ j, f i * g j = (∑ f) * (∑ g)`. -/
lemma vn_sum_sum_mul (f g : Fin N → ℤ) :
    ∑ i, ∑ j, f i * g j = (∑ i, f i) * ∑ j, g j := by
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun i _ => by rw [Finset.mul_sum]

/-- Helper (T15): a triple sum of a product of separated factors is the product of the three
sums. -/
lemma vn_sum3 (f g h : Fin N → ℤ) :
    ∑ i, ∑ j, ∑ k, f i * g j * h k = (∑ i, f i) * (∑ j, g j) * ∑ k, h k := by
  rw [← vn_sum_sum_mul f g, Finset.sum_mul]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun j _ => by rw [Finset.mul_sum]

/-- Helper (T12): `e2` written as a double sum over all ordered index pairs with an `i < j`
guard. -/
lemma e2_eq_sum_sum (y : Fin N → ℤ) :
    e2 y = ∑ i, ∑ j, if i < j then y i * y j else 0 := by
  simp only [e2, Finset.sum_filter, Fintype.sum_prod_type]

/-- Helper (T15): inclusion–exclusion for the "pairwise distinct" indicator of a triple:
`[i,j,k distinct] t = t - [i=j] t - [j=k] t - [i=k] t + 2 [i=j=k] t`. -/
lemma distinct_ind (i j k : Fin N) (t : ℤ) :
    (if i ≠ j ∧ j ≠ k ∧ i ≠ k then t else 0)
      = t - (if i = j then t else 0) - (if j = k then t else 0) - (if i = k then t else 0)
        + (if i = j ∧ j = k then t else 0) + (if i = j ∧ j = k then t else 0) := by
  by_cases h1 : i = j
  · by_cases h2 : j = k
    · simp [h1, h2]
    · simp [h1, h2]
  · by_cases h2 : j = k
    · simp [h1, h2]
    · by_cases h3 : i = k
      · have h4 : k ≠ j := fun e => h1 (h3.trans e)
        simp [h1, h2, h3, h4]
      · simp [h1, h2, h3]

/-- Helper (T15): for every integer family, `3 ∣ (∑ f)^3 - ∑ f^3` (Frobenius mod 3). -/
lemma three_dvd_cube_sum {ι : Type*} [DecidableEq ι] (s : Finset ι) (f : ι → ℤ) :
    (3 : ℤ) ∣ (∑ i ∈ s, f i) ^ 3 - ∑ i ∈ s, f i ^ 3 := by
  refine Finset.induction_on s ?_ ?_
  · simp
  · intro a s ha ih
    rw [Finset.sum_insert ha, Finset.sum_insert ha]
    obtain ⟨c, hc⟩ := ih
    exact ⟨c + f a ^ 2 * (∑ i ∈ s, f i) + f a * (∑ i ∈ s, f i) ^ 2, by linear_combination hc⟩

/-! ### T12 -/

/-- **T12 (Newton–Girard order 2).** For every integer vector `y`,
`(∑ y)² - ∑ y² = 2 e₂(y)`, where `e₂` sums `y i y j` over `i < j`. -/
theorem ng2 (y : Fin N → ℤ) :
    (∑ i, y i) ^ 2 - ∑ i, y i ^ 2 = 2 * e2 y := by
  have hpt : ∀ i j : Fin N, y i * y j = (if i < j then y i * y j else 0)
      + (if j < i then y i * y j else 0) + (if i = j then y i * y j else 0) := by
    intro i j
    rcases lt_trichotomy i j with h | h | h
    · simp [h, h.ne, not_lt.mpr h.le]
    · simp [h]
    · simp [h, h.ne', not_lt.mpr h.le]
  have hsq : (∑ i, y i) ^ 2 = ∑ i, ∑ j, y i * y j := by
    rw [sq, ← vn_sum_sum_mul]
  have hexp : ∑ i, ∑ j, y i * y j = ∑ i, ∑ j, ((if i < j then y i * y j else 0)
      + (if j < i then y i * y j else 0) + (if i = j then y i * y j else 0)) :=
    Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => hpt i j
  have hswap : ∑ i, ∑ j, (if j < i then y i * y j else 0)
      = ∑ i, ∑ j, (if i < j then y i * y j else 0) := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by
      split_ifs <;> ring
  have hdiag : ∑ i, ∑ j, (if i = j then y i * y j else 0) = ∑ i, y i ^ 2 :=
    Finset.sum_congr rfl fun i _ => by simp [sq]
  rw [hsq, hexp]
  simp only [Finset.sum_add_distrib]
  rw [hswap, hdiag, e2_eq_sum_sum]
  ring

/-- **T12 (PSI lane).** For a binary input `x` and lane embedding row `v`, the lane state
satisfies `P1² - P2 = 2 e₂(x ⊙ v)`, i.e. twice the sum of `x i x j v i v j` over `i < j`. -/
theorem psi_lane (v x : Fin N → ℤ) (hx : IsBinary x) :
    psum 1 v x ^ 2 - psum 2 v x = 2 * e2 (fun i => x i * v i) := by
  have h1 : psum 1 v x = ∑ i, x i * v i := by simp only [psum, pow_one]
  have h2 : psum 2 v x = ∑ i, (x i * v i) ^ 2 := by
    unfold psum
    exact Finset.sum_congr rfl fun i _ => by rcases hx i with h | h <;> simp [h]
  have h : (∑ i, x i * v i) ^ 2 - ∑ i, (x i * v i) ^ 2 = 2 * e2 (fun i => x i * v i) :=
    ng2 _
  rw [h1, h2, h]

/-- **T12 (exact halving).** For **every** integer input `x` (binary or not),
`P1² - P2` is even, so the right shift by one in the readout `E = (P1² - P2)/2` is exact. -/
theorem psi_even (v x : Fin N → ℤ) : 2 ∣ psum 1 v x ^ 2 - psum 2 v x := by
  have h1 : psum 1 v x = ∑ i, x i * v i := by simp only [psum, pow_one]
  have hsplit : psum 2 v x
      = ∑ i, (x i * v i) ^ 2 - ∑ i, x i * (x i - 1) * v i ^ 2 := by
    unfold psum
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  have h : (∑ i, x i * v i) ^ 2 - ∑ i, (x i * v i) ^ 2 = 2 * e2 (fun i => x i * v i) :=
    ng2 _
  rw [h1, hsplit]
  have hrest : (2 : ℤ) ∣ ∑ i, x i * (x i - 1) * v i ^ 2 :=
    Finset.dvd_sum fun i _ => Dvd.dvd.mul_right (Int.even_mul_pred_self (x i)).two_dvd _
  obtain ⟨c, hc⟩ := hrest
  exact ⟨e2 (fun i => x i * v i) + c, by linear_combination h + hc⟩

/-! ### T13 -/

/-- **T13 (multi-head PSI readout = all-pairs factorization machine).** For binary `x`,
head `h` of the readout `∑ t, Λ h t (P1_t² - P2_t)` equals twice the sum over unordered pairs
`i < j` of `x i x j Q^h_{ij}`, with pair table `Q^h_{ij} = ∑ t, Λ h t V t i V t j`
(i.e. `Q^h = Vᵀ diag(Λ_h) V`). -/
theorem psi_heads {r m : ℕ} (V : Fin r → Fin N → ℤ) (Lam : Fin m → Fin r → ℤ)
    (x : Fin N → ℤ) (hx : IsBinary x) (h : Fin m) :
    ∑ t, Lam h t * (psum 1 (V t) x ^ 2 - psum 2 (V t) x)
      = 2 * ∑ p ∈ univ.filter (fun p : Fin N × Fin N => p.1 < p.2),
              x p.1 * x p.2 * ∑ t, Lam h t * V t p.1 * V t p.2 := by
  rw [Finset.sum_congr rfl (fun t _ => by rw [psi_lane (V t) x hx])]
  simp only [e2, Finset.mul_sum]
  exact Finset.sum_comm.trans
    (Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun t _ => by ring)

/-! ### T14 -/

/-- **T14 (additivity of power sums).** Each power sum is linear in the input:
`psum k v (x + δ) = psum k v x + psum k v δ`, for all integer `x`, `δ`. -/
theorem psum_add (k : ℕ) (v x δ : Fin N → ℤ) :
    psum k v (x + δ) = psum k v x + psum k v δ := by
  simp [psum, add_mul, Finset.sum_add_distrib]

/-- Helper (T14): the power sum of a single feature `j` is `v j ^ k`. -/
lemma psum_single (k : ℕ) (v : Fin N → ℤ) (j : Fin N) :
    psum k v (Pi.single j 1) = v j ^ k := by
  unfold psum
  rw [Finset.sum_eq_single j (fun b _ hb => by simp [Pi.single_apply, hb])
    (fun h => absurd (Finset.mem_univ j) h)]
  simp

/-- **T14 (adding a feature).** For every integer `x`, adding feature `j` changes the lane
quantity `P1² - P2` by `2 · v j · P1(before)`; equivalently `ΔE = v j · P1(before)`. -/
theorem psi_add_feature (v x : Fin N → ℤ) (j : Fin N) :
    psum 1 v (x + Pi.single j 1) ^ 2 - psum 2 v (x + Pi.single j 1)
      = (psum 1 v x ^ 2 - psum 2 v x) + 2 * (v j * psum 1 v x) := by
  rw [psum_add, psum_add, psum_single, psum_single]
  ring

/-- **T14 (removing a feature).** For every integer `x`, removing feature `j` changes the lane
quantity `P1² - P2` by `-2 · v j · P1(after)`; equivalently `ΔE = -v j · P1(after)`. -/
theorem psi_remove_feature (v x : Fin N → ℤ) (j : Fin N) :
    psum 1 v (x - Pi.single j 1) ^ 2 - psum 2 v (x - Pi.single j 1)
      = (psum 1 v x ^ 2 - psum 2 v x) - 2 * (v j * psum 1 v (x - Pi.single j 1)) := by
  have h1 := psum_add 1 v (x - Pi.single j 1) (Pi.single j 1)
  have h2 := psum_add 2 v (x - Pi.single j 1) (Pi.single j 1)
  rw [sub_add_cancel, psum_single] at h1 h2
  rw [h1, h2]
  ring

/-! ### T15 -/

/-- **T15 (Newton–Girard order 3).** For every integer vector `y`, the ordered distinct-triple
sum satisfies `e3ord y = p₁³ - 3 p₁ p₂ + 2 p₃` with `p_k = ∑ y^k` (and `e3ord = 6 e₃`). -/
theorem ng3 (y : Fin N → ℤ) :
    e3ord y = (∑ i, y i) ^ 3 - 3 * (∑ i, y i) * (∑ i, y i ^ 2) + 2 * ∑ i, y i ^ 3 := by
  have hdist : e3ord y = ∑ i, ∑ j, ∑ k, (y i * y j * y k
      - (if i = j then y i * y j * y k else 0) - (if j = k then y i * y j * y k else 0)
      - (if i = k then y i * y j * y k else 0)
      + (if i = j ∧ j = k then y i * y j * y k else 0)
      + (if i = j ∧ j = k then y i * y j * y k else 0)) := by
    unfold e3ord
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
      Finset.sum_congr rfl fun k _ => distinct_ind i j k _
  have hT : ∑ i, ∑ j, ∑ k, y i * y j * y k = (∑ i, y i) ^ 3 := by
    rw [vn_sum3]
    ring
  have hA : ∑ i, ∑ j, ∑ k, (if i = j then y i * y j * y k else 0)
      = (∑ i, y i * y i) * ∑ k, y k := by
    rw [← vn_sum_sum_mul]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun k _ => by simp
  have hB : ∑ i, ∑ j, ∑ k, (if j = k then y i * y j * y k else 0)
      = (∑ i, y i) * ∑ j, y j * y j := by
    rw [← vn_sum_sum_mul]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
    ring
  have hC : ∑ i, ∑ j, ∑ k, (if i = k then y i * y j * y k else 0)
      = (∑ i, y i * y i) * ∑ j, y j := by
    rw [← vn_sum_sum_mul]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
    ring
  have hD : ∑ i, ∑ j, ∑ k, (if i = j ∧ j = k then y i * y j * y k else 0)
      = ∑ i, y i * y i * y i := by
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_eq_single i]
    · simp
    · intro j _ hji
      simp [Ne.symm hji]
    · intro h
      exact absurd (Finset.mem_univ i) h
  have hX : ∑ i, y i * y i = ∑ i, y i ^ 2 :=
    Finset.sum_congr rfl fun i _ => (sq (y i)).symm
  have hY : ∑ i, y i * y i * y i = ∑ i, y i ^ 3 :=
    Finset.sum_congr rfl fun i _ => by ring
  rw [hdist]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib]
  rw [hT, hA, hB, hC, hD, hY, hX]
  ring

/-- **T15 (divisibility by 6).** For every integer vector `y`,
`6 ∣ p₁³ - 3 p₁ p₂ + 2 p₃`, so the degree-3 readout `e₃ = (p₁³ - 3p₁p₂ + 2p₃)/6` is exact.
Proof: divisibility by 2 from T12 (`ng2`), by 3 from `(∑ y)³ ≡ ∑ y³ (mod 3)`, and
`gcd(2,3) = 1`. -/
theorem ng3_six_dvd (y : Fin N → ℤ) :
    (6 : ℤ) ∣ (∑ i, y i) ^ 3 - 3 * (∑ i, y i) * (∑ i, y i ^ 2) + 2 * ∑ i, y i ^ 3 := by
  have h2 : (2 : ℤ) ∣ (∑ i, y i) ^ 3 - 3 * (∑ i, y i) * (∑ i, y i ^ 2) + 2 * ∑ i, y i ^ 3 :=
    ⟨(∑ i, y i) * e2 y - (∑ i, y i) * (∑ i, y i ^ 2) + ∑ i, y i ^ 3, by
      linear_combination (∑ i, y i) * ng2 y⟩
  have h3 : (3 : ℤ) ∣ (∑ i, y i) ^ 3 - 3 * (∑ i, y i) * (∑ i, y i ^ 2) + 2 * ∑ i, y i ^ 3 := by
    obtain ⟨c, hc⟩ := three_dvd_cube_sum (Finset.univ : Finset (Fin N)) y
    exact ⟨c - (∑ i, y i) * (∑ i, y i ^ 2) + ∑ i, y i ^ 3, by linear_combination hc⟩
  have hcop : IsCoprime (2 : ℤ) 3 := ⟨-1, 1, by norm_num⟩
  rw [show (6 : ℤ) = 2 * 3 by norm_num]
  exact IsCoprime.mul_dvd hcop h2 h3

/-- **T15 (binary inputs).** For a binary input `x` and lane row `v`,
`e3ord (x ⊙ v) = P1³ - 3 P1 P2 + 2 P3` with `P_k = psum k v x`. -/
theorem ng3_binary (v x : Fin N → ℤ) (hx : IsBinary x) :
    e3ord (fun i => x i * v i)
      = psum 1 v x ^ 3 - 3 * psum 1 v x * psum 2 v x + 2 * psum 3 v x := by
  have h : e3ord (fun i => x i * v i) = (∑ i, x i * v i) ^ 3
      - 3 * (∑ i, x i * v i) * (∑ i, (x i * v i) ^ 2) + 2 * ∑ i, (x i * v i) ^ 3 := ng3 _
  have h1 : ∑ i, x i * v i = psum 1 v x := by simp only [psum, pow_one]
  have h2 : ∑ i, (x i * v i) ^ 2 = psum 2 v x := by
    unfold psum
    exact Finset.sum_congr rfl fun i _ => by rcases hx i with h | h <;> simp [h]
  have h3 : ∑ i, (x i * v i) ^ 3 = psum 3 v x := by
    unfold psum
    exact Finset.sum_congr rfl fun i _ => by rcases hx i with h | h <;> simp [h]
  rw [h, h1, h2, h3]

/-! ### T16 -/

/-- **T16 (head mix range).** With `r ≤ 32` lanes, `|Λ| ≤ 127` and `|ê| ≤ 2^15`,
`|∑ t, Λ t ê t| ≤ 32 · 127 · 32768 = 133,169,152 < 2^31`: the int32 head sum cannot
overflow. -/
theorem mix_range {r : ℕ} (Lam e : Fin r → ℤ) (hr : r ≤ 32) (hL : ∀ t, |Lam t| ≤ 127)
    (he : ∀ t, |e t| ≤ 32768) : |∑ t, Lam t * e t| < 2 ^ 31 := by
  calc |∑ t, Lam t * e t| ≤ ∑ t, |Lam t * e t| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _t : Fin r, (127 * 32768 : ℤ) := Finset.sum_le_sum fun t _ => by
        rw [abs_mul]
        exact mul_le_mul (hL t) (he t) (abs_nonneg _) (by norm_num)
    _ = (r : ℤ) * (127 * 32768) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    _ ≤ 32 * (127 * 32768) :=
        mul_le_mul_of_nonneg_right (by exact_mod_cast hr) (by norm_num)
    _ < 2 ^ 31 := by norm_num

/-- **T16 (degree-3 readout range).** If `|p₁| ≤ 480`, `|p₂| ≤ 7200`, `|p₃| ≤ 108000`
(the C1′ ranges for `|A| ≤ 15` and at most 32 active features), then
`|p₁³ - 3 p₁ p₂ + 2 p₃| ≤ 121,176,000 < 2^31`. -/
theorem ng3_range (p1 p2 p3 : ℤ) (h1 : |p1| ≤ 480) (h2 : |p2| ≤ 7200) (h3 : |p3| ≤ 108000) :
    |p1 ^ 3 - 3 * p1 * p2 + 2 * p3| < 2 ^ 31 := by
  have a1 : |p1 ^ 3| ≤ 110592000 := by
    rw [abs_pow]
    calc |p1| ^ 3 ≤ 480 ^ 3 := by gcongr
      _ = 110592000 := by norm_num
  have a2 : |3 * p1 * p2| ≤ 10368000 := by
    rw [abs_mul, abs_mul, (by norm_num : |(3 : ℤ)| = 3)]
    calc 3 * |p1| * |p2| ≤ 3 * 480 * 7200 := by gcongr
      _ = 10368000 := by norm_num
  have a3 : |2 * p3| ≤ 216000 := by
    rw [abs_mul, abs_two]
    calc 2 * |p3| ≤ 2 * 108000 := by gcongr
      _ = 216000 := by norm_num
  have h31 : (2 : ℤ) ^ 31 = 2147483648 := by norm_num
  obtain ⟨b1, c1⟩ := abs_le.mp a1
  obtain ⟨b2, c2⟩ := abs_le.mp a2
  obtain ⟨b3, c3⟩ := abs_le.mp a3
  rw [h31, abs_lt]
  constructor <;> linarith

/-- **T16 (degree-2 lane range).** If `|P1| ≤ 4064` and `0 ≤ P2 ≤ 516128`
(the T9 certificates for `|V| ≤ 127` and at most 32 active features), then
`|P1² - P2| ≤ 4064² < 2^31`. -/
theorem e_lane_range (P1 P2 : ℤ) (h1 : |P1| ≤ 4064) (h2 : 0 ≤ P2) (h3 : P2 ≤ 516128) :
    |P1 ^ 2 - P2| < 2 ^ 31 := by
  have hsq : P1 ^ 2 ≤ 16516096 := by
    calc P1 ^ 2 = |P1| ^ 2 := (sq_abs P1).symm
      _ ≤ 4064 ^ 2 := by gcongr
      _ = 16516096 := by norm_num
  have h31 : (2 : ℤ) ^ 31 = 2147483648 := by norm_num
  rw [h31, abs_lt]
  constructor <;> nlinarith [sq_nonneg P1]

/-! ### T17 -/

/-- Helper (T17): a pair lane `v = e_a + e_b` with `a ≠ b` has power sum `x a + x b` for
every exponent `k ≠ 0`. -/
lemma psum_pair (k : ℕ) (hk : k ≠ 0) (a b : Fin N) (hab : a ≠ b) (x : Fin N → ℤ) :
    psum k ((Pi.single a 1 + Pi.single b 1 : Fin N → ℤ)) x = x a + x b := by
  unfold psum
  have hpt : ∀ i, x i * ((Pi.single a 1 + Pi.single b 1 : Fin N → ℤ) i) ^ k
      = (if i = a then x a else 0) + (if i = b then x b else 0) := by
    intro i
    by_cases h1 : i = a
    · simp [h1, Pi.single_apply, hab]
    · by_cases h2 : i = b
      · simp [h2, Pi.single_apply, Ne.symm hab]
      · simp [h1, h2, Pi.single_apply, zero_pow hk]
  rw [Finset.sum_congr rfl fun i _ => hpt i, Finset.sum_add_distrib]
  simp

/-- Helper (T17): on binary inputs, a pair lane `v = e_a + e_b` (`a ≠ b`) has
`P1² - P2 = 2 x a x b`. -/
lemma lane_pair (a b : Fin N) (hab : a ≠ b) (x : Fin N → ℤ) (hx : IsBinary x) :
    psum 1 ((Pi.single a 1 + Pi.single b 1 : Fin N → ℤ)) x ^ 2
      - psum 2 ((Pi.single a 1 + Pi.single b 1 : Fin N → ℤ)) x = 2 * (x a * x b) := by
  rw [psum_pair 1 one_ne_zero a b hab, psum_pair 2 two_ne_zero a b hab]
  rcases hx a with ha | ha <;> rcases hx b with hb | hb <;> rw [ha, hb] <;> norm_num

/-- Helper (T17): the pair lane `e_a + e_b` attached to an ordered index pair `p = (a, b)`. -/
def pairLane (p : Fin N × Fin N) : Fin N → ℤ := Pi.single p.1 1 + Pi.single p.2 1

/-- Helper (T17): the head weight of the pair lane `p = (a, b)`: `Q a b` if `a < b`, else `0`. -/
def pairWt (Q : Fin N → Fin N → ℤ) (p : Fin N × Fin N) : ℤ :=
  if p.1 < p.2 then Q p.1 p.2 else 0

/-- Helper (T17): summing the pair lanes over all ordered pairs reproduces the pair table
`Q` on binary inputs. -/
lemma pair_sum (Q : Fin N → Fin N → ℤ) (x : Fin N → ℤ) (hx : IsBinary x) :
    ∑ p : Fin N × Fin N, pairWt Q p * (psum 1 (pairLane p) x ^ 2 - psum 2 (pairLane p) x)
      = 2 * ∑ p ∈ univ.filter (fun p : Fin N × Fin N => p.1 < p.2),
          Q p.1 p.2 * x p.1 * x p.2 := by
  rw [Finset.sum_filter, Finset.mul_sum]
  refine Finset.sum_congr rfl fun p _ => ?_
  unfold pairWt pairLane
  split_ifs with h
  · rw [lane_pair p.1 p.2 (ne_of_lt h) x hx]
    ring
  · simp

/-- **T17 (universality of PSI at full width).** Every integer pair table `Q` is reproduced
exactly by a PSI head: there are `r` lanes `V` and head weights `λ` such that for every binary
`x`, `∑ t, λ t (P1_t² - P2_t) = 2 ∑_{i<j} Q i j x i x j`. The construction uses `r = N²` lanes
indexed by ordered pairs `(a, b)` (via `finProdFinEquiv`), pair lanes `V = e_a + e_b` and
`λ = Q a b` when `a < b` (and `λ = 0` otherwise). -/
theorem psi_universal (Q : Fin N → Fin N → ℤ) :
    ∃ (r : ℕ) (V : Fin r → Fin N → ℤ) (lam : Fin r → ℤ), ∀ x : Fin N → ℤ, IsBinary x →
      ∑ t, lam t * (psum 1 (V t) x ^ 2 - psum 2 (V t) x)
        = 2 * ∑ p ∈ univ.filter (fun p : Fin N × Fin N => p.1 < p.2),
            Q p.1 p.2 * x p.1 * x p.2 := by
  refine ⟨N * N, fun t => pairLane (finProdFinEquiv.symm t),
    fun t => pairWt Q (finProdFinEquiv.symm t), fun x hx => ?_⟩
  rw [← pair_sum Q x hx]
  exact Equiv.sum_comp finProdFinEquiv.symm
    (fun p => pairWt Q p * (psum 1 (pairLane p) x ^ 2 - psum 2 (pairLane p) x))

/-! ### T18 -/

/-- **T18(a) (zero heads).** With `Λ = 0` every PSI head output vanishes, for every lane
table `V` and every input `x`. -/
theorem psi_zero_heads {r m : ℕ} (V : Fin r → Fin N → ℤ) (x : Fin N → ℤ) :
    (fun h : Fin m => ∑ t, (0 : Fin m → Fin r → ℤ) h t * (psum 1 (V t) x ^ 2 - psum 2 (V t) x))
      = 0 := by
  funext h
  simp

/-- **T18(b) (fc_0 embedding).** If the extra fc_0 columns `Bz` reading the PSI outputs are
zero, then fc_0 computes exactly NNUE's `A a`. -/
theorem fc0_embed {o m : ℕ} (A : Matrix (Fin o) (Fin L) ℤ) (Bz : Matrix (Fin o) (Fin m) ℤ)
    (a : Fin L → ℤ) (z : Fin m → ℤ) (hB : Bz = 0) :
    A.mulVec a + Bz.mulVec z = A.mulVec a := by
  subst hB
  simp

/-- **T18(c) (affine maps have zero mixed second difference).** For any affine
`f x = c + w₀ x₀ + w₁ x₁` on two indicators, `f(1,1) - f(1,0) - f(0,1) + f(0,0) = 0`. -/
theorem affine_second_diff (c : ℤ) (w : Fin 2 → ℤ) :
    let f := fun x : Fin 2 → ℤ => c + ∑ i, w i * x i
    f ![1, 1] - f ![1, 0] - f ![0, 1] + f ![0, 0] = 0 := by
  intro f
  simp only [f, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons]
  ring

/-- **T18(d) (a PSI lane is not affine).** For the lane `v = (1, 1)` the quantity
`q x = P1² - P2` has mixed second difference `q(1,1) - q(1,0) - q(0,1) + q(0,0) = 2 ≠ 0`,
so by T18(c) it is not affine in the piece indicators. -/
theorem psi_not_affine :
    let q := fun x : Fin 2 → ℤ => psum 1 ![1, 1] x ^ 2 - psum 2 ![1, 1] x
    q ![1, 1] - q ![1, 0] - q ![0, 1] + q ![0, 0] = 2 := by
  intro q
  simp only [q, psum, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons]
  norm_num

end VNNUE
