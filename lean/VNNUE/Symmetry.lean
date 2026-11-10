import Mathlib
import VNNUE.Basic

/-!
# VNNUE.Symmetry  (spec section 4.5, theorems T19-T22)

Board symmetries (horizontal mirror `mirH`, vertical flip `flipV`), their action on the
768 absolute features `Feat = colour × piece × square`, the canonicalization map, and
the equivariance / weight-sharing lemmas.
-/

namespace VNNUE

open Finset

/-! ## Local definitions (exactly as in the spec) -/

/-- Horizontal mirror of a square (file `f ↦ 7 - f`), i.e. `s xor 7`. -/
def mirH (s : Fin 64) : Fin 64 := ⟨8 * (s.val / 8) + (7 - s.val % 8), by have := s.isLt; omega⟩
/-- Vertical flip of a square (rank `r ↦ 7 - r`), i.e. `s xor 56`. -/
def flipV (s : Fin 64) : Fin 64 := ⟨8 * (7 - s.val / 8) + s.val % 8, by have := s.isLt; omega⟩
/-- Absolute feature: colour × piece type × square (768 features). -/
abbrev Feat := Fin 2 × Fin 6 × Fin 64
/-- Horizontal mirror acting on features. -/
def sigH (i : Feat) : Feat := (i.1, i.2.1, mirH i.2.2)
/-- Colour swap with vertical flip acting on features. -/
def sigC (i : Feat) : Feat := (Fin.rev i.1, i.2.1, flipV i.2.2)
/-- Canonical representative: first make the colour `0`, then move to files e-h. -/
def canon (i : Feat) : Feat :=
  let j := if i.1 = 0 then i else sigC i
  if j.2.2.val % 8 < 4 then sigH j else j

/-! ## T19 -/

/-- Spec T19 (Klein four-group on squares): the horizontal mirror `h` and vertical flip `v`
are involutions, they commute, and `h`, `v`, `hv` have no fixed squares. -/
theorem klein_squares :
    (∀ s, mirH (mirH s) = s) ∧ (∀ s, flipV (flipV s) = s) ∧
    (∀ s, mirH (flipV s) = flipV (mirH s)) ∧
    (∀ s, mirH s ≠ s) ∧ (∀ s, flipV s ≠ s) ∧ (∀ s, mirH (flipV s) ≠ s) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro s; have := s.isLt; apply Fin.ext; simp only [mirH]; omega
  · intro s; have := s.isLt; apply Fin.ext; simp only [flipV]; omega
  · intro s; have := s.isLt; apply Fin.ext; simp only [mirH, flipV]; omega
  · intro s h; have := s.isLt; have h' := congrArg Fin.val h; simp only [mirH] at h'; omega
  · intro s h; have := s.isLt; have h' := congrArg Fin.val h; simp only [flipV] at h'; omega
  · intro s h; have := s.isLt; have h' := congrArg Fin.val h
    simp only [mirH, flipV] at h'; omega

/-- Spec T19 (king half-board): exactly 32 squares lie on files e-h, and the mirror `h`
maps files a-d onto files e-h (justifying 32 canonical king buckets). -/
theorem king_half :
    (univ.filter (fun s : Fin 64 => 4 ≤ s.val % 8)).card = 32 ∧
    ∀ s : Fin 64, s.val % 8 < 4 → 4 ≤ (mirH s).val % 8 := by
  refine ⟨by decide, ?_⟩
  intro s hs; have := s.isLt; simp only [mirH]; omega

/-! ## Helper lemmas for T20 -/

/-- Helper (T20): the file of a mirrored square is `7 - file`. -/
theorem mirH_file (s : Fin 64) : (mirH s).val % 8 = 7 - s.val % 8 := by
  have := s.isLt; simp only [mirH]; omega

/-- Spec T19 (king half-board, onto): the mirror `h` maps the 32 squares on files a-d
bijectively onto the 32 squares on files e-h: the image of files a-d under `h` is exactly
files e-h (and `h` is injective, being an involution by `klein_squares`); conversely `h`
sends files e-h into files a-d. -/
theorem king_half_onto :
    (univ.filter (fun s : Fin 64 => s.val % 8 < 4)).image mirH =
      univ.filter (fun s : Fin 64 => 4 ≤ s.val % 8) ∧
    ∀ s : Fin 64, 4 ≤ s.val % 8 → (mirH s).val % 8 < 4 := by
  refine ⟨?_, ?_⟩
  · ext s
    simp only [mem_image, mem_filter, mem_univ, true_and]
    constructor
    · rintro ⟨t, ht, rfl⟩
      rw [mirH_file]; omega
    · intro hs
      refine ⟨mirH s, ?_, klein_squares.1 s⟩
      rw [mirH_file]; omega
  · intro s hs; rw [mirH_file]; omega

/-- Helper (T20): `sigH` is an involution on features. -/
theorem sigH_sigH (i : Feat) : sigH (sigH i) = i := by
  obtain ⟨c, p, s⟩ := i
  simp [sigH, klein_squares.1 s]

/-- Helper (T20): `sigC` is an involution on features. -/
theorem sigC_sigC (i : Feat) : sigC (sigC i) = i := by
  obtain ⟨c, p, s⟩ := i
  simp [sigC, klein_squares.2.1 s]

/-- Helper (T20): `sigH` and `sigC` commute. -/
theorem sigH_sigC (i : Feat) : sigH (sigC i) = sigC (sigH i) := by
  obtain ⟨c, p, s⟩ := i
  simp [sigH, sigC, klein_squares.2.2.1 s]

/-- Helper (T20): colour normalization step of `canon`. -/
def normC (i : Feat) : Feat := if i.1 = 0 then i else sigC i

/-- Helper (T20): file folding step of `canon`. -/
def foldH (j : Feat) : Feat := if j.2.2.val % 8 < 4 then sigH j else j

/-- Helper (T20): `canon` is folding after colour normalization. -/
theorem canon_eq (i : Feat) : canon i = foldH (normC i) := rfl

/-- Helper (T20): in `Fin 2`, `rev c = 0 ↔ c ≠ 0`. -/
theorem fin2_rev_eq_zero (c : Fin 2) : Fin.rev c = 0 ↔ c ≠ 0 := by
  fin_cases c <;> decide

/-- Helper (T20): colour normalization commutes with `sigH`. -/
theorem normC_sigH (i : Feat) : normC (sigH i) = sigH (normC i) := by
  unfold normC
  have h1 : (sigH i).1 = i.1 := rfl
  rw [h1]
  split_ifs
  · rfl
  · exact (sigH_sigC i).symm

/-- Helper (T20): colour normalization is invariant under `sigC`. -/
theorem normC_sigC (i : Feat) : normC (sigC i) = normC i := by
  unfold normC
  have h1 : (sigC i).1 = Fin.rev i.1 := rfl
  by_cases h : i.1 = 0
  · have h2 : ¬ (sigC i).1 = 0 := by rw [h1, fin2_rev_eq_zero]; exact not_not.2 h
    rw [if_neg h2, if_pos h, sigC_sigC]
  · have h2 : (sigC i).1 = 0 := by rw [h1, fin2_rev_eq_zero]; exact h
    rw [if_pos h2, if_neg h]

/-- Helper (T20): `normC` always yields colour `0`. -/
theorem normC_fst (i : Feat) : (normC i).1 = 0 := by
  unfold normC
  split_ifs with h
  · exact h
  · show Fin.rev i.1 = 0
    exact (fin2_rev_eq_zero _).2 h

/-- Helper (T20): file folding is invariant under `sigH`. -/
theorem foldH_sigH (j : Feat) : foldH (sigH j) = foldH j := by
  unfold foldH
  have h1 : (sigH j).2.2.val % 8 = 7 - j.2.2.val % 8 := mirH_file _
  rw [h1, sigH_sigH]
  have := Nat.mod_lt j.2.2.val (by norm_num : 8 > 0)
  split_ifs with ha hb hb <;> first | rfl | omega

/-! ## T20 -/

/-- Spec T20 (free action): on the 768 absolute features, `σ_h`, `σ_c` and `σ_h σ_c` have
no fixed points (`σ_c` changes the colour, `σ_h` has no fixed square). -/
theorem feat_free : ∀ i : Feat, sigH i ≠ i ∧ sigC i ≠ i ∧ sigH (sigC i) ≠ i := by
  intro i
  obtain ⟨c, p, s⟩ := i
  have hc : Fin.rev c ≠ c := by fin_cases c <;> decide
  refine ⟨?_, ?_, ?_⟩
  · intro h
    exact klein_squares.2.2.2.1 s (congrArg (fun k : Feat => k.2.2) h)
  · intro h
    exact hc (congrArg (fun k : Feat => k.1) h)
  · intro h
    exact hc (congrArg (fun k : Feat => k.1) h)

/-- Spec T20 (invariance): `canon` is constant on orbits of the Klein group
generated by `σ_h` and `σ_c`. -/
theorem canon_invariant : ∀ i : Feat, canon (sigH i) = canon i ∧ canon (sigC i) = canon i := by
  intro i
  refine ⟨?_, ?_⟩
  · rw [canon_eq, canon_eq, normC_sigH, foldH_sigH]
  · rw [canon_eq, canon_eq, normC_sigC]

/-- Helper (T20): `canon` lands in colour `0`, files e-h. -/
theorem canon_mem (i : Feat) : (canon i).1 = 0 ∧ 4 ≤ (canon i).2.2.val % 8 := by
  rw [canon_eq]
  unfold foldH
  have h0 := normC_fst i
  split_ifs with h
  · refine ⟨h0, ?_⟩
    rw [show (sigH (normC i)).2.2 = mirH (normC i).2.2 from rfl, mirH_file]
    omega
  · exact ⟨h0, by omega⟩

/-- Helper (T20): `canon` fixes features of colour `0` on files e-h. -/
theorem canon_fix (i : Feat) (h0 : i.1 = 0) (h4 : 4 ≤ i.2.2.val % 8) : canon i = i := by
  rw [canon_eq]
  have : normC i = i := by unfold normC; simp [h0]
  rw [this]
  unfold foldH
  rw [if_neg (by omega)]

/-- Spec T20 (orbit count): `canon` has exactly 192 = 1·6·32 values. Together with
`canon_eq_iff` (the fibres of `canon` are exactly the Klein-group orbits) this shows that
the free Klein-group action has exactly 192 orbits on the 768 features. -/
theorem canon_image_card : (univ.image canon).card = 192 := by
  have himg : univ.image canon =
      ({0} : Finset (Fin 2)) ×ˢ (univ : Finset (Fin 6)) ×ˢ
        (univ.filter (fun s : Fin 64 => 4 ≤ s.val % 8)) := by
    ext ⟨c, p, s⟩
    simp only [mem_image, mem_univ, true_and, mem_product, mem_singleton, mem_filter]
    constructor
    · rintro ⟨k, hk⟩
      have hm := canon_mem k
      rw [hk] at hm
      exact hm
    · rintro ⟨hc, hs⟩
      exact ⟨(c, p, s), canon_fix _ hc hs⟩
  rw [himg, card_product, card_product, king_half.1]
  simp

/-- Helper (T20): orbit membership under the Klein group `{1, σ_h, σ_c, σ_h σ_c}`. -/
def InOrb (i k : Feat) : Prop := k = i ∨ k = sigH i ∨ k = sigC i ∨ k = sigH (sigC i)

/-- Helper (T20): orbit membership is symmetric. -/
theorem inOrb_symm {i k : Feat} (h : InOrb i k) : InOrb k i := by
  rcases h with rfl | rfl | rfl | rfl <;> simp [InOrb, sigH_sigH, sigC_sigC, sigH_sigC]

/-- Helper (T20): orbit membership is transitive. -/
theorem inOrb_trans {i c k : Feat} (h1 : InOrb i c) (h2 : InOrb c k) : InOrb i k := by
  rcases h1 with rfl | rfl | rfl | rfl <;> rcases h2 with rfl | rfl | rfl | rfl <;>
    simp [InOrb, sigH_sigH, sigC_sigC, sigH_sigC]

/-- Spec T20 (canon picks an orbit element): `canon i` lies in the Klein-group orbit of `i`,
i.e. `canon i ∈ {i, σ_h i, σ_c i, σ_h σ_c i}`. -/
theorem canon_mem_orbit (i : Feat) :
    canon i = i ∨ canon i = sigH i ∨ canon i = sigC i ∨ canon i = sigH (sigC i) := by
  rw [canon_eq]
  unfold foldH normC
  split_ifs <;> simp

/-- Spec T20 (fibres of canon = orbits): `canon i = canon k` iff `k` lies in the orbit
`{i, σ_h i, σ_c i, σ_h σ_c i}` of `i`. Hence the orbits are in bijection with the image of
`canon`, which has 192 elements (`canon_image_card`): exactly 192 orbits. -/
theorem canon_eq_iff (i k : Feat) :
    canon i = canon k ↔ k = i ∨ k = sigH i ∨ k = sigC i ∨ k = sigH (sigC i) := by
  constructor
  · intro h
    have a : InOrb i (canon i) := canon_mem_orbit i
    have b : InOrb k (canon k) := canon_mem_orbit k
    rw [← h] at b
    exact inOrb_trans a (inOrb_symm b)
  · rintro (rfl | rfl | rfl | rfl)
    · rfl
    · exact ((canon_invariant i).1).symm
    · exact ((canon_invariant i).2).symm
    · rw [(canon_invariant _).1, (canon_invariant _).2]

/-! ## T21 -/

/-- Spec T21 (equivariance, core lemma): if the weights are tied by
`W (σ i) (π j) = W i j`, then lane `π j` of the accumulator of the transformed input
`x ∘ σ⁻¹` equals lane `j` of the accumulator of `x`:
`∑ i, x (σ⁻¹ i) · W i (π j) = ∑ i, x i · W i j`. -/
theorem acc_equivariant {n L : ℕ} (W : Fin n → Fin L → ℤ) (σ : Equiv.Perm (Fin n))
    (π : Equiv.Perm (Fin L)) (hW : ∀ i j, W (σ i) (π j) = W i j) (x : Fin n → ℤ) (j : Fin L) :
    ∑ i, x (σ.symm i) * W i (π j) = ∑ i, x i * W i j := by
  rw [← Equiv.sum_comp σ (fun i => x (σ.symm i) * W i (π j))]
  simp [hW]

/-- Spec T21 (full accumulator equivariance, `acc(g·x) = ρ(g)·acc(x)`): if the weights are
tied by `W (σ i) (π j) = W i j` and the bias is invariant, `b (π j) = b j` (E9b: `Pb = b`),
then lane `π j` of the accumulator `accV b W` (bias + linear part, from `VNNUE.Basic`) on the
transformed input `x ∘ σ⁻¹` equals lane `j` of the accumulator on `x`. -/
theorem acc_equivariant_accV {n L : ℕ} (b : Fin L → ℤ) (W : Fin n → Fin L → ℤ)
    (σ : Equiv.Perm (Fin n)) (π : Equiv.Perm (Fin L)) (hb : ∀ j, b (π j) = b j)
    (hW : ∀ i j, W (σ i) (π j) = W i j) (x : Fin n → ℤ) (j : Fin L) :
    accV b W (x ∘ σ.symm) (π j) = accV b W x j := by
  unfold accV Phi
  simp only [Pi.add_apply, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Function.comp_apply]
  rw [hb, acc_equivariant W σ π hW x j]

/-- Spec T21 (PSI perspective sharing): if the embedding rows are tied by
`V (P t) (σ i) = V t i`, then the first and second power sums of lane `P t` on the
transformed input `x ∘ σ⁻¹` equal those of lane `t` on `x` (for both P1 and P2). -/
theorem psi_shared {n r : ℕ} (V : Fin r → Fin n → ℤ) (σ : Equiv.Perm (Fin n))
    (P : Equiv.Perm (Fin r)) (hV : ∀ t i, V (P t) (σ i) = V t i) (x : Fin n → ℤ) (t : Fin r) :
    psum 1 (V (P t)) (x ∘ σ.symm) = psum 1 (V t) x ∧
    psum 2 (V (P t)) (x ∘ σ.symm) = psum 2 (V t) x := by
  refine ⟨?_, ?_⟩
  · unfold psum
    rw [← Equiv.sum_comp σ (fun i => (x ∘ σ.symm) i * V (P t) i ^ 1)]
    simp [hV]
  · unfold psum
    rw [← Equiv.sum_comp σ (fun i => (x ∘ σ.symm) i * V (P t) i ^ 2)]
    simp [hV]

/-- The colour flip `σ_c` on features as a permutation (it is an involution). -/
def sigCPerm : Equiv.Perm Feat := Function.Involutive.toPerm sigC sigC_sigC

/-- Spec T21 (PSI sharing for the concrete colour flip `σ_c` on the 768 features): if the
PSI embedding rows are tied by `V (P t) (σ_c i) = V t i` for a lane permutation `P`, then
for every power `k` (in particular P1, P2) the power sum of lane `P t` on the colour-flipped
input `x ∘ σ_c` equals that of lane `t` on `x`, i.e. `P_k^(B) = P_k ∘ P`. -/
theorem psi_shared_sigC {r : ℕ} (V : Fin r → Feat → ℤ) (P : Equiv.Perm (Fin r))
    (hV : ∀ t i, V (P t) (sigC i) = V t i) (x : Feat → ℤ) (t : Fin r) (k : ℕ) :
    ∑ i, x (sigC i) * V (P t) i ^ k = ∑ i, x i * V t i ^ k := by
  rw [← Equiv.sum_comp sigCPerm (fun i => x (sigC i) * V (P t) i ^ k)]
  have happ : ∀ i, sigCPerm i = sigC i := fun _ => rfl
  simp only [happ, sigC_sigC, hV]

/-! ## T22 -/

/-- Spec T22 (regular representation): for an involution `σ` on features, the doubled
table `W' i = (W i, W (σ i))` on lanes `Fin L ⊕ Fin L` is equivariant with lane swap,
`W' (σ i) k = W' i (swap k)`, and its second half computes the accumulator of the
transformed input: `∑ i, x i · W' i (inr j) = ∑ i, x (σ i) · W i j`. -/
theorem regular_rep {n L : ℕ} (W : Fin n → Fin L → ℤ) (σ : Equiv.Perm (Fin n))
    (hσ : ∀ i, σ (σ i) = i) :
    let W' : Fin n → (Fin L ⊕ Fin L) → ℤ := fun i => Sum.elim (W i) (W (σ i))
    (∀ i k, W' (σ i) k = W' i (Equiv.sumComm _ _ k)) ∧
    ∀ x : Fin n → ℤ, ∀ j, (∑ i, x i * W' i (Sum.inr j)) = ∑ i, x (σ i) * W i j := by
  intro W'
  refine ⟨?_, ?_⟩
  · intro i k
    cases k <;> simp [W', hσ]
  · intro x j
    simp only [W', Sum.elim_inr]
    rw [← Equiv.sum_comp σ (fun i => x (σ i) * W i j)]
    simp [hσ]

/-- Spec T22 (integer isotypic split): from the symmetric part `u + v` and the
antisymmetric part `u - v` one recovers `2u` and `2v`:
`(u+v)+(u-v) = 2•u` and `(u+v)-(u-v) = 2•v`. -/
theorem isotypic {L : ℕ} (u v : Fin L → ℤ) :
    (u + v) + (u - v) = 2 • u ∧ (u + v) - (u - v) = 2 • v := by
  constructor <;> abel

/-- Spec T22 (isotypic action, C6/E9a): for an involution `σ` on features and weights `W`,
put `a_± x j = ∑ i, x i · (W i j ± W (σ i) j)` (the accumulators of the symmetrized /
antisymmetrized tables `W ± W∘σ`). Then the mirror acts as `+1` on `a_+` and as `-1` on
`a_-`: `a_+ (x ∘ σ) = a_+ x` and `a_- (x ∘ σ) = -(a_- x)`. So a mirror crossing costs only
one negation pass over the `a_-` half. -/
theorem isotypic_action {n L : ℕ} (W : Fin n → Fin L → ℤ) (σ : Equiv.Perm (Fin n))
    (hσ : ∀ i, σ (σ i) = i) (x : Fin n → ℤ) (j : Fin L) :
    (∑ i, x (σ i) * (W i j + W (σ i) j)) = ∑ i, x i * (W i j + W (σ i) j) ∧
    (∑ i, x (σ i) * (W i j - W (σ i) j)) = -(∑ i, x i * (W i j - W (σ i) j)) := by
  refine ⟨?_, ?_⟩
  · rw [← Equiv.sum_comp σ (fun i => x (σ i) * (W i j + W (σ i) j))]
    simp only [hσ]
    exact Finset.sum_congr rfl (fun i _ => by ring)
  · rw [← Equiv.sum_comp σ (fun i => x (σ i) * (W i j - W (σ i) j)), ← Finset.sum_neg_distrib]
    simp only [hσ]
    exact Finset.sum_congr rfl (fun i _ => by ring)

/-- Spec T22 (the isotypic change of basis is injective over ℤ): `(u, v) ↦ (u+v, u-v)`
is injective on integer lane vectors (ℤ has no 2-torsion). -/
theorem isotypic_inj {L : ℕ} (u v u' v' : Fin L → ℤ) (h1 : u + v = u' + v')
    (h2 : u - v = u' - v') : u = u' ∧ v = v' := by
  refine ⟨funext fun j => ?_, funext fun j => ?_⟩ <;>
  · have a := congrFun h1 j
    have b := congrFun h2 j
    simp only [Pi.add_apply, Pi.sub_apply] at a b
    omega

/-- Spec T22 (the isotypic change of basis is not unimodular): `(u, v) ↦ (u+v, u-v)` is not
surjective onto `ℤ²`; e.g. `(1, 0)` has no integer preimage (`u+v` and `u-v` have equal
parity). -/
theorem isotypic_not_surj : ¬ ∃ u v : ℤ, u + v = 1 ∧ u - v = 0 := by
  rintro ⟨u, v, h1, h2⟩
  omega

end VNNUE
