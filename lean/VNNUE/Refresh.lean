import Mathlib
import VNNUE.Basic

/-!
# VNNUE.Refresh
Spec section 4.6: the low-rank (Woodbury-pattern) king transition (T23) and
source selection / hybrid correctness (T24).
-/

namespace VNNUE

open Finset

variable {R : Type*} [AddCommGroup R] {N L : ℕ}

/-! ## T23. Low-rank king transition -/

/-- **T23 (king transition, Woodbury pattern).** Over any commutative ring `S`, with
bucket weights `W_κ = W0 + U V` and `W_κ' = W0 + U' V`, the bucket-κ′ accumulator of `x'`
equals the bucket-κ accumulator of `x`, plus the κ′ columns applied to the delta `x' - x`,
plus the rank-`r` correction `(U' - U)` applied to `y = V x`.

Scope: the shared part `W0` is the same for κ and κ′, so the identity covers transitions
within one block/region (the C2' region term `R_ρ(κ)` is absorbed into `W0` within a
region). A region change needs `W0 ≠ W0'` and is handled by refresh/Finny instead. Threat and
pawn-pair deltas are additive and orthogonal, so they are not modelled here. The incremental
maintenance of `y` is `king_transition_y_update`. -/
theorem king_transition {S : Type*} [CommRing S] {L r n : ℕ}
    (W0 : Matrix (Fin L) (Fin n) S) (U U' : Matrix (Fin L) (Fin r) S)
    (V : Matrix (Fin r) (Fin n) S) (x x' : Fin n → S) :
    (W0 + U' * V).mulVec x'
      = (W0 + U * V).mulVec x + (W0 + U' * V).mulVec (x' - x)
        + (U' - U).mulVec (V.mulVec x) := by
  simp only [Matrix.add_mulVec, Matrix.sub_mulVec, Matrix.mulVec_sub,
    ← Matrix.mulVec_mulVec]
  abel

/-- **T23 (incremental `y`).** The rank-`r` sketch `y = V x` is maintained incrementally:
after the move, `y' = y + V (x' - x)` equals `V x'`, so the `y` fed into `king_transition`
is the one obtained by the update `y_c += V_K · δ_x`. -/
theorem king_transition_y_update {S : Type*} [CommRing S] {r n : ℕ}
    (V : Matrix (Fin r) (Fin n) S) (x x' : Fin n → S) :
    V.mulVec x + V.mulVec (x' - x) = V.mulVec x' := by
  rw [Matrix.mulVec_sub]; abel

/-- **T23 (factored = merged).** Evaluating the merged bucket matrix `W0 + U V` on `x`
is the same as evaluating the shared part `W0 x` and adding the factored correction
`U (V x)`. -/
theorem factored_eq_merged {S : Type*} [CommRing S] {L r n : ℕ}
    (W0 : Matrix (Fin L) (Fin n) S) (U : Matrix (Fin L) (Fin r) S)
    (V : Matrix (Fin r) (Fin n) S) (x : Fin n → S) :
    (W0 + U * V).mulVec x = W0.mulVec x + U.mulVec (V.mulVec x) := by
  rw [Matrix.add_mulVec, Matrix.mulVec_mulVec]

/-- **T23 (nesting, trivial full-rank embedding).** With inner dimension `n` (`V = 1`,
`W0 = 0`, `U_k = W_k`), the family `W0 + U_k V` contains every family of bucket matrices
`W_k`, so independent king buckets are the special case `r = n`.
NOTE: this does NOT say that the deployed rank `r_K = 16 ≪ n` is as expressive. For `r < n`
the low-rank family is a proper restriction (see `lowrank_proper`), and its cost is measured in
Exp E5. -/
theorem lowrank_generalizes {S : Type*} [CommRing S] {K L n : ℕ}
    (Wk : Fin K → Matrix (Fin L) (Fin n) S) :
    ∃ (W0 : Matrix (Fin L) (Fin n) S) (V : Matrix (Fin n) (Fin n) S)
      (U : Fin K → Matrix (Fin L) (Fin n) S), ∀ k, W0 + U k * V = Wk k :=
  ⟨0, 1, Wk, fun k => by simp⟩

/-- **T23 (the rank restriction is real).** Over any field `F`, if `r < n` there is a
two-bucket family (`W_0 = 0`, `W_1 = 1`, both `n × n`) that no factorization
`W0 + U_k V` with `V : r × n` can represent. If one did, then `(U_1 - U_0) V = 1`, but the
left side has rank `≤ r < n = rank 1`. -/
theorem lowrank_proper {F : Type*} [Field F] {n r : ℕ} (hr : r < n) :
    ¬ ∃ (W0 : Matrix (Fin n) (Fin n) F) (V : Matrix (Fin r) (Fin n) F)
      (U : Fin 2 → Matrix (Fin n) (Fin r) F),
      ∀ k, W0 + U k * V = (![0, 1] : Fin 2 → Matrix (Fin n) (Fin n) F) k := by
  rintro ⟨W0, V, U, h⟩
  have h0 : W0 + U 0 * V = 0 := by simpa using h 0
  have h1 : W0 + U 1 * V = 1 := by simpa using h 1
  have key : (U 1 - U 0) * V = 1 := by
    have e : (U 1 - U 0) * V = (W0 + U 1 * V) - (W0 + U 0 * V) := by
      rw [Matrix.sub_mul]; abel
    rw [e, h0, h1, sub_zero]
  have hle := Matrix.rank_mul_le_right (U 1 - U 0) V
  rw [key, Matrix.rank_one, Fintype.card_fin] at hle
  have hV := Matrix.rank_le_height V
  omega

/-- **T23 (rank restriction, concrete instance).** `lowrank_proper` at `ℚ`, `L = n = 2`,
`r = 1`. -/
theorem lowrank_proper_rat_2_1 :
    ¬ ∃ (W0 : Matrix (Fin 2) (Fin 2) ℚ) (V : Matrix (Fin 1) (Fin 2) ℚ)
      (U : Fin 2 → Matrix (Fin 2) (Fin 1) ℚ),
      ∀ k, W0 + U k * V = (![0, 1] : Fin 2 → Matrix (Fin 2) (Fin 2) ℚ) k :=
  lowrank_proper (by norm_num)

/-- **T23 instantiated at `ℤ`.** The king transition identity over the integers. -/
theorem king_transition_int {L r n : ℕ}
    (W0 : Matrix (Fin L) (Fin n) ℤ) (U U' : Matrix (Fin L) (Fin r) ℤ)
    (V : Matrix (Fin r) (Fin n) ℤ) (x x' : Fin n → ℤ) :
    (W0 + U' * V).mulVec x'
      = (W0 + U * V).mulVec x + (W0 + U' * V).mulVec (x' - x)
        + (U' - U).mulVec (V.mulVec x) :=
  king_transition W0 U U' V x x'

/-- **T23 instantiated at `ZMod 65536`.** The king transition identity holds in
`ZMod 65536` (wrapped int16 lanes). The decoded value equals the true integer accumulator
whenever that accumulator lies in int16 range (see the range certificates T7/T9). -/
theorem king_transition_zmod {L r n : ℕ}
    (W0 : Matrix (Fin L) (Fin n) (ZMod 65536)) (U U' : Matrix (Fin L) (Fin r) (ZMod 65536))
    (V : Matrix (Fin r) (Fin n) (ZMod 65536)) (x x' : Fin n → ZMod 65536) :
    (W0 + U' * V).mulVec x'
      = (W0 + U * V).mulVec x + (W0 + U' * V).mulVec (x' - x)
        + (U' - U).mulVec (V.mulVec x) :=
  king_transition W0 U U' V x x'

/-- **T23 (factored = merged) instantiated at `ℤ`.** -/
theorem factored_eq_merged_int {L r n : ℕ}
    (W0 : Matrix (Fin L) (Fin n) ℤ) (U : Matrix (Fin L) (Fin r) ℤ)
    (V : Matrix (Fin r) (Fin n) ℤ) (x : Fin n → ℤ) :
    (W0 + U * V).mulVec x = W0.mulVec x + U.mulVec (V.mulVec x) :=
  factored_eq_merged W0 U V x

/-- **T23 (factored = merged) instantiated at `ZMod 65536`.** Holds in `ZMod 65536`
(wrapped int16 lanes); decoding to the true integer needs the int16 range (T7/T9). -/
theorem factored_eq_merged_zmod {L r n : ℕ}
    (W0 : Matrix (Fin L) (Fin n) (ZMod 65536)) (U : Matrix (Fin L) (Fin r) (ZMod 65536))
    (V : Matrix (Fin r) (Fin n) (ZMod 65536)) (x : Fin n → ZMod 65536) :
    (W0 + U * V).mulVec x = W0.mulVec x + U.mulVec (V.mulVec x) :=
  factored_eq_merged W0 U V x

/-! ## T24. Source selection and hybrid correctness -/

/-- Candidate accumulator sources. -/
inductive Src | refresh | finny | hybrid | woodbury | ancestor | finnyMirror
  deriving DecidableEq

-- DEVIATION: the spec writes `deriving DecidableEq, Fintype`. `deriving Fintype` fails to
-- elaborate on Lean/Mathlib v4.35.0, so it is replaced with an explicit instance that lists
-- all 6 constructors (sanity-checked below by `Fintype.card Src = 6`). No statement is affected.
/-- Finite enumeration of the candidate sources (written by hand: the `Fintype`
deriving handler produces an ill-typed term in this toolchain). -/
instance : Fintype Src where
  elems := {Src.refresh, Src.finny, Src.hybrid, Src.woodbury, Src.ancestor, Src.finnyMirror}
  complete := by intro x; cases x <;> simp

example : Fintype.card Src = 6 := by decide

/-- Stockfish's candidate set: refresh, Finny cache, and the SF hybrid. -/
def sfSet : Finset Src := {Src.refresh, Src.finny, Src.hybrid}

/-- **T24(a) (monotonicity of the minimum).** For any cost `f` into a linear order, the
minimum over a superset `C₂ ⊇ C₁` of candidates is at most the minimum over `C₁`. -/
theorem inf'_superset {ι α : Type*} [LinearOrder α] (C₁ C₂ : Finset ι) (h : C₁ ⊆ C₂)
    (hne : C₁.Nonempty) (f : ι → α) :
    C₂.inf' (hne.mono h) f ≤ C₁.inf' hne f :=
  Finset.le_inf' hne f (fun _ hb => Finset.inf'_le f (h hb))

/-- **T24(a) (never worse than Stockfish, shared-cost form).** For any cost function on
sources into `WithTop ℕ` (unavailable sources cost `⊤`), the best VNNUE source (minimum over
all `Src`) costs at most the best Stockfish source (minimum over `{refresh, finny, hybrid}`).

Modelling assumptions (this is pure min-monotonicity):
* one per-source cost function is shared by both engines, i.e. refresh/finny/hybrid cost the
  same in VNNUE as in SF. This is NOT automatic, because VNNUE may use merged/factored C2
  tables, pays the C2' readout tax of `r_K·L` MACs, and may hold different cache contents.
  `never_worse_dominated` states the per-source dominance hypothesis explicitly;
* the overhead of choosing the source (computing the six costs) is excluded;
* only the FT-column part of the cost is covered (the PSI term is T25). -/
theorem never_worse (cost : Src → WithTop ℕ) :
    (univ : Finset Src).inf' ⟨Src.refresh, mem_univ _⟩ cost
      ≤ sfSet.inf' ⟨Src.refresh, by simp [sfSet]⟩ cost :=
  inf'_superset sfSet univ (subset_univ _) ⟨Src.refresh, by simp [sfSet]⟩ cost

/-- **T24(a) (any linear order).** The spec's "any cost into a linear order" form: for any
`cost : Src → α` with `α` linearly ordered, the minimum over all sources is at most the
minimum over Stockfish's set. Same modelling assumptions as `never_worse`. -/
theorem never_worse_general {α : Type*} [LinearOrder α] (cost : Src → α) :
    (univ : Finset Src).inf' ⟨Src.refresh, mem_univ _⟩ cost
      ≤ sfSet.inf' ⟨Src.refresh, by simp [sfSet]⟩ cost :=
  inf'_superset sfSet univ (subset_univ _) ⟨Src.refresh, by simp [sfSet]⟩ cost

/-- **T24(a) (never worse, two cost functions).** Suppose VNNUE's cost of each Stockfish
source is at most SF's own cost of that source (`costV s ≤ costSF s` for
`s ∈ {refresh, finny, hybrid}`). This per-source dominance is the hypothesis the engineering
claim really needs. Then VNNUE's best source over all six candidates costs at most SF's
best source. Selection overhead is excluded and only FT columns are covered (PSI: T25). -/
theorem never_worse_dominated {α : Type*} [LinearOrder α] (costV costSF : Src → α)
    (hdom : ∀ s ∈ sfSet, costV s ≤ costSF s) :
    (univ : Finset Src).inf' ⟨Src.refresh, mem_univ _⟩ costV
      ≤ sfSet.inf' ⟨Src.refresh, by simp [sfSet]⟩ costSF :=
  Finset.le_inf' _ _ (fun s hs => (Finset.inf'_le costV (mem_univ s)).trans (hdom s hs))

/-- Local copy of T3 (`accS_any_source`, spec 4.1), re-proved here so that this module
depends only on `VNNUE.Basic`: the set accumulator for `S` is obtained from the one for any
stored set `U` by adding the columns of `S \ U` and removing those of `U \ S`. -/
theorem refresh_accS_any_source (b : Fin L → R) (W : Fin N → Fin L → R)
    (S U : Finset (Fin N)) :
    accS b W S = accS b W U + ∑ i ∈ S \ U, W i - ∑ i ∈ U \ S, W i := by
  have h1 := Finset.sum_sdiff (f := W) (Finset.inter_subset_left : S ∩ U ⊆ S)
  have h2 := Finset.sum_sdiff (f := W) (Finset.inter_subset_right : S ∩ U ⊆ U)
  have e1 : S \ (S ∩ U) = S \ U := by
    ext x; simp only [Finset.mem_sdiff, Finset.mem_inter]; tauto
  have e2 : U \ (S ∩ U) = U \ S := by
    ext x; simp only [Finset.mem_sdiff, Finset.mem_inter]; tauto
  rw [e1] at h1
  rw [e2] at h2
  unfold accS
  rw [← h1, ← h2]
  abel

/-- **T24(b) (hybrid is correct).** Stockfish's hybrid update — previous accumulator,
minus the (corrected) old cache entry, plus the (corrected) new cache entry, plus the
non-PSQ delta `T' - T` (`T` is the threat/pawn-pair component stored in the accumulator) —
reproduces exactly the new accumulator `accS b W' S + T'`. Each corrected cache term equals,
by T3, the true accumulator of the corresponding set, so this form is close to an algebraic
identity. `hybrid_correct_impl` states the implementation order. -/
theorem hybrid_correct (b : Fin L → R) (W W' : Fin N → Fin L → R)
    (S Sprev Cold Cnew : Finset (Fin N)) (T T' : Fin L → R) :
    accS b W' S + T'
      = (accS b W Sprev + T)
        - (accS b W Cold + ∑ i ∈ Sprev \ Cold, W i - ∑ i ∈ Cold \ Sprev, W i)
        + (accS b W' Cnew + ∑ i ∈ S \ Cnew, W' i - ∑ i ∈ Cnew \ S, W' i)
        + (T' - T) := by
  rw [← refresh_accS_any_source b W Sprev Cold, ← refresh_accS_any_source b W' S Cnew]
  abel

/-- **T24(b) (implementation order).** This is SF's actual computation. It first forms
`prev - cacheOld + cacheNew` from the stored accumulator `prev = accS b W Sprev + T` and the
raw cache entries `cacheOld = accS b W Cold`, `cacheNew = accS b W' Cnew`. It then applies
all fix-ups in one combined pass to that difference: `+Σ_{S\Cnew} W' - Σ_{Cnew\S} W'`,
then `-Σ_{Sprev\Cold} W + Σ_{Cold\Sprev} W`, then the threat/pawn-pair delta `T' - T`. The
result is exactly the new accumulator `accS b W' S + T'`. -/
theorem hybrid_correct_impl (b : Fin L → R) (W W' : Fin N → Fin L → R)
    (S Sprev Cold Cnew : Finset (Fin N)) (T T' : Fin L → R) :
    ((accS b W Sprev + T) - accS b W Cold + accS b W' Cnew)
        + ∑ i ∈ S \ Cnew, W' i - ∑ i ∈ Cnew \ S, W' i
        - ∑ i ∈ Sprev \ Cold, W i + ∑ i ∈ Cold \ Sprev, W i
        + (T' - T)
      = accS b W' S + T' := by
  rw [refresh_accS_any_source b W' S Cnew, refresh_accS_any_source b W Sprev Cold]
  abel

/-- **T24(b) instantiated at `ZMod 65536`.** The hybrid identity holds in `ZMod 65536`
(wrapped int16 lanes). The decoded value equals the true integer accumulator whenever that
accumulator lies in int16 range (see the range certificates T7/T9). -/
theorem hybrid_correct_zmod (b : Fin L → ZMod 65536) (W W' : Fin N → Fin L → ZMod 65536)
    (S Sprev Cold Cnew : Finset (Fin N)) (T T' : Fin L → ZMod 65536) :
    accS b W' S + T'
      = (accS b W Sprev + T)
        - (accS b W Cold + ∑ i ∈ Sprev \ Cold, W i - ∑ i ∈ Cold \ Sprev, W i)
        + (accS b W' Cnew + ∑ i ∈ S \ Cnew, W' i - ∑ i ∈ Cnew \ S, W' i)
        + (T' - T) :=
  hybrid_correct b W W' S Sprev Cold Cnew T T'

end VNNUE
