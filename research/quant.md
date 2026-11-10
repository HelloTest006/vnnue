# Quantization, Numerics and Hardware: Research Notes for VNNUE

Scope: integer quantization, fixed-point error analysis, overflow-freedom, Lipschitz bounds of NNUE activations, error propagation, and SIMD/memory performance models. The goal is to pull out formulas that can be stated and proved in Lean 4 + Mathlib. Every factual claim is tagged with a source key ([S1], [R3], ...) from the lists at the end. Claims marked [DERIVED] are elementary consequences that we should prove in Lean rather than cite.

---

## 1. How Stockfish NNUE quantizes (verified from primary sources)

### 1.1 Classic scheme (nnue-pytorch documentation) [S1]

nnue-pytorch `docs/nnue.md`, sections "Quantization" and "The math of quantization":

- **Float-domain activation.** ClippedReLU is `y = min(max(x, 0), 1)`. The upper clip at 1 is there because "aggressive quantization requires reducing the dynamic range of hidden layer inputs."
- **Feature transformer (FT).** The layer only adds, so "it is enough that we multiply the weights and biases by 127." "Both weights and biases are stored as int16." The accumulator is int16. It cannot be int8 because values would overflow before ClippedReLU. Conversion to int8 happens inside ClippedReLU.
- **Quantized ClippedReLU.** "Instead of clamping to 0..1 we clamp to 0..127". The input is int16 or int32 and the output is int8. So the activation scale is s_A = 127.
- **Hidden linear layers.** Inputs are int8 and weights are int8 with scale s_W = 64 "(used in Stockfish)". Accumulation is int32, so biases are int32. The largest representable float weight is `127/64 = 1.984375` and the resolution is `1/64 = 0.015625`. The defining identity is

  `((s_A x)(s_W w) + b s_A s_W) / s_W = s_A y`,

  so the bias is scaled by s_A s_W, the weights by s_W, and the output is divided by s_W. Because s_W is a power of two, the division is an arithmetic right shift (`_mm_srai_epi32(outval, log2_weight_scale)`, i.e. `>> 6`).
- **Output layer.** An extra scale s_O is introduced, where "1 in the float space is equal to s_O internal evaluation units". The identity is

  `((s_A x)(s_W s_O / s_A · w) + b s_W s_O) / s_W = s_O y`,

  with output weights still int8.
- **Trainer clamps that keep the scheme representable.** Hidden weights are clamped to ±127/64. Output weights are clamped to ±127·127/9600, so with s_A = 127 and s_W·s_O/s_A = 9600/127 the weights stay in ±127.
- **Overflow.** "The quantization scheme must be chosen such that no combination of possible active features can exceed the maximum value." With a float accumulator, drift from incremental updates would be bounded by O(MAX_DEPTH), because accumulators live on the search stack and are never un-updated. With integers, updates are exact, but overflow becomes possible.
- **Feature counts.** The "A" (768) feature set has `64·6·2 = 768` features with at most 32 active. A move changes at most 4 inputs (castling), "and the average should be below 3". HalfKP has `64·64·5·2 = 40960` features (plus 64 unused, a Shogi legacy), indexed as `halfkp_idx = piece_square + (p_idx + king_square·10)·64`. Input sparsity is "in the order of 0.1%". The classic 256x2-32-32-1 HalfKP net has about 10M first-layer parameters, about 20 MB quantized. The documented architecture progression is `256x2->32->32->1` → `1024x2->8->32->1`.
- **Quantmoid4.** Approximates sigmoid(4x) with integer outputs in [0, 126]. 126 was chosen as the largest even 8-bit integer, so that x = 0 maps exactly to the midpoint.

### 1.2 Current Stockfish master (fetched 2026-11-09) [S2][S3][S4]

From `src/nnue/nnue_common.h` [S2]:

- `OutputScale = 16`
- `WeightScaleBits = 6`, so s_W = 64
- `CacheLineSize = 64` bytes
- `MaxSimdWidth = 32` bytes (AVX2 register)
- `FtMaxVal = 255` and `HiddenOneVal = 128`
- Types: `TransformedFeatureType = u8`, `BiasType = i16`, `WeightType = i8`, `ThreatWeightType = i8`

From `src/nnue/nnue_architecture.h` [S3]:

- `L1 = 1024`, `L2 = 32`, `L3 = 32`, `LayerStacks = 32`
- Feature sets are `Features::FullThreats`, `Features::PP_3Wide` and `Features::K32H3`. These are newer than HalfKAv2_hm. Their dimensions are in headers not fetched here, so treat them as [UNVERIFIED].
- The hidden stack applies squared and plain clipped ReLUs to fc_0 (1024→32) and concatenates them into 64 values. fc_1 maps 64→32 with dual activations. fc_2 takes all 128 concatenated values and outputs 1, plus a skip term from fc_0. Final scaling uses `multiplier = 600 * OutputScale`.

From `src/nnue/nnue_feature_transformer.h` [S4], the FT output uses a **pairwise product**:

- `sum0 = clamp(acc[j], 0, 255)` and `sum1 = clamp(acc[j + L1/2], 0, 255)`
- `out[j] = (sum0 · sum1) / 512`, which is u8 with range 0..⌊255²/512⌋ = 127
- The x86 path implements the divide as `slli 7` followed by `mulhi_16`, i.e. (a·2^7·b) >> 16 = (a·b) >> 9. Only the first operand needs the lower clip, because `packus` zeroes negatives.

So the int16 accumulator is mapped into the u8 range [0,127], which exactly recovers the classic s_A = 127 for the next layer. The float interpretation is clamp(x,0,1)·clamp(y,0,1), with 255·255/512 ≈ 127.

### 1.3 Why 127 and not 255 for hidden-layer inputs [DERIVED, from x86 ISA semantics]

The x86 `pmaddubsw` (`_mm256_maddubs_epi16`) multiplies unsigned 8-bit by signed 8-bit numbers and adds adjacent pairs into int16 **with signed saturation** [S7].

- With activations in [0,127] and weights in [-127,127], a pair sum is bounded by 2·127·127 = 32258 ≤ 32767, so no saturation can occur.
- With activations in [0,255], the bound is 2·255·127 = 64770, and the instruction would saturate silently.

This is a provable lemma: `∀ a₁ a₂ ∈ [0,127], w₁ w₂ ∈ [-127,127], |a₁w₁ + a₂w₂| ≤ 32258 < 2^15`. VNNI `vpdpbusd` instead accumulates groups of 4 u8·i8 products into int32 [S7], which removes the constraint: 4·255·128 = 130560 ≪ 2^31.

### 1.4 Community trainer conventions (bullet) [S5]

Engines trained with the bullet trainer typically use these values. The sources are secondary (engine READMEs and tutorials), so treat them as conventions, not specifications:

- QA = 255 (FT scale with SCReLU) or 127 (CReLU / pairwise)
- QB = 64 (output weight scale)
- SCALE = 400 (eval units, matching the sigmoid(eval/400) training target)
- SCReLU inference:
  1. clamp to [0, QA]
  2. square, multiply by the output weight and sum over both perspectives
  3. divide by QA
  4. add the bias
  5. multiply by SCALE and divide by QA·QB

Overflow note: v² with v ≤ 255 is 65025, which exceeds the int16 range. A standard trick is to compute (v·w)·v with |w| ≤ 127, so that v·w ≤ 32385 fits in int16, and then use `madd_epi16` into int32. This is why SCReLU nets clip output weights to ±127 even when they are stored as i16.

---

## 2. Integer-arithmetic-only inference (Jacob et al. 2018) [R1]

The affine quantization map is r = S(q − Z), with real scale S > 0 and integer zero-point Z, so that real 0 is exactly representable. For a matrix product r₃ = r₁r₂:

  q₃^{(i,k)} = Z₃ + M Σ_j (q₁^{(i,j)} − Z₁)(q₂^{(j,k)} − Z₂),  with M = S₁S₂/S₃.

- Empirically M ∈ (0,1). It is written M = 2^{−n} M₀ with M₀ ∈ [0.5, 1) and implemented as a 32-bit fixed-point multiply plus a rounding right shift.
- Accumulation uses int32 and biases are int32 with S_bias = S₁S₂ and Z_bias = 0.
- The paper's training uses "simulated quantization" (fake quant) with straight-through gradients. NNUE trainers do the same, implicitly through clamping.

NNUE is the special case Z = 0 (symmetric weights) with power-of-two M = 2^{−6}. That choice makes requantization a single shift, at the cost of a fixed weight range. Related surveys: Krishnamoorthi 2018 [R2] and Gholami et al. 2021 [R3]. Gupta et al. 2015 [R4] show that stochastic rounding allows 16-bit fixed-point training.

---

## 3. Fixed-point and floating-point error analysis (Higham; Wilkinson)

### 3.1 Standard model [R5][R6]

Floating-point arithmetic satisfies fl(x op y) = (x op y)(1+δ) with |δ| ≤ u, where u = 2^{−t} is the unit roundoff. Higham's γ_n is defined as

  γ_n = n u / (1 − n u)  for n u < 1.

For recursive summation of n terms, |ŝ − s| ≤ γ_{n−1} Σ|x_i|. For a dot product, |fl(xᵀy) − xᵀy| ≤ γ_n |x|ᵀ|y| [R5, Ch. 3–4]. The probabilistic analysis of Higham & Mary 2019 [R7] replaces n u by roughly √n·u with high probability.

### 3.2 Integer/fixed-point specialization [DERIVED; provable in Lean]

Integer addition and multiplication are **exact** when no overflow occurs. All error therefore comes from three places:

1. **Weight rounding.** q = round(s·w), so |q − s·w| ≤ 1/2, i.e. |q/s − w| ≤ 1/(2s).
2. **Requantization.** For the shift y = ⌊z / 2^k⌋: z/2^k − 1 < y ≤ z/2^k. Arithmetic right shift is floor division even for negative z, which gives a bias of −1/2 ULP on average. Round-to-nearest uses ⌊(z + 2^{k−1}) / 2^k⌋, which is within 1/2.
3. **Clipping and saturation.** This is not an error if the bounds of Section 4 hold.

**Lemma (accumulated rounding error of the FT).** Let the active set A have |A| ≤ k, and quantize with q_i = round(s·w_i) and q_b = round(s·b). Then

  | (q_b + Σ_{i∈A} q_i)/s − (b + Σ_{i∈A} w_i) | ≤ (k+1)/(2s).

For s = 127 and k = 32, the bound is 33/254 ≈ 0.13 in float units. That is the worst case. Under independent uniform rounding errors, the RMS error is √(33/12)/127 ≈ 0.013.

**Lemma (dot product with quantized weights and activations).** Let |x_i| ≤ X, |w_i| ≤ W, |x̂_i − x_i| ≤ ε_x and |ŵ_i − w_i| ≤ ε_w. Then

  |Σ x̂_i ŵ_i − Σ x_i w_i| ≤ n(X ε_w + W ε_x + ε_x ε_w).

This follows from x̂ŵ − xw = x(ŵ−w) + (x̂−x)w + (x̂−x)(ŵ−w).

---

## 4. Overflow-freedom bounds

### 4.1 Accumulator bound [DERIVED]

Let the bias satisfy |b_j| ≤ B and the weights |W_{ij}| ≤ M. If at most k features are active, then for every output lane j

  |acc_j| = |b_j + Σ_{i∈A} W_{ij}| ≤ B + k·M.

The int16 accumulator is therefore safe if B + kM ≤ 32767.

| Feature set | Max active per perspective (k) | Condition |
|---|---|---|
| HalfKP | 30 (non-king pieces; kings encoded in the index) | B + 30M ≤ 32767 |
| HalfKA / 768 | 32 | B + 32M ≤ 32767 |

With B = 0 and k = 32, the limit is M ≤ 1023 in int16 units, i.e. |w| ≤ 8.06 at s = 127.

A sharper bound is the per-lane column-sum bound: take the max over admissible A of |b_j + Σ_{i∈A} W_ij|. Because pieces have identities, this can be computed exactly per lane in O(#features) by taking, for each piece slot, the max and min weight among its possible squares. This is an easily checkable certificate that a Lean/`decide` or offline script can verify for a concrete net.

### 4.2 Modular-arithmetic insight: intermediate overflow is harmless [DERIVED; key Lean target]

SIMD `vpaddw`/`vpsubw` (`_mm256_add_epi16`) **wrap** modulo 2^16 rather than saturate [S7]. Wrapping integer addition is a ring homomorphism ℤ → ℤ/2^16ℤ. So for any sequence of incremental updates acc ← acc + W_add − W_rem:

  acc_wrapped ≡ b + Σ_{i∈A_final} W_i (mod 2^16).

If the true value lies in [−2^15, 2^15), the wrapped result equals it exactly. **Only the final-state bound (Section 4.1) matters. No bound on the update path is needed**, even though a path through intermediate states could go out of range if adds are applied before removes.

In Lean, this is `ZMod (2^16)` with `Int.cast` a ring hom, plus injectivity of the cast on an interval of length 2^16 (`ZMod.intCast_cast`, `Int.emod_emod_of_dvd`, etc.).

Contrast: **saturating** arithmetic (`adds_epi16`) is not associative, because sat(sat(a+b)+c) ≠ sat(a+sat(b+c)) in general. Incremental updates with saturation are therefore not path-independent. That is a concrete counterexample to put in the paper (e.g. a = 32767, b = 1, c = −1).

### 4.3 Hidden layers

For an int32 accumulator of n inputs, with u8 activations ≤ 127 and i8 weights ≤ 127, |Σ| ≤ n·127·127 + |b|. For n = 1024, this is 16,516,096 ≈ 2^24, safely below 2^31. The only tight constraint is the int16 saturation of `pmaddubsw` pairs (Section 1.3).

---

## 5. Lipschitz bounds and error propagation

### 5.1 Activations [DERIVED]

| Activation | Lipschitz constant | Note |
|---|---|---|
| CReLU(x) = min(max(x,0),1) | 1 | Composition of 1-Lipschitz max/min. |
| SCReLU(x) = CReLU(x)² | 2 | \|a² − b²\| = \|a+b\|\|a−b\| ≤ 2\|a−b\| for a,b ∈ [0,1]. Quantized at scale Q, the constant is 2Q relative to the input scale. |
| Pairwise product p(x,y) = CReLU(x)·CReLU(y) | 1 in each argument | \|p(x,y) − p(x',y')\| ≤ \|x−x'\| + \|y−y'\|, i.e. Lipschitz 1 for ℓ¹ on the pair and 2 for ℓ∞. Proof: ab − a'b' = a(b−b') + (a−a')b'. |
| Quantmoid4 / sigmoid(4x) | 1 | Sigmoid has slope ≤ 1/4. |

### 5.2 Linear layers

For y = Wx + b, Lip_{∞→∞} = ‖W‖_∞ = max_i Σ_j |W_ij| (max row ℓ¹ norm). Lip_{2→2} = σ_max(W). Composition gives Lip(f∘g) ≤ Lip(f)·Lip(g), so the product of per-layer bounds is an upper bound. It can be very loose (Virmaux & Scaman 2018 [R8] show that exact computation is NP-hard and give tighter estimates; Szegedy et al. 2014 [R9] used these products for adversarial sensitivity).

### 5.3 Error propagation recurrence [DERIVED; Lean target]

Let layer ℓ be f_ℓ = σ_ℓ ∘ A_ℓ with float output x_ℓ and quantized output x̂_ℓ. Then

  ‖x̂_ℓ − x_ℓ‖ ≤ L_σ(‖W_ℓ‖·‖x̂_{ℓ−1} − x_{ℓ−1}‖ + ‖ΔW_ℓ‖·‖x̂_{ℓ−1}‖ + ‖Δb_ℓ‖) + r_ℓ,

where r_ℓ is the requantization error (≤ 1 ULP for floor shifts, ≤ 1/2 for rounding). Unrolling gives

  e_L ≤ Σ_ℓ (Π_{m>ℓ} L_m) η_ℓ,

where η_ℓ is the local error injected at layer ℓ. Two consequences:

- Clipping activations into [0,1] keeps ‖x̂_{ℓ−1}‖_∞ ≤ 1, so local errors stay bounded independently of the inputs.
- With 1-Lipschitz activations, errors grow only through ‖W‖ norms. Layers with large row norms amplify quantization error; that is the rationale for weight clipping during training (±127/64).

---

## 6. Hardware: SIMD throughput, memory bandwidth, caches

### 6.1 Roofline model (Williams, Waterman, Patterson 2009) [R10]

  Attainable GFLOP/s = min(Peak compute π, Peak bandwidth β × Operational intensity I),

where I is operations per byte of DRAM (or cache) traffic. The ridge point I* = π/β separates memory-bound from compute-bound kernels. The memory-wall motivation goes back to Wulf & McKee 1995 [R11].

### 6.2 Operational intensity of NNUE kernels [DERIVED]

- **Incremental FT update.** A move with a adds and r removes reads (a+r) weight columns of L1 int16 each, plus the accumulator (2·L1 bytes), and writes the accumulator. The operations are (a+r)·L1 int16 adds, so

  I ≈ (a+r)·L1 / (2L1(a+r) + 4L1) = (a+r) / (2(a+r) + 4) ≤ 0.5 op/byte.

  For a typical move (a+r = 2), I ≈ 0.25. Desktop CPUs have ridge points of order tens of ops per byte for int16 SIMD, so **FT updates are deeply memory/cache-bound**.
- **Size example.** For HalfKAv2_hm (704·32 = 22528 features per perspective [UNVERIFIED for current master]) and L1 = 1024 int16, the FT is about 46 MB, larger than most L2/L3 caches. Each feature column is 2 KB, or 32 cache lines of 64 B. A random-access column read of 2 KB is well suited to hardware prefetch because it is contiguous.
- **Dense hidden layers.** 1024→32 int8 is 32 KB of weights, reused every evaluation. It sits in L1/L2, and I ≈ 1 MAC per weight byte per evaluation. The layer is compute-bound or latency-bound, and sparse-input tricks (skipping zero u8 activations, as in Stockfish's `AffineTransformSparseInput`) cut its work further.

### 6.3 SIMD widths [S7]

| ISA | Register | int16 lanes | Accumulator registers for 1024 int16 (2 KB) |
|---|---|---|---|
| SSE2 | 128 bit | 8 | 128 |
| AVX2 | 256 bit | 16 | 64 |
| AVX-512 | 512 bit | 32 | 32 |

There are 16 architectural vector registers on AVX2 and 32 on AVX-512. Stockfish therefore processes the accumulator in register-sized tiles. Each tile is loaded once, all add and remove columns are applied, and the tile is stored, which minimizes accumulator traffic to 2·L1 bytes per update instead of 2·L1·(a+r).

### 6.4 Cache behavior and accumulator caching [S1][S6]

- **Accumulator stack.** Accumulators live on the search stack, one per ply, and are computed lazily. Undoing a move costs nothing.
- **Accumulator refresh table.** A king move in a king-relative feature set changes every feature. Stockfish (the "Finny tables" idea, introduced 2024 [S6, UNVERIFIED commit reference]) caches, for each king bucket and perspective, the last accumulator together with its piece bitboards. A refresh then becomes a diff against the cached position instead of a full rebuild.
- **Cost model.** A full refresh of 32 features costs 32 column reads (64 KB). A diff refresh costs |Δ| columns.

---

## 7. Precise, Lean-provable statements (candidate theorem list)

1. **Accumulator bound.** For finsets A with A.card ≤ k and |W i| ≤ M: |b + Σ_{i∈A} W i| ≤ |b| + k·M (`Finset.abs_sum_le_sum_abs`, `Finset.sum_le_card_nsmul`).
2. **Path independence mod 2^n.** For any list of ±columns whose net multiset effect is A_final, the cast to ZMod (2^16) equals the cast of the true sum. If the true sum is in [−2^15, 2^15), the signed lift equals the sum.
3. **Saturation is non-associative.** Concrete counterexample, closed by `decide`.
4. **pmaddubsw safety.** 0 ≤ a_i ≤ 127 and |w_i| ≤ 127 ⇒ |a₁w₁ + a₂w₂| ≤ 32258 < 32768. With a_i ≤ 255 the bound fails (witness 255·127·2 = 64770).
5. **Rounding error of sum.** |round(x) − x| ≤ 1/2 ⇒ |Σ round(s w_i)/s − Σ w_i| ≤ k/(2s).
6. **Floor-shift bounds.** For integer z and k: 2^k·⌊z/2^k⌋ ≤ z < 2^k(⌊z/2^k⌋+1) (`Int.ediv_emod_unique`, `Int.emod_nonneg`).
7. **Pairwise FT range.** a, b ∈ [0,255] ⇒ ⌊ab/512⌋ ∈ [0,127], and (a·2^7·b) / 2^16 = ab / 2^9 (exactly, as natural-number division).
8. **Lipschitz lemmas.** CReLU is 1-Lipschitz (`LipschitzWith 1`, via `abs_max_sub_max_le_abs` and the min analogue). Squaring on [0,1] is 2-Lipschitz. The product is Lipschitz in ℓ¹ with constant 1 on [0,1]².
9. **Composition.** `LipschitzWith.comp` gives Lip(f∘g) ≤ Lip f · Lip g, and the error-propagation unrolling (Section 5.3) follows by induction on layers.
10. **Hidden-layer int32 safety.** n·127·127 + |b| < 2^31 for n ≤ 2^17 (with |b| ≤ 2^30, say).

---

## 8. Quantitative numbers at a glance

| Quantity | Value | Source |
|---|---|---|
| FT scale (CReLU era) | 127 | [S1] |
| FT clamp (current pairwise) | 0..255, product /512 → 0..127 | [S4] |
| Hidden weight scale | 64 (WeightScaleBits = 6) | [S1][S2] |
| Max hidden float weight | 127/64 = 1.984375 | [S1] |
| Output weight clamp | ±127²/9600 | [S1] |
| OutputScale; final multiplier | 16; 600·16 | [S2][S3] |
| L1/L2/L3/LayerStacks (master) | 1024/32/32/32 | [S3] |
| Max active (A-768) | 32; changes per move ≤ 4, average < 3 | [S1] |
| HalfKP inputs | 40960 (+64 unused) | [S1] |
| Input sparsity | about 0.1% | [S1] |
| Cache line | 64 B | [S2] |
| pmaddubsw pair bound at 127 | 32258 < 32767 | [DERIVED] |
| Worst-case FT rounding error (k = 32, s = 127) | 33/254 ≈ 0.13 | [DERIVED] |
| FT update operational intensity | ≤ 0.5 int16-op/byte | [DERIVED] |

---

## 9. Implications for VNNUE (concrete design ideas)

1. **Certified-overflow-free accumulators via modular arithmetic.** Specify VNNUE accumulators in ℤ/2^16 with wrapping SIMD adds. Prove (Lean) path independence plus final-state range implies exactness. Then derive the trainer weight clamp from the bound B + kM < 2^15, or from the tighter per-lane max/min-per-piece-slot certificate. This replaces folklore with a theorem and lets the trainer use **larger weights** (up to M ≈ 1023 at k = 32) than intuition-driven clamps.
2. **Per-lane certificate instead of a global clamp.** A per-lane bound only requires that the sum of per-piece-slot extremes fits. A verifier script can check that a trained net satisfies it, and the paper can state the theorem "certificate ⇒ no overflow on any legal position" (approximated by "any ≤ 32-piece multiset").
3. **Exploit int16 headroom by choosing the FT scale.** The rounding error bound (k+1)/(2s) decreases with s. With the final-state-only constraint, s can be raised (e.g. 255 or 511 instead of 127) whenever the certificate allows. This trades range for precision provably.
4. **Lipschitz-budgeted architecture.** Use 1-Lipschitz activations (CReLU, pairwise product on [0,1]², which is ℓ¹-Lipschitz 1) and train with row-ℓ¹ penalties. The end-to-end bound Σ_ℓ Π L_m η_ℓ is then small and provable, and the paper can report the measured quantization error versus the bound.
5. **Stochastic or unbiased requantization.** Floor shifts bias by −1/2 ULP per layer. VNNUE could use round-to-nearest (add 2^{k−1} before shifting, one extra instruction). Prove the |error| ≤ 1/2 lemma and measure the eval-drift reduction.
6. **VNNI-aware activation range.** On AVX-512 VNNI or AVX-VNNI (`vpdpbusd`, int32 accumulation), the 127 cap from `pmaddubsw` saturation disappears, so hidden activations could use the full u8 range 0..255, halving activation quantization error. Keep a 127 path for AVX2. Lemma 4 in Section 7 formalizes exactly when each is safe.
7. **Roofline-driven update design.** FT updates are memory-bound (I ≤ 0.5). VNNUE design levers that reduce bytes per update:
   - (a) a smaller L1 or int8 FT weights with an int16 accumulator, which halves column bytes; the overflow bound then still has to hold;
   - (b) factorized or shared feature columns (a "piece-square" base plus a small king-bucket delta), which shrink the working set to fit L2;
   - (c) fused multi-ply updates (apply several plies' deltas in one tile pass), which raise I toward (a+r)/(2(a+r)+4) with larger a+r.

   All are measurable by experiment. Report bytes/update and nodes/s.
8. **Exact algebraic identity for multi-ply catch-up.** Because the accumulator is a group homomorphism from feature multisets, composing deltas is exact: Δ(p→q) + Δ(q→r) = Δ(p→r), with cancellation of add/remove pairs of the same feature. This can be proved in Lean over `Multiset`/`Finsupp` and used to justify cancelling redundant column reads (e.g. a piece moved and moved back), cutting memory traffic.
9. **Saturation-free proofs as a selling point.** Show the non-associativity counterexample for saturating adds. It explains why NNUE must use wrapping adds, and why VNNUE's certificate is the right correctness notion.

---

## References

Primary software/documentation sources:

- [S1] Official Stockfish developers, "NNUE" documentation, nnue-pytorch repository, `docs/nnue.md`. https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md (fetched 2026-11-09). VERIFIED.
- [S2] Stockfish source, `src/nnue/nnue_common.h`, master branch. https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_common.h (fetched 2026-11-09). VERIFIED.
- [S3] Stockfish source, `src/nnue/nnue_architecture.h`, master branch. https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_architecture.h (fetched 2026-11-09). VERIFIED.
- [S4] Stockfish source, `src/nnue/nnue_feature_transformer.h`, master branch. https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_feature_transformer.h (fetched 2026-11-09). VERIFIED.
- [S5] Community bullet-trainer conventions (QA = 255, QB = 64, SCALE = 400), e.g. https://int0x80-ca.pages.dev/posts/chess-engines-nnue/3-nnue and https://github.com/Bastiball21/cwtch. Secondary sources; the bullet trainer itself is at https://github.com/jw1912/bullet [UNVERIFIED as authoritative spec].
- [S6] Stockfish "accumulator refresh table" / "Finny tables" (2024). [UNVERIFIED: exact commit/PR not checked.]
- [S7] Intel Corporation, Intel 64 and IA-32 Architectures Software Developer's Manual and Intel Intrinsics Guide (PMADDUBSW, PADDW, PADDSW, VPDPBUSD). https://www.intel.com/content/www/us/en/docs/intrinsics-guide/ (instruction semantics standard; not re-fetched).
- [S8] Y. Nasu, "Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi," The 28th World Computer Shogi Championship Appeal Document, 2018 (in Japanese; English translation by D. Rodriguez, 2021). [UNVERIFIED URL.]

Literature:

- [R1] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, D. Kalenichenko, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018, pp. 2704–2713. arXiv:1712.05877.
- [R2] R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference: A whitepaper," arXiv:1806.08342, 2018.
- [R3] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, K. Keutzer, "A Survey of Quantization Methods for Efficient Neural Network Inference," arXiv:2103.13630, 2021.
- [R4] S. Gupta, A. Agrawal, K. Gopalakrishnan, P. Narayanan, "Deep Learning with Limited Numerical Precision," ICML 2015, PMLR 37:1737–1746.
- [R5] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, Philadelphia, 2002. ISBN 978-0-89871-521-7.
- [R6] J. H. Wilkinson, *Rounding Errors in Algebraic Processes*, Prentice-Hall, 1963.
- [R7] N. J. Higham and T. Mary, "A New Approach to Probabilistic Rounding Error Analysis," SIAM J. Sci. Comput. 41(5):A2815–A2835, 2019.
- [R8] A. Virmaux and K. Scaman, "Lipschitz regularity of deep neural networks: analysis and efficient estimation," NeurIPS 2018.
- [R9] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, R. Fergus, "Intriguing properties of neural networks," ICLR 2014. arXiv:1312.6199.
- [R10] S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," Communications of the ACM 52(4):65–76, 2009.
- [R11] W. A. Wulf and S. A. McKee, "Hitting the Memory Wall: Implications of the Obvious," ACM SIGARCH Computer Architecture News 23(1):20–24, 1995.
- [R12] IEEE Std 754-2019, IEEE Standard for Floating-Point Arithmetic, 2019.

Note: R1–R12 are standard, well-known references cited from prior knowledge; bibliographic details (pages/volumes) should be double-checked before final submission but are believed accurate. S1–S4 were fetched and quoted directly on 2026-11-09.
