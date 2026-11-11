# Pre-Change `e7` (`SpeedTier::kSquirrel`) VarDCT Objective Function

This document describes the exact mathematical objective functions, heuristics, distance-dependent scaling laws, and cross-stage mismatches of the pre-change `libjxl` VarDCT encoder at effort `7` (`SpeedTier::kSquirrel`).

Let $d = \text{butteraugli\_distance} > 0$ be the target Butteraugli distance, and let the frame's XYB pixel data be $(X, Y, B)$ before inverse Gaborish (and $(X_{\text{gab}}, Y_{\text{gab}}, B_{\text{gab}})$ after inverse Gaborish).

---

## 1. Initial DC Quantization & Adaptive Quantization Field (`enc_adaptive_quantization.cc`)

### 1.1 `InitialQuantDC(d)`

With `kDcQuant = 1.095924047623553`, `kDcMul = 0.3`, `kDcQuantPow = 0.83`:

$$d_{\text{dc}} = \max\!\left(0.5\, d,\; \min\!\left(d,\; k_{\text{DcMul}} \left(\frac{d}{k_{\text{DcMul}}}\right)^{k_{\text{DcQuantPow}}}\right)\right)$$

$$q_{\text{dc}} = \min\!\left(\frac{k_{\text{DcQuant}}}{d_{\text{dc}}},\; 50.0\right)$$

### 1.2 Pre-Gaborish `initial_quant_masking1x1` ($M_{1\times 1}(x, y)$)

Computed on the pre-Gaborish luma channel $Y(x, y)$ with `match_gamma_offset = 0.019`:

$$\text{base}(x, y) = \frac{1}{4}\big(Y(x-1,y) + Y(x+1,y) + Y(x,y-1) + Y(x,y+1)\big)$$

$$r_{\text{inv}}(v) = \text{RatioOfDerivativesOfCubicRootToSimpleGamma}_{\text{invert}=\text{false}}(v)$$

$$\text{diff}_{1\times 1}(x, y) = \ln\!\Big(1 + \big|r_{\text{inv}}(Y(x, y) + 0.019) \cdot (Y(x, y) - \text{base}(x, y))\big|\Big)$$

$$M^{\text{raw}}_{1\times 1}(x, y) = \frac{1.0}{\text{diff}_{1\times 1}(x, y) + 0.01}$$

where `RatioOfDerivativesOfCubicRootToSimpleGamma` uses `kSGmul = 226.77216153508914`, `kSGmul2 = 1 / 73.377132366608819`, `kSGRetMul = kSGmul2 * 18.6580932135 * kInvLog2e`, `kSGVOffset = 7.7825991679894591`, `kEpsilon = 0.01`:

$$\text{Ratio}_{\text{invert}=\text{false}}(v) = \frac{\text{den}(v)}{\text{num}(v)}, \quad \text{num}(v) = 3 k_{\text{SGRetMul}} k_{\text{SGmul}} v_+^2 + 0.01, \quad \text{den}(v) = k_{\text{InvLog2e}} k_{\text{SGmul}} v_+^3 + k_{\text{SGVOffset}} k_{\text{InvLog2e}} + 0.01$$

with $v_+ = \max(v, 0)$. Finally, $M_{1\times 1}$ is smoothed by `Blur1x1Masking`, a symmetric normalized $5\times 5$ filter `WeightsSymmetric5` with unnormalized weights `w00 = 1.0`, `kFilterMask1x1 = {0.364911248, 0.05, 0.1688888021, 0.221069183, 0.306563504}` (sum $\approx 5.442$).

### 1.3 `InitialQuantField` ($q_{\text{init}}(b_x, b_y)$)

Effective target distance passed to `InitialQuantField`:

$$d_{\text{iqf}} = \begin{cases} d & \text{if } \text{loop\_filter.gab} = \text{true} \\ 0.62\, d & \text{otherwise} \end{cases}$$

Baseline AC scale (`kAcQuant = 0.765`):

$$\text{scale}_{\text{aq}} = \frac{0.765}{d_{\text{iqf}}}, \qquad \text{global\_scale seed: } q_{\text{seed}} = \frac{0.39}{d}$$

For each pixel, $\text{diff}^2(x, y) = \min\!\big((r_{\text{inv}}(Y + 0.019)(Y - \text{base}))^2,\, 0.2\big)$ is mapped via `MaskingSqrt`:

$$m(x, y) = 0.25 \sqrt{\text{diff}^2(x, y)\sqrt{211.66567973503678 \cdot 10^8} + 27.505837037000106}$$

averaged over $4\times 4$ blocks (`pre_erosion`), and combined via `FuzzyErosion` over $3\times 3$ neighborhoods of sorted 4 smallest neighbors $\min_0 \le \min_1 \le \min_2 \le \min_3$:

- $\mu = \max(0, (2.0 - d_{\text{iqf}})/2.0)$, $k_{\text{MulBase}} = \{0.125, 0.1, 0.09, 0.06\}$, $k_{\text{MulAdd}} = \{0, -0.1, -0.09, -0.06\}$, renormalized so $\sum_{i=0}^3 k_i = 0.29959705784054957$.
- Summed across the $2\times 2$ sub-quadrants of each $8\times 8$ block into $\text{aq\_map}[b_y, b_x]$.
- Sets `initial_quant_masking[b_y, b_x] = 1.0 / (aq_map[b_y, b_x] + 0.001)`.

Each $8\times 8$ block exponent `out_val = aq_map[b_y, b_x]` then receives additive log-space modulations:
1. `ComputeMask`: rational mask on $v_1 = \max(0.80061763\, \text{out\_val}, 10^{-3})$.
2. `GammaModulation`: $+ 0.1005613337192697 \log_2 \Big(\frac{1}{128}\sum_{8\times 8} \big(r_{\text{true}}(Y+0.16 - X) + r_{\text{true}}(Y+0.16 + X)\big)\Big)$ using `invert=true`.
3. $\min\!\big(\text{HfModulation},\; \text{BlueModulation}\big)$, where:
   - `HfModulation`: `0.42 - 0.38 * sum_4conn(|ΔY| clamped at 0.0206)`
   - `BlueModulation`: `0.90590805 * clamped_blue_excess`
4. Exponentiated with high-distance damping (`kDampenRampStart = 2.0`, `kDampenRampEnd = 14.0`, $\text{dampen} = \text{clamp}(1 - (d_{\text{iqf}} - 2)/12, 0, 1)$, $\text{base\_level} = 0.48\, \text{scale}_{\text{aq}}$):

$$q_{\text{init}}(b_x, b_y) = 2^{\text{out\_val} \cdot 1.442695041} \cdot (\text{scale}_{\text{aq}} \cdot \text{dampen}) + (1 - \text{dampen}) \cdot (0.48\, \text{scale}_{\text{aq}})$$

---

## 2. AC Strategy Objective Function & Greedy Merge (`enc_ac_strategy.cc`)

For a candidate transform $T$ covering $B = B_x \times B_y$ blocks of $8\times 8$ pixels ($N = 64 B$ samples) at tile/block origin $(x, y)$, on the **post-Gaborish** image $(X_{\text{gab}}, Y_{\text{gab}}, B_{\text{gab}})$ with pass-1 CfL factors `(ytox, 0, ytob)`:

### 2.1 Aggregated Quantizer Scale `quant_norm16`

$$\text{quant\_norm16}(T) = \begin{cases} q_{\text{init}}(b_x, b_y) & B = 1 \\ \max_{(i_x, i_y) \in T} q_{\text{init}}(b_x + i_x, b_y + i_y) & B = 2\;(16\times 8,\; 8\times 16) \\ \left(\dfrac{1}{B} \sum_{(i_x, i_y) \in T} q_{\text{init}}(b_x + i_x, b_y + i_y)^{16}\right)^{1/16} & B \ge 4 \end{cases}$$

### 2.2 `EstimateEntropy` Per-Channel Quantization & Loss

Let $C_c$ be the forward transform `TransformFromPixels` of channel $c \in \{0, 1, 2\}$ ($X, Y, B$), and $C'_c = C_c - \text{cmap}_c \cdot C_1$. Using `InvMatrix(T, c)` (which is $0$ at LLF positions) and `Matrix(T, c)` (without global `x_qm_multiplier` / `b_qm_multiplier` or `AdjustQuantBias`):

$$\text{val}_{c,i} = C'_{c,i} \cdot \text{InvMatrix}(T, c)_i \cdot \text{quant\_norm16}$$

$$\text{rval}_{c,i} = \text{Round}(\text{val}_{c,i}) \quad (\text{nearest-even, NO deadzone widening})$$

$$\epsilon^{\text{freq}}_{c,i} = \text{Matrix}(T, c)_i \cdot (\text{val}_{c,i} - \text{rval}_{c,i}) = \text{quant\_norm16} \cdot \big(C'_{c,i} - \hat{C}'_{c,i}\big)_{\text{non-LLF}}$$

Notice that multiplying by `quant_norm16` inside `val` without dividing back inside `mem[i]` scales spatial error by `quant_norm16`, which is later divided back out in `loss_scalar`! After inverse transform $\epsilon^{\text{pix}}_c = \text{TransformToPixels}_T(\epsilon^{\text{freq}}_c)$:

- Pixel masking offsets: `masku_lut = {12.0, 0.0, 4.0}` for $(X, Y, B)$.
- Channel weights: `kChannelMul = {8.2^8, 1.0^8, 1.03^8}` (giving $X$ a weight factor of $8.2$ inside the $L^8$ norm!).
- Spatial $L^8$ loss accumulation across the $N = 64B$ pixels (excluding clipped right border if `x + ix*8 + dx + Lanes > mask1x1_xsize`):

$$\text{lossc}_c = k_{\text{ChannelMul}}[c] \sum_{(dx, dy) \in T} \Big(\big(M_{1\times 1}(x+dx, y+dy) + \text{masku\_lut}[c]\big)\, \epsilon^{\text{pix}}_c(dx, dy)\Big)^8$$

$$\text{entropy}_c = \text{cost\_delta} \sum_{i=0}^{N-1} \sqrt{|\text{rval}_{c,i}|} + \text{zeros\_mul} \cdot (\text{nbits}_c + \lceil\log_2(\text{nbits}_c + 17)\rceil)$$

where $\text{nbits}_c = \lceil\log_2(\text{nzeros}_c + 1)\rceil + 1$ and $\text{nzeros}_c = \#\{i : \text{rval}_{c,i} \ne 0\}$.

**Multi-block $c=0$ loop-order quirk (`enc_ac_strategy.cc:496-502`)**: at the end of channel $c = 0$ ($X$) when $B \ge 2$, the running accumulators `entropy` (currently containing only $c = 0$) and `loss` (currently $\text{lossc}_0$) are both multiplied by:

$$w_X = 1.0 + \min\!\left(3.0,\; \frac{B}{8.0}\right)$$

before $c = 1, 2$ are added! Combining all channels yields the total block cost $J_{\text{e7}}(T)$:

$$\text{loss\_scalar} = \left(\frac{\text{loss}}{N}\right)^{1/8} \cdot \frac{N}{\text{quant\_norm16}}$$

$$J_{\text{e7}}(T) = \text{entropy\_mul}(T, d) \cdot \text{entropy} + \text{info\_loss\_multiplier} \cdot \text{loss\_scalar}$$

Notice the scaling cancellation in `loss_scalar`: because $\epsilon^{\text{pix}}_c = \text{quant\_norm16} \cdot \delta p_c$ where $|\delta p_c| = O(d)$ in smooth regions, dividing by `quant_norm16` leaves `loss_scalar` proportional to $N \cdot \|\delta p\|_8 = O(N \cdot d)$, which is **1-homogeneous (linear, not quadratic)** in pixel error!

### 2.3 Distance Power Laws & Transform Multipliers

Let $\text{ratio} = \dfrac{d + 0.13731742964354549}{1.13731742964354549}$:

$$\text{info\_loss\_multiplier}(d) = 1.2 \cdot \text{ratio}^{0.33677806662454718}$$

$$\text{zeros\_mul}(d) = 9.3089059022677905 \cdot \text{ratio}^{0.50990926717963703}$$

$$\text{cost\_delta}(d) = 10.833273317067883 \cdot \text{ratio}^{0.36702940662370243}$$

- All $8\times 8$-family transforms normalize by `DCT8`'s `0.8` base multiplier, apply `DCT2X2`/`IDENTITY` high-quality bonus ($-0.4 ((5-d)/5)^2$ for $d < 5$), and multiply their winning score by `mul8x8(d) = 1.0 - 0.4 / (d + 1.4)`.
- Multi-block transform `entropy_mul` biases (applied *only* to the rate term): `DCT16X8`/`8X16` $= 1.21$, `DCT16X16` $= 1.34$, `DCT16X32`/`32X16` $= 1.49$, `DCT32X32` $= 1.48$, `DCT64X32`/`32X64` $= 2.25$, `DCT64X64` $= 2.25$.
- **Greedy selection order**: selects best $8\times 8$ per block, then runs `TryMergeAcs` / `FindBestFirstLevelDivisionForSquare` using a greedy `priority[64]` lock array (`2` for $16\times 8$, `4` for $16\times 32$, `6` for $64\times 32$), plus non-aligned $2\times 2$ and stride-2 (`e8`) / stride-1 (`e9`) $4\times 4$ re-merges.

---

## 3. Chroma-from-Luma (`enc_chroma_from_luma.cc`)

Given candidate transform blocks (first pass with assumed `DCT8` and unit `qq = 1`; second pass with chosen `ac_strategy` and `raw_quant_field` using `q = quantizer.Scale() * 128 * qq`, ZEROED on LLF positions):

For unquantized AC coefficients $y_i = q \cdot \text{inv\_matrix}_{c,i} \cdot C_{1,i}$ and $s_i = q \cdot \text{inv\_matrix}_{c,i} \cdot C_{c,i}$, and base ratio $\beta_c \in \{0.0, \text{kYToBRatio}\}$:

$$r_i(x) = \left(\beta_c + \frac{x}{\text{color\_factor}}\right) y_i - s_i$$

`FindBestMultiplier` (`fast = false` for `e7+`) minimizes the clipped pseudo-Huber + $L^2$ prior via up to 20 damped finite-difference Newton steps (`eps = 100`, stabilizer `0.85`):

$$L_{\text{cfl}}(x) = \frac{1}{3} \sum_{|r_i(x)| < 100} \Big(\big(|r_i(x)| + 1\big)^2 - 1\Big) + 10^{-9}\, N x^2$$

followed by ad-hoc shrinkage toward zero (`towards_zero = 2.6`) and clamping to $[-128, 127]$:

$$m_{\text{out}} = \text{Clamp}\!\Big(\text{round}\big(\text{sign}(x^*)\max(|x^*| - 2.6,\, 0)\big),\; -128,\; 127\Big)$$

---

## 4. Per-Block AQ Adjustments & AC Deadzones (`enc_adaptive_quantization.cc`, `enc_group.cc`)

1. **`AdjustQuantField`**: replaces every multi-block varblock's `initial_quant_field` with `max` ($B = 2$ or $d \le 1.54138$), or a linear blend of `max` and `mean` ($B \ge 4$: $\alpha = \text{clamp}(1 - (d - 1.54138)\cdot 0.56391, 0, 1)$).
2. **`AdjustQuantBlockAC` (`enc_group.cc:104-378`)**: runs six independent heuristic rules sequentially on un-CfL-subtracted $X$, $Y$, $B$ (and takes `max` over channels!):
   - Large-block deadzone reduction: `thres -= clamp(0.003 * B, 0, 0.08)` floored at `0.54`.
   - Flat-Y HF error clamp: if `sum(|v_Y|) * 8 < B` and any HF quadrant has `hfNonZeros == 0 && hfMaxError > 0.46`, `quant += 1` and lower quadrant threshold to `0.9999 * hfMaxError`.
   - Corner HF boost (`mul = {70, 30, 60}`): increases `quant` if highest-frequency row/column dominates.
   - Flat `DCT8` boost: if `sum_hf_nonzeros < 11`, `quant += 1`.
   - Large-block (`>= 16x16`) rounding-error step boost (`kMul1[4][3]`, `kMul2[4][3]`, `kQuantNormalizer = 2.2942708`).
   - Active-area reduction: subtracts `activity = round(min_hf_nonzeros / B)` (capped at `15`, floored at `max(4, quant / 2)`) and adds `0.01 * activity` to HF quadrant deadzones of $Y$.
3. **`QuantizeBlockAC`**:
   - Base quadrant deadzones: `thres_y = {0.58, 0.64, 0.64, 0.64}`, `thres_xb = {0.58, 0.62, 0.62, 0.62}` (minus `0.00744 * B` floored at `0.5` when $B \ge 4$ on $X, B$).
   - **Zero-vs-1 decision** is hard threshold `|val| >= thres[quad] ? round(val) : 0` (independent of neighborhood nonzeros `nzeros`, scan order, or ANS symbol bits); **levels $\ge 1$** simply round without RD bias, though **inverse quantization** uses `kDefaultQuantBias = {1 - 0.05465007, 1 - 0.07005450, 1 - 0.04993510, 0.145}`.

---

## 5. Adaptive Ringing / EPF Control Field (`ComputeARHeuristics`, `enc_heuristics.cc`)

When `epf_iters > 0` and $d \ge \text{kMinButteraugliForDynamicAR}$:
- Evaluates decoded pixel MSE weighted by pre-Gaborish $M_{1\times 1}^2$ (`ComputeBlockL2Distance`, with channel weights `kW = {12.339445295782363, 1.0, 0.2}`) over candidate `epf_sharpness` values $\{0, 4\}$ ($d > 4.5$) or $\{0, 2, 7\}$ ($d \le 4.5$) using full-frame `ReconstructImage`.
- Pass 1 greedily compares `0.99 * error(0)` and other values against causal `top_val`/`left_val`, building histogram counts `histo[top_idx * 3 + left_idx][val]`.
- Pass 2 minimizes **multiplicatively modulated** block error ($\text{error}(\text{val}) \cdot \text{mul}$), where:

$$\text{mul} = \frac{\mathbb{I}[\text{val}=0]\, c_3 + \mathbb{I}[\text{val}\ne 0]}{1 + \dfrac{0.1087690359555803}{\min(d, 5)} \ln\!\left(1 + \frac{\text{histo}[\text{ctx}][\text{val}]}{\text{totals}[\text{ctx}]}\right)}, \quad c_3 = \max\!\big(0.859703389,\; 0.980171988^{\min(d,5)}\big)$$

*(Note: in `enc_heuristics.cc:991`, `histo[ctx][val] / totals[ctx]` is integer division of `size_t` operands, evaluating to $0$ unless all counts fall in one bucket!)*

---

## 6. Cross-Stage Mismatches Addressed by Unified RDO / RDOQ

1. **Decoupled quantization vs. selection**: `EstimateEntropy` uses `Round(val)` (deadzone `0.5`) and `quant_norm16` without `AdjustQuantBias`, `x/b_qm_multiplier`, `raw_quant_field` clamping, `AdjustQuantBlockAC`, or actual `{0.58..0.64}` deadzones.
2. **Greedy lock-in in ACS**: `priority[64]` locks sub-optimal splits in fixed transform-size order rather than optimizing the global tile partition.
3. **Unquantized vs. quantized $Y$ in CfL & pre-CfL `AdjustQuantBlockAC`**: CfL fit unquantized $Y$ (`kStrangeMultiplier = 128`, `towards_zero = 2.6`) while decoder uses quantized $\hat{Y}$; `AdjustQuantBlockAC` ran before CfL subtraction on $X$ and $B$.
4. **Heuristic rate proxies & ignoring zero-density context**: `Sqrt(|q|)` ignores `ZeroDensityContext(nzeros, k, B, prev)` EOB tail savings, custom `coeff_orders`, empirical ANS distributions, and `ACMetadata` signaling costs (`raw_quant_field` transitions).
5. **AQ field oscillation vs. deadzone widening**: varying `raw_quant_field` block-by-block pays two-sided `ACMetadata` residual bits even where holding `qf = prev_qf` with RDOQ deadzone widening produces equivalent rate-distortion cost.
6. **Standalone `e8`/`e9` Butteraugli loop**: `FindBestQuantization` adjusted only the scalar `quant_field` via full-frame Butteraugli roundtrips rather than joint `(ACS, CfL, qf, AC, EPF)` Lagrangian RDO with multi-pass empirical entropy estimation.
