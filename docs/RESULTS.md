# Results

All numbers come from the per-query files in `$SPARSEUP_WORK_DIR/results/` and from the job logs in `logs/`; every
command that produced them is in [`GUIDE.md`](GUIDE.md). Percentages are relative to **SparseUp** in the same column
(`value (% of SparseUp)`). Test sets: MS MARCO dev (6,980 queries), TREC DL19 (43) and DL20 (54), BEIR (NFCorpus 323,
SciFact 300, FiQA 648 queries). Differences below about 0.02 on DL19 / DL20 are noise (few queries). Search is exact
(no ANN). Hyper-parameters, k, λ and the checkpoints were chosen on **validation** (1,000 held-out training queries),
never on dev or BEIR; see section 7 for the choices that were *not* made on validation.

**Summary.**

1. A 256–8192-dimensional concept space learned on top of SparseUp keeps most of its quality: a plain dense `g(d)` with
   k = 8192 reaches 99.0% of SparseUp on dev (frozen) and 99.1% (e2e), k = 2048 about 96% (frozen) / 94% (e2e).
2. Training W matters only for small k (+0.035 MRR@10 at k = 256). From k = 1024 on a random W is as good.
3. Mixed with SparseUp, the fine-tuned model **beats SparseUp in domain**: `hybnoW` k = 1024 reaches dev MRR@10 0.3825
   (**104.1%**, p ≤ 0.0001) and `hyb` k = 2048 0.3788 (103.1%, p ≤ 0.0001), at the same BEIR quality (100.0%).
4. At search time the fine-tuned encoder alone is as good as SparseUp (`ftsparseup`, k = 8192: 100.6% dev, 99.7% BEIR)
   with **0.95×** the space. W is needed only for training.

## 0. Data and baseline

| Check | Expected | Obtained |
|---|---|---|
| `02b`: non-zero entries per passage (total over 8.84M passages) | ~190 (SparseUp blog) | 1,675,900,653 total = **189.5** per passage |
| `02b`: mean non-zero entries per dev query | ~47 (SparseUp blog) | see `logs/enc_msmarco_*.out` |
| `01`: folded vocabulary size C | ~34k | **34,023** |
| `03`: our encoder vs official, mean cosine (fp32 / amp) | 1.0000 / ≈ 0.99+ | see `encoder_check.json` |
| `04`: overlap with the groundtruth of `02b` | 1.0000 | 1.0000 |
| Validation (1,000 held-out queries): SparseUp MRR@10 on its top-100 | – | **0.3487** |

| System | dev MRR@10 | dev nDCG@10 | dev R@10 | dev R@100 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|---|
| SPLADE ensembledistil (previous project) | 0.3828 | 0.4489 | 0.6697 | 0.9093 | 0.9831 | 0.7308 | 0.8724 | 0.7203 | 0.8989 |
| **SparseUp** | 0.3674 | 0.4319 | 0.6492 | 0.9021 | 0.9805 | 0.6940 | 0.7954 | 0.6809 | 0.8486 |

SparseUp is not distilled on MS MARCO, so it is below SPLADE ensembledistil in domain (96% on dev MRR@10); this is the
reference of every percentage below.

BEIR (zero-shot, test split):

| System | NFCorpus nDCG@10 | SciFact nDCG@10 | FiQA nDCG@10 | Mean nDCG@10 | NFCorpus R@100 | SciFact R@100 | FiQA R@100 | Mean R@100 |
|---|---|---|---|---|---|---|---|---|
| **SparseUp** | 0.3510 | 0.7418 | 0.4686 | 0.5205 | 0.3092 | 0.9543 | 0.7651 | 0.6762 |

## 1. Final checkpoints

Chosen on validation. All use `--negatives sparseup --n-hard 7 --neg-rank-min 3`, temperature 1, seed 42.

### 1.1 Frozen (only W trained)

Progression of the validation MRR@10 (SparseUp = 0.3487):

| k | Random W | Round 1 (in-batch only) | Round 2 (+ 7 hard negatives, default hyper-parameters) | Final (grid) | % SparseUp |
|---|---|---|---|---|---|
| 256 | 0.2432 | 0.2467 | 0.2576 | 0.2770 | 79.4% |
| 512 | 0.3026 | 0.2616 | 0.2830 | 0.3095 | 88.8% |
| 1024 | 0.3146 | 0.2723 | 0.2854 | 0.3200 | 91.8% |
| 2048 | 0.3396 | 0.2841 | 0.2994 | 0.3402 | 97.6% |
| 4096 | 0.3380 | 0.2882 | 0.3013 | **0.3534** | **101.3%** |
| 8192 | 0.3447 | 0.2910 | 0.3051 | 0.3492 | 100.1% |

With the default hyper-parameters (lr 1e-3, 10 epochs) training W is worse than leaving it random; the grid on learning rate, epochs and init-std fixes this. Grid: lr ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} ×
epochs ∈ {1, 2, 5} × init-std ∈ {0.02, 1/√k}. Winners:

| k | run | init-std | lr | epochs | best val |
|---|---|---|---|---|---|
| 256 | `frozen_k256_r2_lr1e4_ep2` | 0.02 | 1e-4 | 2 | 0.2770 |
| 512 | `frozen_k512_r2_lr3e4_ep2` | 0.02 | 3e-4 | 2 | 0.3095 |
| 1024 | `frozen_k1024_r2_lr3e5_ep5` | 0.02 | 3e-5 | 5 | 0.3200 |
| 2048 | `frozen_k2048_r2_lr1e5_ep1` | 0.02 | 1e-5 | 1 | 0.3402 |
| 4096 | `frozen_k4096_r2_lr3e4_ep2` | 0.02 | 3e-4 | 2 | 0.3534 |
| 8192 | `frozen_k8192_r2_is0110_lr1e4_ep1` | 0.0110 | 1e-4 | 1 | 0.3492 |

For k = 2048 the winner is at the edge of the grid (lr 1e-5); lr 1e-6 gave 0.3401 < 0.3402, so it is kept.

Second phase, one change at a time on each winner (validation MRR@10): `--neg-rank-min 20` and `--temperature 0.1` / `0.05`
**never help** (e.g. k = 1024: 0.3200 → 0.3188 / 0.3184 / 0.3188; k = 4096: 0.3534 → 0.3416 / 0.3530 / 0.3524). The
starting values (`neg-rank-min 3`, temperature 1) stay.

**Random W** (not trained, init-std = 1/√k, `lr 1e-9 --max-steps 1`): `frozen_k{256,…,8192}_rand`, validation
0.2432, 0.3026, 0.3146, 0.3396, 0.3380, 0.3447.

### 1.2 e2e (W and the SparseUp encoder trained, batch size 32, 1 epoch)

Protocol: lr of W fixed to the best SPLADE value (1e-4 for k = 256, 512; 3e-5 for k ≥ 1024), grid on the encoder
learning rate (`--lr-backbone`) and on init-std ∈ {0.02, 1/√k}; ties (< 0.001) go to the lower encoder lr; the grid is
extended when the winner is on its edge (k = 4096 and 8192 needed learning rates down to 1e-7 / 5e-8).

| k | run | init-std | lr W | lr encoder | best val | frozen val |
|---|---|---|---|---|---|---|
| 256 | `e2e_k256_r2_lr1e4_bb2e5_b32_ep1` | 0.02 | 1e-4 | 2e-5 | 0.3171 | 0.2770 |
| 512 | `e2e_k512_r2_lr1e4_bb5e6_b32_ep1` | 0.02 | 1e-4 | 5e-6 | 0.3349 | 0.3095 |
| 1024 | `e2e_k1024_r2_lr3e5_bb2e5_b32_ep1` | 0.02 | 3e-5 | 2e-5 | 0.3435 | 0.3200 |
| 1024 (W fixed) | `e2e_k1024_r2_Wfix_bb2e5_b32_ep1` | 0.02 | 1e-9 | 2e-5 | 0.3473 | – |
| 2048 | `e2e_k2048_r2_lr3e5_bb5e6_b32_ep1` | 0.02 | 3e-5 | 5e-6 | 0.3442 | 0.3402 |
| 4096 | `e2e_k4096_r2_is0156_lr3e5_bb2e7_b32_ep1` | 0.0156 | 3e-5 | 2e-7 | 0.3465 | 0.3534 |
| 8192 | `e2e_k8192_r2_lr3e5_bb1e7_b32_ep1` | 0.02 | 3e-5 | 1e-7 | 0.3472 | 0.3492 |

Fine-tuning the encoder helps a lot for k ≤ 1024 (+0.040, +0.025, +0.024 over frozen) and not at all for k ≥ 4096,
where the frozen model is already at SparseUp's level. **1 vs 2 epochs** (k ≤ 1024) and the other checks of the
protocol are still to be run: for the time being every e2e uses 1 epoch, as in the SPLADE project.

The "W fixed" control (W left random, only the encoder trained) is as good as, or better than, the full e2e: the
quality of the e2e comes from the encoder, not from W.

## 2. In domain: MS MARCO

Concept Model alone, score `g(q)·g(d)`; B/doc = bytes of the dense `g(d)` in fp16.

| System | k | B/doc | dev MRR@10 | dev nDCG@10 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|---|
| **SparseUp** | – | ~1500 | **0.3674** | **0.4319** | **0.9805** | 0.6940 | 0.7954 | 0.6809 | 0.8486 |
| Random W | 256 | 512 | 0.2394 (65.2%) | 0.2847 (65.9%) | 0.8869 (90.5%) | 0.4723 (68.1%) | 0.6087 (76.5%) | 0.5011 (73.6%) | 0.6653 (78.4%) |
| Frozen | 256 | 512 | 0.2747 (74.8%) | 0.3292 (76.2%) | 0.9363 (95.5%) | 0.5510 (79.4%) | 0.7078 (89.0%) | 0.5660 (83.1%) | 0.7434 (87.6%) |
| e2e | 256 | 512 | 0.3138 (85.4%) | 0.3713 (86.0%) | 0.9562 (97.5%) | 0.6009 (86.6%) | 0.7580 (95.3%) | 0.5893 (86.5%) | 0.7786 (91.8%) |
| Random W | 512 | 1024 | 0.3034 (82.6%) | 0.3614 (83.7%) | 0.9510 (97.0%) | 0.6038 (87.0%) | 0.7199 (90.5%) | 0.6194 (91.0%) | 0.7769 (91.6%) |
| Frozen | 512 | 1024 | 0.3070 (83.6%) | 0.3648 (84.5%) | 0.9586 (97.8%) | 0.6186 (89.1%) | 0.7680 (96.6%) | 0.6121 (89.9%) | 0.7782 (91.7%) |
| e2e | 512 | 1024 | 0.3315 (90.2%) | 0.3938 (91.2%) | 0.9687 (98.8%) | 0.6338 (91.3%) | 0.7870 (98.9%) | 0.6191 (90.9%) | 0.8074 (95.1%) |
| Random W | 1024 | 2048 | 0.3341 (90.9%) | 0.3963 (91.8%) | 0.9689 (98.8%) | 0.6500 (93.7%) | 0.7686 (96.6%) | 0.6385 (93.8%) | 0.7957 (93.8%) |
| Frozen | 1024 | 2048 | 0.3330 (90.6%) | 0.3959 (91.7%) | 0.9720 (99.1%) | 0.6617 (95.3%) | 0.8026 (100.9%) | 0.6502 (95.5%) | 0.8147 (96.0%) |
| e2e | 1024 | 2048 | 0.3490 (95.0%) | 0.4106 (95.1%) | 0.9715 (99.1%) | 0.6535 (94.2%) | 0.8069 (101.4%) | 0.6581 (96.7%) | 0.8204 (96.7%) |
| e2e, W fixed | 1024 | 2048 | 0.3480 (94.7%) | 0.4103 (95.0%) | 0.9707 (99.0%) | 0.6705 (96.6%) | 0.8135 (102.3%) | 0.6633 (97.4%) | 0.8309 (97.9%) |
| Random W | 2048 | 4096 | 0.3523 (95.9%) | 0.4157 (96.2%) | 0.9767 (99.6%) | 0.6742 (97.1%) | 0.7893 (99.2%) | 0.6561 (96.4%) | 0.8374 (98.7%) |
| Frozen | 2048 | 4096 | 0.3528 (96.0%) | 0.4162 (96.4%) | 0.9773 (99.7%) | 0.6757 (97.4%) | 0.7930 (99.7%) | 0.6565 (96.4%) | 0.8408 (99.1%) |
| e2e | 2048 | 4096 | 0.3441 (93.7%) | 0.4064 (94.1%) | 0.9685 (98.8%) | 0.6341 (91.4%) | 0.7971 (100.2%) | 0.6762 (99.3%) | 0.8054 (94.9%) |
| Random W | 4096 | 8192 | 0.3597 (97.9%) | 0.4234 (98.0%) | 0.9783 (99.8%) | 0.6732 (97.0%) | 0.7854 (98.7%) | 0.6761 (99.3%) | 0.8353 (98.4%) |
| Frozen | 4096 | 8192 | 0.3579 (97.4%) | 0.4212 (97.5%) | 0.9788 (99.8%) | 0.6758 (97.4%) | 0.8096 (101.8%) | 0.6802 (99.9%) | 0.8383 (98.8%) |
| e2e | 4096 | 8192 | 0.3586 (97.6%) | 0.4226 (97.8%) | 0.9772 (99.7%) | 0.6794 (97.9%) | 0.7966 (100.2%) | 0.6806 (100.0%) | 0.8485 (100.0%) |
| Random W | 8192 | 16384 | 0.3644 (99.2%) | 0.4281 (99.1%) | 0.9796 (99.9%) | 0.6877 (99.1%) | 0.7880 (99.1%) | 0.6673 (98.0%) | 0.8301 (97.8%) |
| Frozen | 8192 | 16384 | 0.3639 (99.0%) | 0.4267 (98.8%) | 0.9801 (100.0%) | 0.6910 (99.6%) | 0.8083 (101.6%) | 0.6796 (99.8%) | 0.8510 (100.3%) |
| e2e | 8192 | 16384 | 0.3641 (99.1%) | 0.4280 (99.1%) | 0.9792 (99.9%) | 0.6961 (100.3%) | 0.8148 (102.4%) | 0.6775 (99.5%) | 0.8471 (99.8%) |

- On dev, training W matters only for small k: +0.035 at k = 256, +0.004 at k = 512; from k = 1024 on random and trained
  W agree within ±0.002.
- e2e improves on frozen up to k = 1024 and is equal for larger k. The dip of e2e at k = 2048 (0.3441 < 0.3490 of
  k = 1024) is a single run with one seed and is not explained; DL19/DL20 differences of ±0.02 are noise.

## 3. Out of domain: BEIR (zero-shot)

Mean = mean of nDCG@10 (or R@100) over the three datasets.

| Method | k | NFCorpus nDCG@10 | SciFact nDCG@10 | FiQA nDCG@10 | Mean nDCG@10 | Mean R@100 |
|---|---|---|---|---|---|---|
| **SparseUp** | – | **0.3510** | 0.7418 | **0.4686** | **0.5205** (100.0%) | **0.6762** |
| Random W | 256 | 0.1907 (54.3%) | 0.4795 (64.6%) | 0.3032 (64.7%) | 0.3245 (62.3%) | 0.5379 (79.5%) |
| Frozen | 256 | 0.2326 (66.3%) | 0.5314 (71.6%) | 0.3148 (67.2%) | 0.3596 (69.1%) | 0.5644 (83.5%) |
| e2e | 256 | 0.2808 (80.0%) | 0.5582 (75.2%) | 0.3274 (69.9%) | 0.3888 (74.7%) | 0.5922 (87.6%) |
| Random W | 512 | 0.2697 (76.8%) | 0.6138 (82.7%) | 0.3802 (81.1%) | 0.4212 (80.9%) | 0.6076 (89.9%) |
| Frozen | 512 | 0.2785 (79.3%) | 0.6083 (82.0%) | 0.3821 (81.5%) | 0.4230 (81.3%) | 0.5952 (88.0%) |
| e2e | 512 | 0.3106 (88.5%) | 0.6391 (86.2%) | 0.3686 (78.7%) | 0.4394 (84.4%) | 0.6272 (92.7%) |
| Random W | 1024 | 0.3103 (88.4%) | 0.6635 (89.4%) | 0.4225 (90.2%) | 0.4654 (89.4%) | 0.6397 (94.6%) |
| Frozen | 1024 | 0.3116 (88.8%) | 0.6595 (88.9%) | 0.4188 (89.4%) | 0.4633 (89.0%) | 0.6403 (94.7%) |
| e2e | 1024 | 0.3249 (92.6%) | 0.6822 (92.0%) | 0.3888 (83.0%) | 0.4653 (89.4%) | 0.6261 (92.6%) |
| e2e, W fixed | 1024 | 0.3233 (92.1%) | 0.6746 (90.9%) | 0.3824 (81.6%) | 0.4601 (88.4%) | 0.6286 (93.0%) |
| Random W | 2048 | 0.3206 (91.3%) | 0.7000 (94.4%) | 0.4481 (95.6%) | 0.4896 (94.1%) | 0.6611 (97.8%) |
| Frozen | 2048 | 0.3208 (91.4%) | 0.6998 (94.3%) | 0.4490 (95.8%) | 0.4899 (94.1%) | 0.6614 (97.8%) |
| e2e | 2048 | 0.3386 (96.5%) | 0.6798 (91.6%) | 0.3964 (84.6%) | 0.4716 (90.6%) | 0.6268 (92.7%) |
| Random W | 4096 | 0.3438 (97.9%) | 0.7190 (96.9%) | 0.4575 (97.6%) | 0.5068 (97.4%) | 0.6735 (99.6%) |
| Frozen | 4096 | 0.3485 (99.3%) | 0.7233 (97.5%) | 0.4523 (96.5%) | 0.5080 (97.6%) | 0.6719 (99.4%) |
| e2e | 4096 | 0.3440 (98.0%) | 0.7203 (97.1%) | 0.4475 (95.5%) | 0.5039 (96.8%) | 0.6644 (98.3%) |
| Random W | 8192 | 0.3446 (98.2%) | 0.7375 (99.4%) | 0.4611 (98.4%) | 0.5144 (98.8%) | 0.6719 (99.4%) |
| Frozen | 8192 | 0.3459 (98.5%) | **0.7437 (100.3%)** | 0.4618 (98.5%) | 0.5171 (99.4%) | 0.6696 (99.0%) |
| e2e | 8192 | 0.3460 (98.6%) | 0.7428 (100.1%) | 0.4593 (98.0%) | 0.5160 (99.1%) | 0.6718 (99.3%) |

- Out of domain the loss is the same as in domain, or slightly larger; training W is useful only for small k.
- Fine-tuning the encoder (e2e) costs generalisation on FiQA for k ≤ 2048 (e.g. k = 1024: 83.0% vs 89.4% frozen), while
  it helps NFCorpus. For k ≥ 4096 e2e and frozen are equivalent.

## 4. Hybrids with SparseUp

Score `(1 − λ)·SparseUp(q)·SparseUp(d) + λ·g(q)·g(d)`; λ\* chosen on validation among 21 values (0, 0.05, …, 1), then
used unchanged on dev, DL19, DL20 and BEIR. λ = 0 reproduces `04`, λ = 1 the Concept Model alone (both checked).

### 4.1 Frozen and e2e hybrids (`hyb`)

Validation MRR@10 (SparseUp alone = 0.3487):

| Model | k | λ\* | Concept Model alone (λ = 1) | Hybrid (λ\*) | Hybrid − SparseUp |
|---|---|---|---|---|---|
| Frozen | 256 | 0.60 | 0.2646 (75.9%) | 0.3493 (100.2%) | +0.0006 |
| Frozen | 512 | 0.65 | 0.3042 (87.2%) | 0.3551 (101.8%) | +0.0064 |
| Frozen | 1024 | 0.15 | 0.3174 (91.0%) | 0.3502 (100.4%) | +0.0015 |
| Frozen | 2048 | 0.15 | 0.3400 (97.5%) | 0.3497 (100.3%) | +0.0010 |
| Frozen | 4096 | 0.75 | 0.3534 (101.3%) | 0.3564 (102.2%) | +0.0077 |
| Frozen | 8192 | 0.70 | 0.3492 (100.1%) | 0.3512 (100.7%) | +0.0025 |
| e2e | 256 | 0.80 | 0.3085 (88.5%) | 0.3649 (104.6%) | +0.0162 |
| e2e | 512 | 0.85 | 0.3333 (95.6%) | 0.3684 (105.6%) | +0.0197 |
| e2e | 1024 | 0.90 | 0.3412 (97.8%) | 0.3697 (106.0%) | +0.0210 |
| e2e | 2048 | 0.80 | 0.3403 (97.6%) | 0.3715 (106.5%) | +0.0228 |
| e2e | 4096 | 0.50 | 0.3464 (99.3%) | 0.3557 (102.0%) | +0.0070 |
| e2e | 8192 | 0.20 | 0.3472 (99.6%) | 0.3515 (100.8%) | +0.0028 |

(The λ = 1 values differ slightly from the training-time validation of section 1 because `11` re-encodes the validation
queries with the search lengths 128 / 512, while training validates with 64 / 256.)

Official evaluation, in domain:

| System | k | λ\* | dev MRR@10 | dev nDCG@10 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|---|
| **SparseUp** | – | 0 | 0.3674 | 0.4319 | 0.9805 | 0.6940 | 0.7954 | 0.6809 | 0.8486 |
| Hybrid frozen | 256 | 0.60 | 0.3664 (99.7%) | 0.4308 (99.7%) | 0.9810 (100.1%) | 0.6967 (100.4%) | 0.7991 (100.5%) | 0.6757 (99.2%) | 0.8472 (99.8%) |
| Hybrid frozen | 512 | 0.65 | 0.3676 (100.1%) | 0.4315 (99.9%) | 0.9810 (100.1%) | 0.6948 (100.1%) | 0.8086 (101.7%) | 0.6823 (100.2%) | 0.8472 (99.8%) |
| Hybrid frozen | 1024 | 0.15 | 0.3678 (100.1%) | 0.4323 (100.1%) | 0.9803 (100.0%) | 0.6950 (100.1%) | 0.7993 (100.5%) | 0.6851 (100.6%) | 0.8494 (100.1%) |
| Hybrid frozen | 2048 | 0.15 | 0.3680 (100.2%) | 0.4324 (100.1%) | 0.9810 (100.1%) | 0.6899 (99.4%) | 0.7973 (100.2%) | 0.6778 (99.5%) | 0.8499 (100.2%) |
| Hybrid frozen | 4096 | 0.75 | 0.3620 (98.5%) | 0.4257 (98.6%) | 0.9798 (99.9%) | 0.6818 (98.2%) | 0.8070 (101.5%) | 0.6846 (100.5%) | 0.8437 (99.4%) |
| Hybrid frozen | 8192 | 0.70 | 0.3671 (99.9%) | 0.4305 (99.7%) | 0.9802 (100.0%) | 0.6954 (100.2%) | 0.8102 (101.9%) | 0.6799 (99.9%) | 0.8511 (100.3%) |
| Hybrid e2e | 256 | 0.80 | 0.3746 (102.0%) | 0.4395 (101.8%) | 0.9822 (100.2%) | 0.7057 (101.7%) | 0.8064 (101.4%) | 0.7013 (103.0%) | 0.8420 (99.2%) |
| Hybrid e2e | 512 | 0.85 | 0.3718 (101.2%) | 0.4365 (101.1%) | 0.9822 (100.2%) | 0.6984 (100.6%) | 0.8134 (102.3%) | 0.6955 (102.1%) | 0.8472 (99.8%) |
| Hybrid e2e | 1024 | 0.90 | 0.3787 (103.1%) | 0.4444 (102.9%) | 0.9813 (100.1%) | 0.7052 (101.6%) | 0.8231 (103.5%) | 0.7083 (104.0%) | 0.8497 (100.1%) |
| Hybrid e2e | 2048 | 0.80 | 0.3788 (103.1%) | 0.4437 (102.7%) | 0.9834 (100.3%) | 0.7076 (102.0%) | 0.8240 (103.6%) | 0.7147 (105.0%) | 0.8444 (99.5%) |
| Hybrid e2e | 4096 | 0.50 | 0.3688 (100.4%) | 0.4335 (100.4%) | 0.9799 (99.9%) | 0.6854 (98.8%) | 0.8008 (100.7%) | 0.6841 (100.5%) | 0.8545 (100.7%) |
| Hybrid e2e | 8192 | 0.20 | 0.3687 (100.4%) | 0.4333 (100.3%) | 0.9805 (100.0%) | 0.6985 (100.6%) | 0.8079 (101.6%) | 0.6785 (99.6%) | 0.8558 (100.8%) |

Out of domain (λ\* of MS MARCO, not re-tuned):

| System | k | λ\* | NFCorpus nDCG@10 | SciFact nDCG@10 | FiQA nDCG@10 | Mean nDCG@10 | Mean R@100 |
|---|---|---|---|---|---|---|---|
| **SparseUp** | – | 0 | 0.3510 | 0.7418 | 0.4686 | 0.5205 | 0.6762 |
| Hybrid frozen | 256 | 0.60 | 0.3475 (99.0%) | 0.7376 (99.4%) | 0.4640 (99.0%) | 0.5164 (99.2%) | 0.6812 (100.7%) |
| Hybrid frozen | 512 | 0.65 | 0.3509 (100.0%) | 0.7258 (97.8%) | 0.4709 (100.5%) | 0.5159 (99.1%) | 0.6751 (99.8%) |
| Hybrid frozen | 1024 | 0.15 | 0.3498 (99.7%) | 0.7397 (99.7%) | 0.4690 (100.1%) | 0.5195 (99.8%) | 0.6790 (100.4%) |
| Hybrid frozen | 2048 | 0.15 | 0.3498 (99.7%) | 0.7370 (99.4%) | 0.4688 (100.0%) | 0.5185 (99.6%) | 0.6792 (100.4%) |
| Hybrid frozen | 4096 | 0.75 | 0.3515 (100.1%) | 0.7311 (98.6%) | 0.4629 (98.8%) | 0.5152 (99.0%) | 0.6736 (99.6%) |
| Hybrid frozen | 8192 | 0.70 | 0.3482 (99.2%) | 0.7432 (100.2%) | 0.4634 (98.9%) | 0.5183 (99.6%) | 0.6732 (99.6%) |
| Hybrid e2e | 256 | 0.80 | 0.3489 (99.4%) | 0.7386 (99.6%) | 0.4634 (98.9%) | 0.5170 (99.3%) | 0.6775 (100.2%) |
| Hybrid e2e | 512 | 0.85 | 0.3508 (99.9%) | 0.7282 (98.2%) | 0.4602 (98.2%) | 0.5131 (98.6%) | 0.6771 (100.1%) |
| Hybrid e2e | 1024 | 0.90 | 0.3474 (99.0%) | 0.7255 (97.8%) | 0.4573 (97.6%) | 0.5101 (98.0%) | 0.6748 (99.8%) |
| Hybrid e2e | 2048 | 0.80 | 0.3539 (100.8%) | 0.7374 (99.4%) | 0.4696 (100.2%) | 0.5203 (100.0%) | 0.6762 (100.0%) |
| Hybrid e2e | 4096 | 0.50 | 0.3500 (99.7%) | 0.7368 (99.3%) | 0.4641 (99.0%) | 0.5170 (99.3%) | 0.6753 (99.9%) |
| Hybrid e2e | 8192 | 0.20 | 0.3490 (99.4%) | 0.7431 (100.2%) | 0.4685 (100.0%) | 0.5202 (99.9%) | 0.6772 (100.1%) |

- A frozen hybrid is equal to SparseUp (99–100%) in and out of domain: with W trained on top of a frozen encoder there is
  nothing to add.
- The e2e hybrids improve in domain (up to 103.1%, k = 1024 / 2048) and keep BEIR at 98–100%.

### 4.2 Variants of the e2e hybrid

Same e2e checkpoints; SU = original SparseUp, SU_θ = encoder re-trained through the Concept Model.

| Name | Score | Indexes |
|---|---|---|
| `hyb` | `(1−λ)·SU·SU + λ·g·g` | 2 (SU + SU_θ, needed to compute g) |
| `hyb1` | `(1−λ)·SU_θ·SU_θ + λ·g·g` | 1 (SU_θ) |
| `hybnoW` | `(1−λ)·SU·SU + λ·SU_θ·SU_θ` (no W) | 2 |
| `ftsparseup` | `SU_θ·SU_θ` (= `hyb1` at λ = 0 = `hybnoW` at λ = 1) | 1 |

For k = 8192 the λ\* of `hyb1` is 0, so `hyb1` and `ftsparseup` coincide.

Validation MRR@10:

| Variant | k | λ\* | λ = 0 | λ = 1 | Hybrid (λ\*) | Hybrid − SparseUp |
|---|---|---|---|---|---|---|
| hyb1 | 256 | 0.65 | 0.3543 (101.6%) | 0.3085 (88.5%) | 0.3565 (102.2%) | +0.0078 |
| hyb1 | 512 | 0.40 | 0.3564 (102.2%) | 0.3333 (95.6%) | 0.3620 (103.8%) | +0.0133 |
| hyb1 | 1024 | 0.30 | 0.3606 (103.4%) | 0.3412 (97.8%) | 0.3626 (104.0%) | +0.0139 |
| hyb1 | 2048 | 0.50 | 0.3463 (99.3%) | 0.3403 (97.6%) | 0.3509 (100.6%) | +0.0022 |
| hyb1 | 4096 | 0.20 | 0.3530 (101.2%) | 0.3464 (99.3%) | 0.3534 (101.3%) | +0.0047 |
| hyb1 | 8192 | 0.00 | 0.3505 (100.5%) | 0.3472 (99.6%) | 0.3505 (100.5%) | +0.0018 |
| hybnoW | 256 | 0.75 | 0.3487 (100.0%) | 0.3543 (101.6%) | 0.3662 (105.0%) | +0.0175 |
| hybnoW | 512 | 0.70 | 0.3487 (100.0%) | 0.3564 (102.2%) | 0.3637 (104.3%) | +0.0150 |
| hybnoW | 1024 | 0.75 | 0.3487 (100.0%) | 0.3606 (103.4%) | 0.3767 (108.0%) | +0.0280 |
| hybnoW | 2048 | 0.90 | 0.3487 (100.0%) | 0.3463 (99.3%) | 0.3687 (105.7%) | +0.0200 |
| hybnoW | 4096 | 0.90 | 0.3487 (100.0%) | 0.3530 (101.2%) | 0.3548 (101.7%) | +0.0061 |
| hybnoW | 8192 | 0.70 | 0.3487 (100.0%) | 0.3505 (100.5%) | 0.3532 (101.3%) | +0.0045 |

Official evaluation, in domain:

| System | k | λ\* | dev MRR@10 | dev nDCG@10 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|---|
| **SparseUp** | – | 0 | 0.3674 | 0.4319 | 0.9805 | 0.6940 | 0.7954 | 0.6809 | 0.8486 |
| ftsparseup | 256 | 0 | 0.3621 (98.6%) | 0.4266 (98.8%) | 0.9756 (99.5%) | 0.6715 (96.8%) | 0.8073 (101.5%) | 0.6787 (99.7%) | 0.8190 (96.5%) |
| ftsparseup | 512 | 0 | 0.3634 (98.9%) | 0.4279 (99.1%) | 0.9766 (99.6%) | 0.6786 (97.8%) | 0.8143 (102.4%) | 0.6834 (100.4%) | 0.8323 (98.1%) |
| ftsparseup | 1024 | 0 | 0.3691 (100.5%) | 0.4334 (100.3%) | 0.9772 (99.7%) | 0.6645 (95.7%) | 0.8135 (102.3%) | 0.6921 (101.6%) | 0.8276 (97.5%) |
| ftsparseup | 2048 | 0 | 0.3578 (97.4%) | 0.4212 (97.5%) | 0.9714 (99.1%) | 0.6702 (96.6%) | 0.7975 (100.3%) | 0.6835 (100.4%) | 0.8223 (96.9%) |
| ftsparseup | 4096 | 0 | 0.3669 (99.9%) | 0.4315 (99.9%) | 0.9799 (99.9%) | 0.6897 (99.4%) | 0.8038 (101.1%) | 0.6897 (101.3%) | 0.8496 (100.1%) |
| ftsparseup | 8192 | 0 | 0.3695 (100.6%) | 0.4339 (100.5%) | 0.9806 (100.0%) | 0.7009 (101.0%) | 0.7957 (100.0%) | 0.6838 (100.4%) | 0.8486 (100.0%) |
| hyb1 | 256 | 0.65 | 0.3618 (98.5%) | 0.4258 (98.6%) | 0.9760 (99.5%) | 0.6722 (96.9%) | 0.8050 (101.2%) | 0.6784 (99.6%) | 0.8244 (97.1%) |
| hyb1 | 512 | 0.40 | 0.3634 (98.9%) | 0.4278 (99.1%) | 0.9764 (99.6%) | 0.6795 (97.9%) | 0.8154 (102.5%) | 0.6830 (100.3%) | 0.8327 (98.1%) |
| hyb1 | 1024 | 0.30 | 0.3693 (100.5%) | 0.4338 (100.4%) | 0.9778 (99.7%) | 0.6736 (97.1%) | 0.8137 (102.3%) | 0.6877 (101.0%) | 0.8259 (97.3%) |
| hyb1 | 2048 | 0.50 | 0.3562 (97.0%) | 0.4196 (97.2%) | 0.9725 (99.2%) | 0.6576 (94.8%) | 0.8046 (101.2%) | 0.6930 (101.8%) | 0.8196 (96.6%) |
| hyb1 | 4096 | 0.20 | 0.3673 (100.0%) | 0.4317 (100.0%) | 0.9799 (99.9%) | 0.6895 (99.4%) | 0.8010 (100.7%) | 0.6835 (100.4%) | 0.8494 (100.1%) |
| hyb1 | 8192 | 0.00 | 0.3695 (100.6%) | 0.4339 (100.5%) | 0.9806 (100.0%) | 0.7009 (101.0%) | 0.7957 (100.0%) | 0.6838 (100.4%) | 0.8486 (100.0%) |
| hybnoW | 256 | 0.75 | 0.3762 (102.4%) | 0.4415 (102.2%) | 0.9807 (100.0%) | 0.7010 (101.0%) | 0.8159 (102.6%) | 0.7107 (104.4%) | 0.8325 (98.1%) |
| hybnoW | 512 | 0.70 | 0.3744 (101.9%) | 0.4397 (101.8%) | 0.9817 (100.1%) | 0.7016 (101.1%) | 0.8135 (102.3%) | 0.6988 (102.6%) | 0.8538 (100.6%) |
| hybnoW | 1024 | 0.75 | **0.3825 (104.1%)** | **0.4475 (103.6%)** | 0.9825 (100.2%) | 0.7105 (102.4%) | 0.8225 (103.4%) | 0.7158 (105.1%) | 0.8478 (99.9%) |
| hybnoW | 2048 | 0.90 | 0.3779 (102.9%) | 0.4425 (102.5%) | 0.9818 (100.1%) | 0.7012 (101.0%) | 0.8171 (102.7%) | 0.7132 (104.7%) | 0.8355 (98.5%) |
| hybnoW | 4096 | 0.90 | 0.3681 (100.2%) | 0.4328 (100.2%) | 0.9800 (99.9%) | 0.6903 (99.5%) | 0.8027 (100.9%) | 0.6875 (101.0%) | 0.8487 (100.0%) |
| hybnoW | 8192 | 0.70 | 0.3689 (100.4%) | 0.4337 (100.4%) | 0.9809 (100.0%) | 0.6980 (100.6%) | 0.7980 (100.3%) | 0.6852 (100.6%) | 0.8553 (100.8%) |

Out of domain:

| System | k | λ\* | NFCorpus nDCG@10 | SciFact nDCG@10 | FiQA nDCG@10 | Mean nDCG@10 | Mean R@100 |
|---|---|---|---|---|---|---|---|
| **SparseUp** | – | 0 | 0.3510 | 0.7418 | 0.4686 | 0.5205 | 0.6762 |
| ftsparseup | 256 | 0 | 0.3341 (95.2%) | 0.7068 (95.3%) | 0.4301 (91.8%) | 0.4903 (94.2%) | 0.6655 (98.4%) |
| ftsparseup | 512 | 0 | 0.3443 (98.1%) | 0.7214 (97.2%) | 0.4402 (93.9%) | 0.5020 (96.4%) | 0.6695 (99.0%) |
| ftsparseup | 1024 | 0 | 0.3407 (97.1%) | 0.7181 (96.8%) | 0.4302 (91.8%) | 0.4963 (95.4%) | 0.6532 (96.6%) |
| ftsparseup | 2048 | 0 | 0.3452 (98.3%) | 0.7179 (96.8%) | 0.4191 (89.4%) | 0.4941 (94.9%) | 0.6479 (95.8%) |
| ftsparseup | 4096 | 0 | 0.3480 (99.1%) | 0.7354 (99.1%) | 0.4637 (99.0%) | 0.5157 (99.1%) | 0.6757 (99.9%) |
| ftsparseup | 8192 | 0 | 0.3514 (100.1%) | 0.7370 (99.4%) | 0.4686 (100.0%) | 0.5190 (99.7%) | 0.6752 (99.9%) |
| hyb1 | 256 | 0.65 | 0.3322 (94.6%) | 0.7060 (95.2%) | 0.4254 (90.8%) | 0.4879 (93.7%) | 0.6593 (97.5%) |
| hyb1 | 512 | 0.40 | 0.3459 (98.5%) | 0.7199 (97.0%) | 0.4378 (93.4%) | 0.5012 (96.3%) | 0.6710 (99.2%) |
| hyb1 | 1024 | 0.30 | 0.3420 (97.4%) | 0.7153 (96.4%) | 0.4307 (91.9%) | 0.4960 (95.3%) | 0.6534 (96.6%) |
| hyb1 | 2048 | 0.50 | 0.3479 (99.1%) | 0.7054 (95.1%) | 0.4158 (88.7%) | 0.4897 (94.1%) | 0.6394 (94.6%) |
| hyb1 | 4096 | 0.20 | 0.3485 (99.3%) | 0.7356 (99.2%) | 0.4661 (99.5%) | 0.5167 (99.3%) | 0.6744 (99.7%) |
| hyb1 | 8192 | 0.00 | 0.3514 (100.1%) | 0.7370 (99.4%) | 0.4686 (100.0%) | 0.5190 (99.7%) | 0.6752 (99.9%) |
| hybnoW | 256 | 0.75 | 0.3462 (98.6%) | 0.7353 (99.1%) | 0.4598 (98.1%) | 0.5138 (98.7%) | 0.6782 (100.3%) |
| hybnoW | 512 | 0.70 | 0.3498 (99.7%) | 0.7350 (99.1%) | 0.4652 (99.3%) | 0.5167 (99.3%) | 0.6797 (100.5%) |
| hybnoW | 1024 | 0.75 | 0.3516 (100.2%) | 0.7371 (99.4%) | 0.4731 (101.0%) | **0.5206 (100.0%)** | 0.6772 (100.2%) |
| hybnoW | 2048 | 0.90 | 0.3542 (100.9%) | 0.7325 (98.7%) | 0.4612 (98.4%) | 0.5160 (99.1%) | 0.6685 (98.9%) |
| hybnoW | 4096 | 0.90 | 0.3492 (99.5%) | 0.7359 (99.2%) | 0.4660 (99.4%) | 0.5170 (99.3%) | 0.6764 (100.0%) |
| hybnoW | 8192 | 0.70 | 0.3504 (99.8%) | 0.7384 (99.5%) | 0.4698 (100.3%) | 0.5195 (99.8%) | 0.6740 (99.7%) |

- **`hybnoW`** (SparseUp + the fine-tuned SparseUp, no W) is the best of the three new variants at every k, in domain and
  out of domain (and it matches or beats `hyb` at most k): at k = 1024 it gives 0.3825 (104.1%) on dev, 0.5206 (100.0%)
  on BEIR.
- **`ftsparseup`** alone is on par with SparseUp for k ≥ 4096 (and k = 1024 on dev) but loses generality for small k
  (BEIR 94–96%, FiQA down to 89%): the fine-tuned encoder is specialised on MS MARCO. At k = 8192 it is a drop-in
  replacement of SparseUp (100.6% dev, 99.7% BEIR) with 0.95× the space.

## 5. Significance tests

`12_significance.py`: paired bootstrap (10,000 resamples) for the 95% CI of B − A, paired t-test and permutation test;
the reported p is the permutation test. A difference is **significant if p < 0.05 and the CI does not contain zero**
(both ends with the same sign). With Bonferroni, 12 tests → p < 0.0042; 6 tests → p < 0.0083.
With 10,000 permutations the smallest resolvable p is about 1/10,001, so a printed 0.0001 is reported as "≤ 0.0001"
(no permutation was as extreme as the observed difference).

### 5.1 In domain, dev MRR@10 (A = SparseUp, 0.3674)

| k | `hyb` MRR@10 | B − A | 95% CI | p | `hybnoW` MRR@10 | B − A | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| 256 | 0.3746 (102.0%) | +0.0072 | [+0.0037, +0.0106] | 0.0001 | 0.3762 (102.4%) | +0.0088 | [+0.0041, +0.0137] | 0.0008 |
| 512 | 0.3718 (101.2%) | +0.0044 | [+0.0007, +0.0081] | 0.018 | 0.3744 (101.9%) | +0.0070 | [+0.0034, +0.0105] | 0.0001 |
| 1024 | 0.3787 (103.1%) | +0.0113 | [+0.0068, +0.0158] | 0.0001 | **0.3825 (104.1%)** | +0.0151 | [+0.0115, +0.0188] | 0.0001 |
| 2048 | **0.3788 (103.1%)** | +0.0114 | [+0.0080, +0.0147] | 0.0001 | 0.3779 (102.9%) | +0.0105 | [+0.0062, +0.0147] | 0.0001 |
| 4096 | 0.3688 (100.4%) | +0.0014 | [−0.0013, +0.0040] | 0.31 (n.s.) | 0.3681 (100.2%) | +0.0007 | [−0.0025, +0.0039] | 0.65 (n.s.) |
| 8192 | 0.3687 (100.4%) | +0.0013 | [−0.0009, +0.0036] | 0.25 (n.s.) | 0.3689 (100.4%) | +0.0015 | [−0.0008, +0.0037] | 0.21 (n.s.) |

The improvement is significant for k ≤ 2048 (for `hyb` k = 512, p = 0.018, it does not survive Bonferroni) and not
significant for k ≥ 4096, where the Concept Model is at SparseUp's level and has nothing left to add.

### 5.2 Does W add something? `hybnoW` (B) vs `hyb` (A), same k, dev MRR@10

| k | B − A | p |
|---|---|---|
| 256 | +0.0017 | 0.41 |
| 512 | +0.0026 | 0.13 |
| 1024 | +0.0039 | 0.021 |
| 2048 | −0.0009 | 0.55 |
| 4096 | −0.0006 | 0.66 |
| 8192 | +0.0001 | 0.91 |

Except for k = 1024 (p = 0.021, not significant after Bonferroni), carrying W into search or not changes nothing: **W is
needed only for training**; at search time the second SparseUp index is enough.

### 5.3 Out of domain (A = SparseUp, B = system, λ\* of MS MARCO)

Done only to justify "on par" out of domain: no difference is significant (all p ≥ 0.11 and every CI contains zero; the
R@1000 of SciFact is identical, Δ = 0).

| System | Dataset | MRR@10 Δ [95% CI], p | nDCG@10 Δ [95% CI], p | R@1000 Δ [95% CI], p |
|---|---|---|---|---|
| hybnoW k = 1024 | NFCorpus | −0.0107 [−0.0266, +0.0041], 0.17 | +0.0007 [−0.0061, +0.0073], 0.84 | +0.0147 [−0.0029, +0.0327], 0.11 |
| | SciFact | −0.0067 [−0.0202, +0.0058], 0.33 | −0.0047 [−0.0153, +0.0052], 0.38 | 0.0000 [0, 0], 1.00 |
| | FiQA | −0.0006 [−0.0122, +0.0111], 0.92 | +0.0045 [−0.0046, +0.0138], 0.33 | −0.0059 [−0.0136, +0.0019], 0.14 |
| `hyb` k = 2048 | NFCorpus | −0.0060 [−0.0224, +0.0101], 0.48 | +0.0030 [−0.0038, +0.0095], 0.39 | −0.0058 [−0.0215, +0.0099], 0.48 |
| | SciFact | −0.0073 [−0.0241, +0.0088], 0.40 | −0.0043 [−0.0187, +0.0091], 0.54 | 0.0000 [0, 0], 1.00 |
| | FiQA | −0.0063 [−0.0183, +0.0062], 0.32 | +0.0009 [−0.0082, +0.0100], 0.84 | −0.0012 [−0.0088, +0.0059], 0.74 |
| `ftsparseup` k = 8192 | NFCorpus | −0.0058 [−0.0180, +0.0056], 0.34 | +0.0005 [−0.0031, +0.0042], 0.80 | −0.0027 [−0.0062, +0.0008], 0.14 |
| | SciFact | −0.0058 [−0.0144, +0.0018], 0.20 | −0.0047 [−0.0107, +0.0008], 0.12 | 0.0000 [0, 0], 1.00 |
| | FiQA | −0.0012 [−0.0086, +0.0063], 0.76 | −0.0000 [−0.0045, +0.0045], 0.99 | +0.0003 [−0.0013, +0.0020], 0.79 |

(The MRR@10 columns are not significantly lower either, but the point estimates are all slightly negative: out of domain
the gain seen on MS MARCO does not carry over, and the claim is "same quality", not "better".)

### 5.4 `ftsparseup` vs SparseUp in domain (A = SparseUp, B = fine-tuned encoder alone)

Δ = B − A, [95% CI], p (permutation). For k = 8192 the file is `hyb1 … lam0.00` (same system as `ftsparseup`).

| k | MRR@10 | nDCG@10 | R@1000 |
|---|---|---|---|
| 256 | −0.0053 [−0.0111, +0.0007], 0.087 | −0.0053 [−0.0106, +0.0002], 0.056 | −0.0049 [−0.0077, −0.0020], 0.0009 |
| 512 | −0.0040 [−0.0093, +0.0012], 0.14 | −0.0040 [−0.0088, +0.0008], 0.099 | −0.0039 [−0.0066, −0.0013], 0.0026 |
| 1024 | +0.0017 [−0.0040, +0.0075], 0.57 | +0.0015 [−0.0038, +0.0068], 0.58 | −0.0033 [−0.0062, −0.0004], 0.027 |
| 2048 | **−0.0096 [−0.0156, −0.0036], 0.0019** | **−0.0107 [−0.0162, −0.0053], ≤ 0.0001** | **−0.0091 [−0.0123, −0.0059], 0.0001** |
| 4096 | −0.0005 [−0.0040, +0.0029], 0.77 | −0.0004 [−0.0035, +0.0026], 0.78 | −0.0006 [−0.0020, +0.0008], 0.42 |
| 8192 | +0.0021 [−0.0008, +0.0049], 0.15 | +0.0019 [−0.0005, +0.0044], 0.13 | +0.0001 [−0.0010, +0.0012], 0.90 |

- At k = 8192 (and 4096) the fine-tuned encoder alone is **statistically indistinguishable** from SparseUp on all three
  metrics: same quality, not better.
- At k = 2048 it is significantly **worse** on every metric (the 97.4% of section 4.2 is not noise).
- For k ≤ 1024 the MRR@10 / nDCG@10 differences are not significant, but R@1000 is significantly lower (k = 1024 only at
  p = 0.027, not after Bonferroni): the small-k encoders lose recall.

## 6. Space (8.84M passages, raw sizes, 1 GB = 10⁹ bytes)

Sparse index = 8 bytes per non-zero entry (int32 id + float32 value) × entries. SparseUp: 1,675,900,653 entries =
189.5 per passage = 13.4 GB. Dense `g(d)` = 2k bytes per passage (fp16). The index sizes of the fine-tuned encoders are
those printed by `08c`.

| System | What is stored per document | Bytes/doc | Total | vs SparseUp |
|---|---|---|---|---|
| SparseUp | sparse index (~190 terms) | ~1.5 KB | 13.4 GB | ×1.00 |
| frozen / e2e k = 256 | dense `g(d)`, 256 fp16 | 0.5 KB | 4.5 GB | ×0.34 |
| frozen / e2e k = 512 | dense, 512 fp16 | 1.0 KB | 9.1 GB | ×0.68 |
| frozen / e2e k = 1024 | dense, 1024 fp16 | 2.0 KB | 18.1 GB | ×1.35 |
| frozen / e2e k = 2048 | dense, 2048 fp16 | 4.1 KB | 36.2 GB | ×2.7 |
| frozen / e2e k = 4096 | dense, 4096 fp16 | 8.2 KB | 72.4 GB | ×5.4 |
| frozen / e2e k = 8192 | dense, 8192 fp16 | 16.4 KB | 144.9 GB | ×10.8 |
| hybrid frozen (any k) | SparseUp index only (+ W once: 34,023 × k fp32, 35 MB … 1.1 GB) | ~1.5 KB | 13.4 GB | ×1.00 |
| `hyb` / `hybnoW`, e2e k = 256 | SparseUp index + fine-tuned sparse index (~95 terms, 6.7 GB) | ~2.3 KB | 20.1 GB | ×1.50 |
| `hyb` / `hybnoW`, e2e k = 512 | + fine-tuned index (~92 terms, 6.5 GB) | ~2.3 KB | 19.9 GB | ×1.49 |
| `hyb` / `hybnoW`, e2e k = 1024 | + fine-tuned index (~86 terms, 6.1 GB) | ~2.2 KB | 19.5 GB | ×1.46 |
| `hyb` / `hybnoW`, e2e k = 2048 | + fine-tuned index (~78 terms, 5.5 GB) | ~2.1 KB | 18.9 GB | ×1.41 |
| `hyb` / `hybnoW`, e2e k = 4096 | + fine-tuned index (~167 terms, 11.8 GB) | ~2.9 KB | 25.2 GB | ×1.88 |
| `hyb` / `hybnoW`, e2e k = 8192 | + fine-tuned index (~180 terms, 12.7 GB) | ~3.0 KB | 26.1 GB | ×1.95 |
| `hyb1` / `ftsparseup`, e2e k = 256 | only the fine-tuned index | ~0.8 KB | 6.7 GB | ×0.50 |
| `hyb1` / `ftsparseup`, e2e k = 512 | only the fine-tuned index | ~0.7 KB | 6.5 GB | ×0.49 |
| `hyb1` / `ftsparseup`, e2e k = 1024 | only the fine-tuned index | ~0.7 KB | 6.1 GB | ×0.46 |
| `hyb1` / `ftsparseup`, e2e k = 2048 | only the fine-tuned index | ~0.6 KB | 5.5 GB | ×0.41 |
| `hyb1` / `ftsparseup`, e2e k = 4096 | only the fine-tuned index | ~1.3 KB | 11.8 GB | ×0.88 |
| `hyb1` / `ftsparseup`, e2e k = 8192 | only the fine-tuned index | ~1.4 KB | 12.7 GB | ×0.95 |

The e2e encoders at k ≤ 2048 learn much sparser documents (78–95 terms instead of 190) — a side effect of the
training, not an explicit sparsity penalty.

## 7. Final comparison

| System | MS MARCO dev MRR@10 | dev nDCG@10 | BEIR mean nDCG@10 | Indexes (size) |
|---|---|---|---|---|
| SparseUp (original) | 0.3674 | 0.4319 | 0.5205 | 1 (13.4 GB) |
| `hybnoW`, k = 1024 | 0.3825 (+0.0151, +4.1%, p ≤ 0.0001) | 0.4475 (+0.0155, p ≤ 0.0001) | 0.5206 (100.0%) | 2 (19.5 GB, ×1.46) |
| `hyb`, k = 2048 | 0.3788 (+0.0114, +3.1%, p ≤ 0.0001) | 0.4437 (+0.0117, p ≤ 0.0001) | 0.5203 (100.0%) | 2 (18.9 GB, ×1.41) |
| `ftsparseup`, k = 8192 | 0.3695 (+0.0021, 100.6%, p = 0.15) | 0.4339 (100.5%) | 0.5190 (99.7%) | 1 (12.7 GB, ×0.95) |
