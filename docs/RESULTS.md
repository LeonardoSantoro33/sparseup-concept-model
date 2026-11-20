# Results

*To be filled in as the experiments of [`GUIDE.md`](GUIDE.md) complete.* All numbers come from the per-query files in
`$SPARSEUP_WORK_DIR/results/` and the job logs in `logs/`. Percentages are relative to **SparseUp** in the same column.
Test sets: MS MARCO dev (6,980 queries), TREC DL19 (43) and DL20 (54), BEIR (NFCorpus 323, SciFact 300, FiQA 648
queries). Differences below about 0.02 on DL19 / DL20 are noise.

## 0. Data and baseline

| Check | Expected | Obtained |
|---|---|---|
| `02b`: mean non-zero entries per dev query / per passage | ~47 / ~190 (SparseUp blog) | – / – |
| `01`: folded vocabulary size | ~34k | – |
| `03`: our encoder vs official, mean cosine queries / passages (fp32) | 1.0000 / 1.0000 | – / – |
| `03`: same, amp (bf16 backbone, e2e only) | ≈ 0.99+ | – / – |
| `04`: overlap with the groundtruth of `02b` | 1.0000 | – |
| Validation (1,000 held-out queries): SparseUp MRR@10 on its top-100 | – | – |

| System | dev MRR@10 | dev nDCG@10 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|
| SPLADE ensembledistil (previous project) | 0.3828 | 0.4489 | 0.9831 | 0.7308 | 0.8724 | 0.7203 | 0.8989 |
| **SparseUp** | – | – | – | – | – | – | – |

## 1. Final checkpoints

Chosen on validation (1,000 held-out training queries), never on dev or BEIR. All use `--negatives sparseup
--n-hard 7 --neg-rank-min 3`, temperature 1, seed 42.

**Frozen** (only W trained)

| k | run | init-std | lr | epochs | best val |
|---|---|---|---|---|---|
| 256 | | | | | |
| 512 | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |
| 4096 | | | | | |
| 8192 | | | | | |

**Random W** (not trained, init-std = 1/√k)

| k | run | best val |
|---|---|---|
| 256 | `frozen_k256_rand` | |
| 512 | `frozen_k512_rand` | |
| 1024 | `frozen_k1024_rand` | |
| 2048 | `frozen_k2048_rand` | |
| 4096 | `frozen_k4096_rand` | |
| 8192 | `frozen_k8192_rand` | |

**e2e** (W and the SparseUp encoder trained, batch size 32, 1 epoch)

| k | run | init-std | lr W | lr encoder | best val | frozen val |
|---|---|---|---|---|---|---|
| 256 | | | | | | |
| 512 | | | | | | |
| 1024 | | | | | | |
| 1024 (W fixed) | | | 1e-9 | | | |
| 2048 | | | | | | |
| 4096 | | | | | | |
| 8192 | | | | | | |

## 2. In domain: MS MARCO

| System | k | B/doc (fp16) | dev MRR@10 | dev nDCG@10 | dev R@10 | dev R@100 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 | DL20 nDCG@10 | DL20 R@1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SparseUp** | – | ~1500 | | | | | | | | | |
| Random W | 256 … 8192 | | | | | | | | | | |
| Frozen | 256 … 8192 | | | | | | | | | | |
| e2e | 256 … 8192 | | | | | | | | | | |

## 3. Out of domain: BEIR (nDCG@10 and R@100, zero-shot)

| Method | k | NFCorpus | SciFact | FiQA | Mean | % SparseUp |
|---|---|---|---|---|---|---|
| **SparseUp** | sparse | | | | | 100.0% |
| Frozen, W not trained | 256 … 8192 | | | | | |
| Frozen | 256 … 8192 | | | | | |
| e2e | 256 … 8192 | | | | | |
| e2e, W fixed | 1024 | | | | | |

## 4. Hybrids with SparseUp

| Model | k | λ\* | val SparseUp (λ=0) | val CM (λ=1) | val hybrid | dev MRR@10 | dev nDCG@10 | DL19 nDCG@10 | DL20 nDCG@10 | BEIR mean nDCG@10 | Indexes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen | 256 … 8192 | | | | | | | | | | 1 |
| e2e (`hyb`) | 256 … 8192 | | | | | | | | | | 2 |
| `hyb1` | 256 … 8192 | | | | | | | | | | 1 |
| `hybnoW` | 256 … 8192 | | | | | | | | | | 2 |
| `ftsparseup` | 256 … 8192 | 0 | | | | | | | | | 1 |

## 5. Significance tests

| B | A | metric | B − A | 95% CI | p (perm.) | significant? |
|---|---|---|---|---|---|---|
| | `sparseup_baseline_dev` | MRR@10 | | | | |

## 6. Space (8.84M passages, raw sizes)

| System | What is stored per document | Bytes/doc | Total | vs SparseUp |
|---|---|---|---|---|
| SparseUp | sparse index (~190 terms, int32 + float32) | ~1.5 KB | ~13 GB | ×1.00 |
| frozen / e2e k | dense g(d), k fp16 numbers | 2k B | | |
| hybrid frozen | SparseUp index + W (C × k fp32) | | | |
| hybrid e2e | SparseUp index + fine-tuned sparse index | | | |
