# SparseUp Concept Model

**Interpretable concept bottleneck on top of SparseUp (Linkup Research's open learned sparse retriever), trained
contrastively on MS MARCO.** Same research, same protocol and same code structure as the
[SPLADE Concept Model](https://github.com/LeonardoSantoro33/splade-concept-model), with
[`Linkup-Platform/linkup-sparseup-embed-v1`](https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1)
in place of `naver/splade-cocondenser-ensembledistil`.

> Research code developed for a project on interpretable sparse retrieval. Work in progress: results will be added
> as experiments complete (see [`docs/RESULTS.md`](docs/RESULTS.md)).

---

## Overview

[SparseUp](https://www.linkup.so/blog/introducing-sparseup-by-linkup) (Formal, Erbacher, Rulli, 2026) is a
SPLADE-style encoder on a ModernBERT backbone (149M parameters) with three modifications: a **logit shift**
(`log(1 + ReLU(x − 15))`), a **per-position top-k** (each token keeps its 12 largest vocabulary dimensions before
max pooling) and **vocabulary folding** (case and `Ġ` variants of a word share one dimension). Queries and documents
get the prefixes `[Q] ` / `[D] `. Its output lives on ~34k canonical BPE tokens ("folded space", size C).

This project compresses that vector into a small set of **k learned "concepts"** (a Concept Bottleneck, Koh et al.,
2020) and trains the bottleneck for retrieval with a contrastive objective:

$$g(x) = W \cdot \mathrm{SparseUp}(x), \qquad W \in \mathbb{R}^{k \times C}$$

$$\mathrm{score}(q,d) = g(q)\cdot g(d)$$

$$\mathcal{L}(q,d^+) = -\log \frac{e^{s(q,d^+)/\beta}}{e^{s(q,d^+)/\beta} + \sum_i e^{s(q,d_i^-)/\beta}}$$

Two training regimes (**frozen**: only W; **e2e**: W and the SparseUp encoder), in-batch + hard negatives mined with
SparseUp, and hybrids with SparseUp (`(1−λ)·SparseUp(q)·SparseUp(d) + λ·g(q)·g(d)`, λ chosen on validation).
Evaluation on MS MARCO dev, TREC DL 2019/2020 and BEIR (NFCorpus, SciFact, FiQA), with paired significance tests.

## What is different from the SPLADE project

| | SPLADE Concept Model | SparseUp Concept Model |
|---|---|---|
| Encoder | BERT, 30,522 WordPiece dims | ModernBERT, 50,370 BPE dims, folded to ~34k |
| Pre-encoded MS MARCO | provided (Seismic format) | **created by `02b`** with the official implementation, same six files |
| Pooling | `log(1+ReLU(max logits))` | top-k on logits, one `amax` over positions and fold groups, then `log(1+ReLU(x−15))` |
| Prefixes | none | `[Q] ` / `[D] ` (attended, not pooled) |
| Search lengths | query 256 / doc 256 | query 128 / doc 512 (SparseUp's official ones) |
| Training lengths (e2e) | 64 / 256 | 64 / 256 (unchanged) |
| Hyper-parameters, validation, negatives, λ grid, tests | — | identical |
| Env variables | `WORK_DIR`, `FM_DIR` | `SPARSEUP_WORK_DIR`, `SPARSEUP_FM_DIR` (the old ones are ignored) |

## Repository structure

```
.
├── src/
│   ├── 00_smoke_test.py            # unit tests + full pipeline 01-12 on synthetic data with a tiny SparseUp-like model (CPU)
│   ├── 01_prepare_data.py          # read the Seismic .bin files, map token ids to the folded space, cache as CSR
│   ├── 02_download_msmarco.py      # official MS MARCO queries / qrels / passages (+ TREC DL)
│   ├── 02b_encode_msmarco.py       # NEW: encode MS MARCO with the OFFICIAL SparseUp -> the pre-encoded files
│   ├── 03_check_encoder.py         # verify that our encoder reproduces the official vectors
│   ├── 04_baseline_sparseup.py     # SparseUp baseline, exact search, check vs. groundtruth
│   ├── 05_encode_train_queries.py  # SparseUp vectors for the MS MARCO training queries
│   ├── 06_train_concept_model.py   # Concept Model training (frozen / end-to-end, InfoNCE)
│   ├── 07_mine_negatives.py        # hard-negative mining (SparseUp or a trained Concept Model)
│   ├── 08_eval_concept_model.py    # full-corpus evaluation + concept analysis + interventions
│   ├── 08b_eval_streaming.py       # frozen evaluation without dense g(d) (large k)
│   ├── 08c_eval_e2e_streaming.py   # e2e evaluation: re-encodes the corpus with the fine-tuned encoder
│   ├── 09_lce_sparseup.py          # (optional) LCE on SparseUp vectors
│   ├── 10_eval_beir.py             # zero-shot BEIR: baseline, frozen, e2e, fine-tuned encoder, hybrids
│   ├── 11_eval_hybrid.py           # hybrids SparseUp + Concept Model, λ on validation (two / single / now)
│   ├── 12_significance.py          # paired bootstrap + permutation tests
│   │
│   ├── config.py                   # paths (env vars SPARSEUP_WORK_DIR, SPARSEUP_FM_DIR, SPARSEUP_MODEL), lengths
│   ├── sparseup_encoder.py         # SparseUp encoder (memory-efficient pooling, folded space, prefixes)
│   ├── sparseup_reference.py       # literal re-implementation of the official model, used only by the smoke test
│   ├── sparse_io.py  retrieval.py  # CSR / Seismic I/O (also streaming); exact GPU search
│   ├── concept_model.py            # W, InfoNCE, false-negative mask, checkpoints
│   ├── metrics.py  msmarco.py  pipeline.py  utils.py
│   └── lce.py                      # used by the smoke test and by 09
│
├── jobs/                           # generic Slurm job scripts (gpu.sbatch, cpu.sbatch)
├── docs/GUIDE.md                   # step-by-step guide: every command, in order
├── docs/RESULTS.md                 # result tables (to be filled)
├── plots/                          # concept_model_plots.py and the figures (plots/figures/)
├── requirements.txt                # direct dependencies
└── requirements_env.txt            # exact versions of the environment of the SPLADE project (works here too)
```

The scripts write under `$SPARSEUP_WORK_DIR` (default `/datafast/$USER/sparseup_concept`). Checkpoints, embeddings
and caches are not distributed.

## Quickstart

```bash
conda activate lce                         # the environment of the SPLADE project (transformers >= 5.3 needed)
pip install -r requirements.txt
python -u src/00_smoke_test.py             # should end with: ALL TESTS PASSED (14)

python -u src/02_download_msmarco.py --trec-dl
sbatch --time=2-00:00:00 -J enc_msmarco jobs/gpu.sbatch src/02b_encode_msmarco.py
sbatch -J prep jobs/cpu.sbatch src/01_prepare_data.py
srun -p long -c 4 --mem=32G --gpus 1 -t 00:30:00 python -u src/03_check_encoder.py
sbatch -J baseline jobs/gpu.sbatch src/04_baseline_sparseup.py --query-sets dev dl19 dl20
sbatch -J enc_train jobs/gpu.sbatch src/05_encode_train_queries.py
sbatch -J mine_sp jobs/gpu.sbatch src/07_mine_negatives.py --miner sparseup --out sparseup
sbatch -J cm_r2_k1024 jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k 1024 \
       --run-name frozen_k1024_r2 --negatives sparseup --n-hard 7 --neg-rank-min 3
```

The complete sequence, with the exact commands of every experiment, is in [`docs/GUIDE.md`](docs/GUIDE.md).

## Design notes

- **SparseUp pooling.** We take the per-position top-k on the logits and a single `scatter_reduce(amax)` over the
  positions and the fold groups, then apply `log(1 + ReLU(· − θ))`. Because the function is monotone this equals the
  official computation, but avoids the (B, L, 50k) activation tensor in end-to-end training. Checked against the
  official code by `03` and against a literal re-implementation by the smoke test.
- **Reference vectors.** The corpus and the dev queries are encoded once by the official `modeling_splade.py`
  (float32, TF32 off); everything else by our encoder, which must reproduce them (cosine 1.0000).
- **Validation.** The same 1,000 held-out training queries as the SPLADE project (seed 42); dev only for final evaluation.
- **Exact search.** No approximate index (the SparseUp paper uses Seismic): no approximation error to discuss.

## References

- Formal, Erbacher, Rulli, *Introducing SPARSEUP by Linkup Research*, 2026.
- Formal et al., *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*, 2021.
- Sourty et al., *DenseOn with the LateOn: Fully Open Dense and Late-Interaction Models*, 2026.
- Warner et al., *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT)*, 2024.
- Koh et al., *Concept Bottleneck Models*, ICML 2020.
- Izacard et al., *Unsupervised Dense Information Retrieval with Contrastive Learning*, 2021.
- Xiong et al., *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* (ANCE), 2021.
- Lin et al., *How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval*, 2023.
- Bruch et al., *Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations* (Seismic), SIGIR 2024.
- Bajaj et al., *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*, 2016.
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, 2021.

## License

MIT, see [LICENSE](LICENSE). SparseUp is released by Linkup under Apache 2.0.

## Author

Leonardo Santoro, Sapienza University of Rome. Project carried out under academic supervision, using the computing
facilities of ISTI-CNR.
