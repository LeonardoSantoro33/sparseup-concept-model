# Step-by-step guide: how every result is produced

This guide lists, in execution order, the commands of the **SparseUp Concept Model**, on the same Slurm cluster and with
the same protocol as the SPLADE Concept Model (`splade-concept-model`). Every step of that project has its twin here;
the only real difference is at the beginning: **nothing is pre-encoded**, so the SparseUp vectors of MS MARCO are
produced by us (step 1.3, new script `02b`). All commands are run from the repository directory. Result tables go in
[`RESULTS.md`](RESULTS.md).

| Stage | What it does | Script |
|---|---|---|
| Data | MS MARCO + TREC DL download, SparseUp encoding of MS MARCO (official implementation) | `02`, `02b`, `01` |
| Baseline | SparseUp with exact search over all of MS MARCO | `src/04_baseline_sparseup.py` |
| Model | Concept Model `g(x) = W·SparseUp(x)`, score `g(q)·g(d)` | `src/concept_model.py` |
| Training | **frozen** (only W) and **e2e** (W and the SparseUp encoder) variants, InfoNCE | `src/06_train_concept_model.py --mode frozen / e2e` |
| Negatives | in-batch + hard negatives mined with SparseUp (exact search) | `src/07_mine_negatives.py --miner sparseup` |
| Evaluation in domain | MS MARCO dev, TREC DL19, DL20 | `08`, `08b`, `08c` |
| Evaluation out of domain | zero-shot BEIR (NFCorpus, SciFact, FiQA) | `src/10_eval_beir.py` |
| Hybrids | mix with SparseUp, λ chosen on validation | `src/11_eval_hybrid.py` |
| Statistics | paired per-query tests | `src/12_significance.py` |

---

## 0. Before you start (once)

```bash
cd /datafast/leonardo.santoro/codice
unzip sparseup-concept-model.zip && cd sparseup-concept-model
mkdir -p logs                     # Slurm writes the job logs here
conda activate lce                # the environment of the SPLADE project works as it is
python -c "import transformers; print(transformers.__version__)"   # must be >= 5.3 (SparseUp's remote code)
pip install -r requirements.txt   # only if something is missing
```

**Data directories.** Paths come from environment variables with a **`SPARSEUP_` prefix**, so the `WORK_DIR` /
`FM_DIR` of the SPLADE project (still in `~/.bashrc`) are ignored and the two projects can never mix their files.
Defaults in `src/config.py`:

```bash
export SPARSEUP_WORK_DIR=/datafast/$USER/sparseup_concept   # this is already the default on the cluster
export HF_HOME=/datafast/$USER/hf_cache                     # optional: keeps the 750 MB model out of the home quota
```

- `SPARSEUP_WORK_DIR` (`/datafast/leonardo.santoro/sparseup_concept`): same layout as `splade_concept`:
  `beir/ cache/ embeddings/ encoder_check.json lce/ models/ msmarco/ negatives/ results/ runs/`, plus
  **`encoded/`**, the twin of the folder of pre-encoded files that was provided for SPLADE (written by `02b`).
- `SPARSEUP_FM_DIR`: where the pre-encoded files are; default `$SPARSEUP_WORK_DIR/encoded`. Normally not set.

**Space.** SparseUp documents have ~190 non-zero entries (SPLADE ~120): `encoded/documents.bin` and
`cache/docs_folded` are ~13 GB each (SPLADE: 8.5 GB). Plan ~250 GB for the whole set of experiments.

**Model.** [`Linkup-Platform/linkup-sparseup-embed-v1`](https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1)
(Apache 2.0). It is downloaded automatically at the first use (GPU nodes have internet access). It is loaded with
`trust_remote_code=True` because the SparseUp head (logit shift 15, per-position top-k 12, vocabulary folding,
`[Q] `/`[D] ` prefixes) lives in the `modeling_splade.py` file of the repository.

**Submitting jobs.** Same as before:

```bash
sbatch [--export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True] --time=HH:MM:SS -J NAME jobs/gpu.sbatch src/SCRIPT.py --args
sbatch --time=00:15:00 -J NAME jobs/cpu.sbatch src/SCRIPT.py --args
squeue -u $USER ; tail -f logs/NAME_<jobid>.out ; scancel <jobid>
```

**Naming of results.** Every evaluation writes `results/<run>_<set>.json` with per-query metrics. The names are the
same as in the SPLADE project with `splade` → `sparseup`: `sparseup_baseline_dev.json`,
`sparseup_baseline_beir_beir_fiqa.json`, `cm_frozen_k1024_r2_lr3e5_best_dev.json`, `hyb_…`, `hybnoW_…`, `hyb1_…`,
`ftsparseup_…` (was `ftsplade_…`).

### Three things that are different from SPLADE (all handled by the code)

1. **Folded vocabulary.** SparseUp outputs a vector over ModernBERT's 50,370 BPE tokens, but merges the case / `Ġ`
   variants of a word onto one canonical token (`heat`, `Heat`, `Ġheat`, `ĠHeat` → `Ġheat`), so only ~34k dimensions
   can ever be non-zero. All caches, `W` (k × C instead of k × 30,522) and LCE work in this **folded space** of C
   canonical tokens. The token list is in `cache/folded_vocab.json`.
2. **Prefixes.** Queries are encoded as `"[Q] " + text`, documents as `"[D] " + text`; the prefix tokens are attended
   to but not pooled. Every call of the encoder says `kind="query"` or `kind="document"`.
3. **Lengths.** Every encoding used for a search (corpus, dev/DL/BEIR queries and documents, e2e re-encoding) uses
   SparseUp's official lengths, **query 128 / document 512 tokens** (`config.py`). The e2e **training batches**
   keep the lengths of the SPLADE Concept Model (`--q-max-len 64 --d-max-len 256`). MS MARCO passages are almost
   never longer than 256 tokens, so in domain this changes nothing; on BEIR it lets the baseline be the published model.

---

## 1. Data and baseline (Recap – INIZIO)

### 1.1 Smoke test on synthetic data (5–10 minutes, CPU)

```bash
srun -p long -c 4 --mem=16G -t 00:30:00 python -u src/00_smoke_test.py
```

Must end with `ALL TESTS PASSED (14)`. It builds a tiny random "SparseUp-like" model (byte-level BPE with case
variants, 2-layer ModernBERT, shift, top-k, folding, prefixes) loaded with `trust_remote_code` exactly like the real
one, and runs **every script from 01 to 12** on a fake MS MARCO and a fake BEIR dataset. It also checks that our
memory-efficient encoder gives the same vectors as a literal re-implementation of the official code.

### 1.2 Download MS MARCO and TREC DL (~10 min, network, login node is fine)

```bash
python -u src/02_download_msmarco.py --trec-dl
```

Same files as before: `collection.tsv` (8.8M passages), `queries.train.tsv` (808k), `qrels.train.tsv`,
`queries.dev.small.tsv` (6,980), `qrels.dev.small.tsv`, TREC DL 2019/2020 queries and qrels.

### 1.3 Encode MS MARCO with SparseUp — NEW (~4–6 h on an A100 80GB, GPU)

```bash
sbatch --time=2-00:00:00 -J enc_msmarco jobs/gpu.sbatch src/02b_encode_msmarco.py
```

In the SPLADE project the vectors were given to us. Here `02b` produces **the same six files, with the same names and
format** in `sparseup_concept/encoded/`: `documents.bin`, `doc_ids.npy`, `queries.bin` (the 6,980 dev queries),
`queries_ids.npy`, `token_to_id_mapping.json`, `groundtruth_100.tsv`. The vectors are computed with the **official**
SparseUp implementation (`model.encode`, float32, TF32 off): they are the reference. The groundtruth (exact top-100 of
the dev queries) is computed with `torch.sparse`, a different code path from the search of `retrieval.py`, so that
step 1.6 is a real check. The passages are encoded in resumable shards of 500k: if the job dies, resubmit it.
At the end the log prints the mean number of non-zero entries (the SparseUp blog reports ~47 per query and ~190 per
document on MS MARCO).

### 1.4 Prepare the cache (~20 min, CPU)

```bash
sbatch -J prep jobs/cpu.sbatch src/01_prepare_data.py
```

Same as before: converts the Seismic ids of `encoded/` into the folded space and caches the 8.84M documents and
the 6,980 dev queries (`cache/docs_folded`, `cache/dev_queries_folded`). Check `dot-product check: groundtruth=X
recomputed=X` (same number).

### 1.5 Does our encoder reproduce the official vectors? (~5 min, GPU)

```bash
srun -p long -c 4 --mem=32G --gpus 1 -t 00:30:00 python -u src/03_check_encoder.py
```

Everything that is not in `encoded/` (training queries, DL queries, BEIR, the whole e2e variant) is encoded by
**our** memory-efficient encoder (`sparseup_encoder.py`). Must print `GOOD: same model` with mean cosine 1.0000 on
queries and on passages in `fp32`; the `amp` lines (bf16 backbone, used only by the e2e variant, as in the SPLADE
project) are only informative. Saved in `encoder_check.json`.

### 1.6 SparseUp baseline (~15 min, GPU)

```bash
sbatch -J baseline jobs/gpu.sbatch src/04_baseline_sparseup.py --query-sets dev dl19 dl20
```

Must print `Overlap with groundtruth_100 of 02b (top-100): 1.0000 (expected ~1.00)`. If it is low, stop.
SparseUp is not distilled on MS MARCO: expect dev nDCG@10 ≈ 0.42 (from the BEIR-13 averages in the blog), i.e. below
SPLADE ensembledistil (0.4489). **This is the number to beat for the whole project** (every `% of SparseUp` is
computed against these values).

### 1.7 Encode the training queries (~15 min, GPU)

```bash
sbatch -J enc_train jobs/gpu.sbatch src/05_encode_train_queries.py
```

The 503k training queries with a qrel (same set as before). 1,000 of them are held out for validation by script 06
(seed 42): they are **the same 1,000 queries** as in the SPLADE project (same qrels, same seed, same procedure).

---

## 2. Frozen variant: only W is trained (Recap – FROZEN)

Default hyper-parameters (same as the SPLADE project): `--lr 1e-3 --epochs 10 --temperature 1.0 --n-hard 7
--neg-rank-max 100 --val-every 500 --init-std 0.02`; `--neg-rank-min 3` is always passed explicitly (the script default is 0).

### 2.1 Round 1: in-batch negatives only

```bash
sbatch -J cm_r1_k256  jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k 256  --run-name frozen_k256_r1
sbatch -J cm_r1_k512  jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k 512  --run-name frozen_k512_r1
sbatch -J cm_r1_k1024 jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k 1024 --run-name frozen_k1024_r1
```

### 2.2 Mine the hard negatives with SparseUp (~1.5–3 h, GPU)

```bash
sbatch -J mine_sp jobs/gpu.sbatch src/07_mine_negatives.py --miner sparseup --out sparseup
```

### 2.3 Round 2: + 7 hard negatives (ranks 4–100 of SparseUp)

```bash
sbatch -J cm_r2_k256  jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k 256  --run-name frozen_k256_r2  --negatives sparseup --n-hard 7 --neg-rank-min 3
sbatch -J cm_r2_k512  jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k 512  --run-name frozen_k512_r2  --negatives sparseup --n-hard 7 --neg-rank-min 3
sbatch -J cm_r2_k1024 jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k 1024 --run-name frozen_k1024_r2 --negatives sparseup --n-hard 7 --neg-rank-min 3
```

### 2.4 Hyper-parameter search on validation (same grid as the SPLADE project)

All runs: `--negatives sparseup --n-hard 7 --neg-rank-min 3`. Read the best validation MRR@10 in the log
(`new best -> saved`) or in `models/<run>/summary.json`. The questions and the grid are the same as in the SPLADE
project; only the answers may change:

| Question | Runs (k = 1024 unless said) |
|---|---|
| learning rate of W | `--lr 1e-4 --epochs 2` (`frozen_k1024_r2_lr1e4`), `--lr 3e-5 --epochs 2` (`frozen_k1024_r2_lr3e5`) |
| discard more candidates | `--lr 1e-4 --epochs 2 --neg-rank-min 20` |
| temperature | `--lr 3e-5 --temperature 0.1`; `--lr 1e-5 --temperature 0.1`; `--lr 1e-5 --temperature 0.05` |
| k = 256, 512 with low lr | `--lr 3e-5` and `--lr 1e-4` (epochs 2) |
| init-std = 1/√k | k256 `--init-std 0.0625`, k512 `--init-std 0.0442`, k1024 `--init-std 0.03125`, with `--lr 3e-5` / `1e-4` |
| lr × epochs with the new init-std | k256 `--lr 3e-4/1e-3 --epochs 5`; k512 `--lr 3e-4/1e-3 --epochs 5`; k1024 `--lr 1e-4 --epochs 5` |

Example (same naming convention as before: `is` = init-std, `ep` = epochs):

```bash
sbatch -J cm_r2_k512_is0442_lr3e4_ep5 jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k 512 \
       --run-name frozen_k512_r2_is0442_lr3e4_ep5 --negatives sparseup --n-hard 7 --neg-rank-min 3 \
       --init-std 0.0442 --lr 3e-4 --epochs 5
```

### 2.5 k = 2048, 4096, 8192

Round 1 and round 2 with default parameters (`frozen_k2048_r1`, `frozen_k2048_r2`, …), then the learning-rate grid
with `--init-std 1/√k` (0.0221 / 0.0156 / 0.0110) and `--epochs 5 --val-every 250`:

```bash
for k_is in "2048 0.0221 is0221" "4096 0.0156 is0156" "8192 0.0110 is0110"; do
  set -- $k_is
  for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do
    sbatch -J cm_k$1_lr$lr jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k $1 \
           --run-name frozen_k$1_r2_$3_lr${lr/-/}_ep5 --negatives sparseup --n-hard 7 --neg-rank-min 3 \
           --init-std $2 --lr $lr --epochs 5 --val-every 250
  done
done
```

(the run names come out as `frozen_k2048_r2_is0221_lr3e5_ep5`, as in the SPLADE project.)

### 2.6 Random-W baseline (W left at its initialisation, init-std = 1/√k)

```bash
for k_is in "256 0.0625" "512 0.0442" "1024 0.03125" "2048 0.0221" "4096 0.0156" "8192 0.0110"; do
  set -- $k_is
  sbatch -J cm_rand_k$1 jobs/gpu.sbatch src/06_train_concept_model.py --mode frozen --k $1 --run-name frozen_k$1_rand \
         --negatives sparseup --n-hard 7 --neg-rank-min 3 --init-std $2 --lr 1e-9 --max-steps 1
done
```

### 2.7 Official evaluation (dev, DL19, DL20)

For the best checkpoint of each k (and for each `frozen_k*_rand`):

```bash
sbatch -J eval_k1024 jobs/gpu.sbatch src/08_eval_concept_model.py --ckpt frozen_k1024_r2_lr3e5 --query-sets dev dl19 dl20 --explain 5
sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -J eval_stream_k4096 jobs/gpu.sbatch \
       src/08b_eval_streaming.py --ckpt frozen_k4096_r2_is0156_lr3e5_ep5 --query-sets dev dl19 dl20
```

`08` builds dense `g(d)`; on 80 GB it fits up to k = 2048 (k = 4096 needs 72 GB). For larger k use `08b`
(streaming, `g(q)·g(d) = (Wᵀg(q))·SparseUp(d)`), same numbers up to fp16 rounding.
`--explain N` writes `concepts_<name>.tsv` (top tokens of each concept, in the folded vocabulary) and N interventions.

---

## 3. End-to-end variant: W and the SparseUp encoder (Recap – E2E)

Same loss, negatives and protocol, plus `--mode e2e` and a second learning rate for the encoder (`--lr-backbone`;
in the run names it is `bb`). Batch size 32, one epoch (~15,700 steps), bf16 backbone with the MLM head in float32,
gradient checkpointing. ModernBERT (149M) with a 50k-token head is heavier than BERT: expect ~3–5 h per run.

```bash
sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --time=08:00:00 \
       -J cm_e2e_k1024 jobs/gpu.sbatch src/06_train_concept_model.py --mode e2e --k 1024 \
       --run-name e2e_k1024_r2_is03125_lr3e5_bb5e6_b32_ep1 --negatives sparseup --n-hard 7 --neg-rank-min 3 \
       --init-std 0.03125 --lr 3e-5 --lr-backbone 5e-6 --batch-size 32 --epochs 1 \
       --val-every 2000 --val-queries 1000 --grad-checkpointing
```

**Search protocol (identical):** a grid over the learning rate of W and of the encoder at 1 epoch; the best validation
wins, differences below 0.001 are ties (the lower encoder learning rate wins); if the winner is at the edge of the grid,
the grid is extended. Start from the SPLADE winners and their neighbours:

| k | init-std | lr W | lr encoder (`--lr-backbone`) grid to start with |
|---|---|---|---|
| 256 | 0.0625 | 1e-4 | 5e-6, **1e-5**, 2e-5 |
| 512 | 0.0442 | 1e-4 | **5e-6**, 1e-5, 2e-5 |
| 1024 | 0.03125 | 3e-5 | 2e-6, **5e-6**, 1e-5 |
| 2048 | 0.0221 | 3e-5 | 2e-6, **5e-6**, 1e-5 |
| 4096 | 0.0156 | 3e-5 | 5e-7, **1e-6**, 2e-6, 5e-6 |
| 8192 | 0.0110 | 3e-5 | 5e-8, **1e-7**, 2e-7, 1e-6 |

(bold = SPLADE winner.) Then, for k ≤ 1024, 1 vs 2 epochs (`--epochs 2`, `_ep2`), and the control
**e2e with fixed W** at k = 1024: same command with `--lr 1e-9` (run `e2e_k1024_r2_is03125_Wfix_bb5e6_b32_ep1`).

### Evaluation: the whole corpus is re-encoded with the fine-tuned encoder (~2–3 h)

```bash
sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --time=10:00:00 -J eval_e2e_stream_k1024 \
       jobs/gpu.sbatch src/08c_eval_e2e_streaming.py --ckpt e2e_k1024_r2_is03125_lr3e5_bb5e6_b32_ep1 --query-sets dev dl19 dl20
```

It saves the sparse shards of the new encoder in `embeddings/<run>_best_sparse/` (resumable), which `11` and the
hybrids reuse. Run it on every final e2e checkpoint (and on the W-fixed one).

---

## 4. Out of domain: BEIR (Recap – FROZEN ed E2E fuori dominio)

```bash
python -u src/10_eval_beir.py --datasets nfcorpus scifact fiqa --download-only
sbatch --time=01:00:00 -J beir_sparseup jobs/gpu.sbatch src/10_eval_beir.py --datasets nfcorpus scifact fiqa --sparseup
sbatch --time=04:00:00 -J beir_frozen jobs/gpu.sbatch src/10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --ckpts frozen_k256_rand frozen_k512_rand frozen_k1024_rand frozen_k2048_rand frozen_k4096_rand frozen_k8192_rand
sbatch --time=04:00:00 -J beir_trained jobs/gpu.sbatch src/10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --ckpts <the 6 final frozen runs>
sbatch --time=06:00:00 -J beir_e2e jobs/gpu.sbatch src/10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --ckpts <the 6 final e2e runs> e2e_k1024_r2_is03125_Wfix_bb5e6_b32_ep1
```

The BEIR zips come from the same UKP server as before; if it is down, the script takes the same files from the
Hugging Face mirrors `BeIR/<name>` and `BeIR/<name>-qrels`. Caveat for the paper: SparseUp itself was fine-tuned on a
mixture that includes the training splits of some BEIR datasets, so "zero-shot" refers to the Concept Model.

---

## 5. Hybrids (Recap – IBRIDA)

Score `(1-λ)·SparseUp(q)·SparseUp(d) + λ·g(q)·g(d)`. One job per checkpoint: exact search on the 1,000 validation
queries for 21 values of λ (0, 0.05, …, 1) → λ\* = best MRR@10, then dev / DL19 / DL20 with λ\*, plus the sanity
checks λ = 0 (must reproduce `04`) and λ = 1 (must reproduce the Concept Model alone).

```bash
# frozen (only SparseUp's index and W are needed)
sbatch --time=01:00:00 -J hyb_frozen_k256 jobs/gpu.sbatch src/11_eval_hybrid.py --ckpt frozen_k256_r2 --query-sets dev dl19 dl20
# e2e: run 08c on the checkpoint first
sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --time=02:00:00 -J hyb_e2e_k256 \
       jobs/gpu.sbatch src/11_eval_hybrid.py --ckpt e2e_k256_r2_is0625_lr1e4_bb1e5_b32_ep1 --query-sets dev dl19 dl20
```

Then the variants of the IBRIDA recap, for every e2e checkpoint (`--variant single` / `--variant now`):

| Name in the results | `--variant` | Score | Indexes |
|---|---|---|---|
| `hyb_…` | `two` (default) | `(1-λ)·SU·SU + λ·g·g` | 2 |
| `hyb1_…` | `single` | `(1-λ)·SU_θ·SU_θ + λ·g·g` | 1 |
| `hybnoW_…` | `now` | `(1-λ)·SU·SU + λ·SU_θ·SU_θ` (no W) | 2 |
| `ftsparseup_…` | (`single` with λ = 0) | `SU_θ·SU_θ` | 1 |

(SU = original SparseUp, SU_θ = encoder re-trained through the Concept Model.) On BEIR with the λ\* of MS MARCO:

```bash
sbatch --time=02:00:00 -J beir_hyb jobs/gpu.sbatch src/10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --hybrid <e2e_run>:now:<λ*> <e2e_run>:two:<λ*> <e2e_run>:single:<λ*> <frozen_run>:two:<λ*>
sbatch --time=02:00:00 -J beir_ft jobs/gpu.sbatch src/10_eval_beir.py --datasets nfcorpus scifact fiqa --ft-sparseup <e2e runs>
```

---

## 6. Significance tests

```bash
sbatch --time=00:15:00 -J sig_e2e_k256 jobs/cpu.sbatch src/12_significance.py \
       --a sparseup_baseline_dev --b hyb_e2e_k256_r2_is0625_lr1e4_bb1e5_b32_ep1_best_lam0.20_dev
```

Same tests as in the IBRIDA recap: `hyb` vs SparseUp (every k), `hyb` vs `hybnoW`, `hyb1` vs SparseUp,
`ftsparseup` vs SparseUp, and the final system vs SparseUp on dev and on each BEIR dataset
(A = `sparseup_baseline_beir_beir_<dataset>`). A difference is significant if p(perm) < 0.05 and the 95% bootstrap
CI of B − A does not contain zero.

---

## 7. Figures

```bash
python plots/concept_model_plots.py       # writes plots/figures/fig1…fig4 (.png and .pdf)
```

Fill in the dictionaries at the top of the script with the numbers of `RESULTS.md` (missing values are skipped).

---

## 8. Design choices to report

- **Exact search instead of ANN** (the SparseUp paper uses Seismic, approximate): every number is an upper bound for any
  approximate index.
- **Reference vectors from the official code**: the corpus is encoded by `modeling_splade.py` itself (float32); our
  memory-efficient re-implementation is checked against it (step 1.5) before being used for anything.
- **Memory-efficient SparseUp pooling** (`sparseup_encoder.py`): top-k taken on the logits, then a single
  `scatter_reduce(amax)` over positions *and* fold groups, and only then `log(1 + ReLU(x − 15))`. Same result as the
  official code (the function is monotone), without the (B, L, 50k) activation tensor.
- **Folded space**: `W` is k × C with C ≈ 34k canonical tokens (dimensions that SparseUp can never activate are dropped).
- **Clean validation**: the same 1,000 held-out training queries as the SPLADE project; checkpoints, hyper-parameters
  and λ are chosen on them, never on dev or BEIR.
- **False negatives**: the other known positives of the same query are masked out of the loss; the top 3 mining
  candidates are discarded (`--neg-rank-min 3`).

---

## 9. Troubleshooting

| Message | What to do |
|---|---|
| `CUDA out of memory` | lower `--batch-size` (training) or `--q-batch` (search); `--grad-checkpointing` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; for large k use `08b` / `08c` |
| `02b` OOM on long passages | lower `--max-batch-tokens` (default 32768) |
| `trust_remote_code` / `SpladeModel` import errors | `pip install -U "transformers>=5.3"` |
| `03` says WARNING (cosine < 0.99) | do not go on: check the transformers version and that `encoded/` was made by the same model revision |
| `04` overlap low | data and ids are not aligned: rerun `01` |
| `[ERROR] … contains shards made with different settings` | a previous `02b` / `08c` used other options: delete that shard folder and rerun |
