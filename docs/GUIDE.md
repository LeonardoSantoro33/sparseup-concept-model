# Step-by-step guide: how every result is produced

This guide lists, in execution order, the commands of the **SparseUp Concept Model**, on the same Slurm cluster and with
the same protocol as the SPLADE Concept Model (`splade-concept-model`). Every step of that project has its twin here;
the only real difference is at the beginning: **nothing is pre-encoded**, so the SparseUp vectors of MS MARCO are
produced by us (step 1.3, new script `02b`). All commands are run from the repository directory; the examples use the
script name only (`jobs/gpu.sbatch` looks it up in `src/`). The tables of every step are in [`RESULTS.md`](RESULTS.md),
which also says which hyper-parameters won; the winners are written out below so that every result can be reproduced.

| Stage | What it does | Script |
|---|---|---|
| Data | MS MARCO + TREC DL download, SparseUp encoding of MS MARCO (official implementation) | `02`, `02b`, `01` |
| Baseline | SparseUp with exact search over all of MS MARCO | `04_baseline_sparseup.py` |
| Model | Concept Model `g(x) = W·SparseUp(x)`, score `g(q)·g(d)` | `concept_model.py` |
| Training | **frozen** (only W) and **e2e** (W and the SparseUp encoder) variants, InfoNCE | `06_train_concept_model.py --mode frozen / e2e` |
| Negatives | in-batch + hard negatives mined with SparseUp (exact search) | `07_mine_negatives.py --miner sparseup` |
| Evaluation in domain | MS MARCO dev, TREC DL19, DL20 | `08`, `08b`, `08c` |
| Evaluation out of domain | zero-shot BEIR (NFCorpus, SciFact, FiQA) | `10_eval_beir.py` |
| Hybrids | mix with SparseUp, λ chosen on validation | `11_eval_hybrid.py` |
| Statistics | paired per-query tests | `12_significance.py` |

---

## 0. Before you start (once)

```bash
cd /datafast/$USER/codice
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

- `SPARSEUP_WORK_DIR`: same layout as `splade_concept`: `beir/ cache/ embeddings/ encoder_check.json lce/ models/
  msmarco/ negatives/ results/ runs/`, plus **`encoded/`**, the twin of the folder of pre-encoded files that was
  provided for SPLADE (written by `02b`). **The per-query results are in `$SPARSEUP_WORK_DIR/results/`, not in the
  code folder.**
- `SPARSEUP_FM_DIR`: where the pre-encoded files are; default `$SPARSEUP_WORK_DIR/encoded`. Normally not set.

**Space.** SparseUp documents have ~190 non-zero entries (SPLADE ~120): `encoded/documents.bin` (13 GB) and
`cache/docs_folded` are ~13 GB each (SPLADE: 8.5 GB). Every e2e checkpoint also leaves its re-encoded corpus
(`embeddings/<run>_best_sparse/`, 5.5–12.7 GB). Plan ~250 GB for the whole set of experiments.

**Model.** [`Linkup-Platform/linkup-sparseup-embed-v1`](https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1)
(Apache 2.0). It is downloaded automatically at the first use (GPU nodes have internet access). It is loaded with
`trust_remote_code=True` because the SparseUp head (logit shift 15, per-position top-k 12, vocabulary folding,
`[Q] `/`[D] ` prefixes) lives in the `modeling_splade.py` file of the repository.

**Submitting jobs.**

```bash
sbatch [--export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True] --time=HH:MM:SS -J NAME jobs/gpu.sbatch SCRIPT.py --args
sbatch --time=00:15:00 -J NAME jobs/cpu.sbatch SCRIPT.py --args
squeue -u $USER ; tail -f logs/NAME_<jobid>.out ; scancel <jobid>
```

The GPU is an A100 40 GB (A100 80 GB for `02b`); the memory notes below refer to 40 GB.

**Naming of results.** Every evaluation writes `results/<run>_<set>.json` with per-query metrics. The names are the
same as in the SPLADE project with `splade` → `sparseup`: `sparseup_baseline_dev.json`,
`sparseup_baseline_beir_beir_fiqa.json`, `hyb_<e2e run>_best_lam0.80_dev.json`, `hybnoW_…`, `hyb1_…`, `ftsparseup_…`
(was `ftsplade_…`). For `ftsparseup` on dev there is no file of its own: it is the `hyb1_…_lam0.00_dev` file (hyb1 at
λ = 0 is the same system); on BEIR the file is `ftsparseup_<run>_best_beir_<dataset>`.

### Three things that are different from SPLADE (all handled by the code)

1. **Folded vocabulary.** SparseUp outputs a vector over ModernBERT's 50,370 BPE tokens, but merges the case / `Ġ`
   variants of a word onto one canonical token (`heat`, `Heat`, `Ġheat`, `ĠHeat` → `Ġheat`), so only ~34k dimensions
   (C = 34,023) can ever be non-zero. All caches, `W` (k × C instead of k × 30,522) and LCE work in this **folded
   space**. The token list is in `cache/folded_vocab.json`.
2. **Prefixes.** Queries are encoded as `"[Q] " + text`, documents as `"[D] " + text`; the prefix tokens are attended
   to but not pooled. Every call of the encoder says `kind="query"` or `kind="document"`.
3. **Lengths.** Every encoding used for a search (corpus, dev/DL/BEIR queries and documents, e2e re-encoding) uses
   SparseUp's official lengths, **query 128 / document 512 tokens** (`config.py`). The e2e **training batches**
   keep the lengths of the SPLADE Concept Model (`--q-max-len 64 --d-max-len 256`). MS MARCO passages are almost
   never longer than 256 tokens, so in domain this changes nothing; on BEIR it lets the baseline be the published model.

---

## 1. Data and baseline

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
sbatch --time=2-00:00:00 -J enc_msmarco jobs/gpu.sbatch 02b_encode_msmarco.py
```

In the SPLADE project the vectors were given to us. Here `02b` produces **the same six files, with the same names and
format** in `sparseup_concept/encoded/`: `documents.bin`, `doc_ids.npy`, `queries.bin` (the 6,980 dev queries),
`queries_ids.npy`, `token_to_id_mapping.json`, `groundtruth_100.tsv`. The vectors are computed with the **official**
SparseUp implementation (`model.encode`, float32, TF32 off): they are the reference. The groundtruth (exact top-100 of
the dev queries) is computed with `torch.sparse`, a different code path from the search of `retrieval.py`, so that
step 1.6 is a real check. The passages are encoded in resumable shards of 500k: if the job dies, resubmit it.
At the end the log prints the mean number of non-zero entries (the SparseUp blog reports ~47 per query and ~190 per
document on MS MARCO). **Obtained:** 1,675,900,653 entries over 8.84M passages = 189.5 per passage → 13.4 GB.

### 1.4 Prepare the cache (~20 min, CPU)

```bash
sbatch -J prep jobs/cpu.sbatch 01_prepare_data.py
```

Converts the Seismic ids of `encoded/` into the folded space and caches the 8.84M documents and the 6,980 dev
queries (`cache/docs_folded`, `cache/dev_queries_folded`). Check `dot-product check: groundtruth=X recomputed=X`
(same number). The folded vocabulary has 34,023 entries.

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
sbatch -J baseline jobs/gpu.sbatch 04_baseline_sparseup.py --query-sets dev dl19 dl20
```

Must print `Overlap with groundtruth_100 of 02b (top-100): 1.0000`. If it is low, stop. **Obtained:** dev MRR@10
0.3674, nDCG@10 0.4319, R@1000 0.9805 — below SPLADE ensembledistil (0.3828 / 0.4489), because SparseUp is not
distilled on MS MARCO. **This is the number to beat for the whole project** (every `% of SparseUp` is computed against
these values). Output: `results/sparseup_baseline_{dev,dl19,dl20}.json`.

### 1.7 Encode the training queries (~15 min, GPU)

```bash
sbatch -J enc_train jobs/gpu.sbatch 05_encode_train_queries.py
```

The 503k training queries with a qrel (same set as before). 1,000 of them are held out for validation by script 06
(seed 42): they are **the same 1,000 queries** as in the SPLADE project. SparseUp on those candidates:
**validation MRR@10 = 0.3487**.

---

## 2. Frozen variant: only W is trained

Default hyper-parameters (same as the SPLADE project): `--lr 1e-3 --epochs 10 --temperature 1.0 --n-hard 7
--neg-rank-max 100 --val-every 500 --init-std 0.02`; `--neg-rank-min 3` is always passed explicitly (the script
default is 0).

### 2.1 Round 1: in-batch negatives only (all k)

```bash
for k in 256 512 1024 2048 4096 8192; do
  sbatch -J cm_r1_k$k jobs/gpu.sbatch 06_train_concept_model.py --mode frozen --k $k --run-name frozen_k${k}_r1
done
```

### 2.2 Mine the hard negatives with SparseUp (~1.5–3 h, GPU)

```bash
sbatch -J mine_sp jobs/gpu.sbatch 07_mine_negatives.py --miner sparseup --out sparseup
```

For each of the 503k training queries it takes the top-100 documents of SparseUp; the top 3 are discarded (possible
unlabelled positives) and the rest are the hard negatives.

### 2.3 Round 2: + 7 hard negatives (ranks 4–100 of SparseUp), all k

```bash
for k in 256 512 1024 2048 4096 8192; do
  sbatch -J cm_r2_k$k jobs/gpu.sbatch 06_train_concept_model.py --mode frozen --k $k --run-name frozen_k${k}_r2 \
         --negatives sparseup --n-hard 7 --neg-rank-min 3
done
```

With the default hyper-parameters the validation MRR@10 is still below the random W (see `RESULTS.md` 1.1): the grid of
2.4 is needed.

### 2.4 Hyper-parameter search on validation (grid A)

The best checkpoint arrives early (within the first epoch for k ≥ 2048), so epochs ∈ {1, 2, 5}. Grid: lr ∈
{1e-5, 3e-5, 1e-4, 3e-4, 1e-3} × epochs × init-std ∈ {0.02, 1/√k}. For each k one extra column of init-std = 1/√k
(0.0625, 0.0442, 0.03125, 0.0221, 0.0156, 0.0110).

```bash
run() { sbatch -J "$1" jobs/gpu.sbatch 06_train_concept_model.py --mode frozen --run-name "$1" \
        --negatives sparseup --n-hard 7 --neg-rank-min 3 "${@:2}"; }
for k_is in "256 0.0625 is0625" "512 0.0442 is0442" "1024 0.03125 is03125" "2048 0.0221 is0221" "4096 0.0156 is0156" "8192 0.0110 is0110"; do
  set -- $k_is
  for init in 0.02 $2; do
    tag=""; [ "$init" != "0.02" ] && tag="_$3"
    for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do
      for ep in 1 2 5; do
        ve=100; [ $ep = 5 ] && ve=250
        run frozen_k$1_r2${tag}_lr${lr/-/}_ep$ep --k $1 --init-std $init --lr $lr --epochs $ep --val-every $ve
      done
    done
  done
  run frozen_k$1_rand --k $1 --init-std $2 --lr 1e-9 --max-steps 1      # random-W baseline (see 2.6)
done
grep -H "Best validation" logs/frozen_k*_ep*_*.out
grep -H "random W" logs/frozen_k*_rand_*.out
```

Winners (full tables in `RESULTS.md` 1.1):

| k | run | init-std | lr | epochs | val MRR@10 |
|---|---|---|---|---|---|
| 256 | `frozen_k256_r2_lr1e4_ep2` | 0.02 | 1e-4 | 2 | 0.2770 |
| 512 | `frozen_k512_r2_lr3e4_ep2` | 0.02 | 3e-4 | 2 | 0.3095 |
| 1024 | `frozen_k1024_r2_lr3e5_ep5` | 0.02 | 3e-5 | 5 | 0.3200 |
| 2048 | `frozen_k2048_r2_lr1e5_ep1` | 0.02 | 1e-5 | 1 | 0.3402 |
| 4096 | `frozen_k4096_r2_lr3e4_ep2` | 0.02 | 3e-4 | 2 | 0.3534 |
| 8192 | `frozen_k8192_r2_is0110_lr1e4_ep1` | 0.0110 | 1e-4 | 1 | 0.3492 |

At k = 2048 the winner is at the edge of the grid (lr 1e-5); lr 1e-6 gives 0.3401 < 0.3402, so it stays.

### 2.5 Grid B: negatives and temperature (one change at a time on each winner)

```bash
W=(
"frozen_k256_r2_lr1e4_ep2|--k 256 --init-std 0.02 --lr 1e-4 --epochs 2 --val-every 100"
"frozen_k512_r2_lr3e4_ep2|--k 512 --init-std 0.02 --lr 3e-4 --epochs 2 --val-every 100"
"frozen_k1024_r2_lr3e5_ep5|--k 1024 --init-std 0.02 --lr 3e-5 --epochs 5 --val-every 250"
"frozen_k2048_r2_lr1e5_ep1|--k 2048 --init-std 0.02 --lr 1e-5 --epochs 1 --val-every 100"
"frozen_k4096_r2_lr3e4_ep2|--k 4096 --init-std 0.02 --lr 3e-4 --epochs 2 --val-every 100"
"frozen_k8192_r2_is0110_lr1e4_ep1|--k 8192 --init-std 0.0110 --lr 1e-4 --epochs 1 --val-every 100"
)
for w in "${W[@]}"; do
  name=${w%%|*}; args=${w#*|}
  run ${name}_nrm20 $args --neg-rank-min 20       # discard the top 20 candidates instead of 3
  run ${name}_t01   $args --temperature 0.1
  run ${name}_t005  $args --temperature 0.05
done
grep -H "Best validation" logs/frozen_k*_nrm20_*.out logs/frozen_k*_t01_*.out logs/frozen_k*_t005_*.out
```

None of them beats the winner of 2.4: the final values stay `--neg-rank-min 3`, temperature 1.

### 2.6 Random-W baseline

W left at its initialisation (init-std = 1/√k, `--lr 1e-9 --max-steps 1`): the `frozen_k*_rand` runs launched at the
end of the loop of 2.4. It measures how much the training of W adds.

### 2.7 Official evaluation (dev, DL19, DL20)

For the best checkpoint of each k and for each `frozen_k*_rand`. `08` builds the dense `g(d)` and is used up to
k = 1024; for k ≥ 2048 the 40 GB of the GPU are not enough and `08b` (streaming, `g(q)·g(d) = (Wᵀg(q))·SparseUp(d)`,
same numbers up to fp16 rounding) is used.

```bash
for x in "256 frozen_k256_r2_lr1e4_ep2" "512 frozen_k512_r2_lr3e4_ep2" "1024 frozen_k1024_r2_lr3e5_ep5" \
         "2048 frozen_k2048_r2_lr1e5_ep1" "4096 frozen_k4096_r2_lr3e4_ep2" "8192 frozen_k8192_r2_is0110_lr1e4_ep1"; do
  set -- $x
  for ck in "$2" "frozen_k$1_rand"; do
    jn=eval_k$1; [ "$ck" = "frozen_k$1_rand" ] && jn=eval_k$1_rand
    if [ $1 -le 1024 ]; then
      ex=""; [ "$ck" = "$2" ] && ex="--explain 5"
      sbatch -J $jn jobs/gpu.sbatch 08_eval_concept_model.py --ckpt $ck --query-sets dev dl19 dl20 $ex
    else
      sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -J $jn jobs/gpu.sbatch \
             08b_eval_streaming.py --ckpt $ck --query-sets dev dl19 dl20
    fi
  done
done
grep -h "MRR@10=.*nDCG@10" logs/eval_k*.out
```

`--explain N` writes `concepts_<name>.tsv` (top tokens of each concept, in the folded vocabulary) and N interventions
(for k ≤ 1024 only, with the trained W).

---

## 3. End-to-end variant: W and the SparseUp encoder

Same loss, negatives and protocol, plus `--mode e2e` and a second learning rate for the encoder (`--lr-backbone`; in
the run names it is `bb`). Batch size 32, one epoch (~15,700 steps), bf16 backbone with the MLM head in float32,
gradient checkpointing. ModernBERT (149M) with a 50k-token head is heavier than BERT: ~3–5 h per run.

### 3.1 Grid on the encoder learning rate (1 epoch)

The lr of W is fixed to the best SPLADE value (it mattered much less than the encoder lr in the SPLADE project):
1e-4 for k = 256, 512; 3e-5 for k ≥ 1024. Two waves: init-std = 1/√k (`scaled`) and init-std = 0.02. Ties (differences
< 0.001) go to the lower encoder lr; if the winner is at the edge of the grid, the grid is extended (the k = 4096 and
8192 grids went down to 1e-7 and 5e-8).

```bash
e2e() { sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -J "$1" jobs/gpu.sbatch 06_train_concept_model.py \
        --mode e2e --run-name "$1" --negatives sparseup --n-hard 7 --neg-rank-min 3 --batch-size 32 --epochs 1 \
        --val-every 2000 --val-queries 1000 --grad-checkpointing "${@:2}"; }
G=(
"256 0.0625 is0625 1e-4 5e-6 1e-5 2e-5"
"512 0.0442 is0442 1e-4 5e-6 1e-5 2e-5"
"1024 0.03125 is03125 3e-5 2e-6 5e-6 1e-5"
"2048 0.0221 is0221 3e-5 2e-6 5e-6 1e-5"
"4096 0.0156 is0156 3e-5 5e-7 1e-6 2e-6 5e-6"
"8192 0.0110 is0110 3e-5 5e-8 1e-7 2e-7 1e-6"
)
for wave in scaled 0.02; do
  for g in "${G[@]}"; do
    set -- $g; k=$1; sc=$2; tag=$3; lrw=$4; shift 4
    if [ $wave = scaled ]; then init=$sc; t="_$tag"; else init=0.02; t=""; fi
    for bb in "$@"; do
      e2e e2e_k${k}_r2${t}_lr${lrw/-/}_bb${bb/-/}_b32_ep1 --k $k --init-std $init --lr $lrw --lr-backbone $bb
    done
  done
done
grep -H "Best validation" logs/e2e_k*_ep1_*.out
```

Extensions launched after the first grid (same command, other `--lr-backbone` values, only in the cells where the trend
made them useful): k = 256: 5e-5, 1e-4; k = 512: 2e-6, 5e-5, 1e-4; k = 1024: 2e-5, 5e-5; k = 2048: 2e-5, 5e-5;
k = 4096: 5e-8, 1e-7, 2e-7, 1e-5, 2e-5, 5e-5, 1e-4; k = 8192: 5e-7.

Winners:

| k | run | init-std | lr W | lr encoder | val MRR@10 |
|---|---|---|---|---|---|
| 256 | `e2e_k256_r2_lr1e4_bb2e5_b32_ep1` | 0.02 | 1e-4 | 2e-5 | 0.3171 |
| 512 | `e2e_k512_r2_lr1e4_bb5e6_b32_ep1` | 0.02 | 1e-4 | 5e-6 | 0.3349 |
| 1024 | `e2e_k1024_r2_lr3e5_bb2e5_b32_ep1` | 0.02 | 3e-5 | 2e-5 | 0.3435 |
| 2048 | `e2e_k2048_r2_lr3e5_bb5e6_b32_ep1` | 0.02 | 3e-5 | 5e-6 | 0.3442 |
| 4096 | `e2e_k4096_r2_is0156_lr3e5_bb2e7_b32_ep1` | 0.0156 | 3e-5 | 2e-7 | 0.3465 |
| 8192 | `e2e_k8192_r2_lr3e5_bb1e7_b32_ep1` | 0.02 | 3e-5 | 1e-7 | 0.3472 |

### 3.2 Controls

- **e2e with W fixed** (k = 1024): same command as the winner with `--lr 1e-9` (W stays at its random initialisation,
  only the encoder is trained):

```bash
e2e e2e_k1024_r2_Wfix_bb2e5_b32_ep1 --k 1024 --init-std 0.02 --lr 1e-9 --lr-backbone 2e-5      # val 0.3473
```

- **1 vs 2 epochs** for k ≤ 1024 and for W fixed (`--epochs 2`, run names ending in `_ep2`): **not run yet.** All e2e
  runs use 1 epoch (k ≥ 2048 are fixed to 1 epoch, as in the SPLADE project).

### 3.3 Evaluation: the whole corpus is re-encoded with the fine-tuned encoder (~2–3 h each)

```bash
for r in e2e_k256_r2_lr1e4_bb2e5_b32_ep1 e2e_k512_r2_lr1e4_bb5e6_b32_ep1 e2e_k1024_r2_lr3e5_bb2e5_b32_ep1 \
         e2e_k2048_r2_lr3e5_bb5e6_b32_ep1 e2e_k4096_r2_is0156_lr3e5_bb2e7_b32_ep1 e2e_k8192_r2_lr3e5_bb1e7_b32_ep1 \
         e2e_k1024_r2_Wfix_bb2e5_b32_ep1; do
  sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --time=12:00:00 -J eval_${r%%_r2*} \
         jobs/gpu.sbatch 08c_eval_e2e_streaming.py --ckpt $r --query-sets dev dl19 dl20
done
grep -hE "\| (dev|dl19|dl20)\]" logs/eval_e2e_k*.out
```

It saves the sparse shards of the new encoder in `embeddings/<run>_best_sparse/` (resumable, 5.5–12.7 GB), which `11` and
the hybrids reuse; their sizes are the "e2e index" sizes of the space table in `RESULTS.md` 6.

---

## 4. Out of domain: BEIR

```bash
python -u src/10_eval_beir.py --datasets nfcorpus scifact fiqa --download-only
# SparseUp + random W + trained frozen W
sbatch --time=04:00:00 -J beir_frozen jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa --sparseup --ckpts \
  frozen_k256_rand frozen_k256_r2_lr1e4_ep2 frozen_k512_rand frozen_k512_r2_lr3e4_ep2 \
  frozen_k1024_rand frozen_k1024_r2_lr3e5_ep5 frozen_k2048_rand frozen_k2048_r2_lr1e5_ep1 \
  frozen_k4096_rand frozen_k4096_r2_lr3e4_ep2 frozen_k8192_rand frozen_k8192_r2_is0110_lr1e4_ep1
# e2e: one job per checkpoint, the W-fixed one included
for r in e2e_k256_r2_lr1e4_bb2e5_b32_ep1 e2e_k512_r2_lr1e4_bb5e6_b32_ep1 e2e_k1024_r2_lr3e5_bb2e5_b32_ep1 \
         e2e_k2048_r2_lr3e5_bb5e6_b32_ep1 e2e_k4096_r2_is0156_lr3e5_bb2e7_b32_ep1 e2e_k8192_r2_lr3e5_bb1e7_b32_ep1 \
         e2e_k1024_r2_Wfix_bb2e5_b32_ep1; do
  sbatch --time=02:00:00 -J beir_${r%%_r2*} jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa --ckpts $r
done
grep -h "nDCG@10" logs/beir_frozen_*.out ; grep -h "| beir_" logs/beir_e2e_k*.out
```

The BEIR zips come from the same UKP server as before; if it is down, the script takes the same files from the
Hugging Face mirrors `BeIR/<name>` and `BeIR/<name>-qrels`. Caveat for the paper: SparseUp itself was fine-tuned on a
mixture that includes the training splits of some BEIR datasets, so "zero-shot" refers to the Concept Model.

---

## 5. Hybrids

Score `(1-λ)·SparseUp(q)·SparseUp(d) + λ·g(q)·g(d)`. One job per checkpoint: exact search on the 1,000 validation
queries for 21 values of λ (0, 0.05, …, 1) → λ\* = best MRR@10, then dev / DL19 / DL20 with λ\*, plus the sanity
checks λ = 0 (must reproduce `04`) and λ = 1 (must reproduce the Concept Model alone). For the e2e checkpoints `08c`
(step 3.3) must have been run first. BEIR uses the λ\* found on MS MARCO, without re-tuning.

### 5.1 Frozen and e2e (`hyb`)

```bash
H="sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --time=04:00:00"
$H -J hyb_frozen_k256  jobs/gpu.sbatch 11_eval_hybrid.py --ckpt frozen_k256_r2_lr1e4_ep2 --query-sets dev dl19 dl20
$H -J hyb_frozen_k512  jobs/gpu.sbatch 11_eval_hybrid.py --ckpt frozen_k512_r2_lr3e4_ep2 --query-sets dev dl19 dl20
$H -J hyb_frozen_k1024 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt frozen_k1024_r2_lr3e5_ep5 --query-sets dev dl19 dl20
$H -J hyb_frozen_k2048 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt frozen_k2048_r2_lr1e5_ep1 --query-sets dev dl19 dl20
$H -J hyb_frozen_k4096 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt frozen_k4096_r2_lr3e4_ep2 --query-sets dev dl19 dl20
$H -J hyb_frozen_k8192 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt frozen_k8192_r2_is0110_lr1e4_ep1 --query-sets dev dl19 dl20
$H -J hyb_e2e_k256  jobs/gpu.sbatch 11_eval_hybrid.py --ckpt e2e_k256_r2_lr1e4_bb2e5_b32_ep1 --query-sets dev dl19 dl20
$H -J hyb_e2e_k512  jobs/gpu.sbatch 11_eval_hybrid.py --ckpt e2e_k512_r2_lr1e4_bb5e6_b32_ep1 --query-sets dev dl19 dl20
$H -J hyb_e2e_k1024 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt e2e_k1024_r2_lr3e5_bb2e5_b32_ep1 --query-sets dev dl19 dl20
$H -J hyb_e2e_k2048 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt e2e_k2048_r2_lr3e5_bb5e6_b32_ep1 --query-sets dev dl19 dl20
$H -J hyb_e2e_k4096 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt e2e_k4096_r2_is0156_lr3e5_bb2e7_b32_ep1 --query-sets dev dl19 dl20
$H -J hyb_e2e_k8192 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt e2e_k8192_r2_lr3e5_bb1e7_b32_ep1 --query-sets dev dl19 dl20
```

λ\* found (frozen k = 256 … 8192): 0.60, 0.65, 0.15, 0.15, 0.75, 0.70; (e2e): 0.80, 0.85, 0.90, 0.80, 0.50, 0.20.

BEIR (the part after the colon is `variant:λ*`; `two` = `hyb`):

```bash
sbatch --time=02:00:00 -J beir_hyb_frozen_k256 jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --hybrid frozen_k256_r2_lr1e4_ep2:two:0.60
sbatch --time=02:00:00 -J beir_hyb_e2e_k2048 jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --hybrid e2e_k2048_r2_lr3e5_bb5e6_b32_ep1:two:0.80
# ... same pattern for every checkpoint and its λ*
```

### 5.2 Variants of the e2e hybrid

| Name in the results | `--variant` | Score | Indexes |
|---|---|---|---|
| `hyb_…` | `two` (default) | `(1-λ)·SU·SU + λ·g·g` | 2 |
| `hyb1_…` | `single` | `(1-λ)·SU_θ·SU_θ + λ·g·g` | 1 |
| `hybnoW_…` | `now` | `(1-λ)·SU·SU + λ·SU_θ·SU_θ` (no W) | 2 |
| `ftsparseup_…` | (`single` with λ = 0, `10_eval_beir.py --ft-sparseup`) | `SU_θ·SU_θ` | 1 |

SU = original SparseUp, SU_θ = the encoder re-trained through the Concept Model, used directly as a SparseUp (its
sparse vectors are those saved by `08c`). `ftsparseup` is the test of whether the re-trained encoder alone is already a
better SparseUp; on dev it is `hyb1` at λ = 0.

```bash
# in domain: validation of λ, dev, DL19, DL20 (one job per checkpoint and variant)
$H -J hyb1_e2e_k256   jobs/gpu.sbatch 11_eval_hybrid.py --ckpt e2e_k256_r2_lr1e4_bb2e5_b32_ep1 --variant single --query-sets dev dl19 dl20
$H -J hybnoW_e2e_k256 jobs/gpu.sbatch 11_eval_hybrid.py --ckpt e2e_k256_r2_lr1e4_bb2e5_b32_ep1 --variant now    --query-sets dev dl19 dl20
# ... same pattern for k = 512 … 8192 (6 runs per variant, one per e2e checkpoint of 3.1)
```

λ\* found: `hyb1` (k = 256 … 8192) 0.65, 0.40, 0.30, 0.50, 0.20, **0.00**; `hybnoW` 0.75, 0.70, 0.75, 0.90, 0.90, 0.70.
For `hyb1` k = 8192 λ\* = 0, so it coincides with `ftsparseup`.

```bash
# out of domain
sbatch --time=02:00:00 -J beir_ftsparseup_k8192 jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --ft-sparseup e2e_k8192_r2_lr3e5_bb1e7_b32_ep1
sbatch --time=02:00:00 -J beir_hyb1_e2e_k1024 jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --hybrid e2e_k1024_r2_lr3e5_bb2e5_b32_ep1:single:0.30
sbatch --time=02:00:00 -J beir_hybnoW_e2e_k1024 jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
       --hybrid e2e_k1024_r2_lr3e5_bb2e5_b32_ep1:now:0.75
# ... same pattern for the other k (hyb1 at k = 8192 is ftsparseup: no separate run)
```

---

## 6. Significance tests

`12_significance.py --a A --b B` reads `results/A.json` and `results/B.json` (per-query metrics, names without
`.json`), and computes the paired bootstrap (10,000 resamples) CI of B − A, a paired t-test and a permutation test
(the reported p). **Significant if p < 0.05 and the CI does not contain zero** (both ends with the same sign).
Bonferroni thresholds: 12 tests → 0.0042; 6 tests → 0.0083. ~4 s per test; run on the CPU partition. The name of a
run's file is `<method>_<run>_best_lam<λ*>_<set>` (e.g. `hyb_e2e_k2048_r2_lr3e5_bb5e6_b32_ep1_best_lam0.80_dev`);
on BEIR the set is `beir_<dataset>` and the baseline is `sparseup_baseline_beir_beir_<dataset>`. Use the literal
baseline name (`sparseup_baseline_dev`), not a shell variable: an empty variable gives `--a: expected one argument`.

**In domain (dev)** — each block is run for every k = 256 … 8192:

| Block | A (reference) | B (system) | Tests | Question |
|---|---|---|---|---|
| 1 | `sparseup_baseline_dev` | `hyb_e2e_k*_…_lam<λ*>_dev` | 6 | does the hybrid with W beat SparseUp? |
| 2 | `sparseup_baseline_dev` | `hybnoW_e2e_k*_…_lam<λ*>_dev` | 6 | does the hybrid without W beat SparseUp? |
| 3 | `hyb_e2e_k*_…_dev` | `hybnoW_e2e_k*_…_dev` (same k) | 6 | does W add anything at search time? |
| 4 | `sparseup_baseline_dev` | `hyb1_e2e_k*_…_lam<λ*>_dev` (for k = 8192, `…_lam0.00_dev`, the same system as `ftsparseup`) | 6 | is the re-trained encoder alone as good as SparseUp? |

```bash
sbatch --time=00:15:00 -J sig_hyb_k2048 jobs/cpu.sbatch 12_significance.py --a sparseup_baseline_dev \
       --b hyb_e2e_k2048_r2_lr3e5_bb5e6_b32_ep1_best_lam0.80_dev
sbatch --time=00:15:00 -J sig_hybnoW_vs_sparseup_k1024 jobs/cpu.sbatch 12_significance.py --a sparseup_baseline_dev \
       --b hybnoW_e2e_k1024_r2_lr3e5_bb2e5_b32_ep1_best_lam0.75_dev
sbatch --time=00:15:00 -J sig_noW_vs_hyb_k1024 jobs/cpu.sbatch 12_significance.py \
       --a hyb_e2e_k1024_r2_lr3e5_bb2e5_b32_ep1_best_lam0.90_dev \
       --b hybnoW_e2e_k1024_r2_lr3e5_bb2e5_b32_ep1_best_lam0.75_dev
sbatch --time=00:15:00 -J sig_hyb1_k8192 jobs/cpu.sbatch 12_significance.py --a sparseup_baseline_dev \
       --b hyb1_e2e_k8192_r2_lr3e5_bb1e7_b32_ep1_best_lam0.00_dev
# ... same pattern for every k
```

**Out of domain** (9 tests): the three headline systems (`hybnoW` k = 1024, `hyb` k = 2048, `ftsparseup` k = 8192)
against SparseUp on each BEIR dataset; the script reports MRR@10, nDCG@10 and R@1000 in one run.

```bash
sbatch --time=00:15:00 -J sig_beir_nfc_hybnoW1024 jobs/cpu.sbatch 12_significance.py \
       --a sparseup_baseline_beir_beir_nfcorpus \
       --b hybnoW_e2e_k1024_r2_lr3e5_bb2e5_b32_ep1_best_lam0.75_beir_nfcorpus
sbatch --time=00:15:00 -J sig_beir_fiqa_ft8192 jobs/cpu.sbatch 12_significance.py \
       --a sparseup_baseline_beir_beir_fiqa \
       --b ftsparseup_e2e_k8192_r2_lr3e5_bb1e7_b32_ep1_best_beir_fiqa
# ... same pattern for scifact / fiqa and for hyb k = 2048 (…_lam0.80_beir_<dataset>)
```

(`ftsparseup` has no file for dev: use the `hyb1 … lam0.00` one; a missing file gives `FileNotFoundError`.)

---

## 7. Figures

```bash
python plots/concept_model_plots.py       # writes plots/figures/fig1…fig4 (.png and .pdf)
```

Fill in the dictionaries at the top of the script with the numbers of `RESULTS.md` (missing values are skipped; set
`REF_DEV = 0.3674`, `REF_BEIR = 0.5205`, `REF_GB = 13.4`).

---

## 8. Design choices to report

- **Exact search instead of ANN** (the SparseUp paper uses Seismic, approximate): every number is an upper bound for any
  approximate index.
- **Reference vectors from the official code**: the corpus is encoded by `modeling_splade.py` itself (float32); our
  memory-efficient re-implementation is checked against it (step 1.5) before being used for anything.
- **Memory-efficient SparseUp pooling** (`sparseup_encoder.py`): top-k taken on the logits, then a single
  `scatter_reduce(amax)` over positions *and* fold groups, and only then `log(1 + ReLU(x − 15))`. Same result as the
  official code (the function is monotone), without the (B, L, 50k) activation tensor.
- **Folded space**: `W` is k × C with C = 34,023 canonical tokens (dimensions that SparseUp can never activate are dropped).
- **Clean validation**: the same 1,000 held-out training queries as the SPLADE project; checkpoints, hyper-parameters
  and λ are chosen on them, never on dev or BEIR. The *choice of which k to show* as headline system is not made on
  validation (declare it).
- **False negatives**: the other known positives of the same query are masked out of the loss; the top 3 mining
  candidates are discarded (`--neg-rank-min 3`).
- **Space accounting**: sparse index = 8 bytes per non-zero entry (int32 + float32); dense `g(d)` = 2k bytes (fp16);
  1 GB = 10⁹ bytes. The index sizes of the re-trained encoders come from the `08c` logs.
- **One seed** (42) per configuration.

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
| `12_significance.py: --a: expected one argument` | the shell variable holding the name is empty: write the file name literally |
| `FileNotFoundError … _dev.json` in `12` | the result file has another name: `ls $SPARSEUP_WORK_DIR/results/ \| grep <run>` (e.g. `ftsparseup` on dev is `hyb1_…_lam0.00_dev`) |
| results folder not found | it is `$SPARSEUP_WORK_DIR/results/` (default `/datafast/$USER/sparseup_concept/results/`), not the code folder |
