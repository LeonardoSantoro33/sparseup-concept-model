"""
02b_encode_msmarco.py — Encode MS MARCO with the OFFICIAL SparseUp implementation -> the pre-encoded files (FM_DIR).

In the SPLADE project the SPLADE vectors of the 8.8M passages and of the 6,980 dev queries were PROVIDED to us in
Seismic format. For SparseUp nothing is provided, so this script creates THE SAME SIX FILES, with the same names
and the same format, so that every following step (01, 03, 04, ...) is identical to the SPLADE project:

    documents.bin             8,841,823 passages of collection.tsv, encoded with SparseUp ("[D] " prefix, max 512 tokens)
    doc_ids.npy               pid of each row of documents.bin
    queries.bin               the 6,980 queries of queries.dev.small.tsv ("[Q] " prefix, max 128 tokens)
    queries_ids.npy           qid of each row of queries.bin
    token_to_id_mapping.json  token -> Seismic id (tokens sorted alphabetically, as in the SPLADE files);
                              the tokens are the CANONICAL tokens of SparseUp's folded vocabulary (~34k)
    groundtruth_100.tsv       for each dev query, the exact top-100 documents with their score
                              (columns query_row, doc_row, rank, score; no header)

The vectors are produced by the official code (model.encode of modeling_splade.py, trust_remote_code), in float32
with TF32 disabled: they are the REFERENCE vectors. Script 03 then checks that our memory-efficient encoder
(sparseup_encoder.py, used for training queries, DL queries, BEIR and the e2e variant) gives the same vectors.
The groundtruth is computed with torch.sparse (cuSPARSE), a different code path from the embedding_bag search of
retrieval.py: 04 compares the two (overlap ~1.00), exactly as it did with the groundtruth provided for SPLADE.

Run it after 02_download_msmarco.py (it needs collection.tsv and queries.dev.small.tsv). The documents are encoded in
shards of 500k (FM_DIR/_shards): if the job is interrupted, it resumes where it left off.
Time (float32): ~4-6 h on an A100 80GB, ~2-3 h on an H100 (--precision tf32 is ~2x faster, slightly different values).
    sbatch --time=2-00:00:00 -J enc_msmarco jobs/gpu.sbatch src/02b_encode_msmarco.py
For a quick test (first 100k passages, nothing used later):
    python -u src/02b_encode_msmarco.py --max-docs 100000 --out-dir /tmp/$USER/fm_test
"""
import argparse
import json
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_SPARSEUP_MODEL, Paths
from msmarco import Collection, read_queries
from sparse_io import CSR, SeismicWriter, write_seismic_bin
from sparseup_encoder import FoldedVocab, full_fp32, load_sparseup_model
from utils import get_device, human_bytes, log, log_run_header, make_transformers_cpu_safe, save_json, timer

make_transformers_cpu_safe()
from transformers import AutoTokenizer  # noqa: E402


class OfficialEncoder:
    """model.encode of the official implementation, with batches of texts of similar TOKEN length capped by a token
    budget (the official forward builds several (B, L, 50k) float32 tensors) and conversion to Seismic ids."""

    def __init__(self, model, tokenizer, remap, n_cols, batch_size, max_batch_tokens, precision):
        self.model, self.tokenizer, self.remap, self.n_cols = model, tokenizer, remap, n_cols
        self.batch_size, self.max_batch_tokens, self.precision = batch_size, max_batch_tokens, precision
        cfg = model.config
        self.setting = {"query": (getattr(cfg, "query_prefix", "") or "", int(getattr(cfg, "query_max_length", 128))),
                        "document": (getattr(cfg, "document_prefix", "") or "",
                                     int(getattr(cfg, "doc_max_length", 512)))}

    def token_lengths(self, texts, kind, chunk=20_000):
        """Exact number of tokens of each text as the official code sees it (prefix included, truncated)."""
        prefix, max_len = self.setting[kind]
        out = np.empty(len(texts), dtype=np.int64)
        for s in range(0, len(texts), chunk):
            ids = self.tokenizer([prefix + t for t in texts[s:s + chunk]], truncation=True, max_length=max_len,
                                 return_attention_mask=False)["input_ids"]
            out[s:s + len(ids)] = [len(x) for x in ids]
        return out

    def batches(self, texts, kind):
        lens = self.token_lengths(texts, kind)
        order = np.argsort(lens, kind="stable")
        s = 0
        while s < len(order):
            longest = int(lens[order[min(len(order) - 1, s + self.batch_size - 1)]])
            b = int(max(1, min(self.batch_size, self.max_batch_tokens // max(1, longest))))
            yield order[s:s + b]
            s += b

    @torch.no_grad()
    def __call__(self, texts, kind, verbose=True):
        texts = list(texts)
        parts, pos = [], []
        ctx = full_fp32() if self.precision == "fp32" else nullcontext()
        done, step = 0, max(1, len(texts) // 20)
        with ctx:
            for idx in self.batches(texts, kind):
                x = self.model.encode([texts[i] for i in idx], kind=kind, batch_size=len(idx))   # (b, V) float32
                r, c = torch.nonzero(x > 0, as_tuple=True)
                cols = self.remap[c.numpy()]
                if (cols < 0).any():
                    raise SystemExit("[ERROR] the model produced a weight on a non-canonical token id: "
                                     "the folding map does not match (unexpected model version?)")
                counts = np.bincount(r.numpy(), minlength=len(idx))
                indptr = np.zeros(len(idx) + 1, dtype=np.int64)
                np.cumsum(counts, out=indptr[1:])
                parts.append(CSR(indptr, cols.astype(np.int32), x[r, c].numpy().astype(np.float32), self.n_cols))
                pos.append(idx)
                if verbose and (done // step) != ((done + len(idx)) // step):
                    log(f"  encoded {done + len(idx):,}/{len(texts):,} texts")
                done += len(idx)
        csr = CSR.concat(parts)
        inverse = np.empty(len(texts), dtype=np.int64)
        inverse[np.concatenate(pos)] = np.arange(len(texts))
        return csr.subset(inverse)


@torch.no_grad()
def exact_topk_torch_sparse(shard_dirs, q, n_cols, k, dev, q_batch=512):
    """Exact top-k of the queries (CSR) over the documents (CSR shards), with torch.sparse CSR @ dense.
    Independent from retrieval.py (embedding_bag). Returns scores (nq, k), rows (nq, k)."""
    nq = q.n_rows
    Q = torch.from_numpy(q.to_dense()).to(dev)                        # (nq, n_cols)
    best_s = torch.full((nq, k), float("-inf"), device=dev)
    best_i = torch.full((nq, k), -1, dtype=torch.int64, device=dev)
    off = 0
    for d in shard_dirs:
        c = CSR.load(d)
        sp = torch.sparse_csr_tensor(torch.from_numpy(np.asarray(c.indptr, dtype=np.int64)),
                                     torch.from_numpy(np.asarray(c.indices, dtype=np.int64)),
                                     torch.from_numpy(np.asarray(c.data, dtype=np.float32)),
                                     size=(c.n_rows, n_cols)).to(dev)
        for s in range(0, nq, q_batch):
            e = min(nq, s + q_batch)
            S = (sp @ Q[s:e].T.contiguous()).T                          # (b, n_docs_shard)
            kk = min(k, S.shape[1])
            sc, ix = torch.topk(S, kk, dim=1)
            cat_s = torch.cat([best_s[s:e], sc], 1)
            cat_i = torch.cat([best_i[s:e], ix + off], 1)
            top_s, p = torch.topk(cat_s, k, dim=1)
            best_s[s:e], best_i[s:e] = top_s, torch.gather(cat_i, 1, p)
            del S
        off += c.n_rows
        del sp
    return best_s.cpu().numpy(), best_i.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_SPARSEUP_MODEL)
    ap.add_argument("--out-dir", default=None, help="default: FM_DIR (WORK_DIR/encoded)")
    ap.add_argument("--batch-size", type=int, default=256, help="max texts per batch")
    ap.add_argument("--max-batch-tokens", type=int, default=32768,
                    help="max tokens (batch x padded length) per batch: the official forward builds (tokens x 50k) "
                         "float32 tensors (~5 of them at the peak: 32768 tokens -> ~35 GB)")
    ap.add_argument("--shard-size", type=int, default=500_000, help="documents per resumable shard")
    ap.add_argument("--precision", choices=["fp32", "tf32"], default="fp32",
                    help="fp32 = exact float32 as the released model (TF32 off); tf32 = faster, slightly different")
    ap.add_argument("--max-docs", type=int, default=0, help="encode only the first N passages (test)")
    ap.add_argument("--gt-depth", type=int, default=100)
    ap.add_argument("--skip-groundtruth", action="store_true")
    ap.add_argument("--keep-shards", action="store_true", help="do not delete FM_DIR/_shards at the end")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    out = Path(args.out_dir) if args.out_dir else paths.fm
    out.mkdir(parents=True, exist_ok=True)
    dev = get_device()
    log_run_header("02b_encode_msmarco.py", args, dev)
    if args.precision == "fp32":
        log("precision fp32: TF32 disabled during encoding (same numbers as the released float32 model)")

    # ------------------------------------------------------------------ model and folded vocabulary
    model = load_sparseup_model(args.model, dev)
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = model.config
    log(f"SparseUp config: logit_shift={getattr(cfg, 'logit_shift', None)} position_top_k="
        f"{getattr(cfg, 'position_top_k', None)} vocab_fold={getattr(cfg, 'vocab_fold', None)} "
        f"prefixes={getattr(cfg, 'query_prefix', '')!r}/{getattr(cfg, 'document_prefix', '')!r} "
        f"max lengths {getattr(cfg, 'query_max_length', None)}/{getattr(cfg, 'doc_max_length', None)}")
    fv = FoldedVocab.from_model(model, tok, args.model)
    fv.save(paths.folded_vocab)
    log(f"native vocabulary {cfg.vocab_size:,} -> folded vocabulary {len(fv):,} canonical tokens "
        f"(saved to {paths.folded_vocab})")
    sorted_toks = sorted(fv.tokens)
    mapping = {t: i for i, t in enumerate(sorted_toks)}
    remap = np.full(int(cfg.vocab_size), -1, dtype=np.int64)          # native token id -> Seismic id
    remap[fv.canonical_ids] = [mapping[t] for t in fv.tokens]
    (out / "token_to_id_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False))
    n_cols = len(sorted_toks)
    enc = OfficialEncoder(model, tok, remap, n_cols, args.batch_size, args.max_batch_tokens, args.precision)

    # ------------------------------------------------------------------ dev queries
    dev_texts = read_queries(paths.queries_tsv("dev"))
    qids = np.array(list(dev_texts), dtype=np.int64)                 # file order
    if (out / "queries.bin").exists() and (out / "queries_ids.npy").exists():
        log("queries.bin already present")
        from sparse_io import read_seismic_bin
        qcsr = read_seismic_bin(out / "queries.bin", verbose=False)
        qcsr.n_cols = n_cols
        qids = np.load(out / "queries_ids.npy")
    else:
        with timer(f"encoding the {len(qids):,} dev queries"):
            qcsr = enc([dev_texts[int(q)] for q in qids], "query", verbose=False)
        write_seismic_bin(out / "queries.bin", qcsr)
        np.save(out / "queries_ids.npy", qids)
        log(f"queries.bin: mean {qcsr.nnz / max(1, qcsr.n_rows):.1f} non-zero entries per query "
            "(the SparseUp blog reports ~47 on MS MARCO)")

    # ------------------------------------------------------------------ documents (resumable shards)
    collection = Collection(paths.collection_tsv)
    n = len(collection) if not args.max_docs else min(args.max_docs, len(collection))
    pids = np.arange(n, dtype=np.int64)
    if not collection.pid_is_line:
        pids = np.array([collection._line(i)[0] for i in range(n)], dtype=np.int64)
    shard_root = out / "_shards"
    shard_root.mkdir(exist_ok=True)
    meta = {"model": args.model, "n_docs": int(n), "shard_size": int(args.shard_size), "precision": args.precision,
            "max_length": int(getattr(cfg, "doc_max_length", 512)), "revision": getattr(cfg, "_commit_hash", None)}
    meta_path = shard_root / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"[ERROR] {shard_root} contains shards made with different settings: delete it first")
    meta_path.write_text(json.dumps(meta, indent=2))
    shard_dirs = []
    for si, s in enumerate(range(0, n, args.shard_size)):
        e = min(n, s + args.shard_size)
        folder = shard_root / f"shard_{si:04d}"
        shard_dirs.append(folder)
        if (folder / "meta.json").exists():
            continue
        with timer(f"encoding passages {s:,}-{e:,} of {n:,}"):
            csr = enc(collection.get_many(pids[s:e]), "document")
            csr.ids = pids[s:e]
        tmp = folder.with_name(f"{folder.name}.tmp{os.getpid()}")
        csr.save(tmp)
        tmp.rename(folder)
        log(f"  shard {si}: mean {csr.nnz / max(1, csr.n_rows):.1f} non-zero entries per passage "
            "(the SparseUp blog reports ~190 on MS MARCO)")

    with timer("writing documents.bin (streaming)"):
        with SeismicWriter(out / "documents.bin") as w:
            for d in shard_dirs:
                w.append(CSR.load(d))
        np.save(out / "doc_ids.npy", pids)
    log(f"documents.bin: {human_bytes((out / 'documents.bin').stat().st_size)}")

    # ------------------------------------------------------------------ groundtruth (independent exact search)
    if not args.skip_groundtruth:
        enc.model = None
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        with timer(f"groundtruth: exact top-{args.gt_depth} of the dev queries with torch.sparse"):
            scores, rows = exact_topk_torch_sparse(shard_dirs, qcsr, n_cols, min(args.gt_depth, n), dev)
        with open(out / "groundtruth_100.tsv", "w") as f:
            for qi in range(rows.shape[0]):
                for r in range(rows.shape[1]):
                    if rows[qi, r] >= 0:
                        f.write(f"{qi}\t{rows[qi, r]}\t{r}\t{scores[qi, r]:.6f}\n")
        log(f"groundtruth_100.tsv written ({rows.shape[0]:,} queries x {rows.shape[1]})")

    save_json({**meta, "n_queries": int(len(qids)), "n_cols_seismic": n_cols,
               "files": sorted(p.name for p in out.iterdir() if p.is_file())}, out / "encoding_info.json")
    if not args.keep_shards and not args.max_docs:
        shutil.rmtree(shard_root, ignore_errors=True)
        log(f"shards deleted (use --keep-shards to keep them): the vectors are in {out / 'documents.bin'}")
    log(f"Done. Pre-encoded files in {out}. Next: 01_prepare_data.py")


if __name__ == "__main__":
    main()
