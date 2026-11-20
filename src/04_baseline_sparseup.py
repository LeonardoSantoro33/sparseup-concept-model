"""
04_baseline_sparseup.py — Baseline: "plain" SparseUp with exact search over the full MS MARCO corpus.

This is the reference point for the whole project: every variant (Concept Model, LCE, hybrids) must be compared
against these numbers. SparseUp is NOT trained by distillation on MS MARCO (contrastive training on LightOn's mixture),
so it is expected to be weaker than SPLADE++ in domain: the SparseUp blog reports BEIR-13 averages of 55.4 with
MS MARCO and 56.4 without it, i.e. MS MARCO dev nDCG@10 ~ 0.42 (+-0.015, with approximate Seismic search; exact search
can only be equal or higher). For comparison, SPLADE ensembledistil: MRR@10 0.3828, nDCG@10 0.4489, R@1000 0.9831.

Automatic checks:
  - comparison with groundtruth_100.tsv (exact top-100 lists computed by 02b with torch.sparse, a different code path):
    the overlap must be ~1.00, otherwise there is a problem in the data or in the code.

    sbatch -J baseline jobs/gpu.sbatch src/04_baseline_sparseup.py --query-sets dev dl19 dl20
"""
import argparse

import numpy as np

from config import Paths
from metrics import overlap_at_k
from pipeline import docs_to_device, load_docs, report_and_save, sparseup_query_vectors
from retrieval import sparse_search
from sparse_io import read_groundtruth
from utils import get_device, log, log_run_header, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-sets", nargs="+", default=["dev"], choices=["dev", "dl19", "dl20"])
    ap.add_argument("--k", type=int, default=1000, help="number of documents to retrieve per query")
    ap.add_argument("--q-batch", type=int, default=256)
    ap.add_argument("--doc-chunk", type=int, default=1_000_000)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("04_baseline_sparseup.py", args, dev)

    docs = load_docs(paths)
    doc_pids = np.asarray(docs.ids)
    tdocs = docs_to_device(docs, dev)

    for set_name in args.query_sets:
        q, qrels, thr, _ = sparseup_query_vectors(set_name, paths, dev)
        with timer(f"exact SparseUp search on '{set_name}' ({q.n_rows:,} queries)"):
            scores, rows = sparse_search(tdocs, q, args.k, q_batch=args.q_batch, doc_chunk=args.doc_chunk)
        extra = {"avg_query_nnz": float(q.nnz / max(1, q.n_rows))}

        if set_name == "dev" and paths.fm_file("groundtruth").exists():
            gt = read_groundtruth(paths.fm_file("groundtruth"))
            gt = gt.sort_values(["q", "rank"])
            k_gt = int(gt.groupby("q").size().max())
            gt_rows = np.full((q.n_rows, k_gt), -1, dtype=np.int64)
            for qi, grp in gt.groupby("q"):
                if qi < q.n_rows:
                    gt_rows[qi, :len(grp)] = grp["d"].to_numpy()
            kk = min(k_gt, rows.shape[1])
            ov = overlap_at_k(rows, gt_rows, kk)
            # score ties can swap the order: we also compare the top-kk sets
            log(f"Overlap with groundtruth_{k_gt} of 02b (top-{kk}): {ov:.4f}  (expected ~1.00)")
            extra["overlap_groundtruth"] = ov

        report_and_save(paths, "sparseup_baseline", set_name, np.asarray(q.ids), rows, scores, doc_pids,
                        qrels, thr, extra=extra)
    log("Done.")


if __name__ == "__main__":
    main()
