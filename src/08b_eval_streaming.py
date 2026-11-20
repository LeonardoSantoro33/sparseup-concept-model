"""
08b_eval_streaming.py — Official evaluation of a FROZEN checkpoint with the streaming trick (memory independent of k):
    g(q) . g(d) = (W g(q)) . SparseUp(d)
so the dense matrix g(d) of the 8.8M documents is never built (needed for k >= 2048). Same numbers as 08 up to fp16
rounding (08 stores g(d) in fp16).

    sbatch -J eval_stream_k2048 jobs/gpu.sbatch src/08b_eval_streaming.py --ckpt frozen_k2048_r2_is0221_lr3e5_ep5 \
        --query-sets dev dl19 dl20
"""
import argparse
import numpy as np
from concept_model import load_checkpoint
from config import Paths, resolve_ckpt
from pipeline import concept_vectors_from_csr, docs_to_device, load_docs, report_and_save, sparseup_query_vectors
from retrieval import search_with_query_table
from utils import get_device, log, log_run_header, make_transformers_cpu_safe

make_transformers_cpu_safe()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run name (uses .../best) or path")
    ap.add_argument("--query-sets", nargs="+", default=["dev"], choices=["dev", "dl19", "dl20"])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--name", default=None, help="method name in the results")
    ap.add_argument("--q-batch", type=int, default=256)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("08b_eval_streaming.py", args, dev)
    ckpt = resolve_ckpt(args.ckpt, paths)
    head, info = load_checkpoint(ckpt, dev)
    head.eval()
    if info["mode"] != "frozen":
        raise SystemExit("[ERROR] this script is only for frozen checkpoints (use 08 for e2e)")
    name = args.name or f"cm_{ckpt.parent.name}_{ckpt.name}_stream"
    docs = load_docs(paths, mmap=False)
    doc_pids = np.asarray(docs.ids, dtype=np.int64)
    tdocs = docs_to_device(docs, dev)
    W = head.weight.detach().float()
    for set_name in args.query_sets:
        q, qrels, thr, texts = sparseup_query_vectors(set_name, paths, dev)
        qids = np.asarray(q.ids)
        gq = concept_vectors_from_csr(head, q, dev)
        nq = gq.shape[0]
        out_s = np.empty((nq, args.k), dtype=np.float32)
        out_i = np.empty((nq, args.k), dtype=np.int64)
        for b, qs in enumerate(range(0, nq, args.q_batch)):
            qe = min(nq, qs + args.q_batch)
            table = (W @ gq[qs:qe].T).contiguous()
            s, i = search_with_query_table(tdocs, table, args.k)
            out_s[qs:qe] = s.float().cpu().numpy()
            out_i[qs:qe] = i.cpu().numpy()
            if b % 5 == 0:
                log(f"[{set_name}] {qe}/{nq} queries")
        report_and_save(paths, name, set_name, qids, out_i, out_s, doc_pids, qrels, thr,
                        extra={"k_concepts": head.k, "mode": info["mode"], "ckpt": str(ckpt), "search": "streaming"})
    log("Done.")


if __name__ == "__main__":
    main()
