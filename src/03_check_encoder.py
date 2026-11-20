"""
03_check_encoder.py — Does OUR SparseUp encoder reproduce the pre-encoded (official) vectors?

The pre-encoded vectors of FM_DIR were produced by 02b with the OFFICIAL SparseUp implementation
(modeling_splade.py, trust_remote_code). Everything that is not in those files (the 503k training queries,
TREC DL queries, BEIR, the whole end-to-end variant) is encoded with OUR memory-efficient encoder
(sparseup_encoder.py: top-k on the logits + a single max over positions and fold groups). Queries and documents
must "speak the same language", so the two implementations must give the same vectors.

Test (same idea as the SPLADE project, where it identified the checkpoint of the provided files): take dev queries
and passages (we have their texts), encode them with our encoder and compare them with the vectors in
queries.bin / documents.bin. Same model and same algorithm -> mean cosine ~1.0000, Jaccard ~1.
Our encoder is checked twice:
  - fp32 (no TF32)        : used for every frozen-side encoding (05, DL queries, BEIR)   -> must be ~1.0000
  - amp (bf16 backbone)   : used by the e2e variant (training and re-encoding, as in the SPLADE project)
                            -> slightly lower (bf16 rounding), only reported
The choice is saved to WORK_DIR/encoder_check.json and used by the other scripts.

    srun -p long -c 4 --mem=32G --gpus 1 -t 00:30:00 python -u src/03_check_encoder.py
"""
import argparse

import numpy as np

from config import CANDIDATE_SPARSEUP_MODELS, Q_MAX_LEN, D_MAX_LEN, Paths
from msmarco import Collection, read_queries
from sparse_io import CSR
from sparseup_encoder import SparseUpEncoder, top_tokens
from utils import get_device, log, log_run_header, save_json


def compare(ours: CSR, theirs: CSR):
    """Row-by-row cosine, Jaccard of the supports, and ratio of the norms."""
    cos, jac, ratio = [], [], []
    for i in range(ours.n_rows):
        ia, va = ours.row(i)
        ib, vb = theirs.row(i)
        common, xa, xb = np.intersect1d(ia, ib, return_indices=True)
        dot = float((va[xa] * vb[xb]).sum())
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        cos.append(dot / (na * nb) if na > 0 and nb > 0 else 0.0)
        union = len(np.union1d(ia, ib))
        jac.append(len(common) / union if union else 1.0)
        ratio.append(nb / na if na > 0 else np.nan)
    return np.array(cos), np.array(jac), np.array(ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CANDIDATE_SPARSEUP_MODELS)
    ap.add_argument("--n", type=int, default=300, help="number of dev queries to compare")
    ap.add_argument("--n-docs", type=int, default=300, help="number of passages to compare (0 = skip)")
    ap.add_argument("--q-max-len", type=int, default=Q_MAX_LEN)
    ap.add_argument("--d-max-len", type=int, default=D_MAX_LEN)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = Paths()
    dev = get_device()
    log_run_header("03_check_encoder.py", args, dev)

    fm_q = CSR.load(paths.dev_queries_csr)
    texts = read_queries(paths.queries_tsv("dev"))
    qids = np.asarray(fm_q.ids)
    rows = [i for i, q in enumerate(qids.tolist()) if q in texts][: args.n]
    if not rows:
        raise SystemExit("[ERROR] None of the pre-encoded queries has a text in queries.dev.small.tsv: "
                         "check the output of 02b and 01.")
    log(f"Comparing {len(rows)} queries (pre-encoded qids found among dev small: "
        f"{sum(q in texts for q in qids.tolist())}/{len(qids)})")
    theirs_q = fm_q.subset(rows)
    batch_q = [texts[int(qids[i])] for i in rows]

    theirs_d = batch_d = None
    if args.n_docs:
        docs = CSR.load(paths.docs_csr, mmap=True)
        drows = np.sort(np.random.default_rng(args.seed).choice(docs.n_rows, min(args.n_docs, docs.n_rows),
                                                                 replace=False))
        theirs_d = docs.subset(drows)
        collection = Collection(paths.collection_tsv)
        batch_d = collection.get_many(np.asarray(docs.ids)[drows])
        log(f"Comparing {len(drows)} random passages")
        del docs

    results = {}
    for name in args.models:
        try:
            enc = SparseUpEncoder(name).to(dev)
        except Exception as e:  # noqa: BLE001
            log(f"{name}: could not be loaded ({e})")
            continue
        if enc.vocab_size != fm_q.n_cols:
            log(f"{name}: folded vocabulary of {enc.vocab_size} tokens, the cache has {fm_q.n_cols}: different model")
            continue
        res = {}
        for setting, amp in [("fp32", False), ("amp", True)]:
            if amp and dev.type != "cuda":
                continue
            ours = enc.encode_to_csr(batch_q, dev, batch_size=64, max_length=args.q_max_len, amp=amp, verbose=False,
                                     kind="query")
            cos, jac, ratio = compare(ours, theirs_q)
            r = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "jaccard_mean": float(jac.mean()),
                 "scale_median": float(np.nanmedian(ratio))}
            log(f"{name} [{setting}] queries  : mean cosine {cos.mean():.4f} (min {cos.min():.4f}), "
                f"Jaccard {jac.mean():.3f}, scale pre-encoded/ours {np.nanmedian(ratio):.3f}")
            if batch_d is not None:
                ours_d = enc.encode_to_csr(batch_d, dev, batch_size=32, max_length=args.d_max_len, amp=amp,
                                           verbose=False, kind="document")
                cos_d, jac_d, ratio_d = compare(ours_d, theirs_d)
                r.update({"doc_cos_mean": float(cos_d.mean()), "doc_cos_min": float(cos_d.min()),
                          "doc_jaccard_mean": float(jac_d.mean())})
                log(f"{name} [{setting}] passages : mean cosine {cos_d.mean():.4f} (min {cos_d.min():.4f}), "
                    f"Jaccard {jac_d.mean():.3f}")
            res[setting] = r
            if setting == "fp32":
                ia, va = ours.row(0)
                ib, vb = theirs_q.row(0)
                log(f"  query: {batch_q[0]!r}")
                log(f"  ours       : {top_tokens(enc.folded_vocab, ia, va, 8)}")
                log(f"  pre-encoded: {top_tokens(enc.folded_vocab, ib, vb, 8)}")
        results[name] = {**res["fp32"], "settings": res}
        del enc

    if not results:
        raise SystemExit("[ERROR] No model could be loaded (network problem reaching Hugging Face?)")
    best = max(results, key=lambda m: results[m]["cos_mean"])
    info = {"model": best, "scale": results[best]["scale_median"], "results": results}
    save_json(info, paths.encoder_check)
    good = results[best]["cos_mean"] > 0.99 and results[best].get("doc_cos_mean", 1.0) > 0.99
    verdict = "GOOD: same model" if good else \
        "WARNING: our encoder does not reproduce the pre-encoded vectors well (model version? transformers version?)"
    log(f"Selected: {best}  ->  {verdict}")
    log(f"Saved to {paths.encoder_check}")


if __name__ == "__main__":
    main()
