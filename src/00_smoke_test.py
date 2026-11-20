"""
00_smoke_test.py — Tests ALL the code on small SYNTHETIC data, without touching the real data.

Run it first on the cluster: if it ends with "ALL TESTS PASSED", the environment and the code work.
It runs on CPU in a few minutes and writes only to a temporary directory.

    srun -p long -c 4 --mem=16G -t 00:30:00 python -u src/00_smoke_test.py

The real SparseUp checkpoint (750 MB) is NOT needed: the test builds a tiny random "SparseUp-like" model
(byte-level BPE tokenizer with case / 'Ġ' variants, 2-layer ModernBERT, logit shift, per-position top-k,
vocabulary folding, "[Q] " / "[D] " prefixes) whose remote code is sparseup_reference.py, a literal
re-implementation of the official modeling_splade.py, and loads it with trust_remote_code exactly like the real one.

Part 1 - unit tests:
  Seismic format (write/read, also streaming), CSR, exact sparse and dense search vs brute force,
  metrics (vs hand computation and vs ranx if installed), InfoNCE = Eq. (6), false-negative
  mask, THE NUMERICAL EXAMPLE OF SEC. 3.4 OF THE PAPER (LCE), a Concept Model that actually
  learns on data with known structure, and the SparseUp encoder: our memory-efficient encoder = the
  reference implementation (prefix not pooled, folding, top-k), gradients flow end to end.
Part 2 - end-to-end test: builds a fake "MS MARCO" + a fake BEIR dataset and runs scripts 01-12 one after the
  other (02b creates the pre-encoded files with the reference implementation, as on the cluster with the real one).
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from utils import make_transformers_cpu_safe  # noqa: E402

make_transformers_cpu_safe()

PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  [OK] {name}", flush=True)


# =====================================================================
#  PART 1 — unit tests
# =====================================================================
def test_seismic_and_csr(tmp):
    from sparse_io import CSR, SeismicWriter, read_seismic_bin, write_seismic_bin
    rng = np.random.default_rng(0)
    dense = (rng.random((50, 40)) < 0.15) * rng.random((50, 40)).astype(np.float32)
    dense[7] = 0  # empty row
    csr = CSR.from_dense(dense)
    write_seismic_bin(tmp / "x.bin", csr)
    back = read_seismic_bin(tmp / "x.bin", verbose=False)
    back.n_cols = 40
    assert np.allclose(back.to_dense(), dense), "roundtrip Seismic"
    part = read_seismic_bin(tmp / "x.bin", max_vectors=10, verbose=False)
    part.n_cols = 40
    assert np.allclose(part.to_dense(), dense[:10])
    with SeismicWriter(tmp / "y.bin") as w:                 # streaming writer, in 3 blocks
        for a, b in [(0, 13), (13, 14), (14, 50)]:
            w.append(csr.subset(np.arange(a, b)))
    assert (tmp / "y.bin").read_bytes() == (tmp / "x.bin").read_bytes(), "streaming writer = one-shot writer"
    rows = np.array([3, 7, 0, 49, 3])
    assert np.allclose(csr.subset(rows).to_dense(), dense[rows]), "subset/gather"
    csr.ids = np.arange(50) + 1000
    csr.save(tmp / "csr")
    again = CSR.load(tmp / "csr", mmap=True)
    assert np.allclose(again.to_dense(), dense) and again.ids[3] == 1003
    ok("Seismic format (also streaming) + CSR (write, read, subsets, save)")


def test_remap():
    from sparse_io import CSR, build_remap, remap_columns
    tokens = ["apple", "zebra", "Ġing", "unknowntoken"]
    vocab = {"[PAD]": 0, "zebra": 5, "apple": 9, "Ġing": 2}
    remap, unknown = build_remap(tokens, vocab)
    assert remap.tolist() == [9, 5, 2, -1] and unknown == ["unknowntoken"]
    csr = CSR.from_dense(np.array([[1, 0, 2, 3], [0, 4, 0, 0]], dtype=np.float32))
    out = remap_columns(csr, remap, 10)
    d = out.to_dense()
    assert d[0, 9] == 1 and d[0, 2] == 2 and d[1, 5] == 4 and d.sum() == 7
    ok("Seismic id -> folded id conversion")



def test_search():
    from retrieval import TorchCSR, dense_search, project_rows, sparse_search
    from sparse_io import CSR
    rng = np.random.default_rng(1)
    D = (rng.random((300, 60)) < 0.2) * rng.random((300, 60))
    Q = (rng.random((17, 60)) < 0.3) * rng.random((17, 60))
    docs, qs = CSR.from_dense(D), CSR.from_dense(Q)
    td = TorchCSR(docs, torch.device("cpu"))
    S = Q @ D.T
    s, i = sparse_search(td, qs, 25, q_batch=5, doc_chunk=40, verbose=False)
    ref = np.sort(S, axis=1)[:, ::-1][:, :25]
    assert np.allclose(s, ref, atol=1e-5), "sparse search scores"
    assert np.allclose(np.take_along_axis(S, i, 1), s, atol=1e-5), "sparse search indices"
    W = torch.randn(60, 8)
    G = project_rows(td, W, chunk=33, out_dtype=torch.float32)
    assert torch.allclose(G, torch.from_numpy(D).float() @ W, atol=1e-4), "g(d) = W.SparseUp(d)"
    gq = torch.from_numpy(Q).float() @ W
    s2, i2 = dense_search(G, gq, 10, q_batch=4, doc_chunk=70, verbose=False)
    S2 = (gq @ G.T).numpy()
    assert np.allclose(s2, np.sort(S2, 1)[:, ::-1][:, :10], atol=1e-4), "dense search"
    ok("exact sparse and dense search = brute force (also chunked)")


def test_metrics():
    from metrics import evaluate_run
    qrels = {1: {10: 1}, 2: {20: 1, 21: 1}, 3: {30: 3, 31: 1, 32: 2}}
    run = np.array([[5, 10, 6, 7], [20, 9, 8, 21], [31, 30, 4, 32]])
    m, _ = evaluate_run([1, 2, 3], run, qrels, rel_threshold=1, cutoffs_recall=(2, 4), ndcg_cut=3, mrr_cut=3)
    assert abs(m["MRR@3"] - (0.5 + 1 + 1) / 3) < 1e-9
    assert abs(m["R@2"] - (1 + 0.5 + 2 / 3) / 3) < 1e-9
    d = 1 / np.log2(np.arange(2, 6))
    nd1 = d[1] / d[0]
    nd2 = d[0] / (d[0] + d[1])
    nd3 = (1 * d[0] + 3 * d[1]) / (3 * d[0] + 2 * d[1] + 1 * d[2])
    assert abs(m["nDCG@3"] - (nd1 + nd2 + nd3) / 3) < 1e-9
    try:
        from ranx import Qrels, Run, evaluate
        qr = Qrels({str(q): {str(p): r for p, r in v.items()} for q, v in qrels.items()})
        rn = Run({str(q): {str(p): float(10 - j) for j, p in enumerate(row)} for q, row in zip([1, 2, 3], run)})
        rx = evaluate(qr, rn, ["mrr@3", "ndcg@3", "recall@2"])
        assert abs(rx["mrr@3"] - m["MRR@3"]) < 1e-6 and abs(rx["recall@2"] - m["R@2"]) < 1e-6
        assert abs(rx["ndcg@3"] - m["nDCG@3"]) < 1e-6, (rx["ndcg@3"], m["nDCG@3"])
        ok("metrics (MRR, graded nDCG, Recall) = hand computation = ranx")
    except ImportError:
        ok("metrics (MRR, graded nDCG, Recall) = hand computation (ranx not installed)")


def test_lce_paper_example():
    """Reproduces the numbers of Sec. 3.4 of the paper (6-document corpus, alpha=0.5, tau=0.5)."""
    from lce import build_cooccurrence, expand, normalize_cooccurrence, prune
    from sparse_io import CSR
    docs_feats = [[1, 5, 7], [6, 8, 10], [1, 2, 3], [4, 5, 9], [6, 8, 11], [5, 6]]  # F1..F11
    X = np.zeros((6, 11), dtype=np.float32)
    for r, fs in enumerate(docs_feats):
        X[r, [f - 1 for f in fs]] = 1.0
    M, df = build_cooccurrence(CSR.from_dense(X), torch.device("cpu"), chunk=4)
    assert M[5, 7].item() == 2 and M[0, 4].item() == 1 and M[4, 5].item() == 1  # M[F6,F8]=2 etc.
    B = normalize_cooccurrence(M, df, "raw", 6)
    assert float(torch.diagonal(B).abs().sum()) == 0.0
    vq = torch.zeros(1, 11)
    vq[0, 0], vq[0, 3] = 0.94, 0.82
    vext = expand(vq, B, 0.5, "absolute")
    expected = [0.94, 0.47, 0.47, 0.82, 0.88, 0, 0.47, 0, 0.41, 0, 0]
    assert torch.allclose(vext[0], torch.tensor(expected), atol=1e-6), vext
    pruned = prune(vext, "tau:0.5")
    assert torch.allclose(pruned[0], torch.tensor([0.94, 0, 0, 0.82, 0.88, 0, 0, 0, 0, 0, 0]), atol=1e-6)
    d6 = torch.zeros(11)
    d6[4], d6[5] = 0.85, 0.91
    assert abs(float(vq[0] @ d6)) < 1e-9 and abs(float(pruned[0] @ d6) - 0.748) < 1e-6
    for kind in ["cond", "cosine", "ppmi"]:
        Bn = normalize_cooccurrence(M, df, kind, 6)
        assert torch.isfinite(Bn).all() and (Bn >= 0).all()
    top2 = prune(vext, "topk:2")
    assert int((top2 > 0).sum()) == 2
    ok("LCE: reproduced EXACTLY the example of Sec. 3.4 (V_q_ext, pruning, score 0.748)")


def test_infonce_and_mask():
    from concept_model import ConceptHead, false_negative_mask, info_nce_loss
    torch.manual_seed(0)
    q, d = torch.randn(3, 4), torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2])
    loss, _, _ = info_nce_loss(q, d, labels, temperature=0.5)
    s = (q @ d.T) / 0.5
    manual = -np.mean([float(s[i, i] - torch.logsumexp(s[i], 0)) for i in range(3)])
    assert abs(float(loss) - manual) < 1e-5, "Eq. (6)"
    mask = false_negative_mask([[100, 105], [101], [102]], np.array([100, 101, 102, 105, 100]), np.array([0, 1, 2]))
    expected = np.zeros((3, 5), dtype=bool)
    expected[0, 3] = expected[0, 4] = True
    assert (mask == expected).all(), mask
    head = ConceptHead(30, 6)
    x = torch.zeros(2, 30)
    x[0, [1, 5]] = torch.tensor([0.5, 2.0])
    x[1, 7] = 1.0
    bags = head.forward_bags(torch.tensor([1, 5, 7]), torch.tensor([0, 2]), torch.tensor([0.5, 2.0, 1.0]))
    assert torch.allclose(bags, head.forward_dense(x), atol=1e-6)
    ok("InfoNCE = Eq. (6), false-negative mask, sparse g(x) = dense")


def test_concept_model_learns():
    """Data with known structure: 12 'topics', queries and documents use different words of the
    same topic (vocabulary mismatch). The lexical dot product fails; the Concept Model trained
    with InfoNCE must learn to connect them."""
    from concept_model import ConceptHead, csr_batch_to_torch, info_nce_loss
    from sparse_io import CSR
    rng = np.random.default_rng(0)
    T, per, V = 12, 10, 12 * 10
    def vec(topic, part):
        v = np.zeros(V, dtype=np.float32)
        words = topic * per + (np.arange(5) if part == "q" else np.arange(5, 10))
        v[rng.choice(words, 3, replace=False)] = rng.random(3) + 0.5
        return v
    n = 600
    topics = rng.integers(0, T, n)
    Q = CSR.from_dense(np.stack([vec(t, "q") for t in topics]))
    D = CSR.from_dense(np.stack([vec(t, "d") for t in topics]))
    dev = torch.device("cpu")
    head = ConceptHead(V, 16, seed=0)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    tr, va = np.arange(500), np.arange(500, 600)

    def val_mrr():
        with torch.no_grad():
            gq = head.forward_bags(*csr_batch_to_torch(Q, va, dev))
            gd = head.forward_bags(*csr_batch_to_torch(D, va, dev))
            S = gq @ gd.T
            ranks = (S > S.diag()[:, None]).sum(1) + 1
            hit10 = ranks <= 10
            return float(torch.where(hit10, 1.0 / ranks, torch.zeros_like(ranks, dtype=torch.float)).mean())

    before = val_mrr()
    for _ in range(150):
        b = rng.choice(tr, 64, replace=False)
        loss, _, _ = info_nce_loss(head.forward_bags(*csr_batch_to_torch(Q, b, dev)),
                                   head.forward_bags(*csr_batch_to_torch(D, b, dev)), torch.arange(64))
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = val_mrr()
    assert after > before + 0.1, (before, after)
    ok(f"the Concept Model learns to overcome the vocabulary mismatch (MRR {before:.3f} -> {after:.3f})")


# ---------------------------------------------------------------------
#  tiny SparseUp-like model (shared by part 1 and part 2)
# ---------------------------------------------------------------------
WORDS = [f"w{i}" for i in range(120)] + ["chest", "pain", "heart", "sweat", "pressure", "crushing"]


def synth_text(rng, topic, n):
    """Words of one topic (+2 random), ~25% of them capitalized: the BPE vocabulary gets case variants
    ('Ġw5' / 'ĠW5'), so the folding is really exercised."""
    ws = rng.choice(WORDS[topic * 12:(topic + 1) * 12], n).tolist() + rng.choice(WORDS, 2).tolist()
    return " ".join(w.capitalize() if rng.random() < 0.25 else w for w in ws)


def build_tiny_sparseup(mdir, texts, seed=0):
    """Byte-level BPE tokenizer + 2-layer ModernBERT + SparseUp head settings, saved as a remote-code checkpoint
    (config auto_map -> modeling_splade.py = sparseup_reference.py), like the real Linkup checkpoint."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast
    from sparseup_reference import SpladeConfig, SpladeModel, build_fold_index
    mdir = Path(mdir)
    mdir.mkdir(parents=True, exist_ok=True)
    specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=700, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    corpus = list(texts) + [f"[Q] {t}" for t in texts[:50]] + [f"[D] {t}" for t in texts[:50]]
    tk.train_from_iterator(corpus, trainer)
    cls_id, sep_id = tk.token_to_id("[CLS]"), tk.token_to_id("[SEP]")
    tk.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=True),
        processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
                                      special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)])])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
                                  pad_token="[MASK]", mask_token="[MASK]")
    tok.save_pretrained(str(mdir))
    V = len(tok)
    mask_id = tok.convert_tokens_to_ids("[MASK]")
    torch.manual_seed(seed)
    cfg = SpladeConfig(vocab_size=V, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                       max_position_embeddings=256, pad_token_id=mask_id, cls_token_id=cls_id, sep_token_id=sep_id,
                       bos_token_id=cls_id, eos_token_id=sep_id, global_attn_every_n_layers=2, local_attention=16,
                       logit_shift=0.0, position_top_k=4, vocab_fold="case_space", query_prefix="[Q] ",
                       document_prefix="[D] ", query_max_length=32, doc_max_length=64)
    cfg.auto_map = {"AutoConfig": "modeling_splade.SpladeConfig", "AutoModel": "modeling_splade.SpladeModel"}
    cfg._attn_implementation = "sdpa"
    model = SpladeModel(cfg).eval()
    with torch.no_grad():
        model.vocab_fold_index.copy_(build_fold_index(tok))
        # logit shift: the 80th percentile of the logits on some texts (like theta = 15 for the real model)
        enc = tok(list(texts[:32]), padding=True, return_tensors="pt")
        logits = super(SpladeModel, model).forward(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        model.config.logit_shift = float(torch.quantile(logits.flatten()[:200_000].float(), 0.8))
    model.save_pretrained(str(mdir))
    shutil.copy(HERE / "sparseup_reference.py", mdir / "modeling_splade.py")
    n_fold = len(torch.unique(model.vocab_fold_index))
    return mdir, V, n_fold


def test_sparseup_encoder(tmp):
    """Our memory-efficient encoder (top-k on logits, one amax over positions and fold groups) must give the same
    vectors as the literal reference implementation loaded with trust_remote_code; gradients must flow."""
    from transformers import AutoModel
    from sparseup_encoder import SparseUpEncoder
    rng = np.random.default_rng(3)
    texts = [synth_text(rng, int(rng.integers(0, 10)), 10) for _ in range(120)]
    mdir, V, n_fold = build_tiny_sparseup(tmp / "tiny", texts)
    assert n_fold < V, "the fold map must merge something (case variants)"
    ref = AutoModel.from_pretrained(str(mdir), trust_remote_code=True).eval()
    enc = SparseUpEncoder(mdir).eval()
    assert enc.vocab_size == n_fold and enc.position_top_k == 4 and enc.query_prefix == "[Q] "
    dev = torch.device("cpu")
    for kind, sample in [("query", texts[:20]), ("document", texts[20:60])]:
        r = ref.encode(sample, kind=kind)                                          # (n, V), official layout
        with torch.no_grad():
            ours = enc.encode_dense(sample, None, dev, kind=kind)                  # (n, C), folded space
        noncanon = torch.ones(V, dtype=torch.bool)
        noncanon[enc.canonical_ids] = False
        assert float(r[:, noncanon].abs().max()) == 0.0, "reference: non-canonical dims must be 0"
        assert torch.allclose(r[:, enc.canonical_ids], ours, atol=1e-5), \
            float((r[:, enc.canonical_ids] - ours).abs().max())
        assert (ours > 0).sum(1).min() > 0, "empty vectors: logit shift too high"
        csr = enc.encode_to_csr(sample, dev, batch_size=7, kind=kind, verbose=False)
        assert np.allclose(csr.to_dense(), ours.numpy(), atol=1e-6), "encode_to_csr (sorted batches) = encode_dense"
    # the prefix tokens are attended to but NOT pooled; [CLS]/[SEP] are pooled
    ids, mask, pool = enc.tokenize(["w1 w2"], None, dev, kind="query")
    toks = enc.tokenizer.convert_ids_to_tokens(ids[0].tolist())
    not_pooled = [t for t, a, p in zip(toks, mask[0].tolist(), pool[0].tolist()) if a and not p]
    assert "".join(not_pooled).replace("Ġ", "") == "[Q]" and pool[0, 0] == 1, (toks, pool)
    # the logits of the manual head (backbone, then head in float32) = ModernBertForMaskedLM.forward
    ids, mask, pool = enc.tokenize(texts[:5], None, dev, kind="document")
    with torch.no_grad():
        a = enc._head_logits(enc._hidden(ids, mask))
        b = enc._parent_logits(ids, mask)
    assert torch.allclose(a, b, atol=1e-5), "manual head != MaskedLM forward"
    # gradients flow through top-k + scatter amax + folding, and stay finite
    enc.train()
    x = enc(ids, mask, pool)
    (x ** 2).sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    # save / reload a fine-tuned backbone (as 06 does for e2e checkpoints)
    enc.save(tmp / "saved")
    again = SparseUpEncoder(tmp / "saved").eval()
    assert torch.equal(again.canonical_ids, enc.canonical_ids)
    with torch.no_grad():
        assert torch.allclose(again.encode_dense(texts[:4], None, dev, kind="query"),
                              enc.eval().encode_dense(texts[:4], None, dev, kind="query"), atol=1e-6)
    ok(f"SparseUp encoder = reference implementation (vocab {V} -> {n_fold} folded, top-k, prefix not pooled, "
       "gradients, save/reload)")


# =====================================================================
#  PART 2 — synthetic world + scripts 01..12
# =====================================================================
def build_world(root):
    rng = np.random.default_rng(42)
    N = 400
    doc_topic = rng.integers(0, 10, N)
    collection = [synth_text(rng, int(t), 12) for t in doc_topic]
    mdir, _, _ = build_tiny_sparseup(root / "model", collection)

    msm = root / "msm_src"
    msm.mkdir()
    (msm / "collection.tsv").write_text("".join(f"{i}\t{t}\n" for i, t in enumerate(collection)))
    def make_queries(n, start):
        qids = np.arange(start, start + n)
        pos = rng.choice(N, n, replace=False)
        texts = {int(q): synth_text(rng, int(doc_topic[p]), 3) for q, p in zip(qids, pos)}
        qrels = {int(q): {int(p): 1} for q, p in zip(qids, pos)}
        return texts, qrels
    tr_t, tr_r = make_queries(200, 1000)
    dv_t, dv_r = make_queries(40, 5000)
    dl_t, dl_r = make_queries(8, 9000)
    for q in dl_r:  # TREC DL-style graded relevance
        extra = rng.choice(N, 3, replace=False)
        dl_r[q].update({int(e): int(g) for e, g in zip(extra, [3, 2, 1])})
    def wq(path, t):
        Path(path).write_text("".join(f"{q}\t{s}\n" for q, s in t.items()))
    def wr(path, r):
        Path(path).write_text("".join(f"{q}\t0\t{p}\t{v}\n" for q, d in r.items() for p, v in d.items()))
    wq(msm / "queries.train.tsv", tr_t)
    wr(msm / "qrels.train.tsv", tr_r)
    wq(msm / "queries.dev.small.tsv", dv_t)
    wr(msm / "qrels.dev.small.tsv", dv_r)
    with tarfile.open(root / "collectionandqueries.tar.gz", "w:gz") as tar:
        for f in msm.iterdir():
            tar.add(f, arcname=f.name)
    dl_dir = root / "dl"
    dl_dir.mkdir()
    with gzip.open(dl_dir / "msmarco-test2019-queries.tsv.gz", "wt") as f:
        f.write("".join(f"{q}\t{s}\n" for q, s in dl_t.items()))
    (dl_dir / "2019qrels-pass.txt").write_text("".join(f"{q} Q0 {p} {v}\n" for q, d in dl_r.items()
                                                       for p, v in d.items()))

    # a fake BEIR dataset (string ids, title + text), placed where 10_eval_beir.py looks for it
    beir = root / "work" / "beir" / "synth"
    (beir / "qrels").mkdir(parents=True)
    bd_topic = rng.integers(0, 10, 150)
    with open(beir / "corpus.jsonl", "w") as f:
        for i, t in enumerate(bd_topic):
            f.write(json.dumps({"_id": f"doc{i}", "title": synth_text(rng, int(t), 2),
                                "text": synth_text(rng, int(t), 10)}) + "\n")
    with open(beir / "queries.jsonl", "w") as f, open(beir / "qrels" / "test.tsv", "w") as g:
        g.write("query-id\tcorpus-id\tscore\n")
        for j, p in enumerate(rng.choice(150, 25, replace=False)):
            f.write(json.dumps({"_id": f"q{j}", "text": synth_text(rng, int(bd_topic[p]), 3)}) + "\n")
            g.write(f"q{j}\tdoc{p}\t1\n")
    return mdir, dl_dir


def run(script, *args, env):
    cmd = [sys.executable, "-u", str(HERE / script), *map(str, args)]
    print(f"\n$ {script} {' '.join(map(str, args))}", flush=True)
    p = subprocess.run(cmd, env=env, cwd=str(HERE), capture_output=True, text=True)
    tail = "\n".join(p.stdout.strip().splitlines()[-6:])
    print("    " + tail.replace("\n", "\n    "), flush=True)
    if p.returncode != 0:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        raise SystemExit(f"FAILED: {script}")
    return p.stdout


def end_to_end(root):
    mdir, dl_dir = build_world(root)
    work = root / "work"
    env = dict(os.environ, SPARSEUP_WORK_DIR=str(work), SPARSEUP_MODEL=str(mdir), CUDA_VISIBLE_DEVICES="",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_HOME=str(root / "hf_home"))
    env.pop("SPARSEUP_FM_DIR", None)                         # default: WORK_DIR/encoded
    run("02_download_msmarco.py", "--from-tar", root / "collectionandqueries.tar.gz", env=env)
    for f in dl_dir.iterdir():
        shutil.copy(f, work / "msmarco" / f.name)
    out = run("02b_encode_msmarco.py", "--batch-size", 16, "--max-batch-tokens", 300, "--shard-size", 150, env=env)
    assert (work / "encoded" / "documents.bin").exists() and not (work / "encoded" / "_shards").exists(), out
    run("01_prepare_data.py", env=env)
    out = run("03_check_encoder.py", "--models", mdir, "--n", 20, "--n-docs", 30, env=env)
    assert "queries  : mean cosine 1.0000" in out and "passages : mean cosine 1.0000" in out, \
        "our encoder must reproduce the vectors of the reference implementation"
    ok("02b + 01 + 03: pre-encoded files written by the reference implementation, reproduced by our encoder")
    run("04_baseline_sparseup.py", "--query-sets", "dev", "dl19", "--k", 50, env=env)
    res = json.loads((work / "results" / "sparseup_baseline_dev.json").read_text())["summary"]
    assert res["overlap_groundtruth"] > 0.99, res
    ok(f"baseline: overlap with the torch.sparse groundtruth of 02b = {res['overlap_groundtruth']:.3f}")
    run("05_encode_train_queries.py", "--batch-size", 32, env=env)
    common = ["--k", 16, "--batch-size", 16, "--val-queries", 20, "--val-candidates", 20, "--log-every", 5]
    run("06_train_concept_model.py", "--mode", "frozen", "--epochs", 3, "--val-every", 10,
        "--run-name", "fz1", *common, env=env)
    run("07_mine_negatives.py", "--miner", "sparseup", "--out", "sp", "--depth", 30, "--keep", 20, env=env)
    run("07_mine_negatives.py", "--miner", "concept", "--ckpt", "fz1", "--out", "r1", "--depth", 30,
        "--keep", 20, env=env)
    run("06_train_concept_model.py", "--mode", "frozen", "--max-steps", 12, "--val-every", 6, "--n-hard", 3,
        "--negatives", "sp", "r1", "--neg-rank-min", 1, "--init-from", "fz1", "--run-name", "fz2", *common, env=env)
    run("08_eval_concept_model.py", "--ckpt", "fz2", "--query-sets", "dev", "dl19", "--k", 50, "--explain", 2,
        env=env)
    run("08b_eval_streaming.py", "--ckpt", "fz2", "--query-sets", "dev", "--k", 50, env=env)
    a = json.loads((work / "results" / "cm_fz2_best_dev.json").read_text())["summary"]
    b = json.loads((work / "results" / "cm_fz2_best_stream_dev.json").read_text())["summary"]
    assert abs(a["MRR@10"] - b["MRR@10"]) < 0.02, (a["MRR@10"], b["MRR@10"])
    run("06_train_concept_model.py", "--mode", "e2e", "--max-steps", 4, "--val-every", 2, "--n-hard", 1,
        "--negatives", "sp", "--run-name", "e2e1", "--k", 16, "--batch-size", 4, "--val-queries", 5,
        "--val-candidates", 10, "--log-every", 2, "--grad-checkpointing", "--lr-backbone", 1e-3, env=env)
    run("07_mine_negatives.py", "--miner", "concept", "--ckpt", "e2e1", "--out", "e2e_r1", "--depth", 20,
        "--keep", 10, env=env)
    run("08_eval_concept_model.py", "--ckpt", "e2e1", "--query-sets", "dev", "dl19", "--k", 50, "--explain", 1,
        env=env)
    run("08c_eval_e2e_streaming.py", "--ckpt", "e2e1", "--query-sets", "dev", "dl19", "--k", 50,
        "--shard-size", 150, env=env)
    a = json.loads((work / "results" / "cm_e2e1_best_dev.json").read_text())["summary"]
    b = json.loads((work / "results" / "cm_e2e1_best_stream_dev.json").read_text())["summary"]
    assert abs(a["MRR@10"] - b["MRR@10"]) < 0.02, (a["MRR@10"], b["MRR@10"])
    ok("frozen and e2e: training, mining, evaluation (dense and streaming give the same results)")
    run("09_lce_sparseup.py", "--k", 50, "--norms", "raw", "cond", "ppmi", "--alphas", 0.3, "--prunes", "topk:8",
        "tau:0.1", "--demo", "crushing pressure chest sweat", "--examples", 1, "--query-sets", "dev", "dl19", env=env)
    val = ["--val-cache", "seed42_n20_c20_*.npz"]
    out = run("11_eval_hybrid.py", "--ckpt", "fz2", "--query-sets", "dev", "dl19", "--k", 50, *val, env=env)
    for variant in ["two", "single", "now"]:
        run("11_eval_hybrid.py", "--ckpt", "e2e1", "--variant", variant, "--query-sets", "dev", "--k", 50,
            "--lambdas", 0.0, 0.5, 1.0, *val, env=env)
    hyb = sorted((work / "results").glob("hybnoW_e2e1_best_lam*_dev.json"))
    assert hyb, "hybnoW results missing"
    base = json.loads((work / "results" / "sparseup_baseline_dev.json").read_text())["summary"]["MRR@10"]
    lam0 = json.loads((work / "results" / "hyb_fz2_best_validation.json").read_text())["curve"][0]
    assert lam0["lambda"] == 0.0
    ok("hybrids (frozen; e2e two / single / now), lambda chosen on validation")
    run("10_eval_beir.py", "--datasets", "synth", "--sparseup", "--ckpts", "fz2", "e2e1", "--ft-sparseup", "e2e1",
        "--hybrid", "e2e1:now:0.40", "e2e1:two:0.20", "e2e1:single:0.30", "fz2:two:0.10", "--k", 20, env=env)
    out = run("12_significance.py", "--a", "sparseup_baseline_dev", "--b", hyb[0].name, "--iters", 200, env=env)
    assert "paired queries: 40" in out, out
    ok("BEIR (baseline, frozen, e2e, fine-tuned encoder, 4 hybrids) and significance test")
    summary = (work / "results" / "summary.csv").read_text().splitlines()
    assert len(summary) > 20
    ok(f"end-to-end: scripts 01-12 completed, {len(summary) - 1} rows in results/summary.csv "
       f"(synthetic SparseUp baseline dev MRR@10 = {base:.3f})")


def main():
    print("PART 1: unit tests", flush=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_seismic_and_csr(tmp)
        test_remap()
        test_search()
        test_metrics()
        test_lce_paper_example()
        test_infonce_and_mask()
        test_concept_model_learns()
        test_sparseup_encoder(tmp)
    print("\nPART 2: full pipeline on synthetic data", flush=True)
    keep = "--keep" in sys.argv
    root = Path(tempfile.mkdtemp(prefix="sparseup_smoke_"))
    try:
        end_to_end(root)
    finally:
        if keep:
            print(f"(synthetic files left in {root})")
        else:
            shutil.rmtree(root, ignore_errors=True)
    print(f"\nALL TESTS PASSED ({len(PASSED)})", flush=True)


if __name__ == "__main__":
    main()
