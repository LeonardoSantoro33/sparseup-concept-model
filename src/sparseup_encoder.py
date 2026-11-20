"""
sparseup_encoder.py — SparseUp: text -> sparse vector over the FOLDED BPE vocabulary (~34k dimensions).

SparseUp (Linkup Research, 2026; Linkup-Platform/linkup-sparseup-embed-v1) is a SPLADE-style encoder on a
ModernBERT backbone (149M parameters) with three modifications:

    w_j(x) = amax_{v in fold(j)}  max_{i pooled}  topk12_i( log(1 + ReLU(logit_{i,v} - theta)) )

    1) logit shift       theta = 15 (config.logit_shift): log(1 + ReLU(x - theta)) instead of log(1 + ReLU(x))
    2) per-position top-k k = 12 (config.position_top_k): each token keeps only its 12 largest vocabulary
                         dimensions BEFORE the max pooling (it caps the expansion of every token)
    3) vocabulary folding (config.vocab_fold): the case / 'Ġ' variants of the same word ('heat', 'Heat', 'Ġheat',
                         'ĠHeat') are merged onto ONE canonical id, keeping the largest weight (amax)
    + instruction prefixes "[Q] " (queries) and "[D] " (documents): attended to by the backbone but NOT pooled.

The official implementation (modeling_splade.py on the Hugging Face hub, loaded with trust_remote_code) builds the
whole (B, L, 50k) activation tensor. Here we compute exactly the same thing with much less memory (needed for
end-to-end training), as splade_encoder.py did in the SPLADE project:
  - log(1 + ReLU(x - theta)) is monotonically non-decreasing, so the top-k can be taken on the LOGITS;
  - the max over the positions and the amax over the fold group are both max operations, so they are done in a
    single scatter_reduce(amax) of the top-k logits, and log(1 + ReLU(. - theta)) is applied only at the end.
Script 03 checks on dev queries and documents that the result coincides with the official implementation.

OUTPUT SPACE ("folded space"): the canonical token ids (the distinct values of the checkpoint buffer
vocab_fold_index) renumbered 0..C-1 in increasing order of token id. The official model returns a 50,370-dim
vector whose non-canonical dimensions are always 0; dropping them changes nothing in the dot products.
All caches, the concept matrix W (C x k, stored transposed) and LCE live in this space.
"""
import inspect
import json
import shutil
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sparse_io import CSR
from utils import log, make_transformers_cpu_safe

make_transformers_cpu_safe()
from transformers import AutoModel, AutoTokenizer  # noqa: E402


@contextmanager
def full_fp32():
    """Disable TF32 inside the block (exact float32 matmuls, as the official fp32 implementation)."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def load_sparseup_model(name_or_path, device=None):
    """The OFFICIAL SparseUp model (class SpladeModel from modeling_splade.py, a ModernBertForMaskedLM subclass).
    Loaded in float32, as in the released config."""
    kw = {"trust_remote_code": True}
    try:
        model = AutoModel.from_pretrained(str(name_or_path), dtype=torch.float32, **kw)
    except TypeError:          # older transformers: the argument was called torch_dtype
        model = AutoModel.from_pretrained(str(name_or_path), torch_dtype=torch.float32, **kw)
    model.eval()
    return model.to(device) if device is not None else model


def folded_ids_from_index(vocab_fold_index):
    """canonical (native) token ids, sorted = the folded space; compact[v] = folded id of native id v."""
    idx = torch.as_tensor(vocab_fold_index).long().cpu()
    canonical = torch.unique(idx)                          # sorted
    compact = torch.searchsorted(canonical, idx)
    return canonical, compact


def token_strings(tokenizer, ids):
    """Printable and UNIQUE token strings for the given native ids (the token strings are the keys of
    token_to_id_mapping.json): ids beyond the tokenizer (padded vocabulary rows) get a placeholder."""
    ids = [int(i) for i in ids]
    toks = tokenizer.convert_ids_to_tokens(ids)
    out, seen = [], set()
    for i, t in zip(ids, toks):
        t = f"[unused_{i}]" if t is None else str(t)
        if t in seen:
            t = f"{t}#{i}"
        seen.add(t)
        out.append(t)
    return out


class FoldedVocab:
    """Folded id <-> token string. Has convert_ids_to_tokens() like a tokenizer, so the display code
    (concepts, LCE examples, top tokens) works unchanged on folded ids."""

    def __init__(self, canonical_ids, tokens, model=""):
        self.canonical_ids = np.asarray(canonical_ids, dtype=np.int64)
        self.tokens = list(tokens)
        self.model = str(model)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, ids):
        return [self.tokens[int(i)] for i in ids]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"model": self.model, "canonical_ids": self.canonical_ids.tolist(),
                                          "tokens": self.tokens}, ensure_ascii=False))

    @staticmethod
    def from_file(path):
        d = json.loads(Path(path).read_text())
        return FoldedVocab(d["canonical_ids"], d["tokens"], d.get("model", ""))

    @staticmethod
    def from_model(model, tokenizer, name=""):
        canonical, _ = folded_ids_from_index(model.vocab_fold_index)
        return FoldedVocab(canonical.numpy(), token_strings(tokenizer, canonical.tolist()), name)

    @staticmethod
    def load(paths, model_name=None):
        """From WORK_DIR/cache/folded_vocab.json (written by 02b / 01); if missing, from the model (and cached)."""
        if paths.folded_vocab.exists():
            return FoldedVocab.from_file(paths.folded_vocab)
        if model_name is None:
            from config import get_sparseup_model
            model_name = get_sparseup_model(paths)[0]
        enc = SparseUpEncoder(model_name)
        fv = enc.folded_vocab
        try:
            fv.save(paths.folded_vocab)
        except OSError:
            pass
        return fv


class SparseUpEncoder(nn.Module):
    def __init__(self, name_or_path):
        super().__init__()
        self.name_or_path = str(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.mlm = load_sparseup_model(self.name_or_path)
        cfg = self.mlm.config
        self.logit_shift = float(getattr(cfg, "logit_shift", 0.0) or 0.0)
        tk = getattr(cfg, "position_top_k", None)
        self.position_top_k = int(tk) if tk else None
        self.query_prefix = getattr(cfg, "query_prefix", "") or ""
        self.document_prefix = getattr(cfg, "document_prefix", "") or ""
        self.query_max_length = int(getattr(cfg, "query_max_length", 128) or 128)
        self.doc_max_length = int(getattr(cfg, "doc_max_length", 512) or 512)
        self.native_vocab_size = int(cfg.vocab_size)
        fold_on = getattr(cfg, "vocab_fold", None) is not None and hasattr(self.mlm, "vocab_fold_index")
        index = self.mlm.vocab_fold_index if fold_on else torch.arange(self.native_vocab_size)
        canonical, compact = folded_ids_from_index(index)
        self.register_buffer("canonical_ids", canonical, persistent=False)
        self.register_buffer("fold_compact", compact, persistent=False)
        self.vocab_size = int(len(canonical))            # dimension of the output (folded) space
        self.folded_vocab = FoldedVocab(canonical.numpy(), token_strings(self.tokenizer, canonical.tolist()),
                                        self.name_or_path)
        # the logits are computed as decoder(head(backbone(x))): same as ModernBertForMaskedLM.forward,
        # but lets us keep the MLM head in float32 under bf16 autocast (the logits sit around 15-45, where
        # bf16 has a resolution of 0.125-0.25). 00_smoke_test checks the equality with the parent forward.
        self._manual_head = all(hasattr(self.mlm, a) for a in ("model", "head", "decoder"))

    # ------------------------------------------------------------------ forward
    def _hidden(self, input_ids, attention_mask):
        return self.mlm.model(input_ids=input_ids, attention_mask=attention_mask)[0]

    def _head_logits(self, h):
        return self.mlm.decoder(self.mlm.head(h))

    def _parent_logits(self, input_ids, attention_mask):
        """Logits exactly as the official code obtains them: ModernBertForMaskedLM.forward(...).logits."""
        for klass in type(self.mlm).__mro__[1:]:
            if klass.__name__ == "ModernBertForMaskedLM":
                return klass.forward(self.mlm, input_ids=input_ids, attention_mask=attention_mask).logits
        return self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits

    def forward(self, input_ids, attention_mask, pooling_mask=None, amp=False, exact=None, head_chunk=32):
        """(B, C) SparseUp weights in the folded space.
        amp   : backbone in bf16 autocast (CUDA only), MLM head always in float32 (training, e2e encodings)
        exact : float32 without TF32 (default: True when amp is off) -> same numbers as the official fp32 code
        head_chunk: sequences per head/top-k block (bounds the memory of the (b, L, 50k) logits)."""
        if pooling_mask is None:
            pooling_mask = attention_mask
        dev_type = input_ids.device.type
        amp = bool(amp and dev_type == "cuda")
        exact = (not amp) if exact is None else bool(exact)
        prec = full_fp32() if exact else nullcontext()
        B = input_ids.shape[0]
        C = self.vocab_size
        with prec:
            if self._manual_head:
                with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
                    h = self._hidden(input_ids, attention_mask)
                h = h.float()
            pooled = torch.full((B, C), float("-inf"), device=input_ids.device, dtype=torch.float32)
            parts = []
            for s in range(0, B, head_chunk):
                e = min(B, s + head_chunk)
                with torch.autocast(device_type=dev_type, enabled=False):
                    if self._manual_head:
                        logits = self._head_logits(h[s:e])
                    else:
                        logits = self._parent_logits(input_ids[s:e], attention_mask[s:e])
                    logits = logits.float()                                        # (b, L, V)
                    if self.position_top_k is not None and self.position_top_k < logits.shape[-1]:
                        vals, idx = torch.topk(logits, self.position_top_k, dim=-1)  # (b, L, k)
                    else:
                        vals = logits
                        idx = torch.arange(logits.shape[-1], device=logits.device).expand_as(logits)
                    del logits
                    keep = pooling_mask[s:e].bool().unsqueeze(-1)
                    vals = vals.masked_fill(~keep, float("-inf"))
                    tgt = self.fold_compact[idx].reshape(e - s, -1)                  # folded ids
                    parts.append(pooled[s:e].scatter_reduce(1, tgt, vals.reshape(e - s, -1), reduce="amax",
                                                            include_self=True))
            pooled = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
            return torch.log1p(torch.relu(pooled - self.logit_shift))

    # ------------------------------------------------------------------ tokenization
    def prefix_and_length(self, kind, max_length=None):
        if kind == "query":
            return self.query_prefix, int(max_length or self.query_max_length)
        if kind == "document":
            return self.document_prefix, int(max_length or self.doc_max_length)
        raise ValueError(f"kind must be 'query' or 'document', not {kind!r}")

    def tokenize(self, texts, max_length, device, kind="document"):
        """(input_ids, attention_mask, pooling_mask) as the official _tokenize: the prefix is prepended to the text
        (and counts in max_length); its tokens are attended to but excluded from the pooling (special tokens
        such as [CLS] and [SEP] are pooled)."""
        prefix, max_length = self.prefix_and_length(kind, max_length)
        enc = self.tokenizer([prefix + t for t in texts], padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True)
        pool = enc["attention_mask"].clone()
        if prefix:
            in_prefix = (enc["offset_mapping"][..., 0] < len(prefix)) & ~enc["special_tokens_mask"].bool()
            pool = pool.masked_fill(in_prefix, 0)
        return enc["input_ids"].to(device), enc["attention_mask"].to(device), pool.to(device)

    def encode_dense(self, texts, max_length, device, amp=False, kind="document"):
        ids, mask, pool = self.tokenize(texts, max_length, device, kind)
        return self(ids, mask, pool, amp=amp).float()

    @torch.no_grad()
    def encode_to_csr(self, texts, device, batch_size=128, max_length=None, amp=False, ids=None,
                      scale=1.0, verbose=True, kind="document"):
        """Encode a list of texts into a CSR (folded space). Texts are sorted by length
        (less padding = faster) and then restored to their original order."""
        was_training = self.training
        self.eval()
        texts = list(texts)
        order = np.argsort([len(t) for t in texts], kind="stable")
        parts = []
        step = max(1, (len(texts) // batch_size) // 20)
        for b, s in enumerate(range(0, len(texts), batch_size)):
            batch_idx = order[s:s + batch_size]
            x = self.encode_dense([texts[i] for i in batch_idx], max_length, device, amp, kind) * scale
            parts.append(dense_to_csr(x, self.vocab_size))
            if verbose and b % step == 0:
                log(f"  encoded {min(s + batch_size, len(texts)):,}/{len(texts):,} texts")
        if was_training:
            self.train()
        if not parts:
            return CSR(np.zeros(1, dtype=np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32),
                       self.vocab_size, None if ids is None else np.asarray(ids))
        csr_sorted = CSR.concat(parts)
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        out = csr_sorted.subset(inverse)
        out.ids = None if ids is None else np.asarray(ids)
        return out

    def save(self, folder):
        """Save the fine-tuned backbone so that it can be reloaded with trust_remote_code (same class as the
        official model): weights + config + tokenizer + the modeling file of the SpladeModel class."""
        folder = Path(folder)
        self.mlm.save_pretrained(folder)
        self.tokenizer.save_pretrained(folder)
        if not any(folder.glob("modeling*.py")):
            src = inspect.getsourcefile(type(self.mlm))
            if src:
                shutil.copy(src, folder / Path(src).name)


def dense_to_csr(x, n_cols):
    """(B, n_cols) tensor -> CSR of its positive entries."""
    r, c = torch.nonzero(x > 0, as_tuple=True)
    vals = x[r, c]
    counts = torch.bincount(r, minlength=x.shape[0]).cpu().numpy()
    indptr = np.zeros(x.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return CSR(indptr, c.cpu().numpy().astype(np.int32), vals.float().cpu().numpy().astype(np.float32), n_cols)


def top_tokens(vocab, idx, vals, n=15):
    """[(token, weight)] for the highest weights of a sparse vector (folded ids). `vocab` is a FoldedVocab
    (or anything with convert_ids_to_tokens on folded ids)."""
    idx = np.asarray(idx)
    vals = np.asarray(vals)
    order = np.argsort(-vals)[:n]
    toks = vocab.convert_ids_to_tokens([int(i) for i in idx[order]])
    return [(t, float(v)) for t, v in zip(toks, vals[order])]
