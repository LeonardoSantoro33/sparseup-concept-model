"""
sparseup_reference.py — A minimal, deliberately STRAIGHTFORWARD re-implementation of the official SparseUp model
(modeling_splade.py of Linkup-Platform/linkup-sparseup-embed-v1), used ONLY by 00_smoke_test.py.

The smoke test cannot download the real 750 MB checkpoint, so it builds a tiny random ModernBERT with this file as
its remote code (config.json "auto_map" -> modeling_splade.SpladeConfig / SpladeModel) and loads it with
trust_remote_code=True, exactly like the real model. This file follows the official algorithm literally
(full (B, L, V) activation tensor), so the smoke test checks that the memory-efficient sparseup_encoder.py
gives the same vectors. On the cluster, 03_check_encoder.py does the same check against the REAL official code.

Official algorithm (per position i, vocabulary entry v):
    a_{i,v}  = log(1 + ReLU(logit_{i,v} - logit_shift))
    top-k    : a_{i,v} kept only if >= the k-th largest a_{i,.} (ties keep more than k)
    pooling  : max over the positions with pooling_mask = 1 (padding and prefix tokens excluded)
    folding  : w_c = amax over the v with vocab_fold_index[v] = c (output stays V-dimensional)
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM


class SpladeConfig(ModernBertConfig):
    def __init__(self, logit_shift=0.0, position_top_k=None, vocab_fold=None, query_prefix="",
                 document_prefix="", query_max_length=128, doc_max_length=512, **kwargs):
        super().__init__(**kwargs)
        self.logit_shift = logit_shift
        self.position_top_k = position_top_k
        self.vocab_fold = vocab_fold
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length


class SpladeModel(ModernBertForMaskedLM):
    config_class = SpladeConfig

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("vocab_fold_index", torch.arange(config.vocab_size), persistent=True)
        self._tokenizer = None

    def _token_weights(self, input_ids, attention_mask):
        logits = super().forward(input_ids=input_ids, attention_mask=attention_mask).logits
        w = torch.log1p(F.relu(logits - self.config.logit_shift))
        k = self.config.position_top_k
        if k is not None and k < w.shape[-1]:
            cutoff = w.topk(k, dim=-1).values[..., -1:]
            w = w * (w >= cutoff)
        return w

    def _fold(self, sparse):
        index = self.vocab_fold_index.unsqueeze(0).expand_as(sparse)
        return torch.zeros_like(sparse).scatter_reduce(1, index, sparse, reduce="amax", include_self=False)

    def forward(self, input_ids, attention_mask, pooling_mask=None, **kwargs):
        if pooling_mask is None:
            pooling_mask = attention_mask
        w = self._token_weights(input_ids, attention_mask)
        w = w * pooling_mask.to(w.dtype).unsqueeze(-1)
        out = w.max(dim=1).values
        if self.config.vocab_fold is not None:
            out = self._fold(out)
        return out

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        return self._tokenizer

    def _encode_args(self, kind, max_length):
        if kind == "query":
            return self.config.query_prefix, max_length or self.config.query_max_length
        if kind == "document":
            return self.config.document_prefix, max_length or self.config.doc_max_length
        raise ValueError(kind)

    def _tokenize(self, texts, prefix, max_length):
        enc = self._get_tokenizer()([prefix + t for t in texts], padding=True, truncation=True,
                                    max_length=max_length, return_tensors="pt", return_offsets_mapping=True,
                                    return_special_tokens_mask=True)
        pooling_mask = enc["attention_mask"]
        if prefix:
            in_prefix = (enc["offset_mapping"][..., 0] < len(prefix)) & ~enc["special_tokens_mask"].bool()
            pooling_mask = pooling_mask.masked_fill(in_prefix, 0)
        return enc["input_ids"], enc["attention_mask"], pooling_mask, enc["offset_mapping"]

    @torch.inference_mode()
    def encode(self, texts, kind="document", batch_size=32, max_length=None):
        prefix, max_length = self._encode_args(kind, max_length)
        device = next(self.parameters()).device
        out = []
        for s in range(0, len(texts), batch_size):
            ids, attn, pool, _ = self._tokenize(list(texts[s:s + batch_size]), prefix, max_length)
            out.append(self(ids.to(device), attn.to(device), pool.to(device)).float().cpu())
        return torch.cat(out)

    @staticmethod
    def score(queries, documents):
        return queries @ documents.T


def build_fold_index(tokenizer):
    """vocab_fold_index for the synthetic model, following the rule described in the SparseUp blog post:
    tokens with the same lowercase body (case and 'Ġ' variants) are grouped; the canonical member is the
    space-marked lowercase form if it exists, then the bare lowercase form, then any other member.
    (The real model stores its map in the checkpoint: this function is never used on it.)"""
    vocab = tokenizer.get_vocab()
    special = set(tokenizer.all_special_tokens)
    groups = {}
    for tok, i in vocab.items():
        if tok in special:
            continue
        body = tok[1:] if tok.startswith("Ġ") else tok
        groups.setdefault(body.lower(), []).append((tok, i))
    index = list(range(max(vocab.values()) + 1))
    for body, members in groups.items():
        names = {t: i for t, i in members}
        canon = names.get("Ġ" + body, names.get(body, min(i for _, i in members)))
        for _, i in members:
            index[i] = canon
    return torch.tensor(index, dtype=torch.long)
