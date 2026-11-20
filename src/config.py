"""
config.py — Paths and settings shared by ALL scripts.

Set the directories with environment variables (recommended, e.g. in ~/.bashrc):

    export SPARSEUP_WORK_DIR=/datafast/$USER/sparseup_concept
    export SPARSEUP_FM_DIR=$SPARSEUP_WORK_DIR/encoded          # optional: this is already the default

  SPARSEUP_WORK_DIR : YOUR directory, where caches, models and results are saved.
                      It needs space (~150-250 GB for the full set of experiments): do NOT use your home directory.
                      Default: /datafast/$USER/sparseup_concept if /datafast exists, otherwise ./work
  SPARSEUP_FM_DIR   : directory with the pre-encoded SparseUp vectors of MS MARCO in Seismic format
                      (documents.bin, queries.bin, doc_ids.npy, queries_ids.npy, token_to_id_mapping.json,
                      groundtruth_100.tsv). In the SPLADE project these files were PROVIDED to us; for SparseUp
                      nothing is provided, so 02b_encode_msmarco.py creates them with the OFFICIAL SparseUp
                      implementation (trust_remote_code). After that, the folder is only read.

The variables have a SPARSEUP_ prefix ON PURPOSE: WORK_DIR / FM_DIR of the SPLADE project (probably still in your
~/.bashrc) are IGNORED, so this code can never read the SPLADE vectors or write into splade_concept.
"""
import json
import os
from pathlib import Path

USER = os.environ.get("USER", "user")

# Working directory (large). Change it if you do not have write permissions.
_DEFAULT_WORK = Path(f"/datafast/{USER}/sparseup_concept") if Path("/datafast").is_dir() else Path("work")
WORK_DIR = Path(os.environ.get("SPARSEUP_WORK_DIR", str(_DEFAULT_WORK)))

# Pre-encoded SparseUp data (Seismic format). Default: WORK_DIR/encoded (created by 02b).
FM_DIR = Path(os.environ["SPARSEUP_FM_DIR"]) if os.environ.get("SPARSEUP_FM_DIR") else None

# SparseUp model (Linkup Research, Apache 2.0). Script 03 checks that OUR memory-efficient encoder
# (sparseup_encoder.py) reproduces the vectors of the official implementation and saves the choice.
DEFAULT_SPARSEUP_MODEL = os.environ.get("SPARSEUP_MODEL", "Linkup-Platform/linkup-sparseup-embed-v1")
CANDIDATE_SPARSEUP_MODELS = [
    "Linkup-Platform/linkup-sparseup-embed-v1",
]

# Max lengths (in tokens, the "[Q] "/"[D] " prefix included) used for every encoding that ends up
# in a SEARCH: corpus, dev/DL/BEIR queries and documents, end-to-end re-encoding of the corpus.
# They are the official SparseUp evaluation lengths (config.json: query_max_length, doc_max_length),
# so the baseline is exactly the published model.
# The TRAINING batches (06) keep the lengths of the SPLADE Concept Model (query 64, document 256).
Q_MAX_LEN = int(os.environ.get("SPARSEUP_Q_MAX_LEN", 128))
D_MAX_LEN = int(os.environ.get("SPARSEUP_D_MAX_LEN", 512))

# File names in the pre-encoded data directory (same names as the files of the SPLADE project).
FM_FILES = {
    "documents": "documents.bin",
    "queries": "queries.bin",
    "doc_ids": "doc_ids.npy",
    "query_ids": "queries_ids.npy",
    "mapping": "token_to_id_mapping.json",
    "groundtruth": "groundtruth_100.tsv",
}


class Paths:
    """All paths derived from WORK_DIR, in one place."""

    def __init__(self, work_dir=None, fm_dir=None):
        self.work = Path(work_dir) if work_dir else WORK_DIR
        self.fm = Path(fm_dir) if fm_dir else (FM_DIR if FM_DIR is not None else self.work / "encoded")

        # cache (SparseUp vectors in the FOLDED vocabulary space, ~34k dims, see sparseup_encoder.py)
        self.cache = self.work / "cache"
        self.docs_csr = self.cache / "docs_folded"              # 8.8M documents (encoded by 02b)
        self.dev_queries_csr = self.cache / "dev_queries_folded"  # dev queries (encoded by 02b)
        self.train_queries_csr = self.cache / "train_queries_folded"  # training queries (encoded by 05)
        self.val_candidates = self.cache / "val_candidates"
        self.folded_vocab = self.cache / "folded_vocab.json"    # folded id -> token (written by 01/02b)

        # MS MARCO raw text (passages, queries, qrels)
        self.msmarco = self.work / "msmarco"

        # output
        self.negatives = self.work / "negatives"
        self.models = self.work / "models"
        self.embeddings = self.work / "embeddings"
        self.results = self.work / "results"
        self.runs = self.work / "runs"
        self.lce = self.work / "lce"
        self.encoder_check = self.work / "encoder_check.json"
        self.summary_csv = self.results / "summary.csv"

    def fm_file(self, key):
        return self.fm / FM_FILES[key]

    # --- MS MARCO files ---
    @property
    def collection_tsv(self):
        return self.msmarco / "collection.tsv"

    def queries_tsv(self, split):
        names = {
            "train": "queries.train.tsv",
            "dev": "queries.dev.small.tsv",
            "dl19": "msmarco-test2019-queries.tsv.gz",
            "dl20": "msmarco-test2020-queries.tsv.gz",
        }
        return self.msmarco / names[split]

    def qrels_tsv(self, split):
        names = {
            "train": "qrels.train.tsv",
            "dev": "qrels.dev.small.tsv",
            "dl19": "2019qrels-pass.txt",
            "dl20": "2020qrels-pass.txt",
        }
        return self.msmarco / names[split]

    def ensure_work_dir(self):
        """Create WORK_DIR (and subdirectories), with a clear message if that is not possible."""
        try:
            for d in [self.work, self.cache, self.msmarco, self.negatives, self.models,
                      self.embeddings, self.results, self.runs, self.lce]:
                d.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise SystemExit(
                f"\n[ERROR] Cannot create {self.work} (permission denied).\n"
                "Choose a directory you can write to, for example:\n"
                "    export SPARSEUP_WORK_DIR=/datafast/$USER/sparseup_concept\n"
                "or change WORK_DIR in config.py.\n")


def resolve_ckpt(name_or_path, paths):
    """'frozen_k256_r1' -> WORK_DIR/models/frozen_k256_r1/best (if it exists), otherwise the path."""
    p = Path(name_or_path)
    if p.exists():
        return p
    for cand in [paths.models / name_or_path / "best", paths.models / name_or_path / "last",
                 paths.models / name_or_path]:
        if (cand / "concept_head.pt").exists():
            return cand
    raise SystemExit(f"[ERROR] Checkpoint not found: {name_or_path} (also searched in {paths.models})")


def resolve_negatives(name_or_path, paths):
    """'r1' -> WORK_DIR/negatives/r1.npz"""
    p = Path(name_or_path)
    if p.exists():
        return p
    cand = paths.negatives / (name_or_path if name_or_path.endswith(".npz") else name_or_path + ".npz")
    if cand.exists():
        return cand
    raise SystemExit(f"[ERROR] Negatives file not found: {name_or_path} (also tried {cand})")


def get_sparseup_model(paths):
    """SparseUp model to use: the one chosen by 03_check_encoder.py, if available."""
    if "SPARSEUP_MODEL" in os.environ:
        return os.environ["SPARSEUP_MODEL"], 1.0
    if paths.encoder_check.exists():
        info = json.loads(paths.encoder_check.read_text())
        return info["model"], float(info.get("scale", 1.0))
    return DEFAULT_SPARSEUP_MODEL, 1.0
