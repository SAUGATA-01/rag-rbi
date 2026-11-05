"""Step 8: dense retrieval (naive): embedder + embedding cache + numpy cosine index."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from rbi_rag.textproc import tokenize


# ---------------- embedders ----------------
class HashEmbedder:
    """DEBUG/TEST embedder: hashing-trick bag-of-words (koi model download nahi). Asli semantic embedder NAHI hai."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash{dim}"
        self.calls = []  # har encode_passages batch ka size (cache test ke liye)

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 100) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def encode_passages(self, texts):
        self.calls.append(len(texts))
        return np.stack([self._vec(t) for t in texts]).astype(np.float32) if len(texts) else np.zeros((0, self.dim), np.float32)

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)


class STEmbedder:
    """sentence-transformers (CPU). Default: BAAI/bge-small-en-v1.5."""
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", max_seq_length=512, batch_size=32,
                 device="cpu", query_prefix=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("sentence-transformers install nahi hai: "
                               "pip install torch --index-url https://download.pytorch.org/whl/cpu && "
                               "pip install sentence-transformers") from e
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.name = f"{model_name}@{max_seq_length}"
        self.batch_size = batch_size
        self.query_prefix = self.QUERY_PREFIX if query_prefix is None else query_prefix
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def _enc(self, texts, bar):
        v = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                              show_progress_bar=bar, convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32)

    def encode_passages(self, texts):
        return self._enc(texts, bar=len(texts) > 64)

    def encode_queries(self, texts):
        return self._enc([self.query_prefix + t for t in texts], bar=False)


# ---------------- cache ----------------
def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


class EmbeddingCache:
    """key = sha1(passage text). Sirf naye/badle hue texts embed hote hain; baaki disk se."""

    def __init__(self, root, model_name: str):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        self.dir = Path(root) / slug
        self.vec_path, self.keys_path = self.dir / "vectors.npy", self.dir / "keys.json"
        self.vecs = {}
        if self.vec_path.exists() and self.keys_path.exists():
            keys = json.loads(self.keys_path.read_text(encoding="utf-8"))
            mat = np.load(self.vec_path)
            if len(keys) == len(mat):
                self.vecs = {k: mat[i] for i, k in enumerate(keys)}

    def get_matrix(self, texts, embedder):
        keys = [text_key(t) for t in texts]
        dim_mismatch = self.vecs and next(iter(self.vecs.values())).shape[0] != getattr(embedder, "dim", None)
        if dim_mismatch:
            self.vecs = {}
        todo, seen = [], set()
        for k, t in zip(keys, texts):
            if k not in self.vecs and k not in seen:
                seen.add(k)
                todo.append((k, t))
        t0 = time.time()
        if todo:
            new = embedder.encode_passages([t for _, t in todo])
            for (k, _), v in zip(todo, new):
                self.vecs[k] = v
        secs = time.time() - t0
        mat = np.stack([self.vecs[k] for k in keys]).astype(np.float32) if keys else np.zeros((0, 1), np.float32)
        self._save(set(keys))
        return mat, {"chunks": len(texts), "embedded": len(todo), "cached": len(set(keys)) - len(todo),
                     "seconds": round(secs, 1)}

    def _save(self, keep):
        self.vecs = {k: v for k, v in self.vecs.items() if k in keep}  # purane/orphan hatao
        if not self.vecs:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        keys = list(self.vecs)
        tmp_v, tmp_k = self.vec_path.with_suffix(".tmp"), self.keys_path.with_suffix(".tmp")
        with open(tmp_v, "wb") as f:
            np.save(f, np.stack([self.vecs[k] for k in keys]).astype(np.float32))
        tmp_k.write_text(json.dumps(keys), encoding="utf-8")
        os.replace(tmp_v, self.vec_path)
        os.replace(tmp_k, self.keys_path)


# ---------------- index ----------------
class VectorIndex:
    """Normalized vectors par dot product = cosine similarity."""

    def __init__(self, matrix):
        self.M = np.asarray(matrix, dtype=np.float32)

    def top(self, qvec, k):
        s = self.M @ np.asarray(qvec, dtype=np.float32)
        order = np.lexsort((np.arange(len(s)), -s))[:k]
        return [(int(i), float(s[i])) for i in order]