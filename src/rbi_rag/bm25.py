"""Step 8: BM25 scratch se (numpy). Inverted index: term -> (doc ids, term freqs).

score(d, q) = sum over unique query terms t:  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))      (Lucene-style, hamesha >= 0)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


class BM25:
    def __init__(self, docs_tokens, k1: float = 1.5, b: float = 0.75):
        self.N = len(docs_tokens)
        self.k1, self.b = k1, b
        self.dl = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        postings = defaultdict(lambda: ([], []))
        for i, toks in enumerate(docs_tokens):
            for t, c in Counter(toks).items():
                ids, tfs = postings[t]
                ids.append(i)
                tfs.append(c)
        self.idf, self.ids, self.tf = {}, {}, {}
        for t, (ids, tfs) in postings.items():
            df = len(ids)
            self.idf[t] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            self.ids[t] = np.array(ids, dtype=np.int64)
            self.tf[t] = np.array(tfs, dtype=np.float64)
        # length normalization har doc ke liye ek baar
        self.norm = (k1 * (1.0 - b + b * self.dl / self.avgdl)) if self.avgdl > 0 else np.zeros(self.N)

    def scores(self, q_tokens) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float64)
        for t in set(q_tokens):                      # query term ek hi baar count hota hai
            if t not in self.ids:
                continue
            ids, tf = self.ids[t], self.tf[t]
            s[ids] += self.idf[t] * tf * (self.k1 + 1.0) / (tf + self.norm[ids])
        return s

    def top(self, q_tokens, k: int):
        """[(doc_index, score)] descending; ties mein chhota index pehle; score 0 wale nahi."""
        s = self.scores(q_tokens)
        order = np.lexsort((np.arange(self.N), -s))[:k]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]