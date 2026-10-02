"""Retrieval over the property documents: dense cosine, optionally fused with BM25.

Brute-force numpy over ~45 chunks. An approximate index (FAISS, Qdrant) would add a
dependency and a failure mode to speed up a 10x1536 dot product that takes
microseconds -- so it isn't here. The interface is the one a vector DB would sit
behind if the corpus grew.

Two scores travel with every result and they do different jobs:
  dense_score -- raw cosine. Calibrated and comparable across queries, so this is
                 what the out-of-scope similarity gate reads.
  score       -- the ranking score (fused when hybrid is on). Good for ordering,
                 NOT calibrated, so it is never compared to a threshold.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.chunking import Chunk, parse_corpus
from app.embeddings import Embedder

RRF_K = 60  # standard Reciprocal Rank Fusion constant


@dataclass(frozen=True)
class Evidence:
    chunk: Chunk
    score: float        # ranking score (fused if hybrid)
    dense_score: float  # raw cosine similarity -- the gate reads this

    def as_debug(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "citation": self.chunk.citation,
            "source": self.chunk.source,
            "score": round(self.score, 4),
            "dense_score": round(self.dense_score, 4),
            "text": self.chunk.text,
        }


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens, each also emitted as a 5-char prefix.

    The prefix is a deliberately crude stand-in for stemming. It is what makes
    "Has he published anything?" reach the Publication entry: `published` and
    `publication` share the prefix `publi`, which a plain token match misses and a
    real stemmer would handle properly. Cheap, transparent, and good enough for one
    resume -- a larger corpus should use a proper analyser.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = list(words)
    tokens.extend(w[:5] for w in words if len(w) > 5)
    return tokens


class BM25:
    """Okapi BM25 over a handful of documents. ~30 lines, no dependency."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs = [Counter(_tokenise(d)) for d in documents]
        self._lengths = [sum(c.values()) for c in self._docs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._docs else 0.0
        n = len(self._docs)
        document_frequency = Counter(term for doc in self._docs for term in doc)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        terms = _tokenise(query)
        out = np.zeros(len(self._docs), dtype=np.float32)
        for i, doc in enumerate(self._docs):
            length = self._lengths[i] or 1
            total = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self._avg_length)
                total += self._idf.get(term, 0.0) * tf * (self.k1 + 1) / denominator
            out[i] = total
        return out


class PolicyRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        hybrid: bool = True,
        dense_weight: float = 0.5,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.hybrid = hybrid
        # Weight on the dense vote in the fusion. Equal weighting (0.5) measurably hurt
        # education queries -- BM25 has no term overlap for "where did he go to school"
        # and dragged a correct rank-1 hit to rank 5. Swept in eval/run_retrieval_eval.py.
        self.dense_weight = dense_weight
        # Vectors are L2-normalised, so a dot product IS cosine similarity.
        self._matrix = embedder.embed([c.text for c in chunks])
        self._bm25 = BM25([c.text for c in chunks]) if hybrid else None

    @classmethod
    def from_corpus(
        cls,
        path: Path,
        embedder: Embedder,
        hybrid: bool = True,
        dense_weight: float = 0.5,
    ) -> PolicyRetriever:
        return cls(parse_corpus(path), embedder, hybrid=hybrid, dense_weight=dense_weight)

    def search(self, query: str, k: int) -> list[Evidence]:
        if not query.strip() or not self.chunks:
            return []
        dense = self._matrix @ self.embedder.embed([query])[0]

        if self._bm25 is None:
            ranking = dense
        else:
            # Fuse by rank, not by score: BM25 is unbounded and cosine sits in a narrow
            # band, so adding them would let whichever has the wider range dominate.
            lexical = self._bm25.scores(query)
            ranking = np.zeros(len(self.chunks), dtype=np.float32)
            weights = ((dense, self.dense_weight), (lexical, 1.0 - self.dense_weight))
            for scores, weight in weights:
                order = np.argsort(-scores)
                for rank, index in enumerate(order, start=1):
                    ranking[index] += weight / (RRF_K + rank)

        k = min(k, len(self.chunks))
        top = np.argpartition(-ranking, k - 1)[:k]
        top = top[np.argsort(-ranking[top])]
        return [
            Evidence(chunk=self.chunks[i], score=float(ranking[i]), dense_score=float(dense[i]))
            for i in top
        ]
