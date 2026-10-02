"""Retrieval behaviour that the agent depends on, independent of the embedding model."""
from __future__ import annotations

import numpy as np

from app.embeddings import HashingEmbedder, _l2_normalise
from app.retriever import BM25, PolicyRetriever


def test_vectors_are_normalised_so_dot_product_is_cosine():
    matrix = _l2_normalise(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_zero_vector_does_not_produce_nan():
    """A degenerate embedding must read as 'no match', not poison every score."""
    matrix = _l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert not np.isnan(matrix).any()


def test_search_returns_k_results_best_first(offline_retriever):
    hits = offline_retriever.search("pet fee", 3)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_handles_empty_query(offline_retriever):
    assert offline_retriever.search("   ", 3) == []


def test_k_larger_than_corpus_is_clamped(offline_retriever):
    hits = offline_retriever.search("parking", 99)
    assert len(hits) == len(offline_retriever.chunks)


def test_evidence_exposes_both_scores(offline_retriever):
    hit = offline_retriever.search("pet fee", 1)[0]
    debug = hit.as_debug()
    # dense_score is the calibrated one the gate reads; score is for ranking only.
    assert "dense_score" in debug and "score" in debug
    assert debug["citation"] and debug["chunk_id"]
    assert debug["source"], "evidence must name the document it came from"


def test_bm25_prefix_matching_handles_word_endings():
    """The lexical signal exists because dense retrieval misses exact-term questions.
    The 5-character prefix is what lets "refundable" reach a passage about "refund"."""
    bm25 = BM25(["The security deposit is refundable after move-out.",
                 "Quiet hours are 10 PM to 8 AM."])
    assert bm25.scores("is the deposit refunded?")[0] > bm25.scores("is the deposit refunded?")[1]


def test_hybrid_and_dense_expose_the_same_interface():
    from app.chunking import parse_corpus
    from app.config import RENTAL_CORPUS_DIR

    chunks = parse_corpus(RENTAL_CORPUS_DIR)
    embedder = HashingEmbedder()
    for hybrid in (True, False):
        retriever = PolicyRetriever(chunks, embedder, hybrid=hybrid)
        hits = retriever.search("parking", 2)
        assert len(hits) == 2
        assert all(h.dense_score <= 1.0001 for h in hits)
