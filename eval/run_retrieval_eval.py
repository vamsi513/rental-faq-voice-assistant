"""Measure retrieval quality on eval/retrieval_set.json.

Metrics
  Hit@k    : fraction of answerable questions where AT LEAST ONE labeled-relevant chunk
             appears in the top k. This was previously mislabelled "Recall@k" -- it is
             not. Recall@k is the share of all relevant chunks retrieved; Hit@k only
             asks whether any of them made it. The two differ on the 4 questions here
             that have more than one relevant chunk, and Hit@k is the more flattering
             number, so the old name overstated the result.
  Recall@k : the real thing -- mean over questions of
             |relevant retrieved in top k| / |relevant|. Lower than Hit@k whenever a
             question has several valid supporting chunks.
  MRR      : mean of 1/(rank of the first relevant chunk). Rewards putting the right
             passage first. Matters here because voice answers are short, so the top
             chunk dominates what gets said.
  Separation : mean top-1 score on answerable vs unanswerable questions. Drives the
             similarity gate -- the two distributions must be far enough apart that
             one threshold can tell them apart.

Note on k: the agent runs with TOP_K=3, so only the top 3 chunks ever reach the
generator. Hit@5 is reported for diagnosis -- a question that is hit at 5 but not at 3
is one the agent CANNOT currently answer.

Usage
  python eval/run_retrieval_eval.py                 # dense only (default)
  python eval/run_retrieval_eval.py --hybrid        # dense + BM25 fusion
  python eval/run_retrieval_eval.py --offline       # hashing embedder, no API calls
  python eval/run_retrieval_eval.py --sweep         # threshold sweep for MIN_SIMILARITY
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import RENTAL_CORPUS_DIR, settings  # noqa: E402
from app.embeddings import CachedEmbedder, HashingEmbedder, OpenAIEmbedder  # noqa: E402
from app.retriever import PolicyRetriever  # noqa: E402

K_VALUES = (1, 3, 5)


def build_retriever(offline: bool, hybrid: bool, dense_weight: float = 0.5) -> PolicyRetriever:
    if offline:
        embedder = HashingEmbedder()
    else:
        if not settings.openai_api_key:
            sys.exit("OPENAI_API_KEY not set. Re-run with --offline for the stand-in embedder.")
        embedder = CachedEmbedder(
            OpenAIEmbedder(settings.openai_api_key, settings.embedding_model, settings.embedding_dim),
            ROOT / "data/emb_cache.json",
        )
    return PolicyRetriever.from_corpus(RENTAL_CORPUS_DIR, embedder, hybrid=hybrid, dense_weight=dense_weight)


def evaluate(retriever: PolicyRetriever, questions: list[dict], max_k: int = 5) -> dict:
    answerable = [q for q in questions if q["relevant_chunks"]]
    unanswerable = [q for q in questions if not q["relevant_chunks"]]

    hits_at = {k: 0 for k in K_VALUES}
    recall_at: dict[int, list[float]] = {k: [] for k in K_VALUES}
    reciprocal_ranks: list[float] = []
    per_question: list[dict] = []
    answerable_top1: list[float] = []

    for q in answerable:
        results = retriever.search(q["question"], max_k)
        ranked_ids = [e.chunk.chunk_id for e in results]
        relevant = set(q["relevant_chunks"])
        first_rank = next((i + 1 for i, cid in enumerate(ranked_ids) if cid in relevant), None)

        for k in K_VALUES:
            if first_rank is not None and first_rank <= k:
                hits_at[k] += 1
            # True recall: how many of this question's relevant chunks are in the top k.
            found = sum(1 for cid in ranked_ids[:k] if cid in relevant)
            recall_at[k].append(found / len(relevant))
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        answerable_top1.append(results[0].dense_score if results else 0.0)
        per_question.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "first_relevant_rank": first_rank,
                "top1": ranked_ids[0] if ranked_ids else None,
                "top1_dense": round(results[0].dense_score, 4) if results else 0.0,
                "expected": sorted(relevant),
            }
        )

    unanswerable_top1 = []
    for q in unanswerable:
        results = retriever.search(q["question"], max_k)
        unanswerable_top1.append(results[0].dense_score if results else 0.0)
        per_question.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "first_relevant_rank": None,
                "top1": results[0].chunk.chunk_id if results else None,
                "top1_dense": round(results[0].dense_score, 4) if results else 0.0,
                "expected": [],
            }
        )

    n = len(answerable)
    return {
        "n_answerable": n,
        "n_unanswerable": len(unanswerable),
        "hit_at": {f"@{k}": round(hits_at[k] / n, 4) for k in K_VALUES},
        "recall_at": {f"@{k}": round(sum(recall_at[k]) / n, 4) for k in K_VALUES},
        "n_multi_chunk": sum(1 for q in answerable if len(q["relevant_chunks"]) > 1),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "mean_top1_answerable": round(sum(answerable_top1) / n, 4),
        "mean_top1_unanswerable": round(sum(unanswerable_top1) / len(unanswerable), 4)
        if unanswerable
        else None,
        "min_top1_answerable": round(min(answerable_top1), 4),
        "max_top1_unanswerable": round(max(unanswerable_top1), 4) if unanswerable else None,
        "per_question": per_question,
    }


def sweep(retriever: PolicyRetriever, questions: list[dict]) -> None:
    """Pick MIN_SIMILARITY: the gate must admit answerable questions and reject the rest."""
    answerable = [q for q in questions if q["relevant_chunks"]]
    unanswerable = [q for q in questions if not q["relevant_chunks"]]
    a_scores = [retriever.search(q["question"], 1)[0].dense_score for q in answerable]
    u_scores = [retriever.search(q["question"], 1)[0].dense_score for q in unanswerable]

    print(f"\n{'threshold':>10} {'answerable kept':>16} {'unanswerable rejected':>23}")
    best = None
    for t in [round(x * 0.01, 2) for x in range(5, 46)]:
        kept = sum(s >= t for s in a_scores) / len(a_scores)
        rejected = sum(s < t for s in u_scores) / len(u_scores)
        if t * 100 % 2 == 0:
            print(f"{t:>10.2f} {kept:>15.1%} {rejected:>22.1%}")
        # Prefer thresholds that keep every answerable question, then maximise rejection.
        score = (kept == 1.0, rejected, kept)
        if best is None or score > best[0]:
            best = (score, t, kept, rejected)
    _, t, kept, rejected = best
    print(
        f"\nrecommended MIN_SIMILARITY={t:.2f} "
        f"(keeps {kept:.1%} of answerable, rejects {rejected:.1%} of unanswerable)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="hashing embedder, no API calls")
    parser.add_argument("--hybrid", action="store_true", help="fuse dense with BM25")
    parser.add_argument("--dense-weight", type=float, default=0.5, help="dense vote weight in fusion")
    parser.add_argument("--sweep-weight", action="store_true", help="sweep the fusion weight")
    parser.add_argument("--sweep", action="store_true", help="sweep the similarity gate")
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args()

    questions = json.loads((ROOT / "eval/retrieval_set.json").read_text())["questions"]
    retriever = build_retriever(args.offline, args.hybrid, args.dense_weight)

    backend = "hashing-4gram (OFFLINE STAND-IN -- not comparable to real numbers)" if args.offline \
        else f"{retriever.embedder.name} ({retriever.embedder.dim}d)"
    mode = "dense + BM25 fusion" if args.hybrid else "dense only"
    print(f"embedder : {backend}")
    print(f"retrieval: {mode}")
    print(f"corpus   : {len(retriever.chunks)} chunks from {RENTAL_CORPUS_DIR.name}/")

    if args.sweep_weight:
        print("\nfusion weight sweep (dense_weight=1.0 is dense-only):")
        print(f"{'dense_w':>8} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7}")
        for w in [round(x * 0.1, 1) for x in range(0, 11)]:
            r = build_retriever(args.offline, True, w)
            m = evaluate(r, questions)
            print(f"{w:>8.1f} {m['hit_at']['@1']:>7.3f} {m['hit_at']['@3']:>7.3f} "
                  f"{m['hit_at']['@5']:>7.3f} {m['mrr']:>7.3f}")
        return 0

    if args.sweep:
        sweep(retriever, questions)
        return 0

    results = evaluate(retriever, questions)
    print(f"\nanswerable questions: {results['n_answerable']}   unanswerable: {results['n_unanswerable']}")
    print(f"({results['n_multi_chunk']} of them have more than one relevant chunk, which is "
          f"where Hit@k and Recall@k diverge)")
    print(f"\n  {'k':<4}{'Hit@k':>9}{'Recall@k':>11}")
    for k in K_VALUES:
        marker = "   <- the generator only sees these" if k == settings.top_k else ""
        print(f"  {k:<4}{results['hit_at'][f'@{k}']:>9.3f}{results['recall_at'][f'@{k}']:>11.3f}{marker}")
    print(f"\n  MRR {results['mrr']:.3f}")
    print("\nscore separation (drives the similarity gate):")
    print(f"  mean dense top-1, answerable   {results['mean_top1_answerable']:.3f}   (min {results['min_top1_answerable']:.3f})")
    print(f"  mean dense top-1, unanswerable {results['mean_top1_unanswerable']:.3f}   (max {results['max_top1_unanswerable']:.3f})")

    misses = [q for q in results["per_question"] if q["expected"] and q["first_relevant_rank"] != 1]
    if misses:
        print(f"\nnot ranked first ({len(misses)}):")
        for m in misses:
            rank = m["first_relevant_rank"] or "not in top 5"
            print(f"  [{m['id']:<8}] rank={rank:<12} got={m['top1']:<34} {m['question']}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nfull results -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
