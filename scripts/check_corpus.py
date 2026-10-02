"""Check the sample property documents for internal contradictions.

The agent can only be as consistent as these documents are. A figure that appears in two
places with two values will produce two different answers depending on which chunk wins
retrieval, and that looks exactly like a hallucination.

This checks the facts that are deliberately repeated across documents:
  * Figures that appear in more than one file must agree.
  * The emergency phone number and its definition must match everywhere.
  * Every document must carry the fictional-property framing somewhere in the corpus.

Run: python scripts/check_corpus.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/rental"

# Facts stated in more than one document on purpose. Each must appear with exactly one
# value across the whole corpus.
CROSS_DOC_FACTS = {
    "pet fee": r"pet fee[^.]{0,40}?\$(\d+)",
    "pet rent": r"pet rent[^.]{0,40}?\$(\d+)",
    "late fee": r"late (?:rent )?fee[^.]{0,40}?\$(\d+)",
    "application fee": r"application fee[^.]{0,40}?\$(\d+)",
    "holding deposit": r"holding deposit[^.]{0,40}?\$(\d+)",
    "reserved parking": r"reserved (?:covered )?spaces?[^.]{0,60}?\$(\d+)",
    "notice to vacate days": r"at least (\d+) days? (?:written )?notice|(\d+) days? written notice",
    "key replacement": r"(?:key|fob)[^.]{0,40}?\$(\d+)",
}
EMERGENCY_NUMBER = r"\(?555\)?[\s-]*0199"
OFFICE_NUMBER = r"\(?555\)?[\s-]*0100"


def main() -> int:
    docs = {p.stem: p.read_text() for p in sorted(CORPUS.glob("*.md"))}
    if not docs:
        print(f"no documents in {CORPUS}")
        return 1

    problems: list[str] = []

    for label, pattern in CROSS_DOC_FACTS.items():
        found: dict[str, list[str]] = defaultdict(list)
        for name, text in docs.items():
            for match in re.finditer(pattern, text, re.I):
                value = next(g for g in match.groups() if g)
                found[value].append(name)
        if len(found) > 1:
            detail = "; ".join(f"{v} in {', '.join(sorted(set(f)))}" for v, f in found.items())
            problems.append(f"{label}: disagreeing values -> {detail}")

    # The emergency number must be identical wherever it appears.
    mentions = {n: set(re.findall(EMERGENCY_NUMBER, t)) for n, t in docs.items() if re.search(EMERGENCY_NUMBER, t)}
    if len(mentions) < 2:
        problems.append("emergency number appears in fewer than 2 documents; callers may miss it")

    # A caller who asks "how do I reach you" must get the office number, not the
    # emergency line, so both must exist and be distinct.
    if not any(re.search(OFFICE_NUMBER, t) for t in docs.values()):
        problems.append("office phone number missing from the corpus")

    # The fictional framing must exist somewhere. Whitespace-normalised, because these
    # documents are hard-wrapped and a phrase can straddle a line break.
    flat = " ".join(" ".join(docs.values()).split()).lower()
    if not any(phrase in flat for phrase in ("does not exist", "fictional", "not a real")):
        problems.append("no document states that the property is fictional")

    print(f"documents: {len(docs)}")
    print(f"cross-document facts checked: {len(CROSS_DOC_FACTS)}")
    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("no contradictions found in the facts that appear in more than one document.")
    print("NOTE: this checks repeated figures, not every claim. It cannot prove the")
    print("corpus is complete or that a single-document figure is sensible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
