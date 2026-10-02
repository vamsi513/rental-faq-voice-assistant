"""Turn a markdown corpus into retrievable chunks.

Chunk boundaries follow the resume's own structure rather than a fixed token window:
one chunk per project, per role, per degree, and one per flat section. A resume is
already authored as semantically complete units, so splitting on `###`/`##` keeps
each project's bullets with its technology list -- which is exactly what a caller
asks about together ("what did he use in that project?").

Oversized entries are split on bullet boundaries only, never mid-sentence.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 1400  # ~350 tokens: comfortably inside one embedding call, still one topic
RETRIEVABLE_FALSE = re.compile(r"retrievable:\s*false", re.I)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section: str          # "Pet charges", "Monthly rent by floor plan", ...
    entry: str | None     # a `###` sub-entry, or None
    text: str             # what gets embedded and shown as evidence
    source: str = ""      # document id, e.g. "pets" -- which file this came from
    title: str = ""       # that document's H1, for readable citations
    part: int = 1         # >1 only when an oversized entry had to be split
    parts: int = 1

    @property
    def citation(self) -> str:
        """Human-readable provenance, shown in the debug panel -- never spoken.

        Carries the source document so an answer can be traced to a specific policy
        file, which is the point of splitting the corpus across documents.
        """
        base = f"{self.section} › {self.entry}" if self.entry else self.section
        if self.source:
            base = f"{self.source}.md › {base}"
        return f"{base} (part {self.part}/{self.parts})" if self.parts > 1 else base


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def _entry_slug(heading: str) -> str:
    """Short, stable id from an entry heading.

    Headings carry employer and dates ("Machine Learning Intern - Data Engineering,
    Codecasa, India (Mar 2023 - Aug 2023)"); keep only the leading title so chunk ids
    stay readable in eval labels and debug output.
    """
    lead = re.split(r"\s[\u2014\u2013-]\s|,|\(", heading, maxsplit=1)[0]
    return _slug(lead)


def _split_oversized(body: str) -> list[str]:
    """Split on bullet/line boundaries, packing lines until MAX_CHARS."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    parts: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join(current + [line])
        if current and len(candidate) > MAX_CHARS:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts or [body]


def parse_corpus(directory: Path) -> list[Chunk]:
    """Chunk every markdown file in a directory, in filename order.

    Each file is one policy area. Keeping them separate rather than in one giant file
    means a retrieved chunk names the document it came from, so an answer about the pet
    fee is traceable to pets.md rather than to "the corpus".
    """
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        # "05-pets.md" -> source id "pets"
        source = re.sub(r"^\d+[-_]", "", path.stem)
        chunks.extend(parse_document(path, source=source))
    return chunks


def parse_document(path: Path, source: str = "") -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")

    # Track which sections an HTML comment marked as non-retrievable, then drop comments.
    blocked: set[str] = set()
    for comment, following in re.findall(r"(<!--.*?-->)\s*##\s+([^\n]+)", raw, re.S):
        if RETRIEVABLE_FALSE.search(comment):
            blocked.add(following.strip())
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.S)

    # A whole document can be excluded with a retrievable:false comment above its H1.
    first_comment = re.match(r"\s*(<!--.*?-->)", raw, re.S)
    if first_comment and RETRIEVABLE_FALSE.search(first_comment.group(1)) and not re.search(
        r"<!--[^>]*retrievable:\s*false.*?-->\s*##\s", raw, re.S | re.I
    ):
        return []

    title_match = re.search(r"^#\s+(.+)$", text, re.M)
    title = title_match.group(1).strip() if title_match else ""

    chunks: list[Chunk] = []
    # Sections are `## X` up to the next `## ` or EOF.
    for sec_match in re.finditer(r"^##\s+(.+?)\n(.*?)(?=^##\s|\Z)", text, re.S | re.M):
        section = sec_match.group(1).strip()
        body = sec_match.group(2).strip()
        if section in blocked or not body:
            continue

        entries = list(re.finditer(r"^###\s+(.+?)\n(.*?)(?=^###\s|\Z)", body, re.S | re.M))
        if entries:
            for entry_match in entries:
                entry = entry_match.group(1).strip()
                entry_body = entry_match.group(2).strip()
                # The heading carries the dates and employer -- keep it in the embedded
                # text so "when did he work at Codecasa?" can match.
                full = f"{entry}\n{entry_body}"
                pieces = _split_oversized(full) if len(full) > MAX_CHARS else [full]
                for i, piece in enumerate(pieces, start=1):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{source + '--' if source else ''}"
                                     f"{_slug(section)}--{_entry_slug(entry)}--{i}",
                            section=section,
                            entry=entry,
                            text=piece if i == 1 else f"{entry}\n{piece}",
                            source=source,
                            title=title,
                            part=i,
                            parts=len(pieces),
                        )
                    )
        else:
            pieces = _split_oversized(body) if len(body) > MAX_CHARS else [body]
            for i, piece in enumerate(pieces, start=1):
                chunks.append(
                    Chunk(
                        chunk_id=f"{source + '--' if source else ''}{_slug(section)}--{i}",
                        section=section,
                        entry=None,
                        # The document title is embedded with the body so a question
                        # phrased around the topic ("parking", "pets") can match a
                        # section whose own heading does not repeat that word.
                        text=f"{title}\n{section}\n{piece}" if title else f"{section}\n{piece}",
                        source=source,
                        title=title,
                        part=i,
                        parts=len(pieces),
                    )
                )
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Dump chunks to JSON so the knowledge source is inspectable without running the app."""
    path.write_text(
        json.dumps([asdict(c) | {"citation": c.citation} for c in chunks], indent=2),
        encoding="utf-8",
    )
