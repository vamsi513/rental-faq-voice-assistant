"""The knowledge base must chunk on the documents' own structure, with provenance."""
from __future__ import annotations

import pytest


def test_every_document_contributes_chunks(chunks):
    sources = {c.source for c in chunks}
    assert sources == {
        "contact-and-hours", "application", "rent-and-fees", "lease-terms", "pets",
        "parking", "amenities", "maintenance", "move-in-move-out",
    }


def test_the_about_document_is_not_indexed(chunks):
    """It is a note to the reader, not an answer -- marked retrievable:false."""
    assert not any(c.source == "about-this-corpus" for c in chunks)


def test_every_chunk_carries_provenance(chunks):
    for c in chunks:
        assert c.chunk_id and c.text.strip() and c.source
        assert c.source in c.citation
        assert c.section in c.citation


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_a_policy_section_stays_whole(chunks):
    """Pet charges must keep the fee and the rent together: a caller asking about one
    almost always needs the other, and splitting them produced half-answers."""
    pet_charges = next(c for c in chunks if c.chunk_id == "pets--pet-charges--1")
    assert "$250" in pet_charges.text
    assert "$35" in pet_charges.text
    assert "non-refundable" in pet_charges.text


def test_document_title_is_embedded_with_flat_sections(chunks):
    """Without the title, 'what time does the gym open' has no 'amenities' to match."""
    gym = next(c for c in chunks if c.chunk_id == "amenities--hours-and-access--1")
    assert "Amenities" in gym.text


def test_emergency_instructions_are_retrievable(chunks):
    """The single most safety-relevant passage must be in the index, in both places it
    is documented."""
    emergency = [c for c in chunks if "0199" in c.text]
    assert len(emergency) >= 2


@pytest.mark.parametrize("resume_term", ["Vamsi", "AgentIQ", "LangGraph", "resume", "LREC"])
def test_no_resume_content_leaked_into_the_rental_corpus(chunks, resume_term):
    """The adaptation must not answer rental questions with the previous corpus."""
    assert resume_term.lower() not in " ".join(c.text for c in chunks).lower()


def test_oversized_entry_splits_on_line_boundaries():
    """Guards the splitter, since no current section is large enough to trigger it."""
    from app.chunking import MAX_CHARS, _split_oversized

    body = "\n".join(f"- bullet number {i} " + "x" * 120 for i in range(40))
    parts = _split_oversized(body)
    assert len(parts) > 1
    assert all(len(p) <= MAX_CHARS for p in parts)
    assert sum(len(p.splitlines()) for p in parts) == len(body.splitlines())
