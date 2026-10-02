"""Pure helper functions from the graph. No async, no model, no network."""
from __future__ import annotations

from app.graph import _coverage_outline, _invented_entities, _parse_extraction


def test_parse_extraction_reads_the_two_line_format():
    assert _parse_extraction("UNIT: 2B\nISSUE: leaking faucet") == ("2B", "leaking faucet")
    assert _parse_extraction("UNIT: NONE\nISSUE: leaking faucet") == (None, "leaking faucet")
    assert _parse_extraction("") == (None, None)
    assert _parse_extraction("garbage") == (None, None)


def test_coverage_outline_lists_topics_without_the_numbers(offline_retriever):
    outline = _coverage_outline(offline_retriever)
    assert "Pet charges" in outline
    assert "$250" not in outline, "the router must not be able to answer from the outline"


def test_invented_entity_detection():
    from langchain_core.messages import AIMessage, HumanMessage

    history = [HumanMessage(content="What is the pet fee?"),
               AIMessage(content="The pet fee is two fifty per pet.")]
    assert _invented_entities("Is the Parking Garage fee refundable?", history, "is that refundable?")
    assert _invented_entities("Is the pet fee refundable?", history, "is that refundable?") == []
