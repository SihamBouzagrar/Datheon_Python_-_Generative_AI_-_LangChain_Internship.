"""Tests unitaires du pipeline RAG (aucun appel réseau)."""
import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_assistant"))
from ask import format_source, parse_answer, extract_text  # noqa: E402

DOCS = [
    Document(page_content="a", metadata={"source": "data/cours.pdf", "page": 0}),
    Document(page_content="b", metadata={"source": "data/cours.pdf", "page": 11}),
    Document(page_content="c", metadata={"source": "data/notes.txt"}),
]


def test_format_source_adds_one_to_page():
    assert format_source(DOCS[0]) == "cours.pdf, page 1"
    assert format_source(DOCS[1]) == "cours.pdf, page 12"


def test_format_source_without_page():
    assert format_source(DOCS[2]) == "notes.txt"


def test_parse_answer_extracts_cited_docs():
    answer, cited = parse_answer("Réponse.\nSOURCES: 2", DOCS)
    assert answer == "Réponse."
    assert cited == [DOCS[1]]


def test_parse_answer_none():
    answer, cited = parse_answer("Je ne trouve pas.\nSOURCES: aucune", DOCS)
    assert cited == []


def test_parse_answer_ignores_out_of_range():
    _, cited = parse_answer("X\nSOURCES: 1, 9", DOCS)
    assert cited == [DOCS[0]]


def test_extract_text_handles_blocks():
    blocks = [{"type": "text", "text": "Bonjour", "extras": {"signature": "xxx"}}]
    assert extract_text(blocks) == "Bonjour"
    assert extract_text("déjà du texte") == "déjà du texte"