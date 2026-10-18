"""Tests du pipeline d'extraction. Aucun appel API réel : on utilise un faux client LLM."""
import json

import pytest
from pydantic import ValidationError

import json_extractor as je


# ---------- Faux client LLM -------------------------------------------------
class FakeClient(je.LLMClient):
    """Renvoie successivement les réponses fournies, et garde les messages reçus."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def valid_invoice_dict():
    return je.INVOICE_EXAMPLE.output.model_dump(mode="json")


# ---------- Validation Pydantic ---------------------------------------------
def test_few_shot_examples_are_valid():
    for task in je.TASKS.values():
        for ex in task.examples:
            task.schema.model_validate(ex.output.model_dump(mode="json"))


def test_invoice_rejects_wrong_total():
    data = valid_invoice_dict()
    data["total"] = 999.0
    with pytest.raises(ValidationError, match="total"):
        je.Invoice.model_validate(data)


def test_invoice_rejects_due_date_before_issue_date():
    data = valid_invoice_dict()
    data["due_date"] = "2024-01-01"
    with pytest.raises(ValidationError, match="due_date"):
        je.Invoice.model_validate(data)


def test_line_item_rejects_wrong_total():
    with pytest.raises(ValidationError):
        je.LineItem(description="x", quantity=2, unit_price=10, total=50)


def test_resume_deduplicates_skills():
    data = je.RESUME_EXAMPLE.output.model_dump(mode="json")
    data["skills"] = ["Docker", "docker", "Python"]
    assert je.Resume.model_validate(data).skills == ["Docker", "Python"]


# ---------- Nettoyage du JSON -----------------------------------------------
def test_clean_json_removes_markdown_fences():
    raw = '```json\n{"a": 1}\n```'
    assert json.loads(je.clean_json(raw)) == {"a": 1}


def test_clean_json_ignores_text_around_object():
    assert json.loads(je.clean_json('Voici : {"a": 1} merci')) == {"a": 1}


# ---------- Pipeline et boucle de réparation --------------------------------
def test_extract_succeeds_first_try():
    client = FakeClient([json.dumps(valid_invoice_dict())])
    result = je.extract("texte", je.TASKS["invoice"], client, verbose=False)
    assert result.invoice_number == "F-2024-0042"
    assert len(client.calls) == 1


def test_extract_repairs_invalid_output():
    bad = valid_invoice_dict()
    bad["total"] = 1.0
    client = FakeClient([json.dumps(bad), json.dumps(valid_invoice_dict())])
    result = je.extract("texte", je.TASKS["invoice"], client, max_retries=2, verbose=False)
    assert result.total == 20880
    assert len(client.calls) == 2
    # l'erreur de validation a bien été renvoyée au modèle
    assert "invalid" in client.calls[1][-1]["content"].lower()


def test_extract_raises_after_max_retries():
    client = FakeClient(["pas du json"] * 3)
    with pytest.raises(je.ExtractionError):
        je.extract("texte", je.TASKS["invoice"], client, max_retries=2, verbose=False)
    assert len(client.calls) == 3