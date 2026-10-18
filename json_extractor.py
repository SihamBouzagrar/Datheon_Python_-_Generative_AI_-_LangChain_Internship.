#!/usr/bin/env python3
"""
Structured LLM Prompt Pipeline & JSON Extractor
================================================
Transforme un texte non structuré (facture, CV) en JSON strictement validé.

Fonctionnalités
  - Providers : Google Gemini (google-genai) ou OpenAI
  - Templates de prompt paramétrés + exemples few-shot
  - Schémas Pydantic avec règles métier (ex: total = sous-total + taxes)
  - Boucle de réparation : si la sortie est invalide, l'erreur est renvoyée
    au modèle pour qu'il corrige (max N tentatives)

Installation
  pip install pydantic openai google-genai python-dotenv

Clés API (variables d'environnement ou fichier .env)
  OPENAI_API_KEY=...
  GEMINI_API_KEY=...

Exemples
  python json_extractor.py invoice --demo --provider gemini
  python json_extractor.py invoice --file facture.txt --provider openai
  python json_extractor.py resume  --file cv.txt --output cv.json
  python json_extractor.py invoice --demo --show-prompt     # sans appel API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional, Type

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

try:  # .env optionnel
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# =============================================================================
# 1. SCHÉMAS PYDANTIC
# =============================================================================
class Party(BaseModel):
    name: str = Field(min_length=1)
    address: Optional[str] = None
    email: Optional[str] = Field(default=None, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    tax_id: Optional[str] = None


class LineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def check_total(self) -> "LineItem":
        if abs(self.quantity * self.unit_price - self.total) > 0.02:
            raise ValueError(
                f"line '{self.description}': quantity*unit_price "
                f"({self.quantity * self.unit_price:.2f}) != total ({self.total:.2f})"
            )
        return self


class Invoice(BaseModel):
    invoice_number: str = Field(min_length=1)
    issue_date: date = Field(description="format YYYY-MM-DD")
    due_date: Optional[date] = Field(default=None, description="format YYYY-MM-DD")
    vendor: Party
    customer: Party
    currency: str = Field(pattern=r"^[A-Z]{3}$", description="code ISO 4217, ex: MAD, EUR")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax_amount: float = Field(ge=0)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def check_totals(self) -> "Invoice":
        items_sum = sum(i.total for i in self.line_items)
        if abs(items_sum - self.subtotal) > 0.02:
            raise ValueError(f"subtotal ({self.subtotal}) != sum of line items ({items_sum:.2f})")
        if abs(self.subtotal + self.tax_amount - self.total) > 0.02:
            raise ValueError(
                f"total ({self.total}) != subtotal + tax_amount "
                f"({self.subtotal + self.tax_amount:.2f})"
            )
        if self.due_date and self.due_date < self.issue_date:
            raise ValueError("due_date is before issue_date")
        return self


class Experience(BaseModel):
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_date: str = Field(pattern=r"^\d{4}(-\d{2})?$", description="YYYY ou YYYY-MM")
    end_date: Optional[str] = Field(
        default=None,
        pattern=r"^(\d{4}(-\d{2})?|present)$",
        description="YYYY, YYYY-MM ou 'present'",
    )
    highlights: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str = Field(min_length=1)
    degree: str = Field(min_length=1)
    year: Optional[int] = Field(default=None, ge=1950, le=2100)


class Resume(BaseModel):
    full_name: str = Field(min_length=1)
    email: Optional[str] = Field(default=None, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    @field_validator("skills", "languages")
    @classmethod
    def dedupe(cls, values: list[str]) -> list[str]:
        seen, out = set(), []
        for v in values:
            key = v.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(v.strip())
        return out


# =============================================================================
# 2. TEMPLATES DE PROMPT + FEW-SHOT
# =============================================================================
SYSTEM_TEMPLATE = (
    "You are a precise information-extraction engine. "
    "You read unstructured {doc_type} text and return ONLY a single valid JSON object "
    "that conforms exactly to the provided JSON Schema. No markdown, no comments, "
    "no extra keys."
)

USER_TEMPLATE = """\
Extract the data from the {doc_type} below.

RULES:
{rules}
- Output language of free-text fields: keep the original language of the document.
- Use null for optional information that is absent. NEVER invent data.

JSON SCHEMA:
{schema}

{examples}
### NOW EXTRACT
INPUT:
{text}

OUTPUT:
"""


@dataclass
class Example:
    text: str
    output: BaseModel


@dataclass
class Task:
    name: str
    doc_type: str
    schema: Type[BaseModel]
    rules: list[str]
    examples: list[Example]
    demo_text: str


INVOICE_EXAMPLE = Example(
    text="""FACTURE N° F-2024-0042
Date : 15 mars 2024 — Échéance : 14 avril 2024
Émetteur : Atlas Digital SARL, 12 rue Hassan II, Casablanca — contact@atlasdigital.ma
Client : Maroc Retail SA
Prestations :
- Développement site web (1 forfait) ........ 15 000,00 MAD
- Hébergement annuel (2 x 1 200,00 MAD) .... 2 400,00 MAD
Total HT : 17 400,00 MAD | TVA 20% : 3 480,00 MAD | Total TTC : 20 880,00 MAD""",
    output=Invoice(
        invoice_number="F-2024-0042",
        issue_date=date(2024, 3, 15),
        due_date=date(2024, 4, 14),
        vendor=Party(
            name="Atlas Digital SARL",
            address="12 rue Hassan II, Casablanca",
            email="contact@atlasdigital.ma",
        ),
        customer=Party(name="Maroc Retail SA"),
        currency="MAD",
        line_items=[
            LineItem(description="Développement site web", quantity=1, unit_price=15000, total=15000),
            LineItem(description="Hébergement annuel", quantity=2, unit_price=1200, total=2400),
        ],
        subtotal=17400,
        tax_amount=3480,
        total=20880,
    ),
)

RESUME_EXAMPLE = Example(
    text="""SARA BENNANI — Data Analyst
sara.bennani@mail.com | +212 6 12 34 56 78 | Rabat
Profil : analyste orientée données, 3 ans d'expérience en BI.
Expérience
OCP Group — Data Analyst (juin 2021 – présent)
  * Création de dashboards Power BI pour la direction financière
  * Automatisation de rapports avec Python
Éducation : Master Data Science, ENSIAS, 2021
Compétences : Python, SQL, Power BI, python
Langues : Arabe, Français, Anglais""",
    output=Resume(
        full_name="Sara Bennani",
        email="sara.bennani@mail.com",
        phone="+212 6 12 34 56 78",
        location="Rabat",
        summary="Analyste orientée données, 3 ans d'expérience en BI.",
        skills=["Python", "SQL", "Power BI"],
        languages=["Arabe", "Français", "Anglais"],
        experience=[
            Experience(
                company="OCP Group",
                title="Data Analyst",
                start_date="2021-06",
                end_date="present",
                highlights=[
                    "Création de dashboards Power BI pour la direction financière",
                    "Automatisation de rapports avec Python",
                ],
            )
        ],
        education=[Education(institution="ENSIAS", degree="Master Data Science", year=2021)],
    ),
)

DEMO_INVOICE = """INVOICE #INV-889
Issued: 02/09/2025   Payment due: 02/10/2025
From: Nordic Cloud Ltd (VAT: SE556677889901), Storgatan 5, Stockholm. billing@nordiccloud.se
Bill to: Casa Logistics, 45 Bd Zerktouni, Casablanca
Description                    Qty   Unit     Amount
Cloud compute (monthly)         3    120.00   360.00
Premium support                 1    250.00   250.00
Subtotal 610.00 EUR  | VAT 25% 152.50 EUR | TOTAL DUE 762.50 EUR"""

DEMO_RESUME = """Youssef El Amrani
Full-stack developer based in Casablanca. youssef.elamrani@example.com, 0661-223344
Worked at Capgemini Maroc from Jan 2020 to Dec 2022 as Junior Developer (Java, Spring),
then at Inwi as Software Engineer from 2023 until now, building microservices and
leading a team of 4 developers.
Studied Computer Engineering at EMSI (graduated 2019). Skills: Java, Spring Boot,
Docker, Kubernetes, React, docker. Speaks French, English and Arabic."""

TASKS: dict[str, Task] = {
    "invoice": Task(
        name="invoice",
        doc_type="invoice",
        schema=Invoice,
        rules=[
            "- Dates must be ISO format YYYY-MM-DD (interpret ambiguous dates as DD/MM/YYYY).",
            "- Currency must be a 3-letter ISO 4217 code.",
            "- Amounts are plain numbers (no thousands separator, dot as decimal).",
            "- subtotal = sum of line item totals; total = subtotal + tax_amount.",
        ],
        examples=[INVOICE_EXAMPLE],
        demo_text=DEMO_INVOICE,
    ),
    "resume": Task(
        name="resume",
        doc_type="resume (CV)",
        schema=Resume,
        rules=[
            "- Dates are 'YYYY' or 'YYYY-MM'; use 'present' for a current position.",
            "- Skills and languages are short items, without duplicates.",
            "- Experience entries ordered from most recent to oldest.",
        ],
        examples=[RESUME_EXAMPLE],
        demo_text=DEMO_RESUME,
    ),
}


def build_messages(task: Task, text: str) -> tuple[str, str]:
    """Remplit les templates et renvoie (system_prompt, user_prompt)."""
    examples_block = "".join(
        f"### EXAMPLE {i}\nINPUT:\n{ex.text}\n\nOUTPUT:\n"
        f"{json.dumps(ex.output.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n\n"
        for i, ex in enumerate(task.examples, 1)
    )
    system = SYSTEM_TEMPLATE.format(doc_type=task.doc_type)
    user = USER_TEMPLATE.format(
        doc_type=task.doc_type,
        rules="\n".join(task.rules),
        schema=json.dumps(task.schema.model_json_schema(), ensure_ascii=False),
        examples=examples_block,
        text=text.strip(),
    )
    return system, user


# =============================================================================
# 3. CLIENTS LLM
# =============================================================================
Message = dict  # {"role": "user" | "assistant", "content": str}


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, messages: list[Message]) -> str: ...


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY manquante.")
        self.client, self.model = OpenAI(api_key=key), model

    def complete(self, system: str, messages: list[Message]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, *messages],
        )
        return resp.choices[0].message.content or ""


class GeminiClient(LLMClient):
    """Client Gemini basé sur le SDK officiel actuel : google-genai."""

    def __init__(self, model: str = "gemini-flash-latest"):
        from google import genai

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY manquante.")
        self.client, self.model = genai.Client(api_key=key), model

    def complete(self, system: str, messages: list[Message]) -> str:
        from google.genai import types

        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return resp.text or ""


def get_client(provider: str, model: Optional[str]) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(model) if model else OpenAIClient()
    return GeminiClient(model) if model else GeminiClient()


# =============================================================================
# 4. PIPELINE : appel -> parsing -> validation -> réparation
# =============================================================================
class ExtractionError(Exception):
    pass


def clean_json(raw: str) -> str:
    """Retire les ```json ... ``` et le texte autour de l'objet JSON."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start : end + 1] if start != -1 and end != -1 else raw


def extract(
    text: str, task: Task, client: LLMClient, max_retries: int = 2, verbose: bool = True
) -> BaseModel:
    system, user = build_messages(task, text)
    messages: list[Message] = [{"role": "user", "content": user}]
    last_error = ""

    for attempt in range(1, max_retries + 2):
        raw = client.complete(system, messages)
        try:
            data = json.loads(clean_json(raw))
            result = task.schema.model_validate(data)
            if verbose:
                print(f"[ok] extraction validée (tentative {attempt})", file=sys.stderr)
            return result
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            if verbose:
                print(f"[!] tentative {attempt} invalide :\n{last_error}\n", file=sys.stderr)
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous output is invalid. Fix it and return ONLY the "
                        f"corrected JSON object.\nErrors:\n{last_error}"
                    ),
                },
            ]

    raise ExtractionError(f"Échec après {max_retries + 1} tentatives :\n{last_error}")


# =============================================================================
# 5. CLI
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Extracteur JSON structuré piloté par LLM")
    p.add_argument("task", choices=TASKS.keys(), help="type de document")
    p.add_argument("--file", help="fichier texte à analyser (sinon stdin)")
    p.add_argument("--demo", action="store_true", help="utilise un exemple intégré")
    p.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    p.add_argument("--model", help="nom du modèle (sinon défaut du provider)")
    p.add_argument("--retries", type=int, default=2, help="tentatives de réparation")
    p.add_argument("--output", help="écrit le JSON dans ce fichier")
    p.add_argument("--show-prompt", action="store_true", help="affiche le prompt et quitte")
    args = p.parse_args()

    task = TASKS[args.task]
    if args.demo:
        text = task.demo_text
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    if not text.strip():
        sys.exit("Aucun texte à analyser.")

    if args.show_prompt:
        system, user = build_messages(task, text)
        print(f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}")
        return

    client = get_client(args.provider, args.model)
    try:
        result = extract(text, task, client, max_retries=args.retries)
    except ExtractionError as exc:
        sys.exit(str(exc))

    output = result.model_dump_json(indent=2)
    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[ok] sauvegardé dans {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()