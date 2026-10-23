# Datheon Internship: Python + Generative AI & LangChain

Hands-on projects built during the Datheon *Python + Generative AI & LangChain* internship program.

| Task | Project | Status |
|---|---|---|
| 1 | [Structured LLM Prompt Pipeline & JSON Extractor](#task-1--structured-llm-prompt-pipeline--json-extractor) | Done |
| 2 | [Document Q&A Assistant with RAG](#task-2--document-qa-assistant-with-rag--vector-embeddings) | Done |
| 3 | Multi-tool agent | Planned |
| 4 | Streamlit chatbot | Planned |

## Project structure
```
.
├── json_extractor.py        # Task 1: schemas, prompts, LLM clients, pipeline, CLI
├── rag_assistant/           # Task 2
│   ├── data/                # source PDFs (not committed)
│   ├── ingest.py            # phase 1: load, chunk, embed, store
│   ├── ask.py               # phase 2: retrieve, answer, cite pages
│   └── chroma_db/           # vector store (generated, not committed)
├── tests/                   # pytest tests (no real API calls)
├── docs/screenshots/        # demo screenshots
├── requirements.txt
└── .env.example
```

## Installation (both tasks)
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add your API key(s):
```
GEMINI_API_KEY=your_key
```

---

# Task 1 · Structured LLM Prompt Pipeline & JSON Extractor

Turns **unstructured text** (invoices, resumes) into **strictly validated JSON** using an LLM
(Google Gemini or OpenAI), parameterized prompt templates, few-shot examples and Pydantic schemas.

## Features
- Two providers behind one interface: **Gemini** (`google-genai`) 
- **Parameterized prompt templates** (document type, rules, JSON schema and examples injected at runtime)
- **Few-shot examples** generated from Pydantic objects (always valid by construction)
- **Pydantic v2 validation** with business rules (e.g. `total = subtotal + tax`, line totals, date order)
- **Self-repair loop**: if the output is invalid, the validation error is sent back to the model (max N retries)
- Extensible: add a new document type with one schema + one `Task`

## Usage
```bash
# Built-in demo documents
python json_extractor.py invoice --demo --provider gemini

# Your own file, saved to JSON
python json_extractor.py invoice --file invoice.txt --output invoice.json

# Inspect the generated prompt without calling any API
python json_extractor.py invoice --demo --show-prompt
```
Options: `--provider {gemini,openai}`, `--model NAME`, `--retries N`, `--output FILE`.

## Example

**Input**
```
INVOICE #INV-889
Issued: 02/09/2025   Payment due: 02/10/2025
From: Nordic Cloud Ltd (VAT: SE556677889901), Storgatan 5, Stockholm. billing@nordiccloud.se
Bill to: Casa Logistics, 45 Bd Zerktouni, Casablanca
Description                    Qty   Unit     Amount
Cloud compute (monthly)         3    120.00   360.00
Premium support                 1    250.00   250.00
Subtotal 610.00 EUR  | VAT 25% 152.50 EUR | TOTAL DUE 762.50 EUR
```

**Output** (validated)
```json
{
  "invoice_number": "INV-889",
  "issue_date": "2025-09-02",
  "due_date": "2025-10-02",
  "vendor": {
    "name": "Nordic Cloud Ltd",
    "address": "Storgatan 5, Stockholm",
    "email": "billing@nordiccloud.se",
    "tax_id": "SE556677889901"
  },
  "customer": {
    "name": "Casa Logistics",
    "address": "45 Bd Zerktouni, Casablanca",
    "email": null,
    "tax_id": null
  },
  "currency": "EUR",
  "line_items": [
    {"description": "Cloud compute (monthly)", "quantity": 3.0, "unit_price": 120.0, "total": 360.0},
    {"description": "Premium support", "quantity": 1.0, "unit_price": 250.0, "total": 250.0}
  ],
  "subtotal": 610.0,
  "tax_amount": 152.5,
  "total": 762.5
}
```

**Terminal demo**

![Task 1 demo](docs/demo.png)

## How it works
```
raw text → prompt template (rules + schema + few-shot) → LLM → clean_json → json.loads
        → Pydantic validation → valid? yes → result
                                      no  → send error back to the LLM → retry (max N)
```

## Design choices
- **Temperature 0 + native JSON mode** for stable, parseable output.
- **Two validation layers**: syntax (`json.loads`) then content (Pydantic).
- **Provider abstraction** (`LLMClient`) so the pipeline never depends on a specific SDK.
- **Prompt rule "use null, never invent data"** to limit hallucinations on missing fields.

---

# Task 2 · Document Q&A Assistant with RAG & Vector Embeddings

A Retrieval-Augmented Generation (RAG) system that answers questions **strictly from custom PDF documents**
and **cites the source pages**. If the answer is not in the documents, it says so instead of guessing.

## Features
- **PDF/text loading** with LangChain loaders (page number kept as metadata)
- **Recursive chunking** (800 characters, 150 overlap)
- **Local multilingual embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`): free, no API key, works in French and English
- **ChromaDB** persistent vector store (index once, query many times)
- **Strict prompt**: answers only from the retrieved context, fixed refusal sentence otherwise
- **Verifiable citations**: chunks are numbered, the model declares which ones it used, and only those pages are displayed
- **Interactive and one-shot modes** (`-q`)

## Usage
```bash
# 1. Put your PDFs in rag_assistant/data/, then index them (once)
python rag_assistant/ingest.py

# 2a. Interactive mode
python rag_assistant/ask.py

# 2b. One-shot mode
python rag_assistant/ask.py -q "What are volumes used for in Docker?"
```

## How it works
```
INDEXING (ingest.py, once)
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → embeddings → ChromaDB

QUESTION (ask.py, every query)
question → embedding → top-k similar chunks → numbered context + strict prompt
        → Gemini → answer + "SOURCES: n" → parse → display only the cited pages
```

| Component | Choice |
|---|---|
| Loader | `PyPDFLoader` (one page = one document) |
| Splitter | `RecursiveCharacterTextSplitter` (800 / 150) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (local) |
| Vector store | ChromaDB, `k = 4` |
| LLM | Gemini (`gemini-flash-latest`), temperature 0 |

## Demo

**1. Indexing** (4 PDFs, 116 pages, 273 chunks)

![Indexing](docs/screenshots/rag_01_ingest.png)
![Indexing](docs/screenshots/rag_02_ingest.png)

**2. Answer with source page**

![Answer with source](docs/screenshots/rag_02_answer_with_source.png)

**3. Only the page actually used is cited** (not all 4 retrieved chunks)

![Single source](docs/screenshots/rag_03_devops_single_source.png)

**4. Refusal when the information is absent** (no source displayed)

![Refusal](docs/screenshots/rag_04_refusal.png)


## Design choices
- **Retrieval and generation are separated**: the LLM never sees the whole corpus, only the top-k chunks.
- **Numbered context for citations**: showing all retrieved chunks would cite pages the model never used.
- **Local embeddings**: no quota, no cost, reproducible. The same model must be used at indexing and query time.
- **Offline mode** (`HF_HUB_OFFLINE=1`) in `ask.py` once the model is cached, to avoid network retries.
- **`temperature = 0`** for factual, repeatable answers.

## Known limitations
- Scanned PDFs (images) are not supported: no OCR.
- The embedding model truncates inputs beyond ~128 tokens, which may affect 800-character chunks.
- Citations are declared by the LLM: far more faithful than listing all chunks, but not 100% guaranteed.
- Retrieval can vary slightly with the wording of the question.

---

## Tests
```bash
python -m pytest
```
Tests use no real API calls: Task 1 uses a fake LLM client (validators, JSON cleaning, retry loop);
Task 2 tests the pure functions (source formatting, citation parsing, text extraction).

## Next steps (program roadmap)
Task 3: multi-tool agent · Task 4: Streamlit chatbot.

## Author

**Bouzagrar Siham**
- GitHub: [SihamBouzagrar](https://github.com/SihamBouzagrar)
- LinkedIn: [Siham Bouzagrar](https://www.linkedin.com/in/siham-bouzagrar-262711251)

Built as part of the Datheon *Python + Generative AI & LangChain* internship.