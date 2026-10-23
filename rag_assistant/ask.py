"""Phase 2 : questions-réponses sur les documents indexés."""
import argparse
import os
import re
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # modèle déjà en cache : pas de connexion HF

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(Path(__file__).parent.parent / ".env")  # .env à la racine du projet

# ---------- Configuration ----------
CHROMA_DIR = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-flash-latest"  # même modèle que json_extractor.py
TOP_K = 4

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un assistant qui répond UNIQUEMENT à partir du contexte fourni.\n"
     "Règles :\n"
     "- N'utilise aucune connaissance extérieure.\n"
     "- Chaque passage du contexte est numéroté [1], [2], etc.\n"
     "- Si la réponse n'est pas dans le contexte, réponds exactement : "
     "\"Je ne trouve pas cette information dans les documents.\" "
     "et mets une liste de sources vide.\n"
     "- Réponds dans la langue de la question, de façon claire et concise.\n"
     "- Termine TOUJOURS par une dernière ligne au format exact :\n"
     "SOURCES: 2, 4\n"
     "avec UNIQUEMENT les numéros des passages dont tu as réellement "
     "utilisé l'information (ou 'SOURCES: aucune')."),
    ("human", "Contexte :\n{context}\n\nQuestion : {question}"),
])


# ---------- Fonctions utilitaires (pures, testables sans API) ----------
def format_source(doc) -> str:
    """Ex : 'cours.pdf, page 12'."""
    name = Path(doc.metadata.get("source", "inconnu")).name
    if "page" in doc.metadata:
        return f"{name}, page {doc.metadata['page'] + 1}"  # +1 : page 0 -> page 1
    return name


def format_context(docs: list) -> str:
    """Numérote les chunks pour que le LLM puisse les citer."""
    parts = [f"[{i}] ({format_source(doc)})\n{doc.page_content}"
             for i, doc in enumerate(docs, 1)]
    return "\n\n---\n\n".join(parts)


def extract_text(content) -> str:
    """Gemini peut renvoyer une chaîne OU une liste de blocs : on garde le texte."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    ).strip()


def parse_answer(text: str, docs: list) -> tuple[str, list]:
    """Sépare le texte de la ligne 'SOURCES: ...' et retourne les docs cités."""
    match = re.search(r"SOURCES\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return text.strip(), []
    answer = text[:match.start()].strip()
    numbers = {int(n) for n in re.findall(r"\d+", match.group(1))}
    cited = [docs[n - 1] for n in sorted(numbers) if 1 <= n <= len(docs)]
    return answer, cited


# ---------- Pipeline RAG ----------
def build_rag():
    """Construit retriever + chaîne (une seule fois)."""
    if not CHROMA_DIR.exists():
        sys.exit("❌ Base introuvable. Lance d'abord : python rag_assistant\\ingest.py")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("❌ GEMINI_API_KEY manquante dans .env")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=api_key,
    )
    return retriever, PROMPT | llm


def answer_question(question: str, retriever, chain) -> tuple[str, list[str]]:
    """Retourne (réponse, sources citées). Réutilisable dans les tests."""
    docs = retriever.invoke(question)
    raw = chain.invoke({"context": format_context(docs), "question": question})
    answer, cited = parse_answer(extract_text(raw.content), docs)
    return answer, sorted({format_source(d) for d in cited})


def print_result(answer: str, sources: list[str]) -> None:
    print(f"\n💬 {answer}\n")
    if sources:
        print("📚 Sources :")
        for s in sources:
            print(f"   - {s}")
    print()


# ---------- CLI ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant Q&A RAG sur documents PDF")
    parser.add_argument("--question", "-q", help="pose une seule question puis quitte")
    args = parser.parse_args()

    retriever, chain = build_rag()

    if args.question:
        print_result(*answer_question(args.question, retriever, chain))
        return

    print("Assistant prêt. Pose ta question (tape 'q' pour quitter).\n")
    while True:
        question = input("❓ Question : ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question:
            print_result(*answer_question(question, retriever, chain))


if __name__ == "__main__":
    main()