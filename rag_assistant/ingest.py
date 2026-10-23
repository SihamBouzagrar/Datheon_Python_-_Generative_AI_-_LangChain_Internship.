"""Phase 1 : indexation des documents dans ChromaDB."""
import os

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Configuration ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_documents(data_dir: Path) -> list:
    """Charge tous les PDF (et .txt) du dossier data/."""
    documents = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        pages = PyPDFLoader(str(pdf_path)).load()
        print(f"  📄 {pdf_path.name} : {len(pages)} pages")
        documents.extend(pages)

    for txt_path in sorted(data_dir.glob("*.txt")):
        docs = TextLoader(str(txt_path), encoding="utf-8").load()
        print(f"  📝 {txt_path.name} : fichier texte")
        documents.extend(docs)

    return documents


def main() -> None:
    # 1. Chargement
    print("1) Chargement des documents...")
    documents = load_documents(DATA_DIR)
    if not documents:
        print("❌ Aucun document trouvé dans data/. Ajoute des PDF et relance.")
        return

    # 2. Découpage
    print("\n2) Découpage en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]  # supprime les chunks vides
    print(f"  {len(documents)} pages/documents -> {len(chunks)} chunks")

    if not chunks:
        print("❌ Aucun texte extrait. Ton PDF est peut-être un scan (images).")
        return

    # 3. Embeddings + stockage
    print("\n3) Création des embeddings et stockage dans ChromaDB...")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)  # repart de zéro pour éviter les doublons

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"✅ Indexation terminée : {len(chunks)} chunks dans {CHROMA_DIR.name}/")


if __name__ == "__main__":
    main()