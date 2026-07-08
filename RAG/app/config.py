# app/config.py
"""
Módulo de configuración centralizada.
Lee las variables de entorno del archivo .env
"""

import os
from dotenv import load_dotenv

# Carga las variables del archivo .env
load_dotenv()


class Settings:
    """
    Clase que agrupa todas las configuraciones del sistema RAG.
    Usar una clase facilita el acceso y la documentación.
    """

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # ── HuggingFace ──────────────────────────────────────────
    HUGGINGFACEHUB_API_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

    # ── Modelos ──────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    # ── ChromaDB ─────────────────────────────────────────────
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "rag_documents"
    )

    # ── Chunking ─────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ── Recuperación ─────────────────────────────────────────
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))

    # ── Carpeta de documentos ────────────────────────────────
    DOCS_PATH: str = "./docs"

    def validate(self):
        """Verifica que las configuraciones críticas estén presentes."""
        if not self.HUGGINGFACEHUB_API_TOKEN:
            raise ValueError(
                "❌ HUGGINGFACEHUB_API_TOKEN no está configurado en el archivo .env"
            )


# Instancia global que se importa desde otros módulos
settings = Settings()