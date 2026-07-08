"""
Script de ingesta de documentos.

Uso:
    python -m app.ingest

Este script:
  1. Lee todos los PDF y TXT de la carpeta ./docs
  2. Divide el texto en chunks con overlap
  3. Genera embeddings con HuggingFace
  4. Guarda todo en ChromaDB (persistente en disco)
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import settings

def load_documents(docs_path: str) -> list:
    """
    Carga todos los documentos PDF y TXT de una carpeta.

    Args:
        docs_path: Ruta a la carpeta con documentos

    Returns:
        Lista de objetos Document de LangChain
    """
    documents = []
    docs_folder = Path(docs_path)

    if not docs_folder.exists():
        print(f"  La carpeta '{docs_path}' no existe. Créala y añade documentos.")
        return documents

    # Busca archivos soportados
    supported_files = list(docs_folder.glob("*.pdf")) + list(docs_folder.glob("*.txt"))

    if not supported_files:
        print(f"  No se encontraron archivos PDF o TXT en '{docs_path}'")
        return documents

    for file_path in supported_files:
        try:
            print(f" Cargando: {file_path.name}")

            if file_path.suffix.lower() == ".pdf":
                # PyPDFLoader extrae texto de PDFs página por página
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".txt":
                # TextLoader para archivos de texto plano
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                continue

            docs = loader.load()
            documents.extend(docs)
            print(f"    {len(docs)} página(s)/sección(es) cargadas")

        except Exception as e:
            print(f"    Error cargando {file_path.name}: {e}")

    print(f"\n📚 Total documentos cargados: {len(documents)}")
    return documents

def split_documents(documents: list) -> list:
    """
    Divide los documentos en chunks más pequeños.

    RecursiveCharacterTextSplitter intenta dividir por:
    párrafos → oraciones → palabras → caracteres
    Esto preserva mejor el contexto semántico.

    Args:
        documents: Lista de Document de LangChain

    Returns:
        Lista de chunks (también objetos Document)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,        # Máximo de caracteres por chunk
        chunk_overlap=settings.CHUNK_OVERLAP,  # Caracteres compartidos entre chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Prioridad de separadores
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  Documentos divididos en {len(chunks)} chunks")
    print(f"   Tamaño por chunk: ~{settings.CHUNK_SIZE} caracteres")
    print(f"   Solapamiento: {settings.CHUNK_OVERLAP} caracteres")
    return chunks

def create_vector_store(chunks: list) -> Chroma:
    """
    Genera embeddings para cada chunk y los guarda en ChromaDB.

    El proceso:
    1. HuggingFaceEmbeddings convierte cada chunk en un vector numérico
    2. Chroma almacena esos vectores en disco (carpeta chroma_db/)

    Args:
        chunks: Lista de chunks de texto

    Returns:
        Instancia del vector store de ChromaDB
    """
    print(f"\n Generando embeddings con: {settings.EMBEDDING_MODEL}")
    print("   (Esto puede tardar unos minutos la primera vez...)")

    # Configura el modelo de embeddings
    # Este modelo se descarga automáticamente desde HuggingFace (~90MB)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Usa CPU (no requiere GPU)
        encode_kwargs={"normalize_embeddings": True},  # Mejora la similitud coseno
    )

    # Crea o actualiza la base de datos vectorial
    # Si ya existe la colección, la REEMPLAZA por completo
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )

    print(f" Vector store creado en: {settings.CHROMA_DB_PATH}")
    print(f"   Colección: {settings.CHROMA_COLLECTION_NAME}")
    print(f"   Total de vectores almacenados: {len(chunks)}")

    return vector_store

def run_ingestion():
    """
    Función principal que ejecuta el pipeline completo de ingesta.
    """
    print("=" * 60)
    print(" INICIANDO PROCESO DE INGESTA DE DOCUMENTOS")
    print("=" * 60)

    # Valida configuración
    settings.validate()

    # Paso 1: Cargar documentos
    print("\n PASO 1: Cargando documentos...")
    documents = load_documents(settings.DOCS_PATH)

    if not documents:
        print("\n No hay documentos para procesar. Añade archivos a la carpeta ./docs")
        sys.exit(1)

    # Paso 2: Dividir en chunks
    print("\n  PASO 2: Dividiendo en chunks...")
    chunks = split_documents(documents)

    # Paso 3: Crear vector store
    print("\n  PASO 3: Creando vector store...")
    create_vector_store(chunks)

    print("\n" + "=" * 60)
    print(" INGESTA COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("\nAhora puedes iniciar la API con:")
    print("   uvicorn app.main:app --reload")

if __name__ == "__main__":
    run_ingestion()