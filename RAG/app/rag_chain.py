# app/rag_chain.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.config import settings

PROMPT_TEMPLATE = """Eres un experto en biodiversidad y fauna silvestre de Barranquilla (Colombia). 
Usa la información de contexto para responder preguntas sobre especies, qué hacer en caso de avistamientos o emergencias con animales.

Si la respuesta no está en el contexto, di: "No tengo información específica sobre ese animal en mi base de datos, pero te sugiero contactar a Barranquilla Verde."

Contexto:
{context}

Pregunta: {question}
Respuesta:"""

RAG_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Embeddings siguen siendo de HuggingFace (gratis, corren local)."""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    import os

    if not os.path.exists(settings.CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"No se encontró la base de datos en '{settings.CHROMA_DB_PATH}'. "
            "Ejecuta primero: python -m app.ingest"
        )

    return Chroma(
        persist_directory=settings.CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )


def get_llm() -> ChatGroq:
    """LLM via Groq — gratis y ultrarrápido."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=512,
    )


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


_retriever = None
_chain = None


def initialize_chain():
    global _retriever, _chain

    settings.validate()
    print("🔗 Inicializando pipeline RAG...")

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    llm = get_llm()

    _retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVER_K},
    )

    _chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print("✅ Pipeline RAG listo")


def get_answer(question: str) -> dict:
    if _chain is None:
        raise RuntimeError("La cadena RAG no está inicializada.")

    source_documents = _retriever.invoke(question)
    answer = _chain.invoke(question)

    sources = []
    for doc in source_documents:
        sources.append(
            {
                "source": doc.metadata.get("source", "Desconocido"),
                "page": doc.metadata.get("page", None),
                "content_preview": doc.page_content[:200] + "...",
            }
        )

    return {
        "answer": answer.strip(),
        "sources": sources,
    }