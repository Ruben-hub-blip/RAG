"""
Servidor FastAPI para el sistema RAG.

Endpoints:
 GET /     → Health check
 GET /status  → Estado del sistema
 POST /ask    → Hacer una pregunta al RAG
 POST /ingest  → Disparar ingesta de documentos (opcional)

Iniciar servidor:
 uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import time

from app.rag_chain import initialize_chain, get_answer
from app.config import settings

# ─────────────────────────────────────────────────────────────
# MODELOS PYDANTIC (schemas de request/response)
# ─────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
  """Schema para la petición de pregunta."""
  question: str = Field(
    ...,
    min_length=3,
    max_length=1000,
    description="La pregunta a realizar al sistema RAG",
    example="¿Qué debo hacer si encuentro una iguana herida?"  # <--- CAMBIA ESTO
  )

class SourceDocument(BaseModel):
  """Schema para un documento fuente."""
  source: str
  page: int | None = None
  content_preview: str

class AnswerResponse(BaseModel):
  """Schema para la respuesta del RAG."""
  question: str
  answer: str
  sources: list[SourceDocument]
  processing_time_seconds: float

class StatusResponse(BaseModel):
  """Schema para el estado del sistema."""
  status: str
  vector_store_path: str
  vector_store_exists: bool
  llm_model: str
  embedding_model: str
  retriever_k: int

# ─────────────────────────────────────────────────────────────
# LIFECYCLE: arranque y cierre del servidor
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
  """
  Se ejecuta al ARRANCAR el servidor:
   - Valida la configuración
   - Inicializa el pipeline RAG (carga embeddings + ChromaDB)

  Se ejecuta al CERRAR el servidor:
   - Limpieza de recursos
  """
  # ── Startup ──────────────────────────────────────────────
  print("\n" + "=" * 60)
  print("🚀 INICIANDO SERVIDOR RAG API")
  print("=" * 60)

  try:
    initialize_chain()
    print("✅ Servidor listo para recibir peticiones\n")
  except FileNotFoundError as e:
    print(f"\n⚠️ ADVERTENCIA: {e}")
    print("  El servidor arrancará pero /ask dará error hasta que indexes documentos.")
  except Exception as e:
    print(f"\n❌ Error al inicializar: {e}")
    raise

  yield # El servidor está corriendo aquí

  # ── Shutdown ─────────────────────────────────────────────
  print("\n🛑 Cerrando servidor RAG API...")

# ─────────────────────────────────────────────────────────────
# APLICACIÓN FASTAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(
  title="RAG API",
  description="Sistema de Retrieval-Augmented Generation con HuggingFace y ChromaDB",
  version="1.0.0",
  lifespan=lifespan,
  docs_url="/docs",  # Swagger UI automático en http://localhost:8000/docs
  redoc_url="/redoc", # ReDoc en http://localhost:8000/redoc
)

# CORS: permite que otras apps (frontend, etc.) consuman esta API
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],    # En producción, especifica los dominios permitidos
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
  """Health check básico."""
  return {
    "message": "RAG API está funcionando 🤖",
    "docs": "http://localhost:8000/docs"
  }

@app.get("/status", response_model=StatusResponse, tags=["General"])
async def get_status():
  """
  Devuelve el estado actual del sistema:
  - Si ChromaDB existe
  - Modelos configurados
  - Configuración de recuperación
  """
  return StatusResponse(
    status="ok",
    vector_store_path=settings.CHROMA_DB_PATH,
    vector_store_exists=os.path.exists(settings.CHROMA_DB_PATH),
    llm_model=settings.LLM_MODEL,
    embedding_model=settings.EMBEDDING_MODEL,
    retriever_k=settings.RETRIEVER_K,
  )

@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask_question(request: QuestionRequest):
  """
  **Endpoint principal:** Recibe una pregunta y devuelve la respuesta RAG.

  El sistema:
  1. Convierte la pregunta en vector (embedding)
  2. Busca los chunks más relevantes en ChromaDB
  3. Construye un prompt con el contexto encontrado
  4. El LLM de HuggingFace genera la respuesta
  5. Devuelve la respuesta + las fuentes usadas

  **Ejemplo de body:**
  ```json
  {
    "question": "¿Cuáles son los beneficios de los empleados?"
  }
  ```
  """
  start_time = time.time()

  try:
    result = get_answer(request.question)
  except RuntimeError as e:
    raise HTTPException(
      status_code=503,
      detail=f"Sistema RAG no inicializado: {str(e)}. Ejecuta primero la ingesta."
    )
  except FileNotFoundError as e:
    raise HTTPException(
      status_code=503,
      detail=f"Base de datos no encontrada: {str(e)}"
    )
  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f"Error al procesar la pregunta: {str(e)}"
    )

  processing_time = round(time.time() - start_time, 2)

  return AnswerResponse(
    question=request.question,
    answer=result["answer"],
    sources=[SourceDocument(**s) for s in result["sources"]],
    processing_time_seconds=processing_time,
  )

@app.post("/ingest", tags=["Administración"])
async def trigger_ingestion():
  from app.ingest import run_ingestion
  from app.rag_chain import initialize_chain as reinit
  # Elimina la línea 'global _rag_chain' de aquí

  try:
    run_ingestion()
    reinit() # Esto recarga la base de datos en memoria
    return {"message": "✅ Ingesta completada. Vector store actualizado."}
  except Exception as e:
    # Esto te dirá el error real en la respuesta de la API
    raise HTTPException(
      status_code=500,
      detail=f"Error durante la ingesta: {str(e)}"
    )
  

  # En app/main.py añadir:

class Reporte(BaseModel):
    especie: str
    ubicacion: str
    usuario_telegram: str

@app.post("/reportar_avistamiento", tags=["Reportes"])
async def crear_reporte(reporte: Reporte):
    # Aquí guardarías el reporte en una base de datos SQL (ej. SQLite o PostgreSQL)
    print(f"Nuevo reporte: {reporte.especie} en {reporte.ubicacion}")
    return {"status": "success", "message": "Reporte guardado"}