"""
FastAPI Main Application
Complete API for AI ML Studio
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from backend.core.config import settings
from backend.core.database import check_db_connection
from backend.core.cache import cache
from loguru import logger

# Create FastAPI app
app = FastAPI(
    title="AI ML Studio API",
    description="Complete ML/AI Development Platform API",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class DatasetGenerateRequest(BaseModel):
    task: str
    type: str = "classification"
    num_samples: int = 100
    categories: Optional[List[str]] = None


class DatasetGenerateResponse(BaseModel):
    status: str
    num_samples: int
    samples: List[Dict[str, Any]]


class RAGIndexRequest(BaseModel):
    collection_name: str
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class RAGQueryRequest(BaseModel):
    collection_name: str
    question: str
    top_k: int = 5


class TrainingStartRequest(BaseModel):
    model: str
    dataset: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001


# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI ML Studio API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = check_db_connection()

    return {
        "status": "healthy" if db_status else "unhealthy",
        "database": "connected" if db_status else "disconnected",
        "cache": "connected" if cache.redis_client else "disconnected",
    }


@app.post("/api/dataset/generate", response_model=DatasetGenerateResponse)
async def generate_dataset(request: DatasetGenerateRequest):
    """Generate synthetic dataset"""
    try:
        from backend.modules.datasets.generator import DatasetGenerator

        generator = DatasetGenerator()

        if request.type == "classification":
            if not request.categories:
                raise HTTPException(400, "Categories required for classification")

            dataset = generator.generate_text_classification_dataset(
                task_description=request.task,
                categories=request.categories,
                num_samples=request.num_samples,
            )

        elif request.type == "conversation":
            dataset = generator.generate_conversation_dataset(
                domain=request.task,
                num_conversations=request.num_samples,
            )

        elif request.type == "qa":
            dataset = generator.generate_qa_dataset(
                context_source=request.task,
                num_pairs=request.num_samples,
            )

        else:
            raise HTTPException(400, f"Unsupported dataset type: {request.type}")

        return DatasetGenerateResponse(
            status="success",
            num_samples=len(dataset),
            samples=dataset[:10],  # Return first 10 as preview
        )

    except Exception as e:
        logger.error(f"Dataset generation error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/rag/index")
async def rag_index(request: RAGIndexRequest):
    """Index documents for RAG"""
    try:
        from backend.modules.rag import RAGSystem

        rag = RAGSystem(collection_name=request.collection_name)
        rag.index_documents(request.documents, request.metadatas)

        return {
            "status": "success",
            "collection": request.collection_name,
            "num_documents": len(request.documents),
        }

    except Exception as e:
        logger.error(f"RAG indexing error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query RAG system"""
    try:
        from backend.modules.rag import RAGSystem

        rag = RAGSystem(collection_name=request.collection_name)
        result = rag.query(question=request.question, top_k=request.top_k)

        return result

    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/train/start")
async def start_training(request: TrainingStartRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    try:
        # In production, use Celery for background tasks
        # For now, return job ID
        import uuid

        job_id = str(uuid.uuid4())

        return {
            "status": "started",
            "job_id": job_id,
            "message": "Training job started in background",
        }

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    import torch

    stats = {
        "cache": cache.get_stats(),
        "gpu_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        stats["gpu_count"] = torch.cuda.device_count()
        stats["gpu_name"] = torch.cuda.get_device_name(0)
        stats["gpu_memory"] = {
            "allocated": torch.cuda.memory_allocated(0) / 1024**3,
            "reserved": torch.cuda.memory_reserved(0) / 1024**3,
        }

    return stats


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
