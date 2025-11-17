"""
Complete RAG System
Combines retrieval with generation using Claude
"""

from typing import List, Dict, Any, Optional
from .embeddings import EmbeddingModel, get_embedding_model
from .vectordb import VectorDatabase
from .retriever import Retriever
from backend.core.claude_client import claude_client
from loguru import logger
from pathlib import Path
import json


class RAGSystem:
    """Complete RAG system with retrieval and generation"""

    def __init__(
        self,
        collection_name: str = "default",
        embedding_model_type: str = "default",
        top_k: int = 5,
        rerank: bool = True,
    ):
        """
        Initialize RAG system

        Args:
            collection_name: Name of the vector database collection
            embedding_model_type: Type of embedding model to use
            top_k: Number of documents to retrieve
            rerank: Enable reranking
        """
        self.collection_name = collection_name

        # Initialize components
        logger.info(f"Initializing RAG system: {collection_name}")

        self.embedding_model = get_embedding_model(embedding_model_type)
        self.vector_db = VectorDatabase(collection_name=collection_name)
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_db=self.vector_db,
            top_k=top_k,
            rerank=rerank,
        )

        self.claude = claude_client

    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> None:
        """
        Index documents into the vector database

        Args:
            documents: List of documents to index
            metadatas: Optional metadata for each document
            batch_size: Batch size for embedding
        """
        logger.info(f"Indexing {len(documents)} documents...")

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size] if metadatas else None

            # Compute embeddings
            embeddings = self.embedding_model.encode(batch_docs, show_progress=True)

            # Add to vector database
            self.vector_db.add(
                documents=batch_docs,
                embeddings=embeddings.tolist(),
                metadatas=batch_meta,
            )

        logger.info(f"Indexed {len(documents)} documents successfully")

    def index_directory(
        self,
        directory: Path,
        file_patterns: List[str] = ["*.txt", "*.md", "*.pdf"],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Index all documents from a directory

        Args:
            directory: Directory containing documents
            file_patterns: File patterns to match
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        from .document_processor import DocumentProcessor

        processor = DocumentProcessor(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Load and process documents
        documents, metadatas = processor.process_directory(directory, file_patterns)

        # Index documents
        self.index_documents(documents, metadatas)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            question: User question
            top_k: Number of documents to retrieve
            return_sources: Include source documents in response
            conversation_history: Previous conversation context

        Returns:
            Response with answer and optionally sources
        """
        # Retrieve relevant documents
        if conversation_history:
            documents = self.retriever.retrieve_with_context(
                question, conversation_history, top_k=top_k
            )
        else:
            documents = self.retriever.retrieve(question, top_k=top_k)

        # Build context from retrieved documents
        context = self._build_context(documents)

        # Generate response using Claude
        answer = self._generate_answer(question, context, conversation_history)

        # Build response
        response = {"answer": answer, "question": question}

        if return_sources:
            response["sources"] = [
                {
                    "text": doc["text"],
                    "score": doc["score"],
                    "metadata": doc.get("metadata", {}),
                }
                for doc in documents
            ]

        return response

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(doc["text"])
            context_parts.append("")

        return "\n".join(context_parts)

    def _generate_answer(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate answer using Claude"""

        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.

Instructions:
1. Answer the question using ONLY the information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be concise but complete
4. Cite relevant parts of the context when appropriate
5. If asked about something not in the context, politely indicate that"""

        # Build user prompt
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context above."""

        # Add conversation history if provided
        if conversation_history:
            history_text = "\n".join(
                [
                    f"{turn['role']}: {turn['content']}"
                    for turn in conversation_history[-3:]
                ]
            )
            user_prompt = f"Previous conversation:\n{history_text}\n\n{user_prompt}"

        # Generate response
        answer = self.claude.generate(
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=2000,
        )

        return answer

    def chat(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Chat interface for RAG system

        Args:
            message: User message
            conversation_history: Conversation history
            top_k: Number of documents to retrieve

        Returns:
            Response with answer and sources
        """
        response = self.query(
            question=message,
            top_k=top_k,
            return_sources=True,
            conversation_history=conversation_history,
        )

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "collection_name": self.collection_name,
            "document_count": self.vector_db.count(),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimension": self.embedding_model.dimension,
        }

    def reset(self) -> None:
        """Reset the RAG system (delete all documents)"""
        self.vector_db.reset()
        logger.warning(f"RAG system reset: {self.collection_name}")
