"""
Advanced Retriever for RAG
"""

from typing import List, Dict, Any, Optional, Tuple
from .embeddings import EmbeddingModel
from .vectordb import VectorDatabase
from loguru import logger
import numpy as np


class Retriever:
    """Advanced retriever with reranking and filtering"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDatabase,
        top_k: int = 5,
        rerank: bool = True,
    ):
        """
        Initialize retriever

        Args:
            embedding_model: Embedding model
            vector_db: Vector database
            top_k: Number of results to retrieve
            rerank: Enable reranking
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.top_k = top_k
        self.rerank = rerank

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents

        Args:
            query: Query text
            top_k: Number of results (uses default if not provided)
            filter_metadata: Metadata filter

        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or self.top_k

        # Encode query
        query_embedding = self.embedding_model.encode(query)

        # Retrieve from vector DB
        results = self.vector_db.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2 if self.rerank else top_k,  # Get more if reranking
            where=filter_metadata,
        )

        # Format results
        documents = []
        for i in range(len(results["documents"][0])):
            doc = {
                "text": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "id": results["ids"][0][i],
            }
            documents.append(doc)

        # Rerank if enabled
        if self.rerank:
            documents = self.rerank_documents(query, documents)[:top_k]

        return documents

    def rerank_documents(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder or other reranking method

        Args:
            query: Query text
            documents: Retrieved documents

        Returns:
            Reranked documents
        """
        try:
            from sentence_transformers import CrossEncoder

            # Load cross-encoder for reranking
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # Score all documents
            pairs = [[query, doc["text"]] for doc in documents]
            scores = model.predict(pairs)

            # Update scores and sort
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
                doc["original_score"] = doc["score"]
                doc["score"] = float(score)

            documents.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original scores")

        return documents

    def retrieve_with_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with conversation context

        Args:
            query: Current query
            conversation_history: Previous conversation turns
            top_k: Number of results

        Returns:
            Retrieved documents
        """
        # Incorporate conversation history into query
        if conversation_history:
            context_query = self._build_context_query(query, conversation_history)
        else:
            context_query = query

        return self.retrieve(context_query, top_k=top_k)

    def _build_context_query(
        self, query: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Build query with conversation context"""
        # Simple approach: concatenate last few turns
        context_parts = []

        for turn in conversation_history[-3:]:  # Last 3 turns
            if turn.get("role") == "user":
                context_parts.append(turn["content"])

        context_parts.append(query)

        return " ".join(context_parts)

    def multi_query_retrieve(
        self, queries: List[str], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using multiple query variations

        Args:
            queries: List of query variations
            top_k: Total number of results

        Returns:
            Deduplicated and merged results
        """
        top_k = top_k or self.top_k
        all_results = {}

        # Retrieve for each query
        for query in queries:
            results = self.retrieve(query, top_k=top_k)

            for doc in results:
                doc_id = doc["id"]
                if doc_id not in all_results:
                    all_results[doc_id] = doc
                else:
                    # Merge scores (take maximum)
                    all_results[doc_id]["score"] = max(
                        all_results[doc_id]["score"], doc["score"]
                    )

        # Sort by score and return top_k
        results = list(all_results.values())
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]


class HybridRetriever(Retriever):
    """Retriever with hybrid dense + sparse search"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_bm25 = True

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: float = 0.7,  # Weight for dense retrieval
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining dense and sparse methods

        Args:
            query: Query text
            top_k: Number of results
            alpha: Dense weight (1-alpha is sparse weight)

        Returns:
            Hybrid search results
        """
        # For now, just use dense retrieval
        # In production, implement BM25 + dense fusion
        return super().retrieve(query, top_k=top_k)
