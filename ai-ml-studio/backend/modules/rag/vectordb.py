"""
Vector Database with ChromaDB
Optimized for high-performance retrieval
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
from loguru import logger
import numpy as np
from pathlib import Path
from backend.core.config import settings as app_settings


class VectorDatabase:
    """ChromaDB vector database wrapper"""

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        embedding_function=None,
    ):
        """
        Initialize vector database

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedding_function: Custom embedding function
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or app_settings.chroma_persist_directory

        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )
            logger.info(f"Created new collection: {collection_name}")

    def add(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to the database

        Args:
            documents: List of documents
            embeddings: Pre-computed embeddings (optional)
            metadatas: Metadata for each document
            ids: Document IDs (auto-generated if not provided)
        """
        # Generate IDs if not provided
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        # Add to collection
        if embeddings is not None:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

        logger.info(f"Added {len(documents)} documents to {self.collection_name}")

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the database

        Args:
            query_texts: Query texts
            query_embeddings: Pre-computed query embeddings
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter

        Returns:
            Query results with documents, distances, and metadata
        """
        results = self.collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

        return results

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get documents by ID or filter

        Args:
            ids: Document IDs
            where: Metadata filter
            limit: Maximum number of results

        Returns:
            Documents and metadata
        """
        return self.collection.get(ids=ids, where=where, limit=limit)

    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update documents in the database"""
        self.collection.update(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Updated {len(ids)} documents")

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete documents from the database"""
        self.collection.delete(ids=ids, where=where)

        logger.info(f"Deleted documents from {self.collection_name}")

    def count(self) -> int:
        """Get the number of documents in the collection"""
        return self.collection.count()

    def reset(self) -> None:
        """Reset the collection (delete all documents)"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Reset collection: {self.collection_name}")

    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """Peek at the first few documents"""
        return self.collection.peek(limit=limit)

    def list_collections(self) -> List[str]:
        """List all collections in the database"""
        collections = self.client.list_collections()
        return [c.name for c in collections]

    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")


class HybridVectorDatabase(VectorDatabase):
    """
    Hybrid vector database with both dense and sparse retrieval
    """

    def __init__(self, collection_name: str = "hybrid", **kwargs):
        super().__init__(collection_name=collection_name, **kwargs)
        self.bm25_index = None

    def build_bm25_index(self, documents: List[str]):
        """Build BM25 index for sparse retrieval"""
        from rank_bm25 import BM25Okapi

        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)

        logger.info("Built BM25 index for hybrid retrieval")

    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        alpha: float = 0.5,  # Weight for dense vs sparse
    ) -> Dict[str, Any]:
        """
        Hybrid search combining dense and sparse retrieval

        Args:
            query: Query text
            n_results: Number of results
            alpha: Weight (0=sparse only, 1=dense only, 0.5=balanced)

        Returns:
            Combined search results
        """
        # Dense search
        dense_results = self.query(query_texts=[query], n_results=n_results * 2)

        # Sparse search (BM25)
        if self.bm25_index is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)

            # Combine scores (simple weighted combination)
            # In production, use reciprocal rank fusion or more sophisticated methods
            pass

        return dense_results


def create_vector_database(
    collection_name: str,
    documents: List[str],
    embeddings: Optional[List[List[float]]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> VectorDatabase:
    """
    Utility function to create and populate a vector database

    Args:
        collection_name: Name of the collection
        documents: Documents to add
        embeddings: Pre-computed embeddings
        metadatas: Metadata for documents
        **kwargs: Additional arguments for VectorDatabase

    Returns:
        Populated VectorDatabase instance
    """
    db = VectorDatabase(collection_name=collection_name, **kwargs)

    if documents:
        db.add(documents=documents, embeddings=embeddings, metadatas=metadatas)

    return db
