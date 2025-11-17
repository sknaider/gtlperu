"""
Embedding Models for RAG
Optimized for RTX 5090
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
from loguru import logger
from backend.core.config import settings
from backend.core.cache import cache


class EmbeddingModel:
    """Embedding model wrapper with GPU optimization"""

    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        batch_size: int = 256,  # Large batch for RTX 5090
        normalize: bool = True,
    ):
        """
        Initialize embedding model

        Args:
            model_name: Model name from sentence-transformers
            device: Device to use (cuda recommended)
            batch_size: Batch size for encoding
            normalize: Normalize embeddings
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info(f"Loading embedding model: {self.model_name}")

        # Load model with GPU optimization
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Enable optimizations for RTX 5090
        if self.device == "cuda":
            # Enable TF32 for better performance on Ampere/Ada GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: Text or list of texts
            show_progress: Show progress bar
            use_cache: Use cache for repeated texts

        Returns:
            Embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache
        if use_cache and len(texts) == 1:
            cache_key = f"embed:{self.model_name}:{texts[0][:100]}"
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        # Encode with optimized settings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        # Cache single text embeddings
        if use_cache and len(texts) == 1:
            cache_key = f"embed:{self.model_name}:{texts[0][:100]}"
            cache.set(cache_key, embeddings)

        return embeddings

    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """Encode queries (alias for encode)"""
        return self.encode(queries)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents with progress bar"""
        return self.encode(documents, show_progress=True)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)

        return cosine_similarity(embedding1, embedding2)[0][0]

    def batch_similarity(
        self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between query and document embeddings

        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            doc_embeddings: Document embeddings (n_docs, dim)

        Returns:
            Similarity matrix (n_queries, n_docs)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(query_embeddings, doc_embeddings)


# Specialized embedding models
class MultilingualEmbedding(EmbeddingModel):
    """Multilingual embedding model"""

    def __init__(self, **kwargs):
        kwargs["model_name"] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        super().__init__(**kwargs)


class CodeEmbedding(EmbeddingModel):
    """Code-specialized embedding model"""

    def __init__(self, **kwargs):
        kwargs["model_name"] = "microsoft/codebert-base"
        super().__init__(**kwargs)


class FastEmbedding(EmbeddingModel):
    """Fast, lightweight embedding model"""

    def __init__(self, **kwargs):
        kwargs["model_name"] = "sentence-transformers/all-MiniLM-L6-v2"
        super().__init__(**kwargs)


class HighQualityEmbedding(EmbeddingModel):
    """High quality embedding model (slower but better)"""

    def __init__(self, **kwargs):
        kwargs["model_name"] = "sentence-transformers/all-mpnet-base-v2"
        super().__init__(**kwargs)


def get_embedding_model(model_type: str = "default", **kwargs) -> EmbeddingModel:
    """
    Factory function to get embedding model

    Args:
        model_type: Type of embedding model
        **kwargs: Additional arguments

    Returns:
        Embedding model instance
    """
    models = {
        "default": EmbeddingModel,
        "multilingual": MultilingualEmbedding,
        "code": CodeEmbedding,
        "fast": FastEmbedding,
        "quality": HighQualityEmbedding,
    }

    model_class = models.get(model_type, EmbeddingModel)
    return model_class(**kwargs)
