"""
Document Processor for RAG
Handles text extraction and chunking from various file formats
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
from loguru import logger
import re


class DocumentProcessor:
    """Process documents for RAG indexing"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_directory(
        self, directory: Path, file_patterns: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process all documents in a directory

        Args:
            directory: Directory path
            file_patterns: File patterns to match

        Returns:
            Tuple of (documents, metadatas)
        """
        documents = []
        metadatas = []

        directory = Path(directory)

        for pattern in file_patterns:
            for file_path in directory.rglob(pattern):
                try:
                    file_docs, file_metas = self.process_file(file_path)
                    documents.extend(file_docs)
                    metadatas.extend(file_metas)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Processed {len(documents)} chunks from {directory}")

        return documents, metadatas

    def process_file(self, file_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a single file

        Args:
            file_path: Path to file

        Returns:
            Tuple of (document chunks, metadata)
        """
        # Extract text based on file type
        text = self._extract_text(file_path)

        # Chunk the text
        chunks = self.chunk_text(text)

        # Create metadata
        metadatas = [
            {
                "source": str(file_path),
                "chunk": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        return chunks, metadatas

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from file"""
        suffix = file_path.suffix.lower()

        if suffix in [".txt", ".md"]:
            return file_path.read_text(encoding="utf-8")

        elif suffix == ".pdf":
            return self._extract_from_pdf(file_path)

        elif suffix in [".docx", ".doc"]:
            return self._extract_from_docx(file_path)

        else:
            # Try to read as text
            try:
                return file_path.read_text(encoding="utf-8")
            except:
                logger.warning(f"Unsupported file type: {suffix}")
                return ""

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        try:
            import PyPDF2

            text = []
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())

            return "\n".join(text)

        except ImportError:
            logger.warning("PyPDF2 not installed, skipping PDF")
            return ""

    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            import docx

            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])

        except ImportError:
            logger.warning("python-docx not installed, skipping DOCX")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        # Clean text
        text = self._clean_text(text)

        # Split into sentences
        sentences = self._split_sentences(text)

        # Create chunks
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0

                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean text"""
        # Remove multiple whitespaces
        text = re.sub(r"\s+", " ", text)

        # Remove multiple newlines
        text = re.sub(r"\n+", "\n", text)

        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        return [s.strip() for s in sentences if s.strip()]
