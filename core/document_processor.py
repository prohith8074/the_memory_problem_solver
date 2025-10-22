"""
Document processing module for handling PDF, image, and URL inputs.
Supports multiple input types and converts them to processable text chunks.
"""

import asyncio
import logging
import os
import tempfile
from typing import List, Dict, Any, Optional, Union
import base64
import io 
# Document processing libraries
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import pytesseract  # OCR for images
from llama_parse import LlamaParse
import trafilatura  # Web scraping
import requests
from firecrawl import FirecrawlApp

# Local imports
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles processing of various document types."""

    def __init__(self):
        """Initialize the document processor with API clients."""
        self.llama_parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="text",
            verbose=True,
            language='en'
        )
        self.firecrawl = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY', ''))

    async def process_input(self, input_data: Union[str, bytes], input_type: str) -> List[Dict[str, Any]]:
        """
        Process input based on type (pdf, image, url).

        Args:
            input_data: File path/URL or raw bytes
            input_type: Type of input ('pdf', 'image', 'url')

        Returns:
            List of text chunks with metadata
        """
        try:
            if input_type == 'pdf':
                return await self._process_pdf(input_data)
            elif input_type == 'image':
                return await self._process_image(input_data)
            elif input_type == 'url':
                return await self._process_url(input_data)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")

        except Exception as e:
            logger.error(f"Error processing {input_type} input: {e}")
            raise

    async def _process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF file using LlamaParse."""
        logger.info(f"Processing PDF: {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check file size
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise ValueError(f"File size {file_size_mb,".2f"}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB")

        try:
            # Use LlamaParse for intelligent PDF parsing
            documents = await self.llama_parser.aload_data(pdf_path)

            # Convert to chunks with metadata
            chunks = []
            for i, doc in enumerate(documents):
                text_content = doc.text if hasattr(doc, 'text') else str(doc)

                chunks.append({
                    'id': f"pdf_chunk_{i}",
                    'content': text_content,
                    'metadata': {
                        'source': pdf_path,
                        'type': 'pdf',
                        'chunk_index': i,
                        'total_chunks': len(documents)
                    }
                })

            logger.info(f"Processed PDF into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"LlamaParse failed, falling back to PyMuPDF: {e}")
            return await self._process_pdf_fallback(pdf_path)

    async def _process_pdf_fallback(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Fallback PDF processing using PyMuPDF."""
        chunks = []

        with fitz.open(pdf_path) as pdf_doc:
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                text = page.get_text()

                if text.strip():  # Only add non-empty pages
                    # Split into chunks based on settings
                    page_chunks = self._split_text_into_chunks(text, page_num, pdf_path)
                    chunks.extend(page_chunks)

        logger.info(f"Processed PDF (fallback) into {len(chunks)} chunks")
        return chunks

    async def _process_image(self, image_data: Union[str, bytes]) -> List[Dict[str, Any]]:
        """Process image using OCR."""
        logger.info("Processing image with OCR")

        try:
            if isinstance(image_data, str):
                # File path
                image = Image.open(image_data)
            else:
                # Raw bytes
                image = Image.open(io.BytesIO(image_data))

            # Convert to grayscale for better OCR
            image = image.convert('L')

            # Extract text using Tesseract
            text = pytesseract.image_to_string(image)

            if not text.strip():
                return []

            # Create chunks from extracted text
            chunks = self._split_text_into_chunks(text, 0, image_data)
            for chunk in chunks:
                chunk['metadata']['type'] = 'image'

            logger.info(f"Processed image into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise

    async def _process_url(self, url: str) -> List[Dict[str, Any]]:
        """Process URL content using trafilatura or firecrawl."""
        logger.info(f"Processing URL: {url}")

        try:
            # Try firecrawl first (better for modern websites)
            if self.firecrawl:
                try:
                    result = self.firecrawl.scrape_url(url)
                    if result and result.get('content'):
                        text_content = result['content']
                        metadata = {
                            'title': result.get('title', ''),
                            'url': url,
                            'scraper': 'firecrawl'
                        }
                    else:
                        raise Exception("No content from firecrawl")
                except Exception as e:
                    logger.warning(f"Firecrawl failed: {e}")

                    # Fallback to trafilatura
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }

                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()

                    text_content = trafilatura.extract(
                        response.text,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False
                    )

                    if not text_content:
                        raise ValueError("No content extracted from URL")

                    metadata = {
                        'url': url,
                        'scraper': 'trafilatura'
                    }

            else:
                # Use trafilatura only
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                text_content = trafilatura.extract(response.text)
                if not text_content:
                    raise ValueError("No content extracted from URL")

                metadata = {
                    'url': url,
                    'scraper': 'trafilatura'
                }

            # Split into chunks
            chunks = self._split_text_into_chunks(text_content, 0, url)
            for chunk in chunks:
                chunk['metadata'].update(metadata)
                chunk['metadata']['type'] = 'url'

            logger.info(f"Processed URL into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"URL processing failed: {e}")
            raise

    def _split_text_into_chunks(self, text: str, chunk_index: int, source: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks based on settings."""
        if not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + settings.CHUNK_SIZE

            # Find the last complete sentence within the chunk size
            if end < len(text):
                # Look for sentence endings
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)

                if last_period > start + 100:  # Ensure minimum chunk size
                    end = last_period + 1
                elif last_newline > start + 100:
                    end = last_newline
                else:
                    # Find word boundary
                    last_space = text.rfind(' ', start, end)
                    if last_space > start + 100:
                        end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'id': f"{source}_chunk_{chunk_index}_{len(chunks)}",
                    'content': chunk_text,
                    'metadata': {
                        'source': source,
                        'chunk_index': len(chunks),
                        'start_char': start,
                        'end_char': end
                    }
                })

            # Move start position with overlap
            start = end - settings.CHUNK_OVERLAP if end < len(text) else end

        return chunks

    def detect_input_type(self, input_data: Union[str, bytes]) -> str:
        """Detect the type of input data."""
        if isinstance(input_data, str):
            if input_data.startswith(('http://', 'https://')):
                return 'url'
            elif input_data.lower().endswith('.pdf'):
                return 'pdf'
            elif input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                return 'image'
            else:
                # Check if it's a URL
                if '://' in input_data:
                    return 'url'
                else:
                    return 'unknown'

        elif isinstance(input_data, bytes):
            # Try to detect based on magic bytes
            if input_data.startswith(b'%PDF'):
                return 'pdf'
            else:
                return 'image'

        return 'unknown'