"""
Configuration settings for the RAG system.
Centralized configuration management for all external services.
"""

import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # API Keys
        self.COHERE_API_KEY: str = os.getenv('COHERE_API_KEY', '')
        self.GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
        self.LLAMA_CLOUD_API_KEY: str = os.getenv('LLAMA_CLOUD_API_KEY', '')

        # LLM Models
        self.GROQ_MODEL: str = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')
        self.COHERE_MODEL: str = os.getenv('COHERE_MODEL', 'command-r-plus')

        # Vector Database Configuration
        self.QDRANT_URL: str = os.getenv('QDRANT_URL')
        self.QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY')
        self.QDRANT_COLLECTION_NAME: str = os.getenv('QDRANT_COLLECTION_NAME', 'Rag_System')
        self.QDRANT_VECTOR_SIZE: int = int(os.getenv('QDRANT_VECTOR_SIZE', '1024'))
        # Redis Configuration
        self.REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
        self.REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
        self.REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))
        self.REDIS_PASSWORD: str = os.getenv('REDIS_PASSWORD', '')

        # Memory Configuration
        self.SHORT_TERM_MEMORY_TTL: int = int(os.getenv('SHORT_TERM_MEMORY_TTL', '3600'))
        self.MAX_CONVERSATION_HISTORY: int = int(os.getenv('MAX_CONVERSATION_HISTORY', '10'))

        # Document Processing
        self.CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '1000'))
        self.CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '200'))

        # File Upload Settings
        self.MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
        self.ALLOWED_FILE_TYPES: list = ['.pdf', '.txt', '.md', '.docx']

        # Logging
        self.LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE: str = os.getenv('LOG_FILE', 'rag_system.log')

        # Opik Configuration (for LLM tracing)
        self.OPIK_API_KEY: str = os.getenv('OPIK_API_KEY', '')
        self.OPIK_PROJECT_NAME: str = os.getenv('OPIK_PROJECT_NAME', 'rag_system')

    def validate(self) -> None:
        """Validate that all required settings are provided."""
        required_settings = [
            'COHERE_API_KEY',
            'GROQ_API_KEY'
        ]

        missing = []
        for setting in required_settings:
            value = getattr(self, setting)
            if not value:
                missing.append(setting)

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Validate file size
        if self.MAX_FILE_SIZE_MB <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be positive")

        # Validate chunk settings
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")


# Global settings instance
settings = Settings()
