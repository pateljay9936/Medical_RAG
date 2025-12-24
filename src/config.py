import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central Configuration class for the application."""

    # API Keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Pinecone Configuration
    PINECONE_INDEX_NAME = "medical-chatbot"
    PINECONE_CLOUD = "aws"  
    PINECONE_REGION = "us-east-1" 
    PINECONE_METRIC = "cosine"
    PINECONE_DIMENSION = 384

    # Embeddings Configuration
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDINGS_TYPE = "huggingface"
    
    # LLM Configuration
    GEMINI_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.3 
    
    # Document Processing Configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    DATA_PATH = "data/"
    
    # Retrieval Configuration
    RETRIEVAL_K = 3
    SEARCH_TYPE = "similarity"

    @classmethod
    def validate(cls):
        """Validate that all required configuration is present."""
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        return True
