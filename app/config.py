#config.py
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

class Config:
    # Logging Configuration
    LOGGING_LEVEL = logging.INFO
    LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Azure OpenAI LLM Configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # LLM Model Configuration
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    AZURE_OPENAI_MODEL_NAME = "gpt-4o-mini"
    
    # Azure OpenAI Embeddings Configuration
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding-3-small")
    AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = "text-embedding-ada-002"
    AZURE_OPENAI_EMBEDDINGS_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
    
    # Application Configuration
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    
    # Directories
    DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./data")
    STORAGE_DIRECTORY = os.getenv("STORAGE_DIRECTORY", "./storage")
    
    # Embedding Configuration
    EMBEDDING_DIMENSION = 1536  # For text-embedding-ada-002

    @classmethod
    def validate_config(cls):
        """
        Validate critical configuration parameters
        """
        errors = []
        
        # Check critical Azure OpenAI configurations
        if not cls.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is not set")
        
        if not cls.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT is not set")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return True

# Validate configuration on import
Config.validate_config()
