from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from azure_openai_llm import get_azure_llm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_local_embeddings():
    """Configure local HuggingFace embeddings"""
    return HuggingFaceEmbedding(
        model_name="all-MiniLM-L6-v2"  # This is a lightweight, efficient model
    )

def create_index(documents):
    # Create service context with Azure OpenAI
    Settings.llm = get_azure_llm()
    Settings.embed_model = get_local_embeddings()
        
        # Create index with the configured settings
    index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
    return index
