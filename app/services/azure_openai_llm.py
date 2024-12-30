#azure openai llm
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from app.config import Config
def get_azure_llm():
    """Configure Azure OpenAI LLM"""
    return AzureOpenAI(
        model=Config.AZURE_OPENAI_MODEL_NAME,
        engine=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        temperature=0.1
    )
def get_azure_embeddings():
    """Configure Azure OpenAI Embeddings"""
    return AzureOpenAIEmbedding(
        model=Config.AZURE_OPENAI_EMBEDDINGS_MODEL_NAME,
        deployment_name=Config.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
        api_key=Config.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.AZURE_OPENAI_EMBEDDINGS_ENDPOINT,
        api_version=Config.AZURE_OPENAI_API_VERSION
    )
