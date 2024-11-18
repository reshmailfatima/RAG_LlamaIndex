# azure_openai_llm.py
from llama_index.llms.azure_openai import AzureOpenAI
from config import Config
import os

def get_azure_llm():
    """Configure Azure OpenAI LLM"""
    return AzureOpenAI(
        model="gpt-4o-mini",  # or your model name
        engine="gpt4omini",  # Your deployment name
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=0.1
    )