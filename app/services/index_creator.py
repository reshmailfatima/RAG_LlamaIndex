# index_creator.py
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import load_index_from_storage
from app.services.azure_openai_llm import get_azure_llm, get_azure_embeddings
import faiss
import os
import json

def create_index(documents, persist_dir="./storage"):
    """Create index with Azure OpenAI services and local storage"""
    # Configure both LLM and embeddings
    Settings.llm = get_azure_llm()
    Settings.embed_model = get_azure_embeddings()
    
    # Create storage directory if it doesn't exist
    os.makedirs(persist_dir, exist_ok=True)
    
    # Initialize storage context
    storage_context = None
    
    # Check if we have a stored index
    if os.path.exists(os.path.join(persist_dir, "faiss.index")) and \
       os.path.exists(os.path.join(persist_dir, "docstore.json")):
        try:
            # Load existing index
            faiss_index = faiss.read_index(os.path.join(persist_dir, "faiss.index"))
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=persist_dir
            )
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            print(f"Error loading existing index: {e}")
            # If loading fails, we'll create a new index
            pass
    
    # Create new index if we don't have one or loading failed
    dimension = 1536  # Dimension for text-embedding-ada-002
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    # Create new storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    # Save all index data
    try:
        # Save FAISS index
        faiss.write_index(faiss_index, os.path.join(persist_dir, "faiss.index"))
        
        # Persist the storage context
        index.storage_context.persist(persist_dir=persist_dir)
        
    except Exception as e:
        print(f"Error saving index: {e}")
        # Even if saving fails, return the index so it can be used
    
    return index
