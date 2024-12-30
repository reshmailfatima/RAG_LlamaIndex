# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import shutil
from pathlib import Path
import os
import logging
from pydantic import BaseModel

from app.services.document_loader import load_documents
from app.services.query_engine import QueryEngine
from app.services.index_creator import create_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG System API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")

# Global variables
query_engine = None

# Pydantic models
class Query(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

def initialize_rag_system():
    """Initialize or reinitialize the RAG system"""
    global query_engine
    
    try:
        # Ensure directories exist
        for directory in [DATA_DIR, STORAGE_DIR]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")

        # Check for PDF files in the correct data directory
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
        if not pdf_files:
            logger.info(f"No PDF files found in the data directory: {DATA_DIR}")
            return False

        # Load documents from the correct path
        logger.info(f"Loading documents from: {DATA_DIR}")
        documents = load_documents(DATA_DIR)
        
        if not documents:
            logger.info("No documents were successfully loaded.")
            return False

        # Delete existing storage files
        for file in Path(STORAGE_DIR).glob("*"):
            os.remove(file)
            logger.info(f"Deleted storage file: {file}")

        logger.info(f"Creating and persisting index in: {STORAGE_DIR}")
        index = create_index(documents, persist_dir=STORAGE_DIR)
        
        # Initialize query engine
        logger.info("Initializing query engine...")
        query_engine = QueryEngine(index)
        
        return True

    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    if initialize_rag_system():
        logger.info("RAG system initialized successfully")
    else:
        logger.info("RAG system waiting for documents")

@app.post("/upload/", response_model=dict)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload PDF documents to the system"""
    try:
        uploaded_files = []
        # Save uploaded files to the correct data directory
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

            file_path = os.path.join(DATA_DIR, file.filename)
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                uploaded_files.append(file.filename)
                logger.info(f"Successfully uploaded: {file.filename} to {file_path}")
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}")

        # Reinitialize the RAG system
        if initialize_rag_system():
            return {
                "message": f"Successfully uploaded {len(files)} documents and initialized the system",
                "uploaded_files": uploaded_files,
                "data_directory": DATA_DIR
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize system with uploaded documents")

    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", response_model=QueryResponse)
async def query_documents(query: Query):
    """Query the RAG system"""
    global query_engine

    if not query_engine:
        raise HTTPException(status_code=400, detail="System not initialized. Please upload documents first.")

    try:
        answer = query_engine.query(query.question)
        return QueryResponse(answer=str(answer))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your query")

@app.get("/status/")
async def get_status():
    """Get the system status"""
    try:
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
        has_documents = len(pdf_files) > 0
        is_initialized = query_engine is not None

        return {
            "status": "ready" if is_initialized else "waiting_for_documents",
            "documents_loaded": len(pdf_files),
            "document_names": [f.name for f in pdf_files],
            "system_initialized": is_initialized,
            "data_directory": DATA_DIR,
            "storage_directory": STORAGE_DIR
        }
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking system status")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
