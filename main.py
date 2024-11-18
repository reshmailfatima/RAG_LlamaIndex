# main.py
from document_loader import load_documents
from index_creator import create_index
from query_engine import QueryEngine
from pathlib import Path 
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Ensure data directory exists
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created directory: {data_dir}")
            logger.info("Please add your PDF documents to the 'data' directory and run the program again.")
            return

        # Check if there are PDF files in the data directory
        pdf_files = list(Path(data_dir).glob("*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in the 'data' directory.")
            logger.info("Please add your PDF documents to the 'data' directory and run the program again.")
            return

        # Load documents from the data directory
        logger.info("Loading documents...")
        documents = load_documents(data_dir)
        
        if not documents:
            logger.info("No documents were successfully loaded.")
            return

        # Create index
        logger.info("Creating index...")
        index = create_index(documents)
        
        # Initialize query engine
        logger.info("Initializing query engine...")
        query_engine = QueryEngine(index)
        
        # Interactive query loop
        print("\nRAG System Ready! You can now ask questions about your documents.")
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            try:
                answer = query_engine.query(question)
                print("\nAnswer:", answer)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("Sorry, there was an error processing your question. Please try again.")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        print("An error occurred while running the application.")

if __name__ == "__main__":
    main()