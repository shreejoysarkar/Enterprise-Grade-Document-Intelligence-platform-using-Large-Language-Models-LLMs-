import argparse
import sys
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

def run_ingestion(data_dir: str):
    """Run Phase 1-3: Document Processing, Chunking, and Indexing."""
    logger.info("Starting Document Ingestion Pipeline...")
    
    # Imports inside functions to prevent heavy loading for simple commands
    from core.doc_processor_1 import DocumentProcessor
    from core.chunking_2 import DocumentChunker
    from core.embedding_and_indexing_3 import HybridSearchIndexer

    # Phase 1: Process PDFs to Markdown
    processor = DocumentProcessor(input_dir=data_dir)
    processor.process_all()
    
    # Phase 2: Chunk Markdown to JSONL
    chunker = DocumentChunker()
    chunker.chunk_all()
    
    # Phase 3: Embed and Index
    indexer = HybridSearchIndexer()
    indexer.index_all()
    
    logger.info("Ingestion pipeline completed successfully.")

def run_query(query: str):
    """Run Phase 4: Retrieval and Generation."""
    logger.info(f"Running query: '{query}'")
    
    from core.retrieval_and_generation_4 import RAGPipeline
    
    pipeline = RAGPipeline()
    pipeline.generate(query)

def run_chat():
    """Run Phase 4: Interactive Retrieval and Generation Loop."""
    logger.info("Starting interactive RAG chat... (Type 'exit' or 'quit' to stop)")
    
    from core.retrieval_and_generation_4 import RAGPipeline
    pipeline = RAGPipeline()
    
    print("\n--- RAG Chat Initialized ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nQuery: ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query.strip():
                continue
                
            pipeline.generate(query)
            
        except KeyboardInterrupt:
            break
            
    print("\nExiting chat.")

def main():
    parser = argparse.ArgumentParser(description="Doc-Intel Enterprise RAG Platform")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run the ingestion pipeline (PDF -> Index)")
    ingest_parser.add_argument("--data-dir", type=str, default="Data/Input", help="Path to input PDFs")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline once")
    query_parser.add_argument("text", type=str, help="The question to ask")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion(args.data_dir)
    elif args.command == "query":
        run_query(args.text)
    elif args.command == "chat":
        run_chat()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
