import argparse
import sys
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

def run_ingestion(data_dir: str):
    """Run Phase 1-3: Document Processing, Chunking, and Indexing."""
    logger.info("Starting Document Ingestion Pipeline...")
    
    # Imports inside functions to prevent heavy loading for simple commands
    from core.doc_processor_1 import DirectoryProcessor
    from core.chunking_2 import HybridChunker
    from core.embedding_and_indexing_3 import HybridSearchIndexer

    # Phase 1: Process PDFs to Markdown
    processor = DirectoryProcessor(input_dir=data_dir)
    processor.process_directory()
    
    # Phase 2: Chunk Markdown to JSONL
    chunker = HybridChunker()
    chunker.process_directory()
    
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

def main():
    parser = argparse.ArgumentParser(description="Doc-Intel Enterprise RAG Platform")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run the ingestion pipeline (PDF -> Index)")
    ingest_parser.add_argument("--data-dir", type=str, default="Data/Input", help="Path to input PDFs")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("text", type=str, help="The question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion(args.data_dir)
    elif args.command == "query":
        run_query(args.text)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
