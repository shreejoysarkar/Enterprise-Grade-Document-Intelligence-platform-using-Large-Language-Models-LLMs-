"""
This module performs hybrid chunking on parsed markdown documents.
It uses both SentenceChunker and TableChunker from the Chonkie library to process
narrative text and tabular data separately. This ensures tabular structures are preserved
while text segments are chunked along natural sentence boundaries.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from chonkie import MarkdownChef, SentenceChunker, TableChunker, JSONPorter
from utils.logger import get_logger, setup_logging

# Load environment variables
load_dotenv()

# Initialize logger
setup_logging()
logger = get_logger(__name__)


class DocumentChunker:
    """Handles hybrid chunking of Markdown documents using Chonkie chunkers."""

    def __init__(
        self,
        input_dir: Path = Path("Data") / "Output",
        output_dir: Path = Path("Data") / "chunks",
        tokenizer_name: str = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        table_chunk_size: int = 3,
    ):
        """Initializes the DocumentChunker with directories and chunking configurations.

        Args:
            input_dir: Path to the directory containing input Markdown files.
            output_dir: Path to save the output chunk files.
            tokenizer_name: Name of the tokenizer to use (e.g. "gpt2", "cl100k_base").
            chunk_size: Maximum token count for text chunks.
            chunk_overlap: Number of tokens overlapping between text chunks.
            table_chunk_size: Maximum number of rows per table chunk.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tokenizer_name = tokenizer_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_chunk_size = table_chunk_size

        logger.info("Initializing Chonkie MarkdownChef and Chunkers...")
        self.chef = MarkdownChef(tokenizer=self.tokenizer_name)
        
        self.sentence_chunker = SentenceChunker(
            tokenizer=self.tokenizer_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        self.table_chunker = TableChunker(
            tokenizer="row",
            chunk_size=self.table_chunk_size,
        )
        
        self.porter = JSONPorter(lines=True)

    def chunk_document_hybrid(self, file_path: Path) -> list:
        """Processes a single markdown document using both table and sentence chunkers.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Sorted list of Chunk objects representing the document.
        """
        logger.info(f"Parsing document: {file_path.name}")
        doc = self.chef.process(file_path)
        
        content = doc.content
        current_idx = 0
        chunks = []
        
        # Sort tables by start_index to process sequentially
        tables = sorted(doc.tables, key=lambda x: x.start_index)
        
        for table in tables:
            # 1. Chunk narrative text segment before the table
            if table.start_index > current_idx:
                text_segment = content[current_idx:table.start_index]
                if text_segment.strip():
                    text_chunks = self.sentence_chunker.chunk(text_segment)
                    for chunk in text_chunks:
                        chunk.start_index = current_idx + chunk.start_index
                        chunk.end_index = current_idx + chunk.end_index
                        if chunk.metadata is None:
                            chunk.metadata = {}
                        chunk.metadata.update({
                            "type": "text",
                            "source_file": file_path.name
                        })
                        chunks.append(chunk)
            
            # 2. Chunk table content using TableChunker
            table_chunks = self.table_chunker.chunk(table.content)
            for chunk in table_chunks:
                chunk.start_index = table.start_index + chunk.start_index
                chunk.end_index = table.start_index + chunk.end_index
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.update({
                    "type": "table",
                    "source_file": file_path.name
                })
                chunks.append(chunk)
                
            current_idx = table.end_index
            
        # 3. Chunk any narrative text after the last table
        if current_idx < len(content):
            text_segment = content[current_idx:]
            if text_segment.strip():
                text_chunks = self.sentence_chunker.chunk(text_segment)
                for chunk in text_chunks:
                    chunk.start_index = current_idx + chunk.start_index
                    chunk.end_index = current_idx + chunk.end_index
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata.update({
                        "type": "text",
                        "source_file": file_path.name
                    })
                    chunks.append(chunk)
                    
        # 4. Sort chunks by start_index to preserve flow/order
        chunks.sort(key=lambda x: x.start_index)
        
        # 5. Populate unique IDs and merge metadata
        for idx, chunk in enumerate(chunks):
            chunk.id = f"{file_path.stem}_chunk_{idx}"
            if chunk.metadata is None:
                chunk.metadata = {}
            if doc.metadata:
                chunk.metadata.update(doc.metadata)
                
        return chunks

    def chunk_all(self) -> None:
        """Processes all Markdown files in the input directory and exports chunks to the output directory."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            return
            
        md_files = sorted(self.input_dir.glob("*.md"))
        logger.info(f"Found {len(md_files)} Markdown files to process in {self.input_dir}.")
        
        successful = 0
        failed = 0
        
        for md_file in md_files:
            logger.info(f"Processing: {md_file.name}...")
            try:
                # Run the hybrid chunking
                chunks = self.chunk_document_hybrid(md_file)
                
                # Output filename
                output_file = self.output_dir / f"{md_file.stem}_chunks.jsonl"
                
                # Export using JSONPorter
                self.porter.export(chunks, output_file)
                
                logger.info(f"✅ Successfully chunked and saved {len(chunks)} chunks to: {output_file.name}")
                successful += 1
            except Exception as e:
                logger.error(f"❌ Error chunking {md_file.name}: {e}", exc_info=True)
                failed += 1
                
        logger.info("\n🎉 Chunking processing complete!")
        logger.info(f"✅ Successful: {successful} | ❌ Failed: {failed}")

'''
def main():
    chunker = DocumentChunker()
    chunker.chunk_all()


if __name__ == "__main__":
    main()
'''