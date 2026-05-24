'''
This code utilizes the Docling library to convert PDF documents into Markdown format. 
It employs a GPU-accelerated pipeline to process multiple files efficiently. 
The converter is configured to enable remote services, disable OCR, 
and perform detailed table structure and picture description generation for enhanced data extraction.

'''

from dotenv import load_dotenv

load_dotenv()


import os
import sys
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
    AcceleratorDevice
)


class DocumentProcessor:
    """Encapsulates PDF -> Markdown conversion using Docling.

    By default it reads PDFs from Data/Input and writes Markdown to Data/Output.
    """

    def __init__(self, input_dir: Path = Path("Data") / "Input", output_dir: Path = Path("Data") / "Output", use_gpu: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.converter = self._init_converter()

    def _init_converter(self) -> DocumentConverter:
        """Initialize the DocumentConverter with pipeline options.

        Falls back to CPU if GPU initialization fails.
        """
        device = AcceleratorDevice.CUDA if self.use_gpu else AcceleratorDevice.CPU

        try:
            pipeline_options = PdfPipelineOptions(
                enable_remote_services=True,
                do_ocr=False,
                do_table_structure=True,
                generate_picture_images=True,
                do_picture_description=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE
                ),
                accelerator_options=AcceleratorOptions(
                    num_threads=8,
                    device=device
                )
            )
        except Exception:
            # Fallback to a safe CPU configuration
            pipeline_options = PdfPipelineOptions(
                enable_remote_services=True,
                do_ocr=False,
                do_table_structure=True,
                generate_picture_images=True,
                do_picture_description=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE
                ),
                accelerator_options=AcceleratorOptions(
                    num_threads=4,
                    device=AcceleratorDevice.CPU
                )
            )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

        return converter

    def process_all(self) -> None:
        """Process all PDF files found in the input directory and save Markdown to output."""

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.exists():
            print(f"Input directory not found: {self.input_dir}")
            print(f"Please add PDF files to process in: {self.input_dir}")
            try:
                self.input_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created input directory: {self.input_dir}")
            except Exception:
                pass
            return

        pdf_files = sorted(self.input_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process in {self.input_dir}.")

        successful = 0
        failed = 0
        result = None

        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}...")
            try:
                result = self.converter.convert(pdf_file)
                markdown_content = result.document.export_to_markdown()

                output_file = self.output_dir / f"{pdf_file.stem}.md"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)

                print(f"✅ Successfully saved: {output_file.name}")
                successful += 1
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")
                failed += 1

        print("\n🎉 All processing complete!")
        print(f"✅ Successful: {successful} | ❌ Failed: {failed}")


'''
def main():
    processor = DocumentProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()

'''
