import argparse
import os
from pathlib import Path
import fitz
import yaml
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_processor import create_passages, extract_text_with_positions

# Load the project config file
with open("configs.yaml", "r", encoding="utf8") as file:
    project_configs = yaml.safe_load(file)


def clean_file_name(file_name):
    """Function to clean file names of FAISS vector db"""
    cleaned_name = file_name.strip().lower().replace(" ", "_")
    return cleaned_name


def process_single_pdf(pdf_path, output_dir, chunk_size=700, chunk_overlap=70):
    """Process a single PDF file and create its FAISS database."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    faiss_index_path = os.path.join(output_dir, f"{clean_file_name(pdf_name)}")

    if os.path.exists(faiss_index_path):
        print(f"Skipping {pdf_path} - FAISS index already exists at {faiss_index_path}")
        return False

    print(f"\nProcessing PDF: {pdf_path}")
    print(f"Output will be saved to: {faiss_index_path}")

    try:
        pdf_data = fitz.open(pdf_path)
        text_positions = extract_text_with_positions(pdf_data)
        passages = create_passages(pdf_data, text_positions)

        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in passages
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", ""],
        )
        docs = text_splitter.split_documents(documents)

        print(f"Created {len(docs)} document chunks")

        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_documents(docs, embeddings)

        print(f"Saving FAISS index to: {faiss_index_path}")
        docsearch.save_local(faiss_index_path)
        print(f"Successfully processed {pdf_path}")
        return True

    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return False


def create_faiss_db(input_path, output_dir="faiss-vdbs", chunk_size=700, chunk_overlap=70):
    """Create FAISS vector databases from PDF file or directory."""
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            process_single_pdf(str(input_path), output_dir, chunk_size, chunk_overlap)
        else:
            print(f"Skipping {input_path} - not a PDF file")

    elif input_path.is_dir():
        pdf_files = list(input_path.glob("**/*.pdf"))
        total_pdfs = len(pdf_files)

        if total_pdfs == 0:
            print(f"No PDF files found in {input_path}")
            return

        print(f"Found {total_pdfs} PDF files in {input_path}")
        processed, skipped, failed = 0, 0, 0

        for pdf_file in pdf_files:
            result = process_single_pdf(
                str(pdf_file), output_dir, chunk_size, chunk_overlap
            )
            if result:
                processed += 1
            elif os.path.exists(os.path.join(output_dir, f"{pdf_file.stem}")):
                skipped += 1
            else:
                failed += 1

        print("\nProcessing Summary:")
        print(f"Total PDFs found: {total_pdfs}")
        print(f"Successfully processed: {processed}")
        print(f"Skipped (already existed): {skipped}")
        print(f"Failed to process: {failed}")

    else:
        print(f"Error: {input_path} is neither a file nor a directory")


def create_combined_faiss_db(input_path, output_path, chunk_size=700, chunk_overlap=70):
    """Create a combined FAISS index from all PDFs."""
    print("\n🔄 Creating combined FAISS vectorstore from all PDFs...")

    input_path = Path(input_path)
    all_docs = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", ""],
    )

    for pdf_file in input_path.glob("**/*.pdf"):
        try:
            doc = fitz.open(pdf_file)
            text_positions = extract_text_with_positions(doc)
            passages = create_passages(doc, text_positions)
            documents = [
                Document(page_content=p["page_content"], metadata=p["metadata"])
                for p in passages
            ]
            split_docs = text_splitter.split_documents(documents)
            all_docs.extend(split_docs)
            print(f"✅ Processed: {pdf_file.name} → {len(split_docs)} chunks")
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")

    print(f"\n📄 Total chunks: {len(all_docs)}")

    if all_docs:
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(all_docs, embeddings)
        vectordb.save_local(output_path)
        print(f"✅ Combined FAISS index saved to: {output_path}")
    else:
        print("⚠️ No documents processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Create FAISS vector database from PDF file or directory"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=project_configs["pdf-folder-path"],
        help="Directory to find the PDF files (default from configs.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=project_configs["vectordb-folder-path"],
        help="Directory to store the FAISS databases",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=project_configs["chunk-size"],
        help="Character size of text chunks",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=project_configs["chunk-overlap"],
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="If set, combine all PDFs into one FAISS vectorstore",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    if args.combine:
        create_combined_faiss_db(
            args.input_dir,
            args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        create_faiss_db(
            args.input_dir,
            args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )


if __name__ == "__main__":
    main()
