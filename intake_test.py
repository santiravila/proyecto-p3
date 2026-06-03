###########################################################################################

import fitz
import os


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    document = fitz.open(pdf_path)
    full_text = ""

    for page_number in range(len(document)):
        page = document[page_number]
        page_text = page.get_text("text")

        if isinstance(page_text, str):
            full_text += page_text
        elif isinstance(page_text, list):
            full_text += "".join(str(item) for item in page_text)
        elif isinstance(page_text, dict):
            full_text += str(page_text)
        else:
            raise TypeError(
                f"Unsupported text type returned from page.get_text: {type(page_text)}"
            )

    return full_text


if __name__ == "__main__":
    print("Testing PDF extraction")

    try:
        text = extract_text_from_pdf("./single_document/test.pdf")
        print(f"Success. Characters extracted: {len(text)}")
        print(f"First 150 characters:\n{text[:150]}")

    except Exception as e:
        print(f"Error during extraction test: {e}")

###########################################################################################

###########################################################################################

import random
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(text: str):
    """
    Split text into overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)
    return chunks


def display_random_chunks(chunks, sample_size: int = 5):
    """
    Display random chunks for inspection.
    """
    size = min(sample_size, len(chunks))
    selected_chunks = random.sample(chunks, size)

    for i, chunk in enumerate(selected_chunks, 1):
        print(f"\n--- Random Chunk #{i} ---")
        print(chunk)
        print("-" * 60)


if __name__ == "__main__":
    print("\n--- Testing Chunking ---")

    text = extract_text_from_pdf("./single_document/test.pdf")
    chunks = split_text_into_chunks(text)

    print(f"Text split into {len(chunks)} chunks.\n")

    display_random_chunks(chunks, sample_size=5)

###########################################################################################

from qdrant_client import QdrantClient
import ollama

if __name__ == "__main__":
    print("Connecting to Qdrant (Port 6333)...")
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=3)
        client.get_collections()
        print("QDRANT: Operational and listening.")
    except Exception as e:
        print(f"[!] QDRANT: Down. Error: {e}")

    print("-" * 50)

    # 2. Test the AI engine
    print("Connecting to Ollama (Port 11434)...")
    try:
        models = ollama.list()
        print("OLLAMA: Operational and listening.")
        print(f"   Models available in the container: {[m['model'] for m in models.get('models', [])]}")
    except Exception as e:
        print(f"[!] OLLAMA: Down. Error: {e}")
###########################################################################################

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama

def upload_chunks_to_qdrant(pdf_path: str, collection_name: str = "monit_rag_collection"):
    print("[+] Extracting and splitting text from document...")
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    
    client = QdrantClient(host="localhost", port=6333)
    
    if not client.collection_exists(collection_name):
        print(f"[+] Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    
    print(f"[+] Generating embeddings for {len(chunks)} chunks with Ollama...")
    points = []
    for idx, chunk in enumerate(chunks):
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        
        points.append(
            PointStruct(
                id=idx,
                vector=response["embedding"],
                payload={
                    "source": os.path.basename(pdf_path),
                    "content": chunk
                }
            )
        )
    
    print(f"[+] Uploading points to Qdrant container...")
    client.upsert(collection_name=collection_name, points=points)
    print(" Ingestion process completed successfully!")


if __name__ == "__main__":
    print("\n--- Testing Qdrant Initialization & Upload ---")


    try:
        upload_chunks_to_qdrant("./single_document/test.pdf")
        print("\n[!] Verify the data at: http://localhost:6333/dashboard")
    except Exception as e:
        print(f"[!] Error during upload to Qdrant: {e}")

###########################################################################################