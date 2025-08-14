# ingest.py
# This script processes PDF documents, splits them into intelligent chunks,
# and creates a high-quality vector store for retrieval.

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
# Use a more powerful embedding model for better semantic understanding
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Path to your source documents
SOURCE_DOCS_PATH = "./docs"
# Path to save the vector store
VECTOR_STORE_PATH = "./vector_store"

def create_vector_store():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and saves them to a FAISS vector store.
    """
    print("🚀 Starting data ingestion process...")

    # 1. Load documents
    print(f"Loading documents from: {SOURCE_DOCS_PATH}")
    documents = []
    for filename in os.listdir(SOURCE_DOCS_PATH):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(SOURCE_DOCS_PATH, filename))
            documents.extend(loader.load())
    
    if not documents:
        print("❌ No PDF documents found. Please add your PDFs to the 'docs' folder.")
        return

    print(f"✅ Loaded {len(documents)} pages from PDF files.")

    # 2. Split documents into chunks
    # Using a recursive splitter with overlap for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split documents into {len(chunks)} chunks.")

    # 3. Create embeddings
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    # Specify that the model should run on the CPU if no GPU is available
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("✅ Embedding model loaded.")

    # 4. Create and save the vector store
    print("💾 Creating and saving the vector store...")
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"⚠️  Existing vector store found at '{VECTOR_STORE_PATH}'. It will be overwritten.")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store created and saved at: {VECTOR_STORE_PATH}")
    print("🎉 Ingestion complete!")


if __name__ == "__main__":
    # Ensure the source documents directory exists
    if not os.path.exists(SOURCE_DOCS_PATH):
        os.makedirs(SOURCE_DOCS_PATH)
        print(f"Created a '{SOURCE_DOCS_PATH}' directory. Please add your PDF files there.")
    else:
        create_vector_store()
