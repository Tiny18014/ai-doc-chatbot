# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from pathlib import Path
from dotenv import load_dotenv
import time 
import mlflow
# MODIFIED: Added the necessary MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("RAG API Queries")
# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Professional RAG API",
    description="An API for querying documents using a powerful, local extractive QA model.",
    version="2.0.0", # Version updated for new architecture
)

# --- MODEL LOADING (UPGRADED TO EXTRACTIVE QA MODEL) ---
# MODIFIED: Switched to a model specifically fine-tuned for extractive question answering.
# This model is highly accurate at finding answers within a context and avoiding hallucinations.
llm_model_name = "deepset/roberta-base-squad2"
print(f"Loading model: {llm_model_name}")

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(llm_model_name)

# --- PIPELINE CONFIGURATION (UPGRADED) ---
# MODIFIED: Using the "question-answering" pipeline for reliable, extractive answers.
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
)
print("Model and pipeline loaded successfully.")

# --- VECTOR STORE SETUP (NO CHANGE) ---
db_path = "vector_store"
embedding_model_name = "BAAI/bge-base-en-v1.5"
print(f"Loading embedding model for API: {embedding_model_name}")

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Loading FAISS vector store...")
vectorstore = FAISS.load_local(
    db_path,
    embedding_model,
    allow_dangerous_deserialization=True
)
print("Vector store loaded successfully.")

# --- API DEFINITIONS ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):
    """
    Accepts a query, retrieves relevant documents, and extracts a precise answer.
    """
    with mlflow.start_run() as run:
        start_time = time.time()
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})
        
        retrieved_docs = retriever.invoke(request.query)
        context = format_docs(retrieved_docs)

        result = qa_pipeline(question=request.query, context=context)

        # MODIFIED: Lowered confidence threshold to match evaluation script
        answer = result['answer'] if result['score'] > 0.05 else "I do not have enough information to answer this question."
        
        end_time = time.time()
        latency = end_time - start_time

        # --- Log to MLflow ---
        mlflow.log_param("query", request.query)
        mlflow.log_param("top_k", request.top_k)
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("confidence_score", result['score'])
        
        mlflow.log_text(answer, "answer.txt")
        sources_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        mlflow.log_text(sources_text, "sources.txt")

        sources = [
            {"source": doc.metadata.get('source', 'unknown'), "content": doc.page_content}
            for doc in retrieved_docs
        ]
        
        return {
            "query": request.query,
            "answer": answer,
            "sources": sources
        }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Professional Extractive RAG API"}
