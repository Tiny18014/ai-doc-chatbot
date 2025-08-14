AI-Powered Document Q&A Chatbot
This project is a professional, end-to-end question-answering system that allows users to chat with their documents. It leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline, featuring a powerful local AI model, a robust backend API, and a user-friendly web interface. The entire system is containerized with Docker for easy deployment and includes a full suite of MLOps tools for evaluation and experiment tracking.

✨ Features
Accurate, Extractive Q&A: Utilizes a state-of-the-art model (deepset/roberta-base-squad2) to extract precise answers directly from source documents, minimizing hallucinations.

High-Performance Retrieval: Employs a powerful embedding model (BAAI/bge-base-en-v1.5) and a FAISS vector store for fast and highly relevant document retrieval.

Interactive Web Interface: A clean and modern chat interface built with Streamlit, which displays answers and transparently cites the source documents.

Robust Backend API: A scalable and efficient backend built with FastAPI, serving the RAG pipeline.

Quantitative Evaluation: Includes a data-driven evaluation pipeline to objectively measure the system's performance on key metrics like context relevance and answer accuracy.

Experiment Tracking: Integrates with MLflow to log every query, response, and performance metric for monitoring and debugging.

Containerized & Deployable: Fully containerized with Docker and orchestrated with Docker Compose, allowing the entire application to be run with a single command.

🛠️ Tech Stack
Component

Technology

Backend

FastAPI, Uvicorn

Frontend

Streamlit

AI / ML

PyTorch, Transformers, LangChain, Sentence-Transformers

Vector Database

FAISS

MLOps

Docker, Docker Compose, MLflow

Core Models

deepset/roberta-base-squad2 (Q&A), BAAI/bge-base-en-v1.5 (Embeddings)

📂 Project Structure
ai-doc-chatbot/
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docs/
│   └── (Your PDF documents go here)
├── vector_store/
│   └── (FAISS index is generated here)
├── ingest.py
├── evaluate.py
├── evaluation_data.py
└── docker-compose.yml

🚀 Getting Started
Prerequisites
Python 3.10+

Pip

(Optional) Docker Desktop

1. Clone the Repository
git clone https://github.com/your-username/ai-doc-chatbot.git
cd ai-doc-chatbot

2. Set Up the Environment
Create and activate a Python virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install all necessary dependencies:

pip install -r backend/requirements.txt
pip install -r dashboard/requirements.txt
pip install scikit-learn # For evaluation script

3. Ingest Your Data
Place all your PDF documents inside the docs/ folder.

Run the ingestion script to build your vector store:

python ingest.py

4. Run the Application Locally
Start the Backend API:
Open a terminal and run:

uvicorn backend.main:app --reload

Start the Frontend Dashboard:
Open a second terminal and run:

streamlit run dashboard/app.py

Navigate to http://localhost:8501 in your browser to start chatting with your documents.

🐳 Running with Docker
With Docker, you can run the entire application with a single command.

Make sure Docker Desktop is running.

From the project's root directory, run:

docker-compose up --build

Access the Streamlit dashboard at http://localhost:8501.

📊 Evaluation Results
The performance of the RAG pipeline has been quantitatively measured. The improved ingestion process led to excellent context retrieval, which is the foundation of a high-quality RAG system.

Metric

Score

Assessment

Average Context Relevance

0.730

✅ Excellent context retrieval

Average Answer Similarity

0.460

⚠️ Answer accuracy needs improvement

This data-driven approach allows for targeted improvements and demonstrates a professional MLOps workflow.

📈 Experiment Tracking with MLflow
All API queries are automatically logged with MLflow.

After running the application and asking some questions, stop the backend server.

From the backend/ directory, launch the MLflow UI:

mlflow ui

Navigate to http://127.0.0.1:5000 to view a detailed dashboard of all queries, responses, and performance metrics.
