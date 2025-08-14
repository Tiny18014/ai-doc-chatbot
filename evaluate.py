# evaluate.py - Custom RAG Evaluation Script
# This script provides professional evaluation metrics without external dependencies

import os
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from difflib import SequenceMatcher

# --- 1. Load Your RAG Pipeline Components ---
llm_model_name = "deepset/roberta-base-squad2"
print(f"Loading model: {llm_model_name}")

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(llm_model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load the retriever
db_path = "./vector_store"
# MODIFIED: Ensure this matches the model used in ingest.py
embedding_model_name = "BAAI/bge-base-en-v1.5"

print(f"Loading embedding model for evaluation: {embedding_model_name}")
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 2. Custom Evaluation Metrics ---
def calculate_answer_similarity(predicted, expected):
    """Calculate similarity between predicted and expected answers"""
    def clean_text(text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return ' '.join(text.split())
    
    pred_clean = clean_text(predicted)
    exp_clean = clean_text(expected)
    
    similarity = SequenceMatcher(None, pred_clean, exp_clean).ratio()
    return similarity

def calculate_context_relevance(question, context_docs):
    """Calculate how relevant the retrieved context is to the question"""
    question_embedding = embedding_model.embed_query(question)
    
    relevance_scores = []
    for doc in context_docs:
        doc_embedding = embedding_model.embed_query(doc.page_content)
        similarity = cosine_similarity([question_embedding], [doc_embedding])[0][0]
        relevance_scores.append(similarity)
    
    return np.mean(relevance_scores) if relevance_scores else 0.0

def calculate_answer_confidence(qa_result):
    """Extract confidence from QA pipeline result"""
    return qa_result.get('score', 0.0)

def evaluate_rag_performance(questions, ground_truths):
    """Main evaluation function"""
    results = []
    
    for i, (question, expected_answer) in enumerate(zip(questions, ground_truths)):
        print(f"\n--- Evaluating Question {i+1} ---")
        print(f"Question: {question}")
        print(f"Expected: {expected_answer}")
        
        retrieved_docs = retriever.invoke(question)
        context_str = format_docs(retrieved_docs)
        
        qa_result = qa_pipeline(question=question, context=context_str)
        predicted_answer = qa_result['answer'] if qa_result['score'] > 0.1 else "I do not have enough information to answer this question."
        
        print(f"Predicted: {predicted_answer}")
        print(f"Confidence: {qa_result['score']:.3f}")
        
        answer_similarity = calculate_answer_similarity(predicted_answer, expected_answer)
        context_relevance = calculate_context_relevance(question, retrieved_docs)
        answer_confidence = calculate_answer_confidence(qa_result)
        
        result = {
            'question': question,
            'expected_answer': expected_answer,
            'predicted_answer': predicted_answer,
            'answer_similarity': answer_similarity,
            'context_relevance': context_relevance,
            'answer_confidence': answer_confidence,
            'retrieved_docs': [doc.page_content for doc in retrieved_docs]
        }
        results.append(result)
        
        print(f"Answer Similarity: {answer_similarity:.3f}")
        print(f"Context Relevance: {context_relevance:.3f}")
        print(f"Answer Confidence: {answer_confidence:.3f}")
    
    return results

def print_summary_report(results):
    """Print a professional summary report"""
    print("\n" + "="*60)
    print("RAG SYSTEM EVALUATION REPORT")
    print("="*60)
    
    avg_similarity = np.mean([r['answer_similarity'] for r in results])
    avg_relevance = np.mean([r['context_relevance'] for r in results])
    avg_confidence = np.mean([r['answer_confidence'] for r in results])
    
    print(f"\nOVERALL PERFORMANCE METRICS:")
    print(f"• Average Answer Similarity: {avg_similarity:.3f}")
    print(f"• Average Context Relevance: {avg_relevance:.3f}")
    print(f"• Average Answer Confidence: {avg_confidence:.3f}")
    
    print(f"\nDETAILED BREAKDOWN:")
    for i, result in enumerate(results):
        print(f"\nQuestion {i+1}:")
        print(f"   Answer Similarity: {result['answer_similarity']:.3f}")
        print(f"   Context Relevance: {result['context_relevance']:.3f}")
        print(f"   Answer Confidence: {result['answer_confidence']:.3f}")
    
    print(f"\nPERFORMANCE ASSESSMENT:")
    if avg_similarity >= 0.8:
        print("✅ Excellent answer accuracy")
    elif avg_similarity >= 0.6:
        print("✅ Good answer accuracy")
    else:
        print("⚠️  Answer accuracy needs improvement")
    
    if avg_relevance >= 0.7:
        print("✅ Excellent context retrieval")
    elif avg_relevance >= 0.5:
        print("✅ Good context retrieval")
    else:
        print("⚠️  Context retrieval needs improvement")
    
    print("="*60)

# --- 3. Load Evaluation Dataset and Run ---
if __name__ == "__main__":
    from evaluation_data import ground_truth_dataset
    
    print("🚀 Starting RAG System Evaluation...")
    print("Loading evaluation dataset...")
    
    questions = [item["question"] for item in ground_truth_dataset]
    ground_truths = [item["ground_truth"] for item in ground_truth_dataset]
    
    print(f"Loaded {len(questions)} test questions")
    
    results = evaluate_rag_performance(questions, ground_truths)
    
    print_summary_report(results)
    
    print("\n🎯 Evaluation Complete! Use these metrics in your project documentation.")
