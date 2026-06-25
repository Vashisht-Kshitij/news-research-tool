import os
import requests
from dotenv import load_dotenv
from retriever import load_index, retrieve_chunks
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def ask_llm(question: str, context_chunks: list) -> str:
    """
    Send retrieved chunks as context to LLM and get an answer.
    """
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"Source {i+1}: {chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    prompt = f"""You are a helpful news research assistant.
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I cannot find this information in the provided articles."

Context:
{context}

Question: {question}

Answer:"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "google/gemma-4-31b-it:free",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

def research(question: str, index) -> dict:
    """
    Full RAG pipeline: retrieve + generate.
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(question, index, top_k=3)
    
    # Step 2: Generate answer using LLM
    answer = ask_llm(question, chunks)
    
    return {
        "question": question,
        "answer": answer,
        "sources": [chunk["source"] for chunk in chunks],
        "chunks_used": len(chunks)
    }

if __name__ == "__main__":
    print("Loading index...")
    index = load_index()
    
    questions = [
        "What is artificial intelligence and how does it work?",
        "What concerns do artists have about AI?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        result = research(question, index)
        
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {set(result['sources'])}")
        print(f"Chunks used: {result['chunks_used']}")