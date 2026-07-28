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
    Tries multiple models in order if earlier ones fail.
    """
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

    # List of models to try, in order of preference
    models_to_try = [
        "google/gemma-4-31b-it:free",
        "openai/gpt-oss-20b:free"
    ]
    
    last_error = None
    
    for model in models_to_try:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=30
            )
            
            result = response.json()
            
            if "choices" in result:
                return {
                            "answer": result["choices"][0]["message"]["content"],
                            "model_used": model
                        }
            else:
                # This model failed, log it and try next
                error_info = result.get("error", {})
                last_error = error_info.get("message", "Unknown error")
                continue
        
        except requests.exceptions.Timeout:
            last_error = "Request timed out"
            continue
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue
    
    # If we reach here, all models failed
    return {
    "answer": f"All AI models are currently unavailable. Please try again in a few minutes. (Last error: {last_error})",
    "model_used": None
}

def research(question: str, index) -> dict:
    """
    Full RAG pipeline: retrieve + generate.
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(question, index, top_k=3)
    
    # Step 2: Generate answer using LLM
    response = ask_llm(question, chunks)
    answer = response["answer"]
    model = response["model_used"]
    return {
        "question": question,
        "answer": answer,
        "model_used":model,
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
