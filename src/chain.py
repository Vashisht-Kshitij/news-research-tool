import os
import requests
from dotenv import load_dotenv
from retriever import load_index, retrieve_chunks
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def ask_llm(question: str,context_chunks: list)->str:
    """
    Send retrived chunks as context to LLM and get an answer.
    """
    context = "\n\n".join([
        f"Source {i+1}: {chunk['text']}"
        for i,chunk in enumerate(context_chunks)
    ])

    prompt = f"""You are helpful news research assistant.
    Answer the question based on the provided context.
    If the answer is not in the context, say "I cannot find this information in the provided articles."

    Context:
    {context}

    Question: {question}

    Answer:"""

    try:
        response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization":f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json = {
                    "model":"google/gemma-4-31b-it:free",
                    "messages":[
                        {"role":"user","content": prompt}
                    ]
                },
                timeout=30
            )
        result = response.json()

        if "choices" not in result:
            error_info = result.get("error",{})
            error_code = error_info.get("code","unknown")

            if error_code == 429:
                return "The AI model is currently busy due to high demand,Please try again in a moment."
            else:
                return f"Something went wrong while generating the answer.Please try again."

        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "The request took too long.Please try again."
    except requests.exceptions.RequestException as e:
        return "Could not connect to the AI service.Please check your connection and try again."

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
