import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

def load_index(path: str = "index_storage"):
    """
    Load existing index from disk.
    """
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model
    Settings.llm = None
    
    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context)
    return index

def retrieve_chunks(query: str, index, top_k: int = 3):
    """
    Given a query, retrieve the top_k most relevant chunks.
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    
    results = []
    for i, node in enumerate(nodes):
        results.append({
            "rank": i + 1,
            "score": round(node.score, 4),
            "source": node.metadata.get("source", "unknown"),
            "text": node.text[:300]
        })
    
    return results

if __name__ == "__main__":
    print("Loading index...")
    index = load_index()
    
    # Test queries
    queries = [
        "What is artificial intelligence?",
        "What are concerns about AI?",
        "How is AI used in everyday life?"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        results = retrieve_chunks(query, index)
        
        for result in results:
            print(f"\nRank {result['rank']} | Score: {result['score']}")
            print(f"Source: {result['source']}")
            print(f"Text: {result['text']}...")