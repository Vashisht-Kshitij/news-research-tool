import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

def get_embedding_model():
    """
    Load HuggingFace embedding model.
    """
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embed_model

def build_index(documents: list):
    """
    Build a FAISS vector index from documents.
    Splits, embeds, and stores all in one step.
    """
    embed_model = get_embedding_model()
    
    # Set global settings
    Settings.embed_model = embed_model
    Settings.llm = None  # No LLM yet
    
    splitter = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter]
    )
    
    return index

def save_index(index, path: str = "index_storage"):
    """
    Save index to disk so we don't re-embed every run.
    """
    index.storage_context.persist(persist_dir=path)
    print(f"Index saved to {path}")

def load_index(path: str = "index_storage"):
    """
    Load index from disk.
    """
    from llama_index.core import load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context)
    print(f"Index loaded from {path}")
    return index

if __name__ == "__main__":
    # Test the full pipeline
    from loader import load_articles
    
    test_urls = [
        "https://www.bbc.com/news/technology-65855333"
    ]
    
    print("Loading articles...")
    documents = load_articles(test_urls)
    print(f"Loaded {len(documents)} documents")
    
    print("\nBuilding index...")
    index = build_index(documents)
    
    print("\nSaving index...")
    save_index(index)
    
    print("\nIndex built successfully")
    print(f"Index type: {type(index)}")