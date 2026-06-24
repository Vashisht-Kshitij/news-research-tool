import os
import trafilatura
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

def load_articles(urls: list[str]) -> list:
    """
    Load and extract clean article content from URLs.
    Uses trafilatura for main content extraction.
    """
    documents = []
    
    for url in urls:
        # Download and extract main content only
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        
        if text:
            doc = Document(
                text=text,
                metadata={"source": url}
            )
            documents.append(doc)
        else:
            print(f"Warning: Could not extract content from {url}")
    
    return documents


def split_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

if __name__ == "__main__":
    test_urls = [
        "https://www.bbc.com/news/technology-65855333"
    ]
    
    documents = load_articles(test_urls)
    nodes = split_documents(documents)
    
    print(f"Loaded {len(documents)} documents")
    print(f"Split into {len(nodes)} chunks")
    print(f"\nChunk 1 preview:")
    print(nodes[0].text[:300])
    print(f"\nChunk 2 preview:")
    print(nodes[1].text[:300])