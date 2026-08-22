# News Research Tool

Finding one specific fact in a long news article can be slow and frustrating, especially when researching across multiple sources. This tool lets users ask questions about an article and receive context-aware answers, eliminating the need to read the entire piece.

## How It Works

The user submits a news article URL through the Streamlit interface. The article is loaded in `loader.py`, where its content is extracted and split into smaller chunks to prepare it for embedding generation.

In `embeddings.py`, each chunk is converted into a vector embedding and stored in an in-memory vector index for efficient similarity search, which is then persisted to local disk. When a user asks a question, `retriever.py` converts the query into an embedding and retrieves the top-k most relevant chunks from that index.

Finally, the retrieved context, along with the user's question and chat history, is passed to the LLM, which generates a context-aware answer based on the content of the article.

## Features

1. Can answer questions using information pulled from multiple articles at once, not just a single URL.
2. Shows which article URL each answer was sourced from.
3. Remembers previous questions and answers, so follow-up questions work naturally.
4. Falls back automatically — tries Gemma first, then gpt-oss-20b if the first model fails to generate an answer.

## Setup

### 1. Clone the repository and set up a virtual environment

```bash
git clone https://github.com/Vashisht-Kshitij/news-research-tool.git
cd news-research-tool
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies and configure your API key

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:


### 3. Run the app

```bash
streamlit run app.py
```

## Limitations

The application relies on free-tier OpenRouter models. These models may occasionally fail to return a valid response, so a fallback model chain is used to improve reliability.

Responses from free-tier models are often brief by default and may require more specific prompts to obtain detailed explanations.

Article extraction is not perfect. Some websites use dynamic content, paywalls, or unusual HTML structures, so the extracted text may be incomplete or noisy.

Chat history is stored only for the current Streamlit session. Refreshing the page or restarting the app clears the conversation history.