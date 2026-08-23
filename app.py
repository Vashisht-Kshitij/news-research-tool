import os
os.environ["NLTK_DATA"] = "/tmp/nltk_data"
import streamlit as st 
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from loader import load_articles
from embeddings import build_index
from retriever import retrieve_chunks
from chain import ask_llm

st.set_page_config(page_title="News Research Tool", page_icon ="📰")
st.title("📰 News Research Tool")
st.write("Paste news article URLs, then ask questions about them.")

#Sidebar for URL input
st.sidebar.header("Add Articles")
num_urls = st.sidebar.number_input("Number of URLs",min_value=1, max_value=5,value=1)
urls = []
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
    if url:
        urls.append(url)

process_button = st.sidebar.button("process Articles")

#Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "index" not in st.session_state:
    st.session_state.index = None

if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []

# Process articles when button is clicked
if process_button and urls:
    with st.spinner("Processing articles..."):
        try:
            documents = load_articles(urls)
            if documents:
                index = build_index(documents)
                st.session_state.index = index
                st.session_state.processed_urls = urls
                st.sidebar.success(f" Processed {len(documents)} article(s)")
            else:
                st.sidebar.error("Could not extract content from the provided URLs")
        except Exception as e:
            st.sidebar.error(f"Error processing articles: {e}")


# Show currently processed URLs
if st.session_state.processed_urls:
    st.sidebar.write("**Processed articles:**")
    for url in st.session_state.processed_urls:
        st.sidebar.write(f"- {url}")


# Question answering
st.header("Ask a Question")

if st.session_state.index is None:
    st.info("Add and process articles from the sidebar first.")
else:
    question = st.text_input("Your question:")
    ask_button = st.button("Get Answer")

    if ask_button and question:
        with st.spinner("Searching and generating answer..."):
            chunks = retrieve_chunks(question, st.session_state.index, top_k=3)

            chat_history = st.session_state.chat_history

            response = ask_llm(question, chunks,chat_history = chat_history)

            st.session_state.chat_history.append({"question" : question, "answer" : response["answer"]})

            st.subheader("Answer")
            st.write(response["answer"])
            
            if response["model_used"]:
                st.caption(f"Answered by: {response['model_used']}")
            
            st.subheader("Sources")
            sources = set(chunk["source"] for chunk in chunks)
            for source in sources:
                st.write(f"- {source}")


                


