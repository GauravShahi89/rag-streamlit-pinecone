# Streamlit RAG Q&A with Pinecone

A Streamlit app that retrieves relevant chunks from a Pinecone vector index and generates an answer with OpenAI.

## Requirements

- Python 3.10+
- An OpenAI API key
- A Pinecone API key and an existing index named `qa-knowledge-base-1`
- The index must use the `text-embedding-3-small` embedding model (1536 dimensions)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `PINECONE_API_KEY` to your own credentials. Do not commit `.env`.
4. Start the app:

   ```powershell
   streamlit run rag_streamlit_pinecone.py
   ```

The Pinecone index must already contain the knowledge-base vectors. This repository does not include index data or API credentials.
