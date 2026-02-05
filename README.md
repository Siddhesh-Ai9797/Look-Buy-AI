# Look-Buy-AI 🛍️🤖
Multimodal Retrieval-Augmented Generation (RAG) System for Intelligent Product Search

Look-Buy-AI is an end-to-end AI system that enables intelligent product discovery using text, image, and multimodal search. The system combines vector search (FAISS) with large language models (LLMs) to retrieve relevant products and generate context-aware responses, simulating a real-world AI-powered shopping assistant.

## Key Features
- Text-based semantic search using dense embeddings
- Image-based product retrieval using CLIP embeddings
- Multimodal fusion search (text + image)
- Retrieval-Augmented Generation (RAG) with LLMs
- Scalable vector indexing using FAISS
- Interactive UI built with Streamlit

## System Architecture
Product metadata and images are embedded and stored in FAISS indexes. User queries (text or image) are embedded at runtime, relevant products are retrieved using vector similarity search, and the retrieved context is passed to an LLM to generate grounded, context-aware responses.

## Tech Stack
- Python
- PyTorch
- SentenceTransformers
- CLIP & BLIP
- FAISS
- OpenAI API
- Streamlit

## Project Structure
Look-Buy-AI/
- app/app.py        – Streamlit application
- scripts/          – Data processing and indexing
- tools/            – Embedding and caption utilities
- data/             – Ignored (datasets, images, indexes)
- README.md
- .gitignore

## Running the App
Create a virtual environment, install dependencies, set your OpenAI API key, then run:

streamlit run app/app.py

## Notes
Large datasets, embeddings, and FAISS indexes are excluded from this repository and can be rebuilt locally using the provided scripts.

## Author
Siddhesh Patil  
MS in Artificial Intelligence, DePaul University
