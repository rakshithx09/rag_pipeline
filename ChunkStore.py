import os
import pickle
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class ChunkStore:
    _instance = None
    _chunks: List[Document] = []
    _vectorstore: FAISS = None
    _chunks_cache_file = "chunks_cache.pkl"
    _vectorstore_cache_dir = "vectorstore_store"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_chunks(self, pdf_path: str):
        if not self._chunks:
            if os.path.exists(self._chunks_cache_file):
                print(f"Loading chunks from cache: {self._chunks_cache_file}")
                with open(self._chunks_cache_file, "rb") as f:
                    self._chunks = pickle.load(f)
                print(f"Loaded {len(self._chunks)} chunks.")
            else:
                print("No cache found. Performing semantic chunking…")
                self._chunks = load_and_semantic_chunk_pdf(pdf_path)
                with open(self._chunks_cache_file, "wb") as f:
                    pickle.dump(self._chunks, f)
                print(f"Chunked into {len(self._chunks)}. Saved to cache.")

    def get_chunks(self) -> List[Document]:
        return self._chunks

    def load_vectorstore(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        if self._vectorstore is None:
            if os.path.exists(self._vectorstore_cache_dir):
                print(f"Loading vectorstore: {self._vectorstore_cache_dir}")
                self._vectorstore = FAISS.load_local(
                    self._vectorstore_cache_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                print("No vectorstore cache found. Embedding all chunks with Gemini…")
                self._vectorstore = FAISS.from_documents(self._chunks, embeddings)
                self._vectorstore.save_local(self._vectorstore_cache_dir)
                print(f"Saved vectorstore to {self._vectorstore_cache_dir}")

    def get_vectorstore(self) -> FAISS:
        return self._vectorstore

def load_and_semantic_chunk_pdf(pdf_path: str) -> List[Document]:
    # Use nltk for sentence splitting
    from nltk.tokenize import sent_tokenize
    import nltk
    nltk.download('punkt', quiet=True)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = " ".join([page.page_content for page in pages])
    sentences = sent_tokenize(text)

    # Set up Gemini embeddings client
    embeddings_client = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    chunk_texts = []
    current_chunk = []
    current_chunk_text = ""
    prev_embedding = None
    similarity_threshold = 0.75  # Tune as needed

    import numpy as np
    def get_embedding(text: str):
        return embeddings_client.embed_query(text)
    def cosine_sim(vec1, vec2):
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    for sentence in sentences:
        this_sentence_embedding = get_embedding(sentence)
        if prev_embedding is None:
            current_chunk = [sentence]
            current_chunk_text = sentence
            prev_embedding = this_sentence_embedding
        else:
            sim = cosine_sim(this_sentence_embedding, prev_embedding)
            if sim >= similarity_threshold:
                current_chunk.append(sentence)
                current_chunk_text += " " + sentence
                # Always re-embed full chunk for accuracy
                prev_embedding = get_embedding(current_chunk_text)
            else:
                chunk_texts.append(current_chunk_text)
                current_chunk = [sentence]
                current_chunk_text = sentence
                prev_embedding = this_sentence_embedding
    if current_chunk:
        chunk_texts.append(current_chunk_text)
    return [Document(page_content=ctxt) for ctxt in chunk_texts]
