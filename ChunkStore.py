# chunkstore.py
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from io import BytesIO
import tempfile
import os

import faiss
import numpy as np


def load_and_chunk_pdf_from_filelike(file_like: BytesIO, chunk_size: int = 500, chunk_overlap: int = 150) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_like.read())
        tmp_path = tmp.name
    try:
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(pages)
        pure_chunks = [Document(page_content=chunk.page_content, metadata={}) for chunk in chunks]
        return pure_chunks
    finally:
        os.unlink(tmp_path)


class ChunkStore:
    def __init__(self):
        self._chunks: List[Document] = []
        self._index = None

    def load_chunks_from_filelike(self, file_like: BytesIO):
        self._chunks = load_and_chunk_pdf_from_filelike(file_like)
        return self._chunks

    def get_chunks(self) -> List[Document]:
        return self._chunks

    def create_hnsw_vectorstore_from_embeddings(self, chunks, embeddings):
        d = len(embeddings[0])
        emb_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(emb_array)

        index = faiss.IndexHNSWFlat(d, 32)  # M=32 good balance for accuracy/speed
        index.hnsw.efConstruction = 200  # High quality index build
        index.hnsw.efSearch = 128  # Query-time accuracy/speed tradeoff

        index.add(emb_array)

        faiss.omp_set_num_threads(8)  # Adjust to your CPU cores

        self._index = index
        self._chunks = chunks

        return index

    def similarity_search_with_score(self, query: str, k=3, embedding_model=None):
        if not self._index:
            raise ValueError("Index not yet created")

        query_vec = embedding_model.embed_documents([query])[0]
        query_np = np.array([query_vec]).astype('float32')
        faiss.normalize_L2(query_np)

        D, I = self._index.search(query_np, k)
        retrieved_chunks = [self._chunks[i] for i in I[0]]
        scores = D[0].tolist()

        return list(zip(retrieved_chunks, scores))
