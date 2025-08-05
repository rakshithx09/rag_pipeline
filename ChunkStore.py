from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from io import BytesIO
import tempfile
import os

def load_and_chunk_pdf_from_filelike(file_like: BytesIO, chunk_size: int = 600, chunk_overlap: int = 75) -> List[Document]:
    # Save BytesIO to a real temporary file for PyMuPDFLoader
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
        self._vectorstore: FAISS = None

    def load_chunks_from_filelike(self, file_like: BytesIO):
        self._chunks = load_and_chunk_pdf_from_filelike(file_like)
        return self._chunks

    def get_chunks(self) -> List[Document]:
        return self._chunks

    # chunkstore.py
    def create_vectorstore_from_embeddings(self, chunks, embeddings, embedding_model):
        texts = [chunk.page_content for chunk in chunks]
        text_embeddings = list(zip(texts, embeddings))  # this is a list of (text, embedding)
        self._vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embedding_model
        )
        return self._vectorstore



    def get_vectorstore(self) -> FAISS:
        return self._vectorstore
