import os
import pickle
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document




class ChunkStore:
    _instance = None
    _chunks: List[Document] = []
    _vectorstore: FAISS = None
    _chunks_cache_file = "chunks_cache.pkl"
    _vectorstore_cache_dir = "vectorstore_store"   # Use a directory, not a pkl file!


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChunkStore, cls).__new__(cls)
        return cls._instance


    def load_chunks(self, pdf_path: str):
        if not self._chunks:
            if os.path.exists(self._chunks_cache_file):
                print(f"Loading chunks from cache file: {self._chunks_cache_file}")
                with open(self._chunks_cache_file, "rb") as f:
                    self._chunks = pickle.load(f)
                print(f"Loaded {len(self._chunks)} chunks from cache.")
            else:
                print("No chunks cache found. Loading and chunking PDF...")
                self._chunks = load_and_chunk_pdf(pdf_path)
                print(f"Chunked into {len(self._chunks)} pieces.")
                with open(self._chunks_cache_file, "wb") as f:
                    pickle.dump(self._chunks, f)
                print(f"Saved chunks cache to {self._chunks_cache_file}")


    def get_chunks(self) -> List[Document]:
        return self._chunks


    def load_vectorstore(self, embedding_model):
        if self._vectorstore is None:
            if os.path.exists(self._vectorstore_cache_dir):
                print(f"Loading vectorstore from cache folder: {self._vectorstore_cache_dir}")
                self._vectorstore = FAISS.load_local(
                    self._vectorstore_cache_dir,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                print("Vectorstore loaded from cache.")
            else:
                print("No vectorstore cache found. Embedding all chunks...")
                self._vectorstore = FAISS.from_documents(self._chunks, embedding_model)
                print("Embedding complete. Saving vectorstore cache...")
                self._vectorstore.save_local(self._vectorstore_cache_dir)
                print(f"Saved vectorstore cache to {self._vectorstore_cache_dir}")



    def get_vectorstore(self) -> FAISS:
        return self._vectorstore



def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=75)
    chunks = splitter.split_documents(pages)
    # Remove metadata (keep only pure text chunks)
    pure_chunks = [Document(page_content=chunk.page_content, metadata={}) for chunk in chunks]
    return pure_chunks