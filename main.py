import os
import tempfile
import asyncio
import threading
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ChunkStore import ChunkStore
from tag import Tagger, heuristic_chunk_filter, answer_question_with_context, APIKeyManager


app = FastAPI()

AUTH_TOKEN = "8985ca372f1cbd63dedd058f1838e3f03020d6c2f0c063235b74f7962fccfe5a"

# Your multiple API keys
load_dotenv()  # Load .env variables

api_keys_str = os.getenv("API_KEYS", "")
API_KEYS = [key.strip() for key in api_keys_str.split(",") if key.strip()]

MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Thread-safe API Key Manager instance
api_key_manager = APIKeyManager(API_KEYS)

embedding_model_name = "models/embedding-001"


class RunRequest(BaseModel):
    documents: str
    questions: List[str]


@app.post("/hackrx/run")
async def hackrx_run(request: RunRequest, authorization: Optional[str] = Header(None)):
    expected = f"Bearer {AUTH_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        resp = requests.get(request.documents)
        resp.raise_for_status()
        pdf_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp_pdf.write(pdf_bytes)
        tmp_pdf.flush()
        tmp_pdf.close()  # Important on Windows

        chunk_store = ChunkStore()
        chunk_store.load_chunks(tmp_pdf.name)
        all_chunks = chunk_store.get_chunks()

        # Initialize embedding model once with rotating key
        embedding_api_key = api_key_manager.get_next_key()
        embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model_name, api_key=embedding_api_key)
        chunk_store.load_vectorstore(embedding_model)
        vectorstore = chunk_store.get_vectorstore()

        tagger = Tagger(api_key_manager)

        async def answer_single_question(question: str) -> str:
            async with semaphore:
                loop = asyncio.get_running_loop()

                def sync_work():
                    query_tags = tagger.generate_tags_from_query(question)
                    filtered_indices = heuristic_chunk_filter(all_chunks, query_tags, totalMatches=2, threshold=75)
                    if not filtered_indices:
                        filtered_indices = heuristic_chunk_filter(all_chunks, query_tags, totalMatches=2, threshold=90)
                    if not filtered_indices:
                        filtered_indices = heuristic_chunk_filter(all_chunks, query_tags, totalMatches=1, threshold=100)

                    answer = answer_question_with_context(
                        question,
                        vectorstore,
                        all_chunks,
                        filtered_indices,
                        embedding_model=embedding_model,
                        api_key_manager=api_key_manager,
                    )
                    return answer

                return await loop.run_in_executor(None, sync_work)

        tasks = [answer_single_question(q) for q in request.questions]
        answers_list = await asyncio.gather(*tasks)

        return {"answers": answers_list}
    finally:
        os.remove(tmp_pdf.name)
