import os
import tempfile
import asyncio
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ChunkStore import ChunkStore
from tag import answer_question_with_context, generate_tags_from_query, heuristic_chunk_filter

# Initialize embedding model globally (only once per process)
embedding_model_name = "models/embedding-001"
embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model_name)

app = FastAPI()

AUTH_TOKEN = "8985ca372f1cbd63dedd058f1838e3f03020d6c2f0c063235b74f7962fccfe5a"

# Semaphore to limit concurrency of GPU/LLM API calls
MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(
    request: RunRequest,
    authorization: Optional[str] = Header(None)
):
    expected = f"Bearer {AUTH_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Download the PDF document from the URL
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
        tmp_pdf.close()  # Important on Windows for other libs to access the file

        # Initialize ChunkStore and load chunks from the PDF
        chunk_store = ChunkStore()
        chunk_store.load_chunks(tmp_pdf.name)
        all_chunks = chunk_store.get_chunks()

        # Load or create vectorstore (embedding index)
        chunk_store.load_vectorstore(embedding_model)
        vectorstore = chunk_store.get_vectorstore()

        # Async function with semaphore to limit concurrent calls
        async def answer_single_question(question: str) -> str:
            async with semaphore:  # Limit concurrency here
                loop = asyncio.get_running_loop()

                def sync_work():
                    query_tags = generate_tags_from_query(question)
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
                        embedding_model=embedding_model
                    )
                    return answer

                return await loop.run_in_executor(None, sync_work)

        # Process all questions concurrently but limited by semaphore
        tasks = [answer_single_question(q) for q in request.questions]
        answers_list = await asyncio.gather(*tasks)

        # Map questions to their respective answers
        answers = dict(zip(request.questions, answers_list))

    finally:
        os.remove(tmp_pdf.name)

    return {
        "status": "success",
        "answers": answers
    }
