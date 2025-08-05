import os
import asyncio
import hashlib
from io import BytesIO
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel
from langchain.prompts import PromptTemplate

from ChunkStore import ChunkStore
from tag import APIKeyManager

load_dotenv()

api_keys_str = os.getenv("API_KEYS", "")
API_KEYS = [key.strip() for key in api_keys_str.split(",") if key.strip()]
if not API_KEYS:
    raise ValueError("No API_KEYS found in environment variables")

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "default_token_if_any")

app = FastAPI()

MAX_CONCURRENT_REQUESTS = 40  # Tune to your API limits

api_key_manager = APIKeyManager(API_KEYS)
embedding_model_name = "models/embedding-001"
llm_model_name = "gemini-2.0-flash"


class RunRequest(BaseModel):
    documents: str
    questions: List[str]


async def batch_embed_chunks(embedding_model, all_chunks, batch_size=32):
    texts = [chunk.page_content for chunk in all_chunks]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeds = await asyncio.to_thread(embedding_model.embed_documents, batch)
        embeddings.extend(batch_embeds)
    return embeddings


@app.post("/hackrx/run")
async def hackrx_run(request: RunRequest, authorization: Optional[str] = Header(None)):
    expected = f"Bearer {AUTH_TOKEN}"
    print("Request recieved!")


    try:
        async with httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            timeout=30,
        ) as client:
            resp = await client.get(request.documents)
            resp.raise_for_status()
            pdf_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

    doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
    pdf_file = BytesIO(pdf_bytes)

    chunk_store = ChunkStore()
    all_chunks = await asyncio.to_thread(chunk_store.load_chunks_from_filelike, pdf_file)

    embedding_api_key = api_key_manager.get_next_key()
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name, api_key=embedding_api_key
    )

    embeddings = await batch_embed_chunks(embedding_model, all_chunks, batch_size=128)

    vectorstore = await asyncio.to_thread(
        chunk_store.create_vectorstore_from_embeddings,
        all_chunks,
        embeddings,
        embedding_model,
    )

    llm_api_key = api_key_manager.get_next_key()
    llm_model = ChatGoogleGenerativeAI(model=llm_model_name, api_key=llm_api_key)

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Carefully answer the question using ONLY the context provided below.
...
Context:
{context}

Question:
{question}

Answer:
""",
    )

    def get_contexts_for_all():
        results = []
        for question in request.questions:
            docs_scores = vectorstore.similarity_search_with_score(question, k=3)
            if not docs_scores:
                results.append("")
            else:
                docs = [doc for doc, _ in docs_scores]
                context = "\n\n".join([doc.page_content for doc in docs])
                results.append(context)
        return results

    contexts = await asyncio.to_thread(get_contexts_for_all)

    question_inputs = [
        {"context": context, "question": question}
        for context, question in zip(contexts, request.questions)
    ]

    qa_chain = qa_prompt | llm_model
    parallel_chain = RunnableParallel({"answer": qa_chain})

    results = parallel_chain.batch(question_inputs)

    answers_list = []
    for r in results:
        answer = r.get("answer", "")
        if hasattr(answer, "content"):
            answer_text = answer.content.strip()
        elif isinstance(answer, str):
            answer_text = answer.strip()
        else:
            answer_text = str(answer).strip()
        answers_list.append(answer_text)

    # Return a plain JSON array of answer strings as required
    return {"answers": answers_list}
