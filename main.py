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
from tag import APIKeyManager, Tagger, heuristic_chunk_filter

load_dotenv()

api_keys_str = os.getenv("API_KEYS", "")
API_KEYS = [key.strip() for key in api_keys_str.split(",") if key.strip()]
if not API_KEYS:
    raise ValueError("No API_KEYS found in environment variables")

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "default_token_if_any")

app = FastAPI()

MAX_CONCURRENT_REQUESTS = 40  # Tune appropriately

api_key_manager = APIKeyManager(API_KEYS)
embedding_model_name = "models/embedding-001"
llm_model_name = "gemini-2.0-flash"


class RunRequest(BaseModel):
    documents: str
    questions: List[str]


async def batch_embed_chunks(embedding_model, all_chunks, batch_size=32):
    texts = [chunk.page_content for chunk in all_chunks]
    embeddings = []
    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        task = asyncio.to_thread(embedding_model.embed_documents, batch)
        tasks.append(task)
    embeddings_batches = await asyncio.gather(*tasks)
    for batch_embeds in embeddings_batches:
        embeddings.extend(batch_embeds)
    return embeddings


@app.post("/hackrx/run")
async def hackrx_run(request: RunRequest, authorization: Optional[str] = Header(None)):
    expected = f"Bearer {AUTH_TOKEN}"
    print("Request received!")

    try:
        async with httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(max_connections=MAX_CONCURRENT_REQUESTS, max_keepalive_connections=MAX_CONCURRENT_REQUESTS // 2),
            timeout=30,
        ) as client:
            resp = await client.get(request.documents)
            resp.raise_for_status()
            pdf_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

    pdf_file = BytesIO(pdf_bytes)

    chunk_store = ChunkStore()
    all_chunks = await asyncio.to_thread(chunk_store.load_chunks_from_filelike, pdf_file)
    print(f"Total chunks created: {len(all_chunks)}")
    embedding_api_key = api_key_manager.get_next_key()
    embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model_name, api_key=embedding_api_key)

    embeddings = await batch_embed_chunks(embedding_model, all_chunks, batch_size=64)

    tagger = Tagger(api_key_manager)

    async def generate_tags_async(query: str) -> List[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, tagger.generate_tags_from_query, query)

    tags_list = await asyncio.gather(*[generate_tags_async(q) for q in request.questions])

    for idx, (question, tags) in enumerate(zip(request.questions, tags_list), 1):
        print(f"Question {idx}: {question}")
        print(f"Generated Tags: {tags}")

    filtered_contexts = []
    TOP_K = 6
    for idx, (question, tags) in enumerate(zip(request.questions, tags_list), 1):
        filtered_indices = heuristic_chunk_filter(all_chunks, tags, totalMatches=0, threshold=50)
        print(f"Question {idx}: Number of chunks filtered by heuristic filter: {len(filtered_indices)}")
        if not filtered_indices:
            print(f"Question {idx}: No chunks matched heuristic filter")
            filtered_contexts.append("")
            continue

        filtered_chunks = [all_chunks[i] for i in filtered_indices]
        filtered_embeddings = [embeddings[i] for i in filtered_indices]

        # Use CPU-optimized FAISS HNSW vectorstore creation
        temp_vectorstore = chunk_store.create_hnsw_vectorstore_from_embeddings(filtered_chunks, filtered_embeddings)

        # Similarity search top k on temp vectorstore via chunk_store method
        docs_scores = chunk_store.similarity_search_with_score(question, k=TOP_K, embedding_model=embedding_model)
        if not docs_scores:
            print(f"Question {idx}: No top-k chunks found after filtering and similarity search.")
            filtered_contexts.append("")
            continue

        docs = [doc for doc, _ in docs_scores]

        print(f"Question {idx}: Top {len(docs)} chunks selected after similarity search:")
        for c_idx, doc in enumerate(docs, 1):
            preview = doc.page_content
            print(f"  Chunk {c_idx}: {preview}...")

        combined_context = "\n\n".join(doc.page_content for doc in docs)
        filtered_contexts.append(combined_context)

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

    question_inputs = [
        {"context": context, "question": question} for context, question in zip(filtered_contexts, request.questions)
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

    return {"answers": answers_list}
