import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import json
from rapidfuzz import fuzz

from ChunkStore import ChunkStore



load_dotenv()

llm_model_name = "gemini-2.0-flash"
embedding_model_name = "models/embedding-001"
tag_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # for final QA

embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model_name)

# --- Helper: Parse and clean tags robustly ---
def parse_and_clean_tags(tag_response_str: str) -> List[str]:
    try:
        tags = json.loads(tag_response_str)
        cleaned_tags = []
        for tag in tags:
            if isinstance(tag, str):
                t = tag.lower().strip()
                t = t.strip('[]"\'')

                cleaned_tags.append(t)
        return cleaned_tags
    except Exception:
        raw = tag_response_str.replace('```',"")
        raw = raw.strip('[] \n\r')
        parts = raw.split(',')
        cleaned = []
        for p in parts:
            t = p.strip().strip('[]"\'').lower()
            if t:
                cleaned.append(t)
        return cleaned

def generate_tags_from_query(query: str) -> List[str]:
    tag_generation_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
You will receive a user query. Generate a list of exactly 15 single-word tags or keywords that best represent the key topics in the query.

Five of these tags must be unique and highly relevant to the query itself.
The remaining Ten tags should be synonyms, closely related terms, or common text variations that expand on those unique keywords to provide broader coverage and improve approximate matching.

When a key concept in the query is represented as a compound word or concatenated form (e.g., 'roomrent', 'sublimits', 'copayment'), split it into its meaningful component words (e.g., 'room', 'rent', 'sub', 'limit', 'co', 'payment') if that makes each sub-concept independently relevant for retrieval or matching.

Include common alternative spellings, plurals, and related morphological forms of your tags (e.g., singular and plural forms, verb/noun forms).

Consider terms or keywords that might appear with minor typos or variations in the documents so they can be matched fuzzily.

Return ONLY a JSON array of lowercase tags/keywords, with NO markdown, no extra explanations, and NO quotes inside the strings.

Example: [generate, keywords, tags, produce, create, make, terms, labels, identifiers, relevant, related, unique, distinct, original, exclusive]

User query:
{query}

"""
    )

    tag_chain = RunnableSequence(tag_generation_prompt, tag_llm)
    tag_response = tag_chain.invoke({"query": query})

    # Extract content string from AIMessage object or use str
    if hasattr(tag_response, "content"):
        tag_text = tag_response.content
    else:
        tag_text = str(tag_response)

    tags = parse_and_clean_tags(tag_text)
    print(f"Raw tag response:\n{tag_text}\n")
    print(f"Cleaned tags:\n{tags}\n")
    return tags

def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    # Remove metadata (keep only pure text chunks)
    pure_chunks = [Document(page_content=chunk.page_content, metadata={}) for chunk in chunks]
    return pure_chunks


def heuristic_chunk_filter(chunks: List[Document], query_tags: List[str], totalMatches=1, threshold=75):
    """
    Filter chunks where at least `totalMatches` tags match fuzzily with score >= threshold.
    """
    filtered_indices = []
    lowercase_tags = [tag.lower() for tag in query_tags]

    for idx, chunk in enumerate(chunks):
        text_lower = chunk.page_content.lower()

        # Count number of tags that fuzzily match chunk with score >= threshold
        matches = 0
        for tag in lowercase_tags:
            # Compute fuzzy partial ratio to handle substrings, typos, spacing
            score = fuzz.partial_ratio(tag, text_lower)
            if score >= threshold:
                matches += 1
            if matches >= totalMatches:
                break

        if matches >= totalMatches:
            filtered_indices.append(idx)

    return filtered_indices


def adaptive_k_results(similarity_scores, threshold_gap=0.15, min_k=1, max_k=3):
    n = min(max_k, len(similarity_scores))
    if n <= min_k:
        return n

    k = min_k
    for i in range(1, n):
        gap = similarity_scores[i-1] - similarity_scores[i]
        if gap < threshold_gap:
            k = i + 1
        else:
            break
    return min(k, max_k)

def answer_question_with_context(
    query: str,
    vectorstore: FAISS,
    all_chunks: List[Document],
    filtered_indices: List[int],
    k_results: int = 3,
    embedding_model=None
) -> str:
    if not filtered_indices:
        return "No relevant chunks found to answer the query."

    filtered_chunks = [all_chunks[i] for i in filtered_indices]
    filtered_vectorstore = FAISS.from_documents(all_chunks, embedding_model)

    # Use similarity_search_with_score instead of raw _faiss_index.search()
    retrieved_docs_with_scores = filtered_vectorstore.similarity_search_with_score(query, k=k_results)

    if not retrieved_docs_with_scores:
        return "No relevant chunks retrieved."

    # unpack docs and scores
    retrieved_docs, scores = zip(*retrieved_docs_with_scores)

    # adaptive k based on scores (assuming scores are similarity scores, not distances)
    adaptive_k = adaptive_k_results(scores, threshold_gap=0.10, min_k=1, max_k=k_results)
    print(f"\nAdaptive k_results chosen: {adaptive_k}")

    final_docs = list(retrieved_docs)[:adaptive_k]

    for i, chunk in enumerate(final_docs, 1):
        preview = chunk.page_content[:200].replace('\n', ' ')
        print(f"Chunk {i} preview: {preview}...")
        print("-" * 80)

    combined_context = "\n\n".join(doc.page_content for doc in final_docs)

    # Your QA prompt chain logic as before...

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert insurance policy assistant. Carefully answer the question using ONLY the context provided below.

- If the answer is directly stated or clearly implied by the context, provide a detailed and precise answer based on that information.
- If the context does not explicitly mention the answer or relevant information, respond with "No" without attempting to infer or guess.
- When multiple relevant points exist (e.g., conditions, exceptions, or steps), include them all in your answer.
- Use clear and professional language appropriate for insurance policy explanations.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    qa_chain = RunnableSequence(qa_prompt, llm)
    answer = qa_chain.invoke({"context": combined_context, "question": query})

    if hasattr(answer, "content"):
        return answer.content.strip()
    return str(answer).strip()


# --- Main pipeline flow ---
def main_pipeline(pdf_file: str, user_query: str, all_chunks: List[Document], vectorstore: FAISS,embedding_model):
    print("Generating tags from query...")
    query_tags = generate_tags_from_query(user_query)
    print(f"Generated tags (with synonyms): {query_tags}")

    print("Filtering chunks heuristically based on query tags with at least 3 matches...")
    filtered_indices = heuristic_chunk_filter(all_chunks, query_tags, totalMatches=2, threshold=75)
    print(f"Chunks kept after filtering (3 matches required): {len(filtered_indices)}")



    if not filtered_indices:
        print("No chunks found with 3 matches, trying with 2 matches...")
        filtered_indices = heuristic_chunk_filter(all_chunks, query_tags, totalMatches=2, threshold=90)
        print(f"Chunks kept after filtering (2 matches required): {len(filtered_indices)}")



    if not filtered_indices:
        print("No chunks found with 2 matches, trying with 1 match...")
        filtered_indices = heuristic_chunk_filter(all_chunks, query_tags, totalMatches=1, threshold=100)
        print(f"Chunks kept after filtering (1 match required): {len(filtered_indices)}")

    print("Answering question based on filtered chunks...")
    answer = answer_question_with_context(user_query, vectorstore, all_chunks, filtered_indices,embedding_model=embedding_model)
    print("\n=== FINAL ANSWER ===")
    print(answer)



if __name__ == "__main__":
    PDF_PATH = "b.pdf"  # Your PDF path

    # Singleton instance
    chunk_store = ChunkStore()

    # Load or retrieve chunks
    chunk_store.load_chunks(PDF_PATH )
    all_chunks = chunk_store.get_chunks()

    # Load or create cached vectorstore using your embedding model
    chunk_store.load_vectorstore(embedding_model)
    vectorstore = chunk_store.get_vectorstore()

    USER_QUERY = "How does the policy define a 'Hospital'? what is the criteria"

    # Pass vectorstore and chunks to your pipeline - your main_pipeline will need to accept vectorstore:
    main_pipeline(PDF_PATH, USER_QUERY, all_chunks, vectorstore,embedding_model)










