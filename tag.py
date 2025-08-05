import json
from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from rapidfuzz import fuzz
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


class APIKeyManager:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.index = 0

    def get_next_key(self) -> str:
        key = self.keys[self.index]
        self.index = (self.index + 1) % len(self.keys)
        return key


class Tagger:
    def __init__(self, api_key_manager: APIKeyManager, llm_model_name="gemini-2.0-flash-lite", embedding_model_name="models/embedding-001"):
        self.api_key_manager = api_key_manager
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name

    def _create_tag_llm(self):
        key = self.api_key_manager.get_next_key()
        return ChatGoogleGenerativeAI(model=self.llm_model_name, api_key=key)

    def _create_embedding_model(self):
        key = self.api_key_manager.get_next_key()
        return GoogleGenerativeAIEmbeddings(model=self.embedding_model_name, api_key=key)

    def parse_and_clean_tags(self, tag_response_str: str) -> List[str]:
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

    def generate_tags_from_query(self, query: str) -> List[str]:
        tag_llm = self._create_tag_llm()
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

        tag_text = getattr(tag_response, "content", str(tag_response))

        tags = self.parse_and_clean_tags(tag_text)
        print(f"Raw tag response:\n{tag_text}\n")
        print(f"Cleaned tags:\n{tags}\n")
        return tags


def heuristic_chunk_filter(chunks: List[Document], query_tags: List[str], totalMatches=1, threshold=75) -> List[int]:
    filtered_indices = []
    lowercase_tags = [tag.lower() for tag in query_tags]
    for idx, chunk in enumerate(chunks):
        text_lower = chunk.page_content.lower()

        matches = 0
        for tag in lowercase_tags:
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
        gap = similarity_scores[i - 1] - similarity_scores[i]
        if gap < threshold_gap:
            k = i + 1
        else:
            break
    return min(k, max_k)


def answer_question_with_context(
    query: str,
    vectorstore,
    all_chunks: List[Document],
    filtered_indices: List[int],
    k_results: int = 3,
    embedding_model=None,
    llm_model_name="gemini-2.0-flash",
    api_key_manager: APIKeyManager = None,
) -> str:
    if not filtered_indices:
        return "No relevant chunks found to answer the query."

    filtered_chunks = [all_chunks[i] for i in filtered_indices]

    # Create a filtered vectorstore with embeddings refreshed under the correct API key
    # Rotate API key for embeddings if provided
    if api_key_manager:
        embedding_key = api_key_manager.get_next_key()
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=embedding_key)

    filtered_vectorstore = FAISS.from_documents(filtered_chunks, embedding_model)

    retrieved_docs_with_scores = filtered_vectorstore.similarity_search_with_score(query, k=k_results)
    if not retrieved_docs_with_scores:
        return "No relevant chunks retrieved."

    retrieved_docs, scores = zip(*retrieved_docs_with_scores)

    adaptive_k = adaptive_k_results(scores, threshold_gap=0.10, min_k=1, max_k=k_results)
    print(f"\nAdaptive k_results chosen: {adaptive_k}")

    final_docs = list(retrieved_docs)[:adaptive_k]

    for i, chunk in enumerate(final_docs, 1):
        preview = chunk.page_content[:200].replace('\n', ' ')
        print(f"Chunk {i} preview: {preview}...")
        print("-" * 80)

    combined_context = "\n\n".join(doc.page_content for doc in final_docs)

    # Rotate API key for QA LLM
    if api_key_manager:
        llm_api_key = api_key_manager.get_next_key()
        llm = ChatGoogleGenerativeAI(model=llm_model_name, api_key=llm_api_key)
    else:
        llm = ChatGoogleGenerativeAI(model=llm_model_name)

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

    return getattr(answer, "content", str(answer)).strip()
