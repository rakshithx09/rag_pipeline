import json
from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from rapidfuzz import fuzz
from langchain_google_genai import ChatGoogleGenerativeAI

class APIKeyManager:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.index = 0

    def get_next_key(self) -> str:
        key = self.keys[self.index]
        self.index = (self.index + 1) % len(self.keys)
        return key

class Tagger:
    def __init__(self, api_key_manager: APIKeyManager, llm_model_name="gemini-2.0-flash-lite"):
        self.api_key_manager = api_key_manager
        self.llm_model_name = llm_model_name

    def _create_tag_llm(self):
        key = self.api_key_manager.get_next_key()
        return ChatGoogleGenerativeAI(model=self.llm_model_name, api_key=key)

    def parse_and_clean_tags(self, tag_response_str: str) -> List[str]:
        try:
            tags = json.loads(tag_response_str)
            return [t.strip('[]"\' ).').lower() for t in tags if isinstance(t, str)]
        except json.JSONDecodeError:
            raw = tag_response_str.replace('```',"")
            return [p.strip('[]"\' ).').lower() for p in raw.split(',') if p.strip()]

    def generate_tags_from_query(self, query: str) -> List[str]:
        tag_llm = self._create_tag_llm()
        tag_generation_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You will receive a user query. Generate a list of exactly 15 single-word tags or keywords that best represent the key topics in the query.
...
User query:
{query}
"""
        )
        tag_chain = RunnableSequence(tag_generation_prompt, tag_llm)
        tag_response = tag_chain.invoke({"query": query})
        tag_text = getattr(tag_response, "content", str(tag_response))
        return self.parse_and_clean_tags(tag_text)

def heuristic_chunk_filter(chunks: List[Document], query_tags: List[str], totalMatches=1, threshold=75) -> List[int]:
    filtered_indices = []
    lowercase_tags = [tag.lower() for tag in query_tags]
    for idx, chunk in enumerate(chunks):
        text_lower = chunk.page_content.lower()
        matches = sum(fuzz.partial_ratio(tag, text_lower) >= threshold for tag in lowercase_tags)
        if matches >= totalMatches:
            filtered_indices.append(idx)
    return filtered_indices

def adaptive_k_results(similarity_scores, threshold_gap=0.15, min_k=1, max_k=3):
    n = min(max_k, len(similarity_scores))
    if n <= min_k:
        return n
    k = min_k
    for i in range(1, n):
        if similarity_scores[i - 1] - similarity_scores[i] < threshold_gap:
            k = i + 1
        else:
            break
    return min(k, max_k)

def answer_question_with_context(query: str, vectorstore, all_chunks: List[Document], k_results: int = 3, embedding_model=None, llm=None) -> str:
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(query, k=k_results)
    if not retrieved_docs_with_scores:
        return "No relevant chunks retrieved."

    retrieved_docs, scores = zip(*retrieved_docs_with_scores)
     #adaptive_k = adaptive_k_results(scores, threshold_gap=0.10, min_k=1, max_k=k_results)
    final_docs = list(retrieved_docs)
    combined_context = "\n\n".join(doc.page_content for doc in final_docs)

    if llm is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

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
"""
    )
    qa_chain = RunnableSequence(qa_prompt, llm)
    answer = qa_chain.invoke({"context": combined_context, "question": query})
    return getattr(answer, "content", str(answer)).strip()
