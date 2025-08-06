import json
from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from rapidfuzz import fuzz
from langchain_google_genai import ChatGoogleGenerativeAI
import re


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
            return [t.strip() for t in tags if isinstance(t, str)]
        except json.JSONDecodeError:
            cleaned = tag_response_str.strip().strip('`').replace('\n', '')
            bracketed = re.search(r'\[.*\]', cleaned)
            if bracketed:
                content = bracketed.group(0)
                content = content.replace("'", '"')
                try:
                    tags = json.loads(content)
                    return [t.strip() for t in tags if isinstance(t, str)]
                except json.JSONDecodeError:
                    splitted = content.strip('[]').split(',')
                    return [re.sub(r'[\'\"\s]', '', s) for s in splitted if s.strip()]
            else:
                splitted = cleaned.split(',')
                return [re.sub(r'[\'\"\s]', '', s) for s in splitted if s.strip()]

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
