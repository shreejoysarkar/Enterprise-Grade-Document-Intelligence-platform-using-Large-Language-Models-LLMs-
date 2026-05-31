"""
Retrieval & Generation Module (Phase 4)

This module handles RAG (Retrieval-Augmented Generation) queries.
It performs the following steps:
1. Retrieves top N chunks using Hybrid Search (Dense + Sparse).
2. Re-ranks the chunks using a Cross-Encoder to find the top K most relevant.
3. Prompts a local LLM (Ollama) with the reranked context and the user query.
4. Streams the response.
"""

import sys
from typing import Any

import ollama
from sentence_transformers import CrossEncoder

from core.embedding_and_indexing_3 import HybridSearchIndexer
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self, reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the RAG Pipeline with a Re-ranker and Ollama."""
        self.settings = get_settings()
        self.llm_model = self.settings.llm_model
        
        logger.info("Initializing Hybrid Search Indexer...")
        self.indexer = HybridSearchIndexer()
        
        logger.info(f"Loading CrossEncoder re-ranker: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name)
        logger.info("Re-ranker loaded successfully.")

    def retrieve_and_rerank(self, query: str, retrieve_top_n: int = 20, keep_top_k: int = 4) -> list[dict[str, Any]]:
        """Retrieve chunks via Hybrid Search and re-rank them."""
        logger.info(f"Retrieving top {retrieve_top_n} candidates from Pinecone...")
        
        # 1. Base Retrieval
        raw_results = self.indexer.hybrid_search(
            query=query, 
            top_k=retrieve_top_n, 
            alpha=0.7  # 70% Semantic, 30% Keyword
        )
        
        if not raw_results:
            logger.warning("No results found in index.")
            return []
            
        # 2. Re-ranking
        logger.info("Re-ranking candidates...")
        # Prepare pairs for cross-encoder: (query, document_text)
        pairs = [[query, match["metadata"].get("text", "")] for match in raw_results]
        
        # Get scores
        scores = self.reranker.predict(pairs)
        
        # Combine scores with results
        for match, score in zip(raw_results, scores):
            match["rerank_score"] = float(score)
            
        # Sort by rerank score descending
        reranked_results = sorted(raw_results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Keep top K
        final_results = reranked_results[:keep_top_k]
        logger.info(f"Retained top {keep_top_k} results after re-ranking.")
        
        return final_results
        
    def build_prompt(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        """Format the retrieved chunks into a prompt for the LLM."""
        context_str = ""
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk["metadata"].get("source_file", "Unknown")
            text = chunk["metadata"].get("text", "")
            context_str += f"--- Document {i} (Source: {source}) ---\n{text}\n\n"
            
        prompt = (
            "You are an expert AI assistant for an enterprise document intelligence platform.\n"
            "Synthesize a clear, concise, and accurate answer to the user's question based strictly on the provided context below.\n"
            "Keep your response to the point, avoiding unnecessary repetition or overly long explanations.\n"
            "If the answer cannot be found in the context, clearly state that you do not know.\n"
            "Use inline citations to reference your sources (e.g., 'Apple faces data protection risks [aapl-20230930.md]').\n\n"
            "CONTEXT DOCUMENTS:\n"
            f"{context_str}\n"
            "USER QUESTION:\n"
            f"{query}\n\n"
            "ANSWER:\n"
        )
        return prompt

    def generate(self, query: str, stream: bool = True):
        """Run the full RAG pipeline."""
        # 1. Retrieve & Re-rank
        contexts = self.retrieve_and_rerank(
            query, 
            retrieve_top_n=20, 
            keep_top_k=self.settings.retrieval_k
        )
        
        if not contexts:
            return "I couldn't find any relevant documents to answer your question."
            
        # 2. Build Prompt
        prompt = self.build_prompt(query, contexts)
        
        # 3. Call Ollama
        logger.info(f"Generating answer using local Ollama model: {self.llm_model}...")
        
        messages = [
            {"role": "system", "content": "You are a helpful, precise, and concise document assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Options to speed up generation and prevent repeating loops
        options = {
            "temperature": self.settings.llm_temperature if self.settings.llm_temperature > 0 else 0.1,
            "num_predict": 512,  # Limit the max generation length to reduce time
            "repeat_penalty": 1.15  # Help prevent the model from looping/stuttering
        }
        
        try:
            response = ollama.chat(
                model=self.llm_model, 
                messages=messages, 
                stream=stream,
                options=options
            )
            
            if stream:
                print(f"\n[{self.llm_model}] Answer:\n", end="")
                for chunk in response:
                    print(chunk['message']['content'], end="", flush=True)
                print("\n")
                return None
            else:
                return response['message']['content']
                
        except Exception as e:
            logger.error(f"Failed to generate response with Ollama: {e}")
            return f"Error: Could not connect to Ollama. Make sure Ollama is running and the model '{self.llm_model}' is pulled."


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Summarize the key points of the available documents."
        
    pipeline = RAGPipeline()
    pipeline.generate(query)


if __name__ == "__main__":
    main()
