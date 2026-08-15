from typing import Dict, List, Tuple
from config import RELEVANCE_THRESHOLD, TOP_K
from src.vector_store import VectorStore


class RAGRetriever:
    """Retriever class responsible for querying the vector store, computing similarity scores,

    and formatting context for downstream LLM generation.

    """

    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        filter_threshold: bool = True,
    ) -> List[Dict]:
        """Retrieves relevant document chunks for a query and calculates similarity scores."""
        raw_hits = self.vector_store.search(query, k=top_k)
        if not raw_hits:
            return []

        processed_hits = []
        for hit in raw_hits:
            distance = hit.get("distance", 1.0)
            # Distance to similarity calculation
            # For normalized embeddings in Chroma, cosine distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - distance)) if distance <= 1.0 else max(0.0, 1.0 / (1.0 + distance))
            
            hit_data = {
                "document": hit["document"],
                "metadata": hit["metadata"],
                "distance": distance,
                "similarity": similarity,
            }

            if not filter_threshold or similarity >= relevance_threshold:
                processed_hits.append(hit_data)

        # Sort by highest similarity first
        processed_hits.sort(key=lambda x: x["similarity"], reverse=True)
        return processed_hits

    def format_context(self, hits: List[Dict]) -> Tuple[str, List[Dict]]:
        """Formats retrieved hits into a clean context string for LLM prompts and extracts sources list."""
        if not hits:
            return "", []

        context_blocks = []
        sources = []

        for item in hits:
            meta = item["metadata"]
            source_name = meta.get("source", "Unknown Document")
            page_info = f" | Page: {meta.get('page')}" if meta.get("page") else ""
            sim_info = f" (Relevance: {item['similarity']:.2f})"
            
            header = f"--- Source: {source_name}{page_info}{sim_info} ---"
            block = f"{header}\n{item['document']}"
            context_blocks.append(block)

            sources.append(
                {
                    "source": source_name,
                    "page": meta.get("page"),
                    "similarity": item["similarity"],
                    "snippet": item["document"][:150] + "..." if len(item["document"]) > 150 else item["document"],
                }
            )

        context_str = "\n\n".join(context_blocks)
        return context_str, sources
