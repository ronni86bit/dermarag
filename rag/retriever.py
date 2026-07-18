"""
DermaRAG Retriever Module

This module handles loading the FAISS index and retrieving relevant documents
based on a query string. It generates queries from predicted conditions and
retrieves the top-k most relevant documents.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DermatologyRetriever:
    """
    A retriever class for the dermatology knowledge base using FAISS.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize the retriever.

        Args:
            index_path: Path to the FASS index file
            metadata_path: Path to the metadata JSON file
            model_name: Name of the sentence-transformers model for encoding queries
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Load the FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(self.metadata)} documents")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve the top-k most relevant documents for a given query.

        Args:
            query: The search query string
            k: Number of results to return (default: 5)

        Returns:
            List of dictionaries containing retrieval results with keys:
            - condition: The medical condition
            - text: The document text
            - source_url: Source URL
            - score: Similarity score (higher is better)
            - rank: Rank position (1-based)
        """
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity

        # Search the index
        scores, indices = self.index.search(
            query_embedding.astype('float32'), k
        )

        # Format results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:  # FAISS returns -1 for empty slots when index is empty
                continue

            meta = self.metadata[idx]
            results.append({
                'condition': meta['condition'],
                'text': meta['text'],
                'source_url': meta['source_url'],
                'score': float(score),
                'rank': rank
            })

        return results

    def retrieve_by_condition(self, condition: str, k: int = 5) -> List[Dict]:
        """
        Convenience method to retrieve documents by generating a query from a condition.

        Args:
            condition: The medical condition name
            k: Number of results to return

        Returns:
            List of retrieval results (same format as retrieve method)
        """
        query = f"symptoms treatment causes {condition}"
        return self.retrieve(query, k)


# For backward compatibility with the original specification
def retrieve(query: str, k: int = 5) -> List[Dict]:
    """
    Legacy function for retrieving documents.
    Initializes the retriever with default paths and performs retrieval.

    Args:
        query: The search query string
        k: Number of results to return

    Returns:
        List of retrieval result dictionaries
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, 'index', 'faiss.index')
    metadata_path = os.path.join(base_dir, 'index', 'metadata.json')

    retriever = DermatologyRetriever(index_path, metadata_path)
    return retriever.retrieve(query, k)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test with a sample query
    try:
        results = retrieve("symptoms treatment causes melanoma", k=3)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['condition']}] {result['text'][:100]}...")
            print(f"   Source: {result['source_url']}")
            print(f"   Score: {result['score']:.4f}\n")
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        print("Make sure to run build_index.py first to create the FAISS index.")