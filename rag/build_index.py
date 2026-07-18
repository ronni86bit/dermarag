"""
DermaRAG Knowledge Base Index Builder

This script builds a FAISS vector store index from a dermatology knowledge base CSV file.
It uses sentence-transformers to generate embeddings and stores them in a FAISS index
for efficient similarity search.
"""

import os
import json
import logging
from typing import List, Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_faiss_index(
    csv_path: str,
    index_path: str,
    metadata_path: str,
    model_name: str = 'all-MiniLM-L6-v2'
) -> None:
    """
    Build FAISS index from dermatology knowledge base CSV.

    Args:
        csv_path: Path to the input CSV file with columns: condition, text, source_url
        index_path: Path where FAISS index will be saved
        metadata_path: Path where metadata JSON will be saved
        model_name: Name of the sentence-transformers model to use for embeddings
    """
    logger.info(f"Loading knowledge base from {csv_path}")

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_columns = ['condition', 'text', 'source_url']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    # Combine condition and text for richer embeddings
    # Format: "Condition: [condition]. [text]"
    documents = []
    for _, row in df.iterrows():
        doc = f"Condition: {row['condition']}. {row['text']}"
        documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents from knowledge base")

    # Load the embedding model
    logger.info(f"Loading sentence-transformers model: {model_name}")
    model = SentenceTransformer(model_name)

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    # Normalize embeddings for cosine similarity (using inner product)
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype('float32'))

    logger.info(f"Built FAISS index with {index.ntotal} vectors of dimension {dimension}")

    # Save the index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    logger.info(f"Saved FAISS index to {index_path}")

    # Save metadata (original data plus the document text used for embedding)
    metadata = []
    for idx, row in df.iterrows():
        metadata.append({
            'id': int(idx),
            'condition': row['condition'],
            'text': row['text'],
            'source_url': row['source_url'],
            'embedded_text': documents[idx]
        })

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metadata to {metadata_path}")
    logger.info("Index building completed successfully!")


if __name__ == "__main__":
    # Define paths relative to script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'dermatology_kb.csv')
    index_path = os.path.join(base_dir, 'index', 'faiss.index')
    metadata_path = os.path.join(base_dir, 'index', 'metadata.json')

    build_faiss_index(csv_path, index_path, metadata_path)