#!/usr/bin/env python3
"""
Verification script for DermaRAG project structure
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.isfile(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (MISSING)")
        return False

def check_dir_exists(dirpath, description):
    """Check if a directory exists and report status."""
    if os.path.isdir(dirpath):
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: {dirpath} (MISSING)")
        return False

def main():
    print("Verifying DermaRAG project structure...")
    print("=" * 50)

    base_dir = Path(__file__).parent
    all_good = True

    # Check main files
    all_good &= check_file_exists(base_dir / "app.py", "Main Streamlit app")
    all_good &= check_file_exists(base_dir / "README.md", "README documentation")
    all_good &= check_file_exists(base_dir / "requirements.txt", "Python requirements")
    all_good &= check_file_exists(base_dir / "Makefile", "Build automation")
    all_good &= check_file_exists(base_dir / ".env.example", "Environment variable template")

    # Check model directory
    model_dir = base_dir / "model"
    all_good &= check_dir_exists(model_dir, "Model directory")
    all_good &= check_file_exists(model_dir / "inference.py", "Model inference code")

    # Check rag directory
    rag_dir = base_dir / "rag"
    all_good &= check_dir_exists(rag_dir, "RAG directory")
    all_good &= check_file_exists(rag_dir / "build_index.py", "Index building script")
    all_good &= check_file_exists(rag_dir / "retriever.py", "Retrieval logic")
    all_good &= check_file_exists(rag_dir / "generator.py", "Generation logic")

    # Check data directory
    data_dir = base_dir / "data"
    all_good &= check_dir_exists(data_dir, "Data directory")
    all_good &= check_file_exists(data_dir / "dermatology_kb.csv", "Knowledge base CSV")

    # Check index directory
    index_dir = base_dir / "index"
    all_good &= check_dir_exists(index_dir, "Index directory")
    all_good &= check_file_exists(index_dir / ".gitkeep", "Git placeholder for index")

    print("=" * 50)
    if all_good:
        print("✓ All checks passed! Project structure is complete.")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())