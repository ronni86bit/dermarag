#!/usr/bin/env python3
"""
Basic syntax check for DermaRAG modules
"""

import ast
import sys
import traceback

def check_syntax(filepath):
    """Check Python syntax for a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        print(f"✓ Syntax OK: {filepath}")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax Error in {filepath}:{e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return False

def main():
    print("Checking syntax of Python files...")
    print("=" * 50)

    base_dir = "/c/Ronni/Projects/RAG DermaAI/dermarag"
    files_to_check = [
        "app.py",
        "model/inference.py",
        "rag/build_index.py",
        "rag/retriever.py",
        "rag/generator.py",
        "verify_structure.py"
    ]

    all_good = True
    for file_path in files_to_check:
        full_path = f"{base_dir}/{file_path}"
        if not check_syntax(full_path):
            all_good = False

    print("=" * 50)
    if all_good:
        print("✓ All files have valid Python syntax!")
        return 0
    else:
        print("✗ Some files have syntax errors.")
        return 1

if __name__ == "__main__":
    sys.exit(main())