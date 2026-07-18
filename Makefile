# Makefile for DermaRAG
# Provides convenient commands for development and deployment

.PHONY: help build-index run deploy test clean

# Default target
help:
	@echo "DermaRAG Makefile"
	@echo "================="
	@echo "Available commands:"
	@echo "  make build-index   - Build the FAISS knowledge base index"
	@echo "  make run           - Run the Streamlit application locally"
	@echo "  make deploy        - Deploy to Hugging Face Spaces (requires HF CLI)"
	@echo "  make test          - Run tests (if implemented)"
	@echo "  make clean         - Remove temporary files and cache"
	@echo ""
	@echo "Example usage:"
	@echo "  make build-index   # First time setup"
	@echo "  make run           # Start the application"

# Build the FAISS index from the knowledge base CSV
build-index:
	@echo "Building FAISS index from knowledge base..."
	@python -m rag.build_index

# Run the Streamlit application
run:
	@echo "Starting DermaRAG application..."
	@streamlit run app.py

# Deploy to Hugging Face Spaces
# Requires: huggingface_hub CLI and logged in (huggingface-cli login)
deploy:
	@echo "Deploying to Hugging Face Spaces..."
	@echo "Make sure you've installed huggingface_hub and logged in:"
	@echo "  pip install huggingface_hub"
	@echo "  huggingface-cli login"
	@huggingface-cli upload dermarag/ . --repo-type=space --commit-message="Auto-deploy via Makefile"

# Run tests (placeholder for future implementation)
test:
	@echo "Running tests..."
	@echo "No tests implemented yet. Add tests to tests/ directory."

# Clean up temporary files and cache
clean:
	@echo "Cleaning up temporary files..."
	@rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
	@rm -rf .streamlit/
	@rm -rf .ipynb_checkpoints/
	@find . -name "*.pyc" -delete
	@find . -name ".DS_Store" -delete
	@echo "Cleanup complete."

# Help target is default
.DEFAULT_GOAL := help