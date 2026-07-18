# 🩺 DermaRAG — AI-Powered Skin Condition Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> A multimodal RAG pipeline combining EfficientNetB0 image classification with FAISS vector search and Groq LLaMA 3 to provide grounded, cited explanations for skin conditions.

🚀 **[Live Demo → Share at streamlit.io/cloud](https://dermarag.streamlit.app)** (to be deployed)

---

## Overview

DermaRAG is an end-to-end multimodal AI system that takes 1–3 skin lesion images, classifies the likely condition using a Siamese EfficientNetB0 architecture, and generates a grounded clinical explanation using Retrieval-Augmented Generation (RAG). Instead of relying on an LLM's memory — which can hallucinate medical facts — DermaRAG retrieves verified information from a curated dermatology knowledge base and grounds every response in that retrieved evidence, with source citations.

The system was built as an extension of the SCIN skin disease prediction research, combining computer vision, semantic search, and large language model generation into a single deployed pipeline. It demonstrates how RAG can be applied in high-stakes domains where hallucination has real consequences.

---

## Architecture
┌─────────────────────────────────────────────────────────┐
 │ USER UPLOADS 1-3 IMAGES │
 └──────────────────┬──────────────────────────────────────┘
 ↓
 ┌─────────────────────────────────────────────────────────┐
 │ SIAMESE EFFICIENTNETB0 (3-input, shared) │
 │ img1 → EfficientNetB0 ─┐ │
 │ img2 → EfficientNetB0 ─┼→ Weighted Fusion → Dense(256)│
 │ img3 → EfficientNetB0 ─┘ → Dropout(0.3) │
 │ → Dense(25, softmax) │
 └──────────────────┬──────────────────────────────────────┘
 ↓
 Top-3 Predicted Conditions + Confidence Scores
 ↓
 ┌─────────────────────────────────────────────────────────┐
 │ QUERY GENERATION │
 │ "symptoms treatment causes [Top Condition]" │
 └──────────────────┬──────────────────────────────────────┘
 ↓
 ┌─────────────────────────────────────────────────────────┐
 │ SEMANTIC RETRIEVAL (RAG) │
 │ Query → sentence-transformers → vector embedding │
 │ FAISS index search → Top-5 relevant chunks │
 │ Dermatology KB (25 conditions, clinical text) │
 └──────────────────┬──────────────────────────────────────┘
 ↓
 ┌─────────────────────────────────────────────────────────┐
 │ GROQ LLAMA 3 GENERATION │
 │ Prediction + Retrieved chunks → Grounded response │
 │ Citations [1][2][3] → Source URLs displayed │
 └──────────────────┬──────────────────────────────────────┘
 ↓
 ┌─────────────────────────────────────────────────────────┐
 │ STREAMLIT UI │
 │ Top-3 predictions │ Confidence bars │ Sources │ Explanation │
 │ ⚠️ Medical disclaimer on every response │
 └─────────────────────────────────────────────────────────┘

---

## Key Features

- **Multimodal input** — accepts 1–3 skin lesion images, handles missing images with zero-tensor masking
- **Siamese EfficientNetB0** — shared-weight 3-input architecture with weighted feature fusion
- **25-class classification** — covers the most common dermatological conditions
- **FAISS semantic retrieval** — fast approximate nearest-neighbor search over embedded clinical text
- **Hallucination-resistant generation** — LLaMA 3 answers only from retrieved context, never from memory
- **Source citations** — every generated response cites [1][2][3] with source URLs
- **Auto-builds index** — FAISS index builds automatically on first Streamlit Community Cloud startup
- **CPU-only deployment** — runs on Streamlit Community Cloud, no GPU required
- **Medical disclaimer** — prominent on every response

---

## Model Performance

Results from the SCIN dataset (5,033 real-world dermatology cases, 25 conditions, multilabel stratified split):

### Tabular Models (metadata only)
| Model | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Ridge Regressor | 35.8% | 63.6% | 77.0% |
| LightGBM | 36.2% | 64.8% | 77.0% |
| ElasticNet | 35.2% | 64.8% | 77.0% |
| KNN Regressor | 33.7% | 60.6% | 68.1% |
| Feedforward NN | 33.5% | 62.6% | 75.2% |

### Image Models
| Model | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Custom Siamese CNN | 37.5% | 62.8% | 72.4% |
| EfficientNetB0 (feature extractor) | 37.2% | 69.1% | **80.5%** |
| EfficientNetB0 (fine-tuned) | 33.1% | 63.2% | 76.8% |

### Final Fusion Model (image + tabular metadata)
| Metric | Score |
|---|---|
| Top-1 Accuracy | 37.5% |
| Top-3 Accuracy | 65.4% |
| **Top-5 Accuracy** | **79.6%** |
| Mean KL Divergence | 2.26 |

---

## Tech Stack

| Category | Technology |
|---|---|
| Vision Model | EfficientNetB0 (TensorFlow/Keras, ImageNet weights) |
| Vector Store | FAISS (faiss-cpu) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Groq LLaMA 3 (llama3-8b-8192) |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |
| Dataset | Google SCIN (5,033 dermatology cases) |

---

## Project Structure

dermarag/
 ├── app.py # Main Streamlit application
 ├── model/
 │ └── inference.py # EfficientNetB0 loading, preprocessing, prediction
 ├── rag/
 │ ├── build_index.py # One-time FAISS index builder
 │ ├── retriever.py # Semantic retrieval from FAISS index
 │ └── generator.py # Groq LLaMA 3 grounded generation
 ├── data/
 │ └── dermatology_kb.csv # Dermatology knowledge base (condition, text, source_url)
 ├── index/
 │ └── .gitkeep # Index files built at runtime, not committed
 ├── .env.example # Environment variable template
 ├── requirements.txt # Python dependencies
 ├── Makefile # Convenience commands
 └── README.md # This file

---

## Local Setup

### Prerequisites
- Python 3.10+
- Git

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/ronni86bit/dermarag.git
cd dermarag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and add your Groq API key:
# GROQ_API_KEY=gsk_your_key_here

# 5. Build the knowledge base index (one-time setup)
make build-index

# 6. Run the app
make run
```

Open **http://localhost:8501** in your browser.

---

## How to Use

1. **Upload** 1–3 clear, well-lit images of the skin area
2. Click **"Analyze Images"**
3. View **top-3 predicted conditions** with confidence bars
4. Expand **retrieved medical sources** [1][2][3] to see the clinical evidence
5. Read the **AI-generated explanation** grounded in retrieved sources
6. Note the **medical disclaimer** — this is educational only

---

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (if not already done).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign up/log in.
3. Click "New app", select your repository, branch, and set the main file path to `app.py`.
4. Under "Advanced settings", set the Python version to **3.11**.
5. Add a secret named `GROQ_API_KEY` with your Groq API key.
6. Click "Deploy!". The app will build and deploy automatically.
   - The FAISS index will be built on first startup (takes about 60 seconds).


## Dataset

This project uses the **Google SCIN (Skin Condition Image Network)** dataset:
- 5,033 real-world dermatology cases submitted via Google Search
- Dermatologist-assigned soft labels (weighted probability distributions)
- 25 skin conditions after filtering rare classes (<50 instances)
- Multilabel stratified split for train/test

Dataset: [github.com/google-research-datasets/scin](https://github.com/google-research-datasets/scin)

---

## Limitations

- **Model weights** — uses ImageNet-pretrained EfficientNetB0 as feature extractor; loads custom weights from model/dermal_weights.h5 if available, otherwise uses random weights for the classification head. For production, replace with trained weights from the SCIN study.
- **Sample knowledge base** — `dermatology_kb.csv` contains sample clinical text; expand with a full PubMed API scrape for production
- **Not a medical device** — outputs are educational only, not clinical diagnoses
- **CPU inference** — slower than GPU; ~5–10 seconds per analysis on Streamlit Community Cloud
- **25 conditions only** — limited to conditions present in the SCIN dataset

---

## Future Improvements

- Upload trained model weights to Hugging Face Model Hub (optional)
- Expand knowledge base with full PubMed API scrape (free, no key needed)
- Add Grad-CAM visualization to highlight which image regions influenced prediction
- MLOps pipeline for model versioning and monitoring (MLflow + Docker)
- Fine-tune LLM on dermatology Q&A pairs
- Add user feedback loop to improve retrieval quality over time

---

## Author

**Cherukuri Rohith**
B.Tech CSE (Data Science), CVR College of Engineering, Hyderabad
ML Intern @ Astra Microwave Products Ltd.

- GitHub: [@ronni86bit](https://github.com/ronni86bit)
- LinkedIn: [linkedin.com/in/cherukurirohith](https://linkedin.com/in/cherukurirohith)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

⚠️ **Medical Disclaimer**: DermaRAG is an educational tool only. It does not provide medical diagnosis, treatment advice, or clinical decisions. Always consult a qualified dermatologist or healthcare professional for any skin concerns.