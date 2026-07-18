"""
DermaRAG — AI-Powered Skin Condition Assistant

Main Streamlit application for the DermaRAG project.
Provides a user interface for uploading skin lesion images,
getting predictions, retrieving relevant medical information,
and generating grounded explanations using Groq's Llama3 model.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import sys
from urllib.parse import urlparse

# Import our custom modules
from model.inference import predict_image
from rag.retriever import retrieve
from rag.generator import generate_explanation

# Auto-build FAISS index on first startup (required for Streamlit Community Cloud)
from pathlib import Path as _Path
import subprocess as _subprocess

_INDEX_PATH = _Path("index/faiss.index")
_METADATA_PATH = _Path("index/metadata.json")

if not _INDEX_PATH.exists() or not _METADATA_PATH.exists():
    import streamlit as st
    with st.spinner("🔨 Building knowledge base for first time... (~60 seconds)"):
        _subprocess.run([sys.executable, "rag/build_index.py"], check=True)
    st.success("✅ Knowledge base ready!")
    st.rerun()
st.set_page_config(
    page_title="DermaRAG — AI-Powered Skin Condition Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_DIR = Path(__file__).parent / "model"
DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"
KB_PATH = DATA_DIR / "dermatology_kb.csv"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# Condition labels (should match the model's training)
CONDITION_LABELS = [
    "Acne", "Actinic Keratosis", "Atopic Dermatitis",
    "Basal Cell Carcinoma", "Contact Dermatitis", "Eczema",
    "Folliculitis", "Fungal Infection", "Herpes Zoster",
    "Hyperpigmentation", "Impetigo", "Keloid",
    "Lichen Planus", "Melanoma", "Molluscum Contagiosum",
    "Nail Disease", "Psoriasis", "Rosacea",
    "Seborrheic Dermatitis", "Seborrheic Keratosis",
    "Squamous Cell Carcinoma", "Tinea", "Urticaria",
    "Vitiligo", "Warts"
]


def display_header():
    """Display the application header and description."""
    st.title("🩺 DermaRAG — AI-Powered Skin Condition Assistant")
    st.markdown("""
    DermaRAG combines deep learning image analysis with retrieval-augmented generation
    to provide educational information about skin conditions. Upload 1-3 images of a skin
    concern to get AI-powered insights backed by medical knowledge sources.
    """)

def display_sidebar():
    """Display the sidebar with app information."""
    with st.sidebar:
        st.header("About DermaRAG")
        st.markdown("""
        This application uses:
        - **EfficientNetB0** for image feature extraction
        - **FAISS** for efficient similarity search
        - **Sentence Transformers** for text embeddings
        - **Groq's Llama3 model** for generating explanations
        - **Streamlit** for the user interface
        """)

        st.header("How It Works")
        st.markdown("""
        1. **Upload** 1-3 images of a skin condition
        2. **Analyze** - The AI extracts features and predicts potential conditions
        3. **Retrieve** - Relevant medical information is fetched from our knowledge base
        4. **Generate** - Groq's Llama3 creates a grounded explanation with citations
        5. **Learn** - Review the explanation with source references
        """)

        st.header("⚠️ Important Disclaimer")
        st.markdown("""
        This tool is for **educational purposes only**.
        It does not provide medical diagnosis or treatment advice.
        Always consult a qualified healthcare professional for medical concerns.
        """)

        st.header("🔗 Resources")
        st.markdown("- [GitHub Repository](https://github.com/yourusername/dermarag)")
        st.markdown("- [Dermatology Knowledge Base](#)")

def check_index_exists():
    """Check if the FAISS index exists."""
    return INDEX_PATH.exists() and METADATA_PATH.exists()

def display_no_index_warning():
    """Display warning when index is not built."""
    st.warning(
        """
        ⚠️ **Knowledge base not found!**
        Please run `make build-index` first to create the FAISS index from the knowledge base.
        The application will still work for image predictions, but retrieval and generation
        features will be limited until the index is built.
        """
    )

def main():
    """Main application function."""
    
    # Display header and sidebar
    display_header()
    display_sidebar()

    # Check if index exists
    index_exists = check_index_exists()

    # File uploader
    st.subheader("📤 Upload Skin Images")

    uploaded_files = st.file_uploader(
        "Choose 1-3 images of a skin condition",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload clear, well-lit images of the skin area concerned. Maximum 3 files."
    )

    # Limit to 3 files
    if uploaded_files and len(uploaded_files) > 3:
        st.warning("Please upload a maximum of 3 images. Only the first 3 will be used.")
        uploaded_files = uploaded_files[:3]

    # Display uploaded images
    if uploaded_files:
        st.subheader("🖼️ Uploaded Images")

        # Create columns for image display
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image {idx+1}", use_column_width=True)

    # Analyze button
    analyze_button = st.button(
        "🔍 Analyze Images",
        type="primary",
        disabled=not uploaded_files,
        help="Click to analyze the uploaded images"
    )

    # Process when button is clicked
    if analyze_button and uploaded_files:
        with st.spinner("Analyzing images..."):
            # Save uploaded files temporarily
            temp_paths = []
            for uploaded_file in uploaded_files:
                # Create a temporary file
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_paths.append(temp_path)

            try:
                # Get predictions
                predictions = predict_image(temp_paths)

                # Display results
                st.subheader("🔍 Analysis Results")
                st.subheader("Top Predictions")
                for i, (condition, confidence) in enumerate(predictions):
                    with st.container():
                        st.subheader(f"{i+1}. {condition}")
                        st.metric(label="Confidence", value=f"{confidence:.1%}")
                        st.progress(confidence)

                # Use top prediction for retrieval and generation
                top_condition, top_confidence = predictions[0]

                # Only proceed with RAG if index exists
                if index_exists:
                    with st.spinner("Retrieving relevant medical information..."):
                        # Generate query from top prediction
                        query = f"symptoms treatment causes {top_condition}"
                        retrieved_chunks = retrieve(query, k=5)

                    # Generate explanation
                    with st.spinner("Generating explanation with Groq..."):
                        # Prepare image description (simple placeholder)
                        image_description = f"Skin lesion image(s) showing characteristics consistent with {top_condition}"

                        explanation = generate_explanation(
                            condition=top_condition,
                            confidence=top_confidence,
                            retrieved_chunks=retrieved_chunks,
                            image_description=image_description
                        )

                    st.subheader(f"📝 Medical Explanation for {top_condition}")
                    st.write(explanation)

                    # Display sources
                    st.subheader("📚 Medical Sources")
                    if retrieved_chunks:
                        for i, chunk in enumerate(retrieved_chunks, start=1):
                            with st.expander(f"Source {i}: {chunk['condition']}"):
                                st.write(chunk['text'])
                                parsed = urlparse(chunk['source_url'])
                                source_name = parsed.netloc
                                st.markdown(f"[{source_name}]({chunk['source_url']})")
                    else:
                        st.info("No relevant information found in the knowledge base.")
                else:
                    # Show prediction results without RAG
                    st.subheader("📝 Prediction Information")
                    st.info("""
                    Knowledge base not available. Showing predictions only.
                    To get detailed explanations with medical sources, please run `make build-index`
                    to create the FAISS index from the dermatology knowledge base.
                    """)

                    # Show basic info about top prediction
                    st.markdown(f"**Predicted Condition:** {top_condition}")
                    st.markdown(f"**Confidence:** {top_confidence:.1%}")

                    # Provide generic information based on condition
                    st.markdown("**General Information:**")
                    st.markdown(f"""
                    {top_condition} is a skin condition that requires professional evaluation.
                    Please consult a dermatologist for proper diagnosis and treatment.
                    """)

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please check your inputs and try again.")

            finally:
                # Clean up temporary files
                for temp_path in temp_paths:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass  # File already deleted or doesn't exist

    # Display disclaimer at the bottom
    disclaimer_message = """
    ⚠️ **Important Disclaimer:** This tool is for educational purposes only.
    The AI-generated explanations are based on retrieved medical information and
    should not be considered medical advice. Always consult a qualified dermatologist
    or healthcare professional for diagnosis, treatment, and medical decisions.
    """
    st.warning(disclaimer_message)

if __name__ == "__main__":
    main()