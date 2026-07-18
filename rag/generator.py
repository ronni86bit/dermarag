"""
DermaRAG Generator Module

This module handles generating explanations using Groq's LLM (Llama3-8b-8192),
grounded in retrieved dermatology knowledge base information.
"""

import os
import logging
from typing import List, Dict, Union
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_groq() -> Groq:
    """
    Configure the Groq client with the API key from environment variables or Streamlit secrets.

    Returns:
        Groq: Configured Groq client
    """
    api_key = None
    # Try to get from Streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
    except ImportError:
        pass  # Streamlit not available

    # If not found in secrets, try environment variable
    if not api_key:
        api_key = os.getenv('GROQ_API_KEY')

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file or Streamlit secrets."
        )
    return Groq(api_key=api_key)


def groq_generate(
    condition: str,
    confidence: float,
    retrieved_chunks: List[Dict],
    image_description: str = None
) -> str:
    """
    Generate a grounded explanation using Groq's LLM.

    Args:
        condition: The predicted medical condition
        confidence: Confidence score of the prediction (0-1)
        retrieved_chunks: List of retrieved documents from the knowledge base
        image_description: Optional description of the image contents

    Returns:
        Generated explanation string with citations and disclaimer
    """
    try:
        # Initialize Groq client
        client = configure_groq()

        # Prepare the context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"[{i}] Condition: {chunk['condition']}\n"
                f"Information: {chunk['text']}\n"
                f"Source: {chunk['source_url']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Build the prompt
        prompt_parts = [
            f"You are a knowledgeable dermatology assistant. "
            f"Based on the following medical information, explain the predicted condition "
            f"'{condition}' with {confidence:.1%} confidence.",
            "",
            "RETRIEVED MEDICAL INFORMATION:",
            context,
            "",
            "INSTRUCTIONS:",
            "1. Provide a clear, accurate explanation of the condition based SOLELY on the provided information.",
            "2. Do not add information that is not present in the retrieved chunks.",
            "3. Cite your sources using bracket numbers [1], [2], etc. corresponding to the source numbers above.",
            "4. Explain symptoms, causes, and treatment approaches as covered in the sources.",
            "5. If the information is insufficient to cover certain aspects, state that clearly.",
            "6. End with the standard medical disclaimer.",
            "",
            "FORMAT YOUR RESPONSE AS:",
            "- Brief explanation of the condition",
            "- Key symptoms mentioned in sources",
            "- Possible causes/risk factors",
            "- General treatment approaches",
            "- Citations throughout",
            "",
            "⚠️ DISCLAIMER: This information is for educational purposes only and is not a medical diagnosis. "
            "Consult a qualified dermatologist for proper diagnosis and treatment."
        ]

        # Add image description if provided
        if image_description:
            prompt_parts.insert(
                2,
                f"IMAGE DESCRIPTION: {image_description}"
            )

        prompt = "\n".join(prompt_parts)

        # Generate the response
        logger.info(f"Generating explanation for {condition} with {confidence:.1%} confidence")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",  # Using Llama 3 8B model
            temperature=0.3,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        # Extract the text
        explanation = chat_completion.choices[0].message.content

        logger.info("Successfully generated explanation")
        return explanation

    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        # Fallback response in case of API error
        fallback = f"""
Based on the retrieved information, the predicted condition is {condition}
with {confidence:.1%} confidence.

However, I encountered an error while generating the detailed explanation.
Please consult the retrieved sources directly for more information.

Sources consulted:
{chr(10).join([f"[{i+1}] {chunk['condition']}: {chunk['text'][:100]}... ({chunk['source_url']})"
               for i, chunk in enumerate(retrieved_chunks)])}

⚠️ DISCLAIMER: This information is for educational purposes only and is not a medical diagnosis.
Consult a qualified dermatologist for proper diagnosis and treatment.
        """
        return fallback.strip()


# For backward compatibility with the original specification
def generate_explanation(
    condition: str,
    confidence: float,
    retrieved_chunks: List[Dict],
    image_description: str = None
) -> str:
    """
    Backward compatible function name.
    """
    return groq_generate(condition, confidence, retrieved_chunks, image_description)


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)

    # Mock retrieved chunks for testing
    mock_chunks = [
        {
            "condition": "Melanoma",
            "text": "Melanoma is a type of skin cancer that develops from melanocytes. Warning signs include changes in the size, shape, or color of moles.",
            "source_url": "https://www.dermnetnz.org/topics/melanoma"
        },
        {
            "condition": "Melanoma",
            "text": "Risk factors for melanoma include UV exposure, fair skin, family history, and having many moles.",
            "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/"
        }
    ]

    # Note: This will fail without a valid GROQ_API_KEY
    # To test, set the environment variable or create a .env file
    try:
        result = groq_generate(
            condition="Melanoma",
            confidence=0.87,
            retrieved_chunks=mock_chunks
        )
        print("Generated explanation:")
        print(result)
    except Exception as e:
        print(f"Expected error (no API key): {e}")
        print("To test properly, set GROQ_API_KEY in your environment or .env file")