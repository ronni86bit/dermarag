"""
DermaRAG Model Inference Module

This module handles the image processing and classification using a Siamese EfficientNetB0 architecture.
It loads a pre-trained model (with ImageNet weights) and provides prediction functionality for
1-3 skin lesion images.

The model architecture follows a Siamese network with three branches (for up to three images),
each branch being an EfficientNetB0 with global average pooling, followed by a shared dense
head for classification into 25 skin condition classes.

If trained weights are available locally, they are loaded to improve accuracy.
"""

import os
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import streamlit as st


@st.cache_resource
def load_model() -> Model:
    """
    Load and cache the Siamese EfficientNetB0 model with classification head.

    Returns:
        tf.keras.Model: The loaded model ready for inference
    """
    # Define the input shape for EfficientNetB0
    input_shape = (224, 224, 3)

    # Create three input branches (for up to 3 images)
    input_1 = tf.keras.Input(shape=input_shape, name='image1')
    input_2 = tf.keras.Input(shape=input_shape, name='image2')
    input_3 = tf.keras.Input(shape=input_shape, name='image3')

    # Base EfficientNetB0 model (without top, with average pooling)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )

    # Freeze the base model
    base_model.trainable = False

    # Process each image through the base model
    features_1 = base_model(input_1)
    features_2 = base_model(input_2)
    features_3 = base_model(input_3)

    # Concatenate features from all three branches
    concatenated = tf.keras.layers.Concatenate()([features_1, features_2, features_3])

    # Classification head
    x = Dense(256, activation='relu')(concatenated)
    x = Dropout(0.3)(x)
    outputs = Dense(25, activation='softmax', name='predictions')(x)

    # Create the model
    model = Model(
        inputs=[input_1, input_2, input_3],
        outputs=outputs,
        name='dermarag_siamese_efficientnet'
    )

    # Load trained weights if available
    weights_path = os.path.join(os.path.dirname(__file__), 'dermal_weights.h5')
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Loaded custom weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
    else:
        print("No custom weights found. Using ImageNet weights for base and random weights for head.")

    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess a single image for model input.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image as numpy array with shape (224, 224, 3)
    """
    # Load image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(image_paths: List[str]) -> List[Tuple[str, float]]:
    """
    Predict skin condition from 1-3 images.

    Args:
        image_paths: List of paths to image files (1-3 images)

    Returns:
        List of tuples (condition_name, confidence) sorted by confidence descending
    """
    # Load the model
    model = load_model()

    # Prepare inputs (always 3 images, missing ones are zeros)
    inputs = []
    for i in range(3):
        if i < len(image_paths):
            # Preprocess the image
            img_array = preprocess_image(image_paths[i])
            inputs.append(img_array)
        else:
            # Missing image -> zeros tensor
            inputs.append(np.zeros((1, 224, 224, 3), dtype=np.float32))

    # Make prediction
    predictions = model.predict(inputs, verbose=0)[0]

    # Get class names (should match training order)
    # In a real implementation, this would come from training metadata
    class_names = [
        'Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis',
        'Dermatofibroma', 'Eczema', 'Fungal Infection', 'Glomus Tumor',
        'Hair Disorders', 'Haemangioma', 'Hyperhidrosis', 'Infected Wound',
        'Lichen Planus', 'Lupus Erythematosus', 'Melanoma', 'Molluscum Contagiosum',
        'Mycetoma', 'Nail Disorders', 'Papillomatosis', 'Psoriasis',
        'Seborrheic Keratosis', 'Syphilis', 'Systemic Lupus Erythematosus',
        'Vasculitis', 'Vitiligo'
    ]

    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        (class_names[i], float(predictions[i]))
        for i in top_indices
    ]

    return top_predictions