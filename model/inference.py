"""
DermaRAG Model Inference Module

This module handles the image processing and classification using a Siamese EfficientNetB0 architecture
as trained in the SCIN_Model notebook. The model loads pre-trained weights from the checkpoint
file and provides prediction functionality for 1-3 skin lesion images.

The architecture matches the one used in the notebook:
- Three input branches (for up to 3 images)
- Each branch passes through an EfficientNetB0 (include_top=False, weights='imagenet', pooling='avg')
- Features are combined via a weighted average (handling missing images via masking)
- Followed by dense layers: 256 (ReLU) -> BatchNorm -> Dropout(0.4) -> 128 (ReLU) -> Dropout(0.3) -> Dense(num_classes, softmax)
"""

import os
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda, Add, Multiply
from tensorflow.keras.models import Model
import streamlit as st


def build_effnet_siamese_model(num_classes: int) -> Model:
    """
    Build the Siamese EfficientNetB0 model with feature extraction and classification head.
    This mirrors the architecture from the SCIN_Model notebook.

    Args:
        num_classes: Number of output classes.

    Returns:
        tf.keras.Model: The constructed model.
    """
    IMG_SIZE = 224

    # Define three input branches
    img1_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img1")
    img2_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img2")
    img3_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img3")

    # Create the EfficientNetB0 base model (shared across branches)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling='avg'
    )
    # Note: The base model trainability is set later when loading weights.
    # For inference, we can set it to non-trainable.
    base_model.trainable = False

    # Function to compute mask for handling missing images
    def compute_mask(x):
        # Sum over height, width, and channels -> shape (batch,)
        mask = tf.reduce_sum(x, axis=[1, 2, 3])
        # Cast to float32: 0.0 if all zeros (missing image), >0 otherwise
        mask = tf.cast(mask > 0, tf.float32)
        # Expand dims to allow broadcasting: (batch, 1)
        return tf.expand_dims(mask, -1)

    # Process each branch
    # Branch 1
    features_1 = base_model(img1_input)
    mask_1 = Lambda(compute_mask, name="mask1")(img1_input)
    masked_features_1 = Multiply()([features_1, mask_1])

    # Branch 2
    features_2 = base_model(img2_input)
    mask_2 = Lambda(compute_mask, name="mask2")(img2_input)
    masked_features_2 = Multiply()([features_2, mask_2])

    # Branch 3
    features_3 = base_model(img3_input)
    mask_3 = Lambda(compute_mask, name="mask3")(img3_input)
    masked_features_3 = Multiply()([features_3, mask_3])

    # Combine features: weighted average by number of valid images
    sum_features = Add()([masked_features_1, masked_features_2, masked_features_3])
    sum_masks = Add()([mask_1, mask_2, mask_3])
    # Avoid division by zero: add a small epsilon
    sum_masks = Lambda(lambda x: x + 1e-6)(sum_masks)
    # Features after averaging
    averaged_features = Lambda(lambda x: x[0] / x[1])([sum_features, sum_masks])

    # Classification head
    x = Dense(256, activation='relu')(averaged_features)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name="predictions")(x)

    # Create model
    model = Model(
        inputs=[img1_input, img2_input, img3_input],
        outputs=outputs,
        name='effnet_siamese_classifier'
    )

    return model


@st.cache_resource
def load_model() -> Model:
    """
    Load and cache the Siamese EfficientNetB0 model with pre-trained weights.

    Returns:
        tf.keras.Model: The loaded model ready for inference.
    """
    # Number of classes determined from the SCIN dataset after filtering
    num_classes = 41  # As computed from the notebook after removing classes with <30 instances

    # Build the model architecture
    model = build_effnet_siamese_model(num_classes)

    # Define paths to the weights file
    weights_path_h5 = os.path.join(os.path.dirname(__file__), 'model_checkpoints', 'custom_siamese.weights.h5')
    weights_path_keras = os.path.join(os.path.dirname(__file__), 'model_checkpoints', 'custom_siamese.weights.keras')

    # Check for weights file (prefer .keras if exists, else .h5)
    weights_path = None
    if os.path.exists(weights_path_keras):
        weights_path = weights_path_keras
    elif os.path.exists(weights_path_h5):
        weights_path = weights_path_h5

    if weights_path:
        try:
            model.load_weights(weights_path)
            print(f'Loaded custom weights from {weights_path}')
        except Exception as e:
            print(f'Warning: Could not load weights from {weights_path}: {e}')
            print('Using ImageNet weights for base and random weights for head.')
    else:
        print('No custom weights found. Using ImageNet weights for base and random weights for head.')

    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess a single image for model input.
    Matches the preprocessing used in the notebook: resizing to 224x224 and applying
    EfficientNet preprocessing.

    Args:
        image_path: Path to the image file. Empty string indicates a missing image.

    Returns:
        Preprocessed image as numpy array with shape (1, 224, 224, 3).
    """
    if image_path == "":
        # Return a zero array of the correct shape for missing images
        return np.zeros((1, 224, 224, 3), dtype=np.float32)

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
        image_paths: List of paths to image files (1-3 images).
                     Missing images should be represented as empty strings.

    Returns:
        List of tuples (condition_name, confidence) sorted by confidence descending.
    """
    # Load the model
    model = load_model()

    # Prepare inputs (always 3 images, missing ones are empty strings)
    # We will preprocess each image to get a tensor of shape (1, 224, 224, 3)
    inputs = []
    for i in range(3):
        if i < len(image_paths):
            img_array = preprocess_image(image_paths[i])
            inputs.append(img_array)
        else:
            # Missing image -> zeros tensor
            inputs.append(np.zeros((1, 224, 224, 3), dtype=np.float32))

    # Make prediction
    predictions = model.predict(inputs, verbose=0)[0]  # Shape: (num_classes,)

    # Get class names in the same order as used during training
    # This list must match the order of the classes in the training data.
    class_names = [
        'Abrasion, scrape, or scab', 'Abscess', 'Acne', 'Acute and chronic dermatitis',
        'Acute dermatitis, NOS', 'Allergic Contact Dermatitis', 'CD - Contact dermatitis',
        'Cellulitis', 'Drug Rash', 'Eczema', 'Erythema multiforme', 'Folliculitis',
        'Granuloma annulare', 'Herpes Simplex', 'Herpes Zoster', 'Hypersensitivity',
        'Impetigo', 'Insect Bite', 'Irritant Contact Dermatitis', 'Keratosis pilaris',
        'Leukocytoclastic Vasculitis', 'Lichen Simplex Chronicus', 'Lichen planus/lichenoid eruption',
        'Miliaria', 'Molluscum Contagiosum', 'O/E - ecchymoses present', 'Photodermatitis',
        'Pigmented purpuric eruption', 'Pityriasis rosea', 'Psoriasis', 'Purpura', 'Rosacea',
        'SCC/SCCIS', 'Scabies', 'Seborrheic Dermatitis', 'Stasis Dermatitis', 'Tinea',
        'Tinea Versicolor', 'Urticaria', 'Verruca vulgaris', 'Viral Exanthem'
    ]

    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        (class_names[i], float(predictions[i]))
        for i in top_indices
    ]

    return top_predictions