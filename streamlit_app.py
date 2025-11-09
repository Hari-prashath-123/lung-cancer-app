import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np
import json
import os

# Page configuration
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")

# Constants and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "lung_cancer_20251109-095332.keras"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

@st.cache_resource
def load_model():
    """Load the trained Keras model from disk."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}: {e}")
        st.stop()

@st.cache_resource
def load_class_names():
    """Load and normalize class names from JSON file.

    Supports either a list like ["normal", "malignant"] or a dict
    like {"0": "normal", "1": "malignant"}.
    """
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            data = json.load(f)
        # Normalize to list[str]
        if isinstance(data, list):
            class_names = [str(x) for x in data]
        elif isinstance(data, dict):
            # Sort by numeric key when possible
            try:
                class_names = [v for k, v in sorted(data.items(), key=lambda kv: int(str(kv[0])))]
            except Exception:
                class_names = [v for k, v in sorted(data.items(), key=lambda kv: str(kv[0]))]
        else:
            class_names = [str(data)]
        return class_names
    except Exception as e:
        st.error(f"Failed to load class names from {CLASS_NAMES_PATH}: {e}")
        st.stop()

# Load model and class names
model = load_model()
class_names = load_class_names()

# Title and subtitle
st.title("🫁 Lung Cancer Prediction")
st.markdown("### Upload a CT scan image to predict the likelihood of lung cancer.")

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a CT scan image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

with col2:
    st.subheader("Prediction Result")
    prediction_placeholder = st.empty()

# Initialize session state for image and prediction
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None

# Handle file upload
if uploaded_file is not None:
    st.session_state.uploaded_image = Image.open(uploaded_file)

# Image display (reduced size)
if st.session_state.uploaded_image is not None:
    # Allow user-controlled preview size (default 300px)
    with st.container():
        preview_col, control_col = st.columns([3, 1])
        with control_col:
            size = st.slider("Preview width", min_value=150, max_value=512, value=300, step=10, key="img_preview_width")
        with preview_col:
            st.image(
                st.session_state.uploaded_image,
                caption="Uploaded CT Scan",
                width=size,
            )

# Create button columns
button_col1, button_col2 = st.columns(2)

with button_col1:
    predict_button = st.button(
        "🔍 Predict",
        key="predict_btn",
        use_container_width=True,
        type="primary"
    )

with button_col2:
    clear_button = st.button(
        "🗑️ Clear",
        key="clear_btn",
        use_container_width=True
    )

# Handle prediction
if predict_button and st.session_state.uploaded_image is not None:
    with st.spinner("Processing image..."):
        # Determine expected input size from the model (fallback to 224x224)
        input_shape = model.input_shape
        # If the model has multiple inputs, pick the first
        if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]
        try:
            target_h = int(input_shape[1]) if input_shape and len(input_shape) >= 3 and input_shape[1] else 224
            target_w = int(input_shape[2]) if input_shape and len(input_shape) >= 3 and input_shape[2] else 224
        except Exception:
            target_h, target_w = 224, 224

        # Ensure image is RGB
        img = st.session_state.uploaded_image
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize
        processed_image = img.resize((target_w, target_h))

        # Convert to numpy array
        image_array = np.array(processed_image, dtype=np.float32)

        # Handle grayscale case (already converted to RGB above, but keep a safeguard)
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.ndim == 3 and image_array.shape[-1] == 4:
            # Drop alpha channel if present (shouldn't after convert, but safe)
            image_array = image_array[..., :3]

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Apply DenseNet preprocessing with a safe fallback
        try:
            preprocessed = preprocess_input(image_array.copy())
        except Exception:
            # Fallback to simple [0,1] scaling
            preprocessed = image_array / 255.0

        # Make prediction
        prediction_output = model.predict(preprocessed)

        # Normalize output to predicted index and confidence
        pred = np.array(prediction_output)
        if pred.ndim == 2 and pred.shape[1] == 1:
            # Binary sigmoid output
            p1 = float(pred[0, 0])
            predicted_index = 1 if p1 >= 0.5 else 0
            confidence = p1 if predicted_index == 1 else 1.0 - p1
        elif pred.ndim == 2 and pred.shape[1] >= 2:
            # Softmax / logits for multi-class
            probs = pred[0]
            try:
                # If not already softmaxed, apply softmax for a nicer confidence
                exp = np.exp(probs - np.max(probs))
                probs = exp / np.sum(exp)
            except Exception:
                pass
            predicted_index = int(np.argmax(probs))
            confidence = float(np.max(probs))
        else:
            # Unexpected output shape; attempt to flatten
            flat = pred.flatten()
            if flat.size == 1:
                p1 = float(flat[0])
                predicted_index = 1 if p1 >= 0.5 else 0
                confidence = p1 if predicted_index == 1 else 1.0 - p1
            else:
                predicted_index = int(np.argmax(flat))
                confidence = float(np.max(flat))

        # Map to class name safely
        if 0 <= predicted_index < len(class_names):
            predicted_class = class_names[predicted_index]
        else:
            predicted_class = f"class_{predicted_index}"

        st.session_state.prediction = str(predicted_class)
        st.session_state.confidence = float(confidence)

# Handle clear button
if clear_button:
    st.session_state.uploaded_image = None
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.rerun()

# Display prediction result
if st.session_state.prediction is not None:
    with prediction_placeholder.container():
        label = str(st.session_state.prediction)
        label_lower = label.lower()
        if any(x in label_lower for x in ["malig", "cancer", "positive", "abnormal"]):
            st.error(f"⚠️ Prediction: **{label.upper()}**")
        else:
            st.success(f"✓ Prediction: **{label.upper()}**")

        st.metric(
            "Confidence Score",
            f"{st.session_state.confidence:.1%}"
        )
elif st.session_state.uploaded_image is not None:
    with prediction_placeholder.container():
        st.info("Click the Predict button to analyze the image.")
