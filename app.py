import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e6f4ea;
        border: 1px solid #ceead6;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_veg_model.h5")

with st.spinner("Loading model, please wait..."):
    model = load_model()

class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/color/96/ingredients.png", width=80)
    st.header("About App")
    st.write("This application uses a Convolutional Neural Network (CNN) to classify 36 different fruits and vegetables.")
    st.markdown("---")
    st.markdown("**Developer:** Sikandar Ali")

# Main Interface Header
st.title("🍎 Fruit & Vegetable Classifier")
st.markdown("Upload an image of a fruit or vegetable to let the model identify it!")
st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader("Upload image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    image = Image.open(uploaded_file)
    
    with col1:
        # Fixed width to prevent image from breaking the layout
        st.image(image, caption='Uploaded Image', width=250)
        
    with col2:
        st.write("### Analysis Results")
        with st.spinner("Model is predicting..."):
            # Preprocess image
            image_resized = image.resize((128, 128))
            img_array = np.array(image_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = float(tf.nn.softmax(predictions[0])[predicted_class_idx]) * 100
        
        st.markdown(f"""
            <div class="prediction-box">
                <h3 style='color: #137333; margin:0;'>Prediction: {predicted_class.capitalize()}</h3>
                <p style='font-size: 18px; color: #5f6368; margin-top: 10px;'>Confidence: <b>{confidence:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)