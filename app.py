# image_app.py

import streamlit as st
from PIL import Image
from object_detection import create_caption_audio

# Streamlit Configuration
st.set_page_config(page_title="🧠 Visual Insight App", layout="wide")
st.title("📸 AI-based Visual Insight Generator")

st.markdown("Drop your image below and let AI narrate what's inside:")

# File Upload
image_data = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if image_data:
    image = Image.open(image_data)
    st.image(image, caption="Input Preview", use_column_width=True)

    if st.button("🧠 Analyze and Narrate"):
        st.write("Processing image and generating insights...")
        description, audio_path = create_caption_audio(image)

        st.success("AI Description:")
        st.write(description)

        st.audio(audio_path)
