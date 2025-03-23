import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances
import os
from sklearn.cluster import KMeans

# ✅ Load the saved model
model_path = "outfit_recommendation_model.h5"
model = load_model(model_path)

# ✅ Function to extract dominant colors
def extract_colors(image, n_colors=5):
    image_resized = cv2.resize(image, (200, 200))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    colors = np.round(kmeans.cluster_centers_).astype(int)
    return colors

# ✅ Function to preprocess and extract features
def extract_features(image, model):
    image_resized = cv2.resize(image, (28, 28))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_normalized = image_gray / 255.0
    image_reshaped = image_normalized.reshape(1, 28, 28, 1)
    features = model.predict(image_reshaped)
    return features

# ✅ Streamlit UI
st.title("👗 Outfit Recommendation System")
st.write("Upload your *Tops* and *Bottoms*, and let the system suggest the best pairing!")

# 🔹 Upload top and bottom images
top_file = st.file_uploader("Upload a Top Image", type=["jpg", "jpeg", "png"])
bottom_file = st.file_uploader("Upload a Bottom Image", type=["jpg", "jpeg", "png"])

if top_file and bottom_file:
    top_image = cv2.imdecode(np.frombuffer(top_file.read(), np.uint8), 1)
    bottom_image = cv2.imdecode(np.frombuffer(bottom_file.read(), np.uint8), 1)

    top_colors = extract_colors(top_image)
    bottom_colors = extract_colors(bottom_image)

    top_features = extract_features(top_image, model)
    bottom_features = extract_features(bottom_image, model)

    top_combined = np.hstack((top_features.flatten(), top_colors.flatten()))
    bottom_combined = np.hstack((bottom_features.flatten(), bottom_colors.flatten()))

    distance = euclidean_distances([top_combined], [bottom_combined])[0][0]

    col1, col2 = st.columns(2)
    with col1:
        st.image(top_image, caption="Top Image", use_column_width=True)
    with col2:
        st.image(bottom_image, caption="Bottom Image", use_column_width=True)

    st.write(f"🔥 *Matching Score:* {100 - distance:.2f}%")
    if distance < 5:
        st.success("✅ Great Match! These outfits pair well together.")
    else:
        st.warning("⚠️ Not the best match. Try a different combination!")