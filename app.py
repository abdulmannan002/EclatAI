
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd
import tempfile
import os

# Import the classify_and_explain function from me.py
from classifier import classify_and_explain

# Import the create_pdf_report function from report_generator.py
from report_generator import create_pdf_report

from login import main as login_page

#from shap import generate_shap_report,load_resnet_model,load_svm_model  # Import from shap.py

# Set page config for a cleaner look
st.set_page_config(page_title="ÉclatAI", layout="wide", initial_sidebar_state="auto")

# Call the login page
login_page()
# Check if the user is logged in and on the correct page
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Redirecting to login page...")
    st.session_state['page'] = "login"
    st.stop()

# Main app content after login
st.title(f"Welcome to ÉclatAI, {st.session_state['username']}!")
st.write("This is the main content of the app. You are now logged in.")

# Your app's logic for file upload, image classification, and report generation.

# Load the pre-trained SVM model
svm_classifier = joblib.load('models/svm_classifier (1).pkl')

# Load and preprocess image
def load_and_preprocess_image(img):
    img = image.load_img(img, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Feature extraction model using ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_model = Model(inputs=base_model.input, outputs=x)

# Function to convert DataFrame to CSV
@st.cache
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


# Main function
def main():
    st.title("ÉclatAI: Image Classification and Explainability")
    st.write("Upload an image to classify as **real** or **AI-generated**.")

    st.markdown("---")

    uploaded_file = st.file_uploader("🔍 Choose an image to classify...", type="jpg")

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Display uploaded image
        st.image(temp_file_path, caption='Uploaded Image', width=300)

        # "Classify" button
        if st.button("Classify"):
            # Preprocess the uploaded image
            img = load_and_preprocess_image(temp_file_path)
            features = feature_model.predict(img)

            # Temporary directories for saving files
            with tempfile.TemporaryDirectory() as tmpdirname:
                try:
                    # Call classify_and_explain function from me.py
                    classify_and_explain(
                        image_file=temp_file_path,
                        svm_model_path='models/svm_classifier (1).pkl',
                        save_dir=tmpdirname
                    )

                    # Load and display the heatmap image
                    heatmap_path = os.path.join(tmpdirname, 'superimposed_heatmap.jpg')
                    st.image(heatmap_path, caption="Grad-CAM Heatmap", width=300)

                    # Load the explanation report
                    report_path = os.path.join(tmpdirname, 'classification_report.txt')
                    if os.path.exists(report_path):
                        try:
                            with open(report_path, 'r', encoding='utf-8') as file:
                                explanation = file.read()
                        except UnicodeDecodeError:
                            with open(report_path, 'r', encoding='latin-1') as file:
                                explanation = file.read()
                        # Display explanation report
                        st.subheader("📄 Explanation Report")
                        st.text(explanation)
                    else:
                        st.error("Explanation report not found.")

                    # Generate PDF report
                    predicted_label = "Real" if svm_classifier.predict(feature_model.predict(load_and_preprocess_image(temp_file_path)))[0] == 1 else "AI-Generated-Image"
                    pdf_buffer = create_pdf_report(predicted_label, heatmap_path, explanation, temp_file_path)

                    # PDF download
                    st.download_button(
                        label="⬇️ Download Report as PDF",
                        data=pdf_buffer,
                        file_name='classification_report.pdf',
                        mime='application/pdf'
                    )

                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
