import streamlit as st
import requests
from PIL import Image
import io

FASTAPI_URL = "https://defect-detection-system-gbea.onrender.com"

st.set_page_config(
    page_title="Defect Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Industrial Defect Detection System")

st.write(
    "Upload a casting image to classify it as **Defective** or **Non-Defective**."
)

st.info(
    "ℹ️ Note: This application is hosted on Render's free tier. "
    "The first prediction after inactivity may take up to a minute while the server wakes up."
)
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=300)

    if st.button("Analyse Image"):

        files = {
            "image_predict": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
            f"{FASTAPI_URL}/analyse",
            files=files
        )
    

        if response.status_code == 200:

            result = response.json()

            with col2:

                st.subheader("Prediction")

                st.success(result["Prediction"])

                st.metric(
                    "Confidence",
                    f"{result['Confidence (%)']:.2f}%"
                )

        else:

            st.error("Prediction failed.")

        uploaded_file.seek(0)

        files = {
            "Heatmap_gen": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
            f"{FASTAPI_URL}/generate",
            files=files
        )

        if response.status_code == 200:

            st.subheader("Grad-CAM")

            st.image(
                response.content,
                caption="Gradcam Heatmap",
                width = 300
            )

        else:

            st.error("Could not generate Grad-CAM.")