import streamlit as st
import requests
from PIL import Image
import time

FASTAPI_URL = "https://defect-detection-system-gbea.onrender.com"

st.set_page_config(
    page_title="Defect Detection System",
    page_icon="🔍",
    layout="wide"
)

# --- Cold start detection + blocking modal ---

if "server_warm" not in st.session_state:
    st.session_state.server_warm = False

@st.dialog("Waking up the server")
def cold_start_dialog():
    st.write("This app runs on a free-tier server that sleeps when idle. It's starting up now — this can take up to a minute. Please wait, don't close this tab.")
    with st.spinner("Connecting..."):
        while True:
            try:
                r = requests.get(f"{FASTAPI_URL}/health", timeout=5)
                if r.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
    st.session_state.server_warm = True
    st.rerun()

def ensure_server_awake():
    if st.session_state.server_warm:
        return True
    # Quick probe first — if server's already warm, skip the modal entirely
    try:
        r = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        if r.status_code == 200:
            st.session_state.server_warm = True
            return True
    except requests.exceptions.RequestException:
        pass
    # Server didn't respond quickly — show blocking modal and wait
    cold_start_dialog()
    return False  # dialog will rerun the app once ready

# --- Main app ---

st.title("🔍 Industrial Defect Detection System")
st.write("Upload a casting image to classify it as **Defective** or **Non-Defective**.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=300)

    if st.button("Analyse Image"):
        if not ensure_server_awake():
            st.stop()  # modal is showing / server not ready yet, halt this run

        files = {
            "image_predict": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        with st.spinner("Analysing image..."):
            response = requests.post(f"{FASTAPI_URL}/analyse", files=files)

        if response.status_code == 200:
            result = response.json()
            with col2:
                st.subheader("Prediction")
                st.success(result["Prediction"])
                st.metric("Confidence", f"{result['Confidence (%)']:.2f}%")
        else:
            st.error("Prediction failed.")

        uploaded_file.seek(0)
        files = {
            "Heatmap_gen": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        with st.spinner("Generating Grad-CAM..."):
            response = requests.post(f"{FASTAPI_URL}/generate", files=files)

        if response.status_code == 200:
            st.subheader("Grad-CAM")
            st.image(response.content, caption="Gradcam Heatmap", width=300)
        else:
            st.error("Could not generate Grad-CAM.")