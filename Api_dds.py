from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import tensorflow as tf
from PIL import Image
import numpy as np
from Grad_cam import generate_gradcam
cnn = tf.keras.models.load_model(r"Defect_detection_system.keras")
app = FastAPI()

def Image_Analyse(upload: UploadFile = File(...)):      # Helper function
    
    if upload.content_type not in ["image/jpeg","image/png"]:
        return {"Error":"Upload a JPG or PNG image."}
    
    img = Image.open(upload.file)

    img = img.convert("RGB")
    img = img.resize((128,128))  # Resize to input size
    img = np.array(img) # Convert to an array
    img = img/255  # Normalize

    img = np.expand_dims(img, axis=0)

    return img

@app.post("/analyse")
def Prediction(image_predict: UploadFile = File(...)):
    img = Image_Analyse(image_predict)
    predict = cnn.predict(img)
    output = -1
    confidence = 100 * predict[0][0]
    if confidence > 55:                   # Threshold value : 55 %
        output = "Non-Defective"
    elif confidence < 55:
        output = "Defective"
        confidence = 100 - confidence
    
    return {"Prediction": output, 
            "Confidence (%)": round(float(confidence),4)}

@app.post("/generate")

def Heatmap_generate(Heatmap_gen: UploadFile=File(...)):
    img = Image_Analyse(Heatmap_gen)

    hm = generate_gradcam(cnn=cnn,image_jpg = img)

    return Response(content = hm,media_type="image/png")