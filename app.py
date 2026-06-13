from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os

app = FastAPI()
model = YOLO('yolov8n.pt')

@app.post("/predict")
async def predict(file:UploadFile = File (...)):
    temp_filename = "temp_image.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    results = model(temp_filename)
    detected = []
    for r in results:
        for box in r.boxes:
            detected.append(model.names[int(box.cls[0])])
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return {"detected": detected}

@app.get("/")
def home():
    return {"message": "AI model is ready"}
