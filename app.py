from fastapi import FastAPI, UploadFile, File
from object_detection import VisionAssistant
from translations import translate
import shutil
import os

app = FastAPI()

assistant = VisionAssistant(model_path='yolov8n.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = "temp_image.jpg"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        detected_objects = assistant.detect_objects(temp_filename)
        arabic_objects = [translate(obj) for obj in detected_objects]
        return {
            "message": "Image processed successfully",
            "detected_english": detected_objects,
            "detected_arabic": arabic_objects,
            "count": len(detected_objects)
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/")
def read_root():
    return {"message": "Vision Assistant API is running!"}
