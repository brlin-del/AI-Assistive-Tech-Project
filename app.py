from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from object_detection import VisionAssistant
from translations import translate
import shutil
import os

app = FastAPI()

assistant = VisionAssistant(model_path='yolov8n.pt')

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

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
