from fastapi import FastAPI, UploadFile, File
from object_detection import VisionAssistant
import shutil
import os

app = FastAPI()

# Load VisionAssistant once at startup
assistant = VisionAssistant(model_path='weights/yolov8n.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = "temp_image.jpg"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        detected_objects = assistant.detect_objects(temp_filename)
        
        return {
            "message": "Image processed successfully",
            "detected": detected_objects,
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
