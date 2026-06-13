# AI-Powered Assistive Technology - Vision Assistant

## Overview
An AI application to help blind and visually impaired individuals detect objects in real-time using YOLOv8.

## Tech Stack
- Python, YOLOv8, FastAPI

## How to Run
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## API Endpoints
- GET `/` — Check if server is running
- POST `/predict` — Upload an image and get detected objects
