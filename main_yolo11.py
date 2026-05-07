import os

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from yolo11.detector_image import detect_image
from yolo11.detector_video import detect_video_yolov5_format

app = FastAPI(title="YOLO11 Helmet & License Plate API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "YOLO11 API is running!"}


@app.post("/yolo11/detect")
async def detect_image_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        api_item = detect_image(temp_path, clear_existing=True)
        return JSONResponse(content={"success": True, "results": [api_item]})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/yolo11/detect-video")
async def detect_video_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        results = detect_video_yolov5_format(
            video_path=temp_path,
            frame_stride=20,
        )
        return JSONResponse(content={"success": True, "results": results})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run("main_yolo11:app", host="0.0.0.0", port=8001, reload=True)
