import uvicorn
import torch
import cv2
import os
import glob
import time
import random
import requests
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from VideoDetector import start_detecttion



from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, set_logging
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


app = FastAPI(title="Helmet & License Plate Detection API", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

weights_path = './runs/train/finalModel/weights/best.pt'
device = select_device('cpu') 
half = device.type != 'cpu'

set_logging()
model = attempt_load(weights_path, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(448, s=stride)
if half:
    model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


def run_detection(source):
    cudnn = torch.backends.cudnn
    cudnn.benchmark = True

    os.makedirs('laneOutput', exist_ok=True)
    existingOutputs = []
    imgCounter = 0
    results = []

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            p = Path(path)

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)

                    if names[c] == 'Rider':
                        x1, y1, x2, y2 = map(int, xyxy)
                        roi = im0s[y1:y2, x1:x2]
                        rider_helmet_status = None
                        rider_lp_status = None
                        rider_lp_number = None
                        no_of_passengers = 0

                        if roi.size > 0:
                            cv2.imwrite('rider.png', roi)
                            rid_dataset = LoadImages('rider.png', img_size=imgsz, stride=stride)

                            for rid_path, rid_img, rid_im0s, rid_vid_cap in rid_dataset:
                                rid_img = torch.from_numpy(rid_img).to(device)
                                rid_img = rid_img.half() if half else rid_img.float()
                                rid_img /= 255.0
                                if rid_img.ndimension() == 3:
                                    rid_img = rid_img.unsqueeze(0)

                                rid_pred = model(rid_img, augment=False)[0]
                                rid_pred = non_max_suppression(rid_pred, 0.25, 0.45)

                                for rid_det in rid_pred:
                                    if len(rid_det):
                                        rid_det[:, :4] = scale_coords(rid_img.shape[2:], rid_det[:, :4], rid_im0s.shape).round()
                                        for *xyxy, rid_conf, rid_cls in rid_det:
                                            rid_c = int(rid_cls)
                                            if names[rid_c] == "Helmet":
                                                rider_helmet_status = True
                                                no_of_passengers += 1
                                            elif names[rid_c] == "No Helmet":
                                                rider_helmet_status = False
                                                no_of_passengers += 1
                                            elif names[rid_c] == "LP":
                                                x1, y1, x2, y2 = map(int, xyxy)
                                                lp_roi = roi[y1:y2, x1:x2]
                                                cv2.imwrite('rider_lp.png', lp_roi)

                                                # Call Plate Recognizer API
                                                with open("rider_lp.png", 'rb') as fp:
                                                    response = requests.post(
                                                        'https://api.platerecognizer.com/v1/plate-reader/',
                                                        data=dict(regions=['in']),
                                                        files=dict(upload=fp),
                                                        headers={'Authorization': 'Token 286a048d66a997b41b9d39278b1446816eee9fe1'}
                                                    )
                                                    try:
                                                        rider_lp_number = response.json()['results'][0]['plate']
                                                        rider_lp_status = True
                                                    except Exception:
                                                        rider_lp_status = False

                                                os.remove('rider_lp.png')

                        result = {
                            "rider_bbox": [x1, y1, x2, y2],
                            "helmet_status": rider_helmet_status,
                            "license_plate_detected": rider_lp_status,
                            "plate_number": rider_lp_number,
                            "num_passengers": no_of_passengers
                        }

                        results.append(result)

                        cv2.imwrite(f'laneOutput/Det_{imgCounter}.png', roi)
                        imgCounter += 1

    return results


@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    """Upload an image and run YOLO detection."""
    try:
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        detections = run_detection(temp_path)

        os.remove(temp_path)

        return JSONResponse(content={"success": True, "results": detections})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.get("/")
def root():
    return {"message": "Helmet & License Plate Detection API is running!"}


@app.post("/detect-video/")
async def detect_video(file: UploadFile = File(...)):
    """
    Upload a video file and run YOLO-based video detection (Helmet, LP, Rider).
    Returns detection summary JSON.
    """
    try:
        
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        
        start_detecttion(file=temp_path)

        
        output_files = sorted(glob.glob("output/*.txt"))
        results = []

        for txt_path in output_files:
            with open(txt_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 5:
                    results.append({
                        "image_path": lines[0].strip(),
                        "helmet_status": lines[1].strip(),
                        "lp_status": lines[2].strip(),
                        "plate_number": lines[3].strip(),
                        "num_passengers": lines[4].strip()
                    })

        
        os.remove(temp_path)

        return JSONResponse(content={"success": True, "results": results})

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
