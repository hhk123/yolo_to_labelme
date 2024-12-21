

import os
import json

import cv2
from ultralytics import YOLO

from utils_ import cut_video_by_frame, get_video_info

# model_path = "models/AI-ModelScope/YOLO11/yolo11x.pt"
model_path = 'weights/best.pt'
model_path = 'weights_m/best.pt'
os.path.exists(model_path)
model_detect = YOLO(model_path)



pic_path = '/mnt/e/tennis/1807-label/4dc57401-ad76-4cc7-b96b-f6a2da9fa648-1/4028.jpg'
res = model_detect(pic_path, imgsz=1024*4, save=True)
