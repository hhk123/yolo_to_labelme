import os
from utils_ import cut_video_by_frame

if __name__ == '__main__':
    from ultralytics import YOLO

    model_path = 'weights_m/best.pt'
    os.path.exists(model_path)
    model_detect = YOLO(model_path)
    video_path = '/data/hhk/data/swing_partial3/fabaa955-26d4-4020-ac16-c3a7035fbda5-1.mp4'
    i = 0
    for r in model_detect.predict(source=video_path, save=True, stream=True, save_txt=True, imgsz=1024*3, verbose=False):
        i += 1
        print(r)
        if i >= 100000:
            break
