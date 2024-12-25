import os
from utils_ import cut_video_by_frame

'''
if __name__ == '__main__':
    video_path = '/data/hhk/data/swing_partial3/fabaa955-26d4-4020-ac16-c3a7035fbda5-1.mp4'
    from utils_ import cut_video_by_frame
    input_path = video_path
    start_frame = 14*60*60+50*60
    end_frame = start_frame + 60*10
    output_path = f'__video/fabaa955-26d4-4020-ac16-c3a7035fbda5-1__{start_frame}_{end_frame}.mp4'
    cut_video_by_frame(input_path, output_path, start_frame, end_frame)
'''



if __name__ == '__main__':
    from ultralytics import YOLO

    model_path = 'weights_m/best.pt'
    os.path.exists(model_path)
    model_detect = YOLO(model_path)
    # video_path = '__video/fabaa955-26d4-4020-ac16-c3a7035fbda5-1__52200_end_frame.mp4'
    video_path = '__video/fabaa955-26d4-4020-ac16-c3a7035fbda5-1__53400_54000.mp4'
    
if __name__ == '__main__':
    i = 0
    for r in model_detect.track(source=video_path, save=True, stream=True, save_txt=True, imgsz=1024*3, verbose=False):
        i += 1
        print(r)
        if i >= 100000:
            break
    i = 0
    for r in model_detect.predict(source=video_path, save=True, stream=True, save_txt=True, imgsz=1024*3, verbose=False):
        i += 1
        print(r)
    # new_track_thresh 
    i = 0
    for r in model_detect.track(source=video_path, save=True, stream=True, save_txt=True, imgsz=1024*3, verbose=False,  tracker='modified-bytetrack.yaml'):
        i += 1
        print(r)
        
    # new_track_thresh 
    i = 0
    # for r in model_detect.track(source=video_path, save=True, stream=True, save_txt=True, imgsz=1024*3, verbose=False,  tracker='_botsort.yaml'):
    for r in model_detect(source=video_path, save=True, stream=True, save_txt=True, imgsz=(1920, 1088), verbose=False,  tracker='_botsort.yaml'):
        
        if i % 100 == 0:
            print(r)
            print(i)
        i += 1
    print(r)
        # print(r)
        