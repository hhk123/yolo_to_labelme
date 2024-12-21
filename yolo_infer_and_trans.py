import os 
from tqdm import tqdm
from ultralytics import YOLO

from trans_tools import convert_yolo11_to_labelme

model_path = 'weights_m/best.pt'
os.path.exists(model_path)
model_detect = YOLO(model_path)
if __name__ == '__main__':
    frame_save_dir0 = '/mnt/'
    # # yolo推理结果存放路径
    # txt_folder = 'runs/detect/predict/labels'
    img_folder = '/mnt/e/tennis/youxuan/2403-label'
    
if __name__ == '__main__':
    txt_folder =  '/mnt/e/tennis/youxuan/2403-label_txt'
    os.makedirs(txt_folder, exist_ok=True)
    # 把抽帧的图片都进行yolo推理；并且把json放在和图片同一个文件夹
    jpg_files = os.listdir(img_folder)
    jpg_files = [_ for _ in jpg_files if _.endswith('.jpg')]
    print(len(jpg_files))


    for jpg_file in tqdm(jpg_files):
        jpg_path = os.path.join(img_folder, jpg_file)

        res = model_detect(jpg_path, save_txt=True, imgsz=1024*3) # 推理结果默认存放在 'runs/detect/predict/labels'
    
if __name__ == '__main__':
    txt_folder = 'runs/detect/predict3/labels'
    save_folder = '/mnt/e/tennis/youxuan/2403-label_jsons'
    os.makedirs(save_folder, exist_ok=True)
    len(os.listdir(txt_folder))
    len(os.listdir(save_folder))
    
    convert_yolo11_to_labelme(txt_folder, img_folder, save_folder)