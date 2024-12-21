import os 
import shutil
from tqdm import tqdm
from ultralytics import YOLO

from trans_tools import convert_yolo11_to_labelme




if __name__ == '__main__':
    model_path = 'weights/best.pt'
    os.path.exists(model_path)
    model_detect = YOLO(model_path)
    img_father_folder = '/mnt/e/tmp/tmp2' # 图片文件夹在此目录下，图片文件夹下是图片
    
    txt_folder = 'runs/detect/predict/labels'
    shutil.rmtree('runs') # 删除中间结果路径
    img_folder_basenames = os.listdir(img_father_folder)
    for img_folder_basename in img_folder_basenames:
        img_folder = os.path.join(img_father_folder, img_folder_basename)

        # 把抽帧的图片都进行yolo推理；并且把json放在和图片同一个文件夹
        jpg_files = os.listdir(img_folder)
        jpg_files = [_ for _ in jpg_files if _.endswith('.jpg')]
        print(img_folder, len(jpg_files))


        for jpg_file in tqdm(jpg_files):
            jpg_path = os.path.join(img_folder, jpg_file)

            res = model_detect(jpg_path, save_txt=True, imgsz=1024*3, verbose=False) # 推理结果默认存放在 'runs/detect/predict/labels'
        
        convert_yolo11_to_labelme(txt_folder, img_folder, img_folder)
        shutil.rmtree(txt_folder) # 删除中间结果路径




