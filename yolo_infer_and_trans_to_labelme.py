
import os
import json

import cv2
from ultralytics import YOLO

from utils_ import cut_video_by_frame, get_video_info

# model_path = "models/AI-ModelScope/YOLO11/yolo11x.pt"
model_path = 'weights/best.pt'
os.path.exists(model_path)
model_detect = YOLO(model_path)

# 原始模型80个分类
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# 新训练的模型，3个分类
LABELS = ['person', 'sports ball', 'tennis racket']

def yolo11_to_labelme(txt_file, img_file, save_dir, labels=LABELS):
          
    """
          
    将YOLO11格式的分割标签文件转换为Labelme格式的JSON文件。
          
    参数：
          
    - txt_file (str): YOLO11标签的txt文件路径。
          
    - img_file (str): 对应的图像文件路径。
          
    - save_dir (str): JSON文件保存目录。
          
    - labels (list): 类别标签列表。
          
    """
          
    # 读取图像，获取图像尺寸
          
    img = cv2.imread(img_file)
          
    height, width, _ = img.shape
          
 
          
    # 创建Labelme格式的JSON数据结构
          
    labelme_data = {
          
        "version": "2.4.4",
          
        "flags": {},
          
        "shapes": [],
          
        "imagePath": os.path.basename(img_file),
          
        "imageHeight": height,
          
        "imageWidth": width,
          
        "imageData": None  # 可以选择将图像数据转为base64后嵌入JSON
          
    }
          
 
          
    # 读取YOLO11标签文件
    
    with open(txt_file, "r") as file:
          
        for line in file.readlines():
          
            data = line.strip().split()
          
            class_id = int(data[0])  # 类别ID
          
            points = list(map(float, data[1:]))  # 获取多边形坐标
          
 
          
            # 将归一化坐标转换为实际像素坐标
          
            rectangle = []
          
            x, y, w, h = points # 中心点坐标
            x, w = x * width, w * width
            y, h = y * height, h * height
            left_up = [x-w/2, y-h/2]
            right_up = [x + w/2, y-h/2]
            right_bottom = [x + w/2, y + h/2]
            left_bottom = [x-w/2, y + h/2]
            # rectangle += [left_bottom, right_bottom, right_up, left_up]
            rectangle += [left_up, right_up, right_bottom, left_bottom]
          
 
          
            # 定义多边形区域
          
            shape = {
          
                "label": labels[class_id],  # 使用直接定义的类别名称
          
                "points": rectangle,
          
                "group_id": None,
          
                "shape_type": "rectangle",  # 分割使用多边形
          
                "flags": {}
          
            }
          
            labelme_data["shapes"].append(shape)
          
 
          
    # 保存为labelme格式的JSON文件
          
    save_path = os.path.join(save_dir, os.path.basename(txt_file).replace(".txt", ".json"))
          
    with open(save_path, "w") as json_file:
          
        json.dump(labelme_data, json_file, indent=4)
          
 
          
def convert_yolo11_to_labelme(txt_folder, img_folder, save_folder):
          
    """
          
    读取文件夹中的所有txt文件，将YOLO11标签转为Labelme的JSON格式。
          
    参数：
          
    - txt_folder (str): 存放YOLO11 txt标签文件的文件夹路径。
          
    - img_folder (str): 存放图像文件的文件夹路径。
          
    - save_folder (str): 保存转换后的JSON文件的文件夹路径。
          
    """
          
    labels = LABELS  # 直接使用定义好的标签
          
 
          
    if not os.path.exists(save_folder):
          
        os.makedirs(save_folder)
          
 
          
    for txt_file in os.listdir(txt_folder):
          
        if txt_file.endswith(".txt"):
          
            txt_path = os.path.join(txt_folder, txt_file)
          
            img_file = txt_file.replace(".txt", ".jpg")  # 假设图像为.png格式
          
            img_path = os.path.join(img_folder, img_file)
          
 
          
            # 检查图像文件是否存在
          
            if os.path.exists(img_path):
          
                yolo11_to_labelme(txt_path, img_path, save_folder, labels)
          
                print(f"已成功转换: {txt_file} -> JSON文件")
          
            else:
          
                print(f"图像文件不存在: {img_path}")
          

def video_to_images(input_path, output_folder, out_name, start_frame=0, frame_rate=1, quality=50):
    """
    将视频文件的每一帧保存为单独的图片文件。

    :param input_path: 输入视频文件的路径
    :param output_folder: 输出图片文件夹的路径
    :param frame_rate: 每秒保存的帧数（默认为每秒保存一帧）
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # 获取视频的帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算间隔帧数
    interval = int(video_fps / frame_rate)
    
    frame_count = 0
    saved_frame_count = start_frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        saved_frame_count += 1
        if frame_count % interval == 0:
            # 构建输出文件名
            output_path = os.path.join(output_folder, f"{out_name}_{saved_frame_count:04d}.jpg")
            # 保存帧为图片
            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            
        
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count} frames")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Saved {saved_frame_count} frames to {output_folder}")


def time_to_frame(video_path, time_start):
    fps, duration, n_frames, width, height = get_video_info(video_path)
    minute = round(time_start)
    second = int((time_start - minute) * 60)
    start_frame = int((minute*60+second) * fps)
    return start_frame

# 使用示例
if __name__ == '__main__':


    video_dir = '/data/hhk/data/swing_partial3' # 视频的根目录
    frame_save_dir0 = '/data/hhk/data/frames' # 切出的图片保留的文件夹
    os.makedirs(frame_save_dir0, exist_ok=True)
    
    # 视频名称（不带后缀）：从几分几秒开始截取。
    video_time_d = {'71e5574b-f98e-4724-a06d-61dd44d5dca3-1': 8.30,
                    '618e2e4c-1cac-486e-bc6e-2b3d2cae40e9-1': 9.38,
                    '98084b36-ddb9-43b7-94a1-75abf85707a5-1': 5.56,
                    'a12b477f-23b9-4736-9e4d-c7a0ca80426b-1': 19.25,
                    'd191412c-1f01-4727-a43d-825cdb899202-1': 12.28,
                    }
    
    # 切取到不同的文件夹
    for video_name, time_start in video_time_d.items():
        video_path = os.path.join(video_dir, video_name+'.mp4')
        output_folder = os.path.join(frame_save_dir0, video_name)
        os.makedirs(output_folder, exist_ok=True)
        out_name = video_name
        input_path = video_path
        start_frame = time_to_frame(video_path, time_start)
        end_frame = start_frame + 10000 # 截取10000帧
        video_output_path = os.path.join(frame_save_dir0, f"{video_name}_{start_frame}_{end_frame}.mp4")
         
        # cut_video_by_frame(input_path, video_output_path, start_frame, end_frame)
        video_to_images(video_output_path, output_folder, out_name, frame_rate=6, quality=40) #quality保存图片质量小于100，使图片不至于过大
    
    # yolo推理结果存放路径
    txt_folder = 'runs/detect/predict/labels'
    # 把抽帧的图片都进行yolo推理；并且把json放在和图片同一个文件夹
    import shutil
    for video_name, time_start in video_time_d.items():

        output_folder = os.path.join(frame_save_dir0, video_name)
        jpg_files = os.listdir(output_folder)
        jpg_files = [_ for _ in jpg_files if _.endswith('.jpg')]
        
        img_folder = output_folder
        save_folder = output_folder
        
        for jpg_file in jpg_files:
            jpg_path = os.path.join(output_folder, jpg_file)
            res = model_detect(jpg_path, save_txt=True, imgsz=1024*3) # 推理结果默认存放在 'runs/detect/predict/labels'
            
        convert_yolo11_to_labelme(txt_folder, img_folder, save_folder)
        # shutil.rmtree(txt_folder)

# if __name__ == '__main__':
    # pic_path = '/data/hhk/data/frames/d191412c-1f01-4727-a43d-825cdb899202-1_0011.jpg'  
    # res = model_detect(pic_path, save_txt=True)
    # res[0].save()
    # res[0].show()
    # yolo_text = 'runs/detect/predict/labels/d191412c-1f01-4727-a43d-825cdb899202-1_0011.txt'
    # txt_folder = 'runs/detect/predict/labels'
    # img_folder = '/data/hhk/data/frames'
    # save_folder = 'runs' # 转化成X-anylabling的json的格式的文件存放位置；
    # convert_yolo11_to_labelme(txt_folder, img_folder, save_folder)
    # txt_file = 'runs/detect/predict/labels/d191412c-1f01-4727-a43d-825cdb899202-1_0011.txt'