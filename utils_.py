import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import VideoFileClip
from collections import defaultdict

def cut_video_by_frame(input_path, output_path, start_frame, end_frame):
    """
    根据帧数截取视频的一部分并保存为新的MP4文件。

    :param input_path: 输入视频文件的路径
    :param output_path: 输出视频文件的路径
    :param start_frame: 截取片段的开始帧数
    :param end_frame: 截取片段的结束帧数
    """
    # 加载视频文件
    video = VideoFileClip(input_path)
    
    # 获取视频的帧率
    fps = video.fps
    
    # 将帧数转换为时间
    start_time = start_frame / fps
    end_time = end_frame / fps
    
    # 截取视频片段
    clip = video.subclip(start_time, end_time)
    
    # 保存截取的视频片段为新的MP4文件
    clip.write_videofile(output_path)
    

def get_video_info(input_path):
    video = VideoFileClip(input_path) 
    # 获取视频的帧率
    fps = video.fps
    duration = video.duration
    n_frames = duration * fps
    width = video.w
    height = video.h
    
    return fps, duration, n_frames, width, height





# 创建一个示例的 NumPy 数组（这里假设您已经有一个 NumPy 数组）
# 这里创建一个简单的黑色图像
data = np.zeros((100, 100, 3), dtype=np.uint8)
def np_to_jpg(np_array, jpg_path):
    # 将 NumPy 数组转换为 PIL 图像
    img = Image.fromarray(np_array, 'RGB')

    # 保存图像为 JPG 文件
    img.save(jpg_path)
    
def get_person_boxes(detect_res):
    person_d = defaultdict(list)
    for i_frame, detect_res0 in enumerate(detect_res):
        boxes = detect_res0.boxes
        xyxy = boxes.xyxy
        if len(xyxy) == 0:
            continue
        if boxes.cls is None or boxes.id is None:
            continue
        cls = boxes.cls.tolist()
        ids = boxes.id.tolist()
        xyxy = boxes.xyxy
        for i in range(len(cls)):
            if cls[i] == 0:
                # if ids[i] not in person_d:
                #     person_d[ids[i]] = [(i_frame, xyxy[i])]
                xyxy_ = xyxy[i].tolist()
                xyxy_ = list(map(int, xyxy_))
                person_d[ids[i]].append((i_frame, xyxy_))
    return person_d
def clip_person(strim_output_path, xyxys, output_path=''):
    if not output_path:
        video_name = os.path.splitext(os.path.basename(strim_output_path))[0]
        output_path = os.path.join('_output/person', f"{video_name}.mp4")
    # 读取原始视频
    max_width, max_height = 100, 100
    for _, (x1, y1, x2, y2) in xyxys:
        max_width = max(max_width, x2-x1)
        max_height = max(max_height, y2-y1)
    max_wh = int(max(max_width, max_height))
    video_path = strim_output_path
    

    cap = cv2.VideoCapture(video_path)
    # 设置输出视频的编解码器、帧率等参数
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (max_wh, max_wh))

    # 矩形区域的坐标（左上角和右下角）
    
    n_frame = -1
    while cap.isOpened():
        ret, frame = cap.read()
        n_frame += 1
        if not ret:
            break
        if n_frame > len(xyxys)-1:
            break
        source_n_frame, (x1, y1, x2, y2) = xyxys[n_frame]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if source_n_frame > n_frame:
            continue
        img = np.zeros((max_wh, max_wh, 3), dtype=np.uint8)
        # 提取矩形区域
        cropped_frame = frame[y1:y2, x1:x2, :]
        img[0:(y2-y1), 0:(x2-x1)] = cropped_frame
        # 写入输出视频
        out.write(img)
       
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return n_frame
def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    
    参数:
    box1: 第一个边界框，格式为 (x_min, y_min, x_max, y_max)
    box2: 第二个边界框，格式为 (x_min, y_min, x_max, y_max)
    
    返回:
    iou: 两个边界框的IoU
    """
    # 计算交集区域
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # 如果没有交集，返回0
    if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
        return 0.0

    # 计算交集面积
    inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

    # 计算每个边界框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area

    return iou
