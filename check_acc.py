import json
import os

gt_dir = '/mnt/e/tennis/checked_frame/4d5144bf-2e8a-4f3a-acf7-7964bb6b6596-1'
label_dir = '/mnt/e/tennis/frame2/1807-label/4d5144bf-2e8a-4f3a-acf7-7964bb6b6596-1'

json_names = os.listdir(label_dir)
json_names = [name for name in json_names if name.endswith('.json')]
n = len(json_names)
acc = 0
for json_name in json_names:
    label_path = os.path.join(label_dir, json_name)
    gt_path = os.path.join(gt_dir, json_name)
    with open(label_path, 'r') as f:
        lable_d = json.load(f)
    with open(gt_path, 'r') as f:
        gt_d = json.load(f) 
    if lable_d == gt_d:
        acc += 1
        
acc/n