
# coco是从左上角是原点
import os
import json
import shutil

from tqdm import tqdm
from PIL import Image

def get_label_d():
    lablestr = '''
    0: person
    1: bicycle
    2: car
    3: motorcycle
    4: airplane
    5: bus
    6: train
    7: truck
    8: boat
    9: traffic light
    10: fire hydrant
    11: stop sign
    12: parking meter
    13: bench
    14: bird
    15: cat
    16: dog
    17: horse
    18: sheep
    19: cow
    20: elephant
    21: bear
    22: zebra
    23: giraffe
    24: backpack
    25: umbrella
    26: handbag
    27: tie
    28: suitcase
    29: frisbee
    30: skis
    31: snowboard
    32: sports ball
    33: kite
    34: baseball bat
    35: baseball glove
    36: skateboard
    37: surfboard
    38: tennis racket
    39: bottle
    40: wine glass
    41: cup
    42: fork
    43: knife
    44: spoon
    45: bowl
    46: banana
    47: apple
    48: sandwich
    49: orange
    50: broccoli
    51: carrot
    52: hot dog
    53: pizza
    54: donut
    55: cake
    56: chair
    57: couch
    58: potted plant
    59: bed
    60: dining table
    61: toilet
    62: tv
    63: laptop
    64: mouse
    65: remote
    66: keyboard
    67: cell phone
    68: microwave
    69: oven
    70: toaster
    71: sink
    72: refrigerator
    73: book
    74: clock
    75: vase
    76: scissors
    77: teddy bear
    78: hair drier
    79: toothbrush
    '''
    labels = [_.strip().rstrip() for _ in lablestr.split('\n') if _]
    labels = [_ for _ in labels if _]

    num_cls_d = {}
    cls_num_d = {}
    for _ in labels:
        num, cls = _.split(': ')
        num_cls_d[num] = cls
        cls_num_d[cls] = num
    print(len(cls_num_d), len(num_cls_d))
    # 增加自己标注的类别
    person_num = cls_num_d['person']
    cls_num_d['p'] = person_num
    cls_num_d['r'] = person_num
    cls_num_d["r'"] = person_num
    
    ball = cls_num_d['sports ball'] 
    cls_num_d['q'] = ball
    cls_num_d['ball'] = ball

    racket = cls_num_d['tennis racket'] 
    cls_num_d['p'] = racket   

    return cls_num_d, num_cls_d

cls_num_d, num_cls_d = get_label_d()


def shape_to_coco(shape, image_size):
    '''
    points = [[1511.3800048828125, 586.1802978515625], [1667.766357421875, 586.1802978515625], [1667.766357421875, 930.5776977539062], [1511.3800048828125, 930.5776977539062]]
    left_up, right_up, right_bottom, left_bottom
    '''
    w, h = image_size
    points = shape['points']
    left_up, right_up, right_bottom, left_bottom = points
    left = left_up[0]
    up = left_up[1]
    right = right_bottom[0]
    bottom = right_bottom[1]

    wn = (right-left)/w
    hn = (bottom-up)/h
    center_xn = (right+left)/2/w
    center_yn = (bottom+up)/2/h

    wn = round(wn, 7)
    hn = round(hn, 7)
    center_xn = round(center_xn, 7)
    center_yn = round(center_yn, 7)

    cls = shape['label']
    if cls not in cls_num_d:
        print(cls)
        print('not in cls_num_d')
    num = cls_num_d[cls]
    coco_str = f"{num} {center_xn} {center_yn} {wn} {hn}"
    return coco_str


def json_to_coco_text(json_path, image_size):

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    shapes = json_data['shapes']
    shape = shapes[0]
    text_str = ''
    for shape in shapes:
        coco_str = shape_to_coco(shape, image_size)
        text_str += coco_str + '\n'
    return text_str


def labelme_to_yolo11(labelme_dir, out_txt_dir, out_jpg_dir):
    def list_filenames(labelme_dir):
        vidoe_name = os.path.basename(labelme_dir)
        file_names = os.listdir(labelme_dir)
        file_names = [_.split('.')[0] for _ in file_names if _.endswith('.json')]
        return vidoe_name, file_names
    
    vidoe_name, file_names = list_filenames(labelme_dir)

    for file_name in tqdm(file_names):

        json_path = os.path.join(labelme_dir, file_name+'.json')
        jpg_path = os.path.join(labelme_dir, file_name+'.jpg')
        image = Image.open(jpg_path)
        image_size = image.size # (宽, 高)
        text_str = json_to_coco_text(json_path, image_size)

        new_file_name = vidoe_name+'__'+file_name
        text_path = os.path.join(out_txt_dir, new_file_name + '.txt')
        new_jpg_path = os.path.join(out_jpg_dir, new_file_name + '.jpg')
        with open(text_path, 'w') as f:
            f.write(text_str)
        
        _ = shutil.copy(jpg_path, new_jpg_path)
    n_pics = len(os.listdir(out_jpg_dir))
    n_txt = len(os.listdir(out_txt_dir))
    print('图片数量: ', n_pics)
    print('标注数量: ', n_txt)

def filter_yolo11_cls(txt_str, keep_d):

    coco_strs = txt_str.split('\n')
    num_point_strs = [_.split(' ', 1) for _ in coco_strs]
    filtered_txt_str = ''
    for i in range(len(coco_strs)):
        if num_point_strs[i][0] in keep_d:
            filtered_txt_str += keep_d[num_point_strs[i][0]] + ' ' + num_point_strs[i][1]
            filtered_txt_str += '\n'
    return filtered_txt_str

def get_keep_d():
    keep_cls = ['person', 'sports ball', 'tennis racket']
    keep_num = [cls_num_d[_] for _ in keep_cls]
    keep_num.sort()
    keep_d = {keep_num[i]: str(i) for i in range(len(keep_num))}
    return keep_d

keep_d = get_keep_d()

def filter_yolo11_cls_dir(input_dir, output_dir, keep_d):
    txt_names = os.listdir(input_dir)
    for txt_name in tqdm(txt_names):
        txt_path = os.path.join(input_dir, txt_name)
        new_txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, 'r') as f:
            txt_str = f.read()
        filtered_txt_str = filter_yolo11_cls(txt_str, keep_d)
        if len(filtered_txt_str) < 4:
            continue
        with open(new_txt_path, 'w') as f:
            f.write(filtered_txt_str)
    print(f'共{len(os.listdir(output_dir))}个新标签文件')
    
if __name__ == '__main__':

    out_jpg_dir = '/mnt/e/tennis2/jpgs'
    out_label_dir = '/mnt/e/tennis2/labels'
    out_label_filter_dir = '/mnt/e/tennis2/labels_filter'
    
    os.makedirs(out_jpg_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_label_filter_dir, exist_ok=True)
    labelme_root_dir = '/mnt/e/tennis/frame'
    labelme_dirs = os.listdir(labelme_root_dir)
    labelme_dirs = [_ for _ in labelme_dirs if not _.endswith('.zip')]
    for labelme_dir0 in labelme_dirs:
        labelme_dir = os.path.join(labelme_root_dir, labelme_dir0)
        labelme_to_yolo11(labelme_dir, out_label_dir, out_jpg_dir)
        filtered_txt_str = filter_yolo11_cls_dir(out_label_dir, out_label_filter_dir, keep_d)
        
    '''
    tmp = os.listdir(labelme_dir)
    tmp = [_ for _ in tmp if _.endswith('.json')]
    tmp[386]
    '''
    train_label_dir = '/mnt/e/tennis/coco8/coco/labels/train2017'
    val_label_dir = '/mnt/e/tennis/coco8/coco/labels/val2017'
    train_filter_label_dir = '/mnt/e/tennis/coco8/coco/label_filter/train2017'
    val_filter_label_dir = '/mnt/e/tennis/coco8/coco/label_filter/val2017'
    os.makedirs(train_filter_label_dir, exist_ok=True)
    os.makedirs(val_filter_label_dir, exist_ok=True)
    filtered_txt_str = filter_yolo11_cls_dir(train_label_dir, train_filter_label_dir, keep_d)
    filtered_txt_str = filter_yolo11_cls_dir(val_label_dir, val_filter_label_dir, keep_d)
    # os.path.getsize(tmp_video_path)

if __name__ == '__main__':
    # 复制筛选过的图片到文件夹
    import os, shutil
    from tqdm import tqdm

    train_filter_label_dir = '/mnt/e/tennis/coco8/coco_filtered/labels/train2017'
    val_filter_label_dir = '/mnt/e/tennis/coco8/coco_filtered/labels/val2017'
    train_image_dir = '/mnt/e/tennis/coco8/coco/images/train2017'
    val_image_dir = '/mnt/e/tennis/coco8/coco/images/val2017'
    len(os.listdir(train_image_dir))
    len(os.listdir(val_image_dir))
    train_filter_image_dir = '/mnt/e/tennis/coco8/coco_filtered/images/train2017'
    val_filter_image_dir = '/mnt/e/tennis/coco8/coco_filtered/images/val2017'
    os.makedirs(train_filter_image_dir, exist_ok=True)
    os.makedirs(val_filter_image_dir, exist_ok=True)
    train_txts = os.listdir(train_filter_label_dir) 
    len(train_txts)
    for train_txt in tqdm(train_txts):
        file_name = train_txt.split('.')[0]
        pic_name = file_name + '.jpg'
        source_pic_path = os.path.join(train_image_dir, pic_name)
        target_pic_path = os.path.join(train_filter_image_dir, pic_name)
        _ = shutil.copy(source_pic_path, target_pic_path)
        
    val_txts = os.listdir(val_filter_label_dir)
    for val_txt in tqdm(val_txts):
        file_name = val_txt.split('.')[0]
        pic_name = file_name + '.jpg'
        source_pic_path = os.path.join(val_image_dir, pic_name)
        target_pic_path = os.path.join(val_filter_image_dir, pic_name)
        _ = shutil.copy(source_pic_path, target_pic_path)
        
    print(len(os.listdir(train_filter_image_dir)), len(os.listdir(val_filter_image_dir)))
