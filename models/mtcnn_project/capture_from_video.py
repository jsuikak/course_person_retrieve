#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 15:37
# @Author  : Ssozh
# @FileName: build_train_bank.py
# @Software: PyCharm
# @Blog    ：https://www.cnblogs.com/SsoZhNO-1/
import sys
import os
import cv2
import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import datetime

from PIL import Image
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='take a picture from video')
parser.add_argument('--video_name', '-i', default='IPartment2.mp4',type=str,help='the dataset source')
parser.add_argument('--sampling_freq',default=10,type=int,help="sampling frequency in the video")
parser.add_argument('--only_frame',default=False,action='store_true',help="only save the frames without face")

args = parser.parse_args()
SAMPLING_FREQ = args.sampling_freq
# 当前路径
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
video_data_path = Path(os.path.join(ROOT_PATH,"video",args.video_name))
print("video path: {}".format(video_data_path))

if not video_data_path.exists():
    raise FileNotFoundError("{}没有放在./video/目录下".format(args.video_name))
save_path = os.path.join(ROOT_PATH, "output")
save_path = Path(save_path)
if not save_path.exists():
    save_path.mkdir()
save_path = save_path / "video_output"
if not save_path.exists():
    save_path.mkdir()
save_path = save_path / args.video_name.split(".")[0]
if not save_path.exists():
    save_path.mkdir()
else:
    print("You have detected this video")
    del_list = os.listdir(str(save_path))
    for f in del_list:
        file_path = os.path.join(str(save_path), f)
        if os.path.isfile(file_path):
            os.remove(file_path)


# 获取video
video = cv2.VideoCapture(str(video_data_path))
# 视频帧率
fps = video.get(cv2.CAP_PROP_FPS)
# 视频总帧数
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
# 视频宽度,视频高度
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("VIDEO INFO:")
print("FPS0"
      ".:{0}\nframeCount:{1}\nsize:{2}x{3}\n".format(fps,frameCount,size[0],size[1]))

mtcnn = MTCNN()  # ValueError: Object arrays cannot be loaded when allow_pickle=False ->pip install numpy==1.16.2
success, frame = video.read()
index = 0
tic = time.time()

while success:
    if index % 10 == 0:
        p = Image.fromarray(frame[..., ::-1]) # rgb ->bgr
        if args.only_frame:
            filename = str(save_path / '{}.jpg'.format(index))
            cv2.imencode('.jpg', frame)[1].tofile(filename)
        else:   
            try:
                # mtcnn.align()函数会返回一个裁剪好的人脸图像，如果没有检测到人脸，则会抛出异常
                warped_face = np.array(mtcnn.align(p))[..., ::-1]  # bgr -> rgb
                # print('net forward time: {:.4f}'.format(time.time() - tic))
                filename = str(save_path / '{}.jpg'.format(index))
                cv2.imencode('.jpg', warped_face)[1].tofile(filename)
                # print('face captured')
            except:
                print('{} frame:no face captured'.format(index))
    index= index + 1
    success, frame = video.read()

# 释放视频资源
video.release()
