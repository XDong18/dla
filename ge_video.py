import os
import cv2
from os import listdir
from os.path import join, split


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)

file_dir='0322_val_0000'
img_list = listdir(file_dir)
img_list = sorted(img_list)
img_list.pop()
img_list.pop()

out_path = 'videos'
out_fn = 'knn.mp4'
mkdir(out_path)

video=cv2.VideoWriter(join(out_path, out_fn), cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 10, (1920,1200))  #定义保存视频目录名称及压缩格式，fps=10,像素为1280*720
for fn in img_list:
    img=cv2.imread(join(file_dir, fn))  #读取图片
    img=cv2.resize(img,(1920,1200)) #将图片转换为1280*720
    video.write(img)   #写入视频

video.release()
