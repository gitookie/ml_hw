import numpy as np
import torch
import cv2 as cv
import os
from utils.process import *

# groundtruch里的左右是指我看到的左右，比如我看到的在左边的眼睛其实是他的右眼

def load_and_cut(dir_path, txt_path, one_channel=1):
    """加载图像, 裁取人脸并resize, 同时返回integral imgs。
    one_channel指示是否转化成单通道的
    最终以numpy数组的形式返回"""

    faces = []
    integral_imgs = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        all_lines = 2000
        for line in lines[:2000]:   
            line = line.strip().split()
            img_name = line[0]
            bbox = [float(coord) for coord in line[1:]]
            bbox = [int(x) for x in bbox]
            img_path = os.path.join(dir_path, img_name)
            image = cv.imread(img_path)
            
            # 定义人脸的框的左上角和右下角的顶点
            # 注意这里直接取所有x坐标的最值，所有y坐标的最值即可
            # 如果想当然的认为左眼的x，y坐标最小，右眼的x坐标最大，嘴巴的y坐标最大，
            # 遇到倾斜的人脸就会出问题
            left_top_x = min(bbox[i] for i in range(0, len(bbox), 2))
            left_top_y = min(bbox[i] for i in range(1, len(bbox), 2))
            right_bottom_x = max(bbox[i] for i in range(0, len(bbox), 2))
            right_bottom_y = max(bbox[i] for i in range(1, len(bbox), 2))
            

            # 这里根据给定的眼睛，鼻子，嘴巴的坐标，稍微扩大一点得到近似的人脸的框
            # (考虑到只用这几个坐标的话，框会偏小)
            # 扩展时注意不能超出图像边界

            # x方向扩展(这里图像的第一个维度表示y方向，第二个维度表示x方向)
            len_x = right_bottom_x - left_top_x
            left_top_x = max(0, left_top_x - int(0.5 * 0.8 * len_x))
            right_bottom_x = min(image.shape[1], right_bottom_x + int(0.5 * 0.8 * len_x))
            
            # y方向扩展
            len_y = right_bottom_y - left_top_y
            left_top_y = max(0, left_top_y - int(0.5 * 0.8 * len_y))
            right_bottom_y = min(image.shape[0], right_bottom_y + int(0.5 * 0.8 * len_y))

            # 调试用
            # cv.circle(image, (left_top_x, left_top_y), 5, (0, 0, 255), -1)
            # cv.circle(image, (right_bottom_x, right_bottom_y), 5, (0, 255, 0), -1)
            
            # 裁出人脸部分
            cur_image = image[left_top_y:right_bottom_y, left_top_x:right_bottom_x]
            
            # if cur_image.shape == None: 调试用
            #     print(img_name)
            
            # 统一大小
            cur_image = cv.resize(cur_image, (304, 312))
            # print('cur image shape', cur_image.shape)
            if one_channel:
                cur_image = cv.cvtColor(cur_image, cv.COLOR_BGR2GRAY)
                integral_img = get_integral_image_without_channels(cur_image)
            else:
                integral_img = get_integral_image(cur_image)
            faces.append(cur_image)
            integral_imgs.append(integral_img)
            if idx % 5 == 0:
                print(f'loading data: {idx + 1} / {all_lines}')
            idx += 1
            # if cur_image.shape[2] == 1:调试用。经调试，所有的图像都是三通道的
            #     print('error')
            # cv.imshow('Image', cur_image)
            # cv.waitKey(0)
    faces_array = np.stack(faces, axis=0)
    integrals = np.stack(integral_imgs, axis=0)
    return faces_array, integrals
            
def get_negative_samples(dir_path, txt_path, one_channel=1):
    """从数据里抽一些负样本, 最终以numpy数组的形式返回"""

    neg_samples = []
    integral_imgs = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        all_lines = 1000
        idx = 0
        for line in lines[:1000]:   
            line = line.strip().split()
            img_name = line[0]
            bbox = [float(coord) for coord in line[1:]]
            bbox = [int(x) for x in bbox]
            img_path = os.path.join(dir_path, img_name)
            image = cv.imread(img_path)

            left_top_x = min(bbox[i] for i in range(0, len(bbox), 2))
            left_top_y = min(bbox[i] for i in range(1, len(bbox), 2))
            right_bottom_x = max(bbox[i] for i in range(0, len(bbox), 2))
            right_bottom_y = max(bbox[i] for i in range(1, len(bbox), 2))
            

            # 这里根据给定的眼睛，鼻子，嘴巴的坐标，稍微扩大一点得到近似的人脸的框
            # (考虑到只用这几个坐标的话，框会偏小)
            # 扩展时注意不能超出图像边界

            # x方向扩展(这里图像的第一个维度表示y方向，第二个维度表示x方向)
            len_x = right_bottom_x - left_top_x
            left_top_x = max(0, left_top_x - int(0.5 * 0.8 * len_x))
            right_bottom_x = min(image.shape[1], right_bottom_x + int(0.5 * 0.8 * len_x))
            
            # y方向扩展
            len_y = right_bottom_y - left_top_y
            left_top_y = max(0, left_top_y - int(0.5 * 0.8 * len_y))
            right_bottom_y = min(image.shape[0], right_bottom_y + int(0.5 * 0.8 * len_y))
            rows, cols, channels = image.shape
            if rows - 50 > right_bottom_y and cols - 50 > right_bottom_x:
                sample = image[rows - 50:, cols - 50:]
                sample = cv.resize(sample, [304, 312])
                if one_channel:
                    sample = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
                    integral_img = get_integral_image_without_channels(sample)
                else:
                    integral_img = get_integral_image(sample)
                neg_samples.append(sample)
                integral_imgs.append(integral_img)
                # cv.imshow('neg sample', sample)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
            if idx % 5 == 0:
                print(f'getting negative samples:{idx + 1}/{all_lines}')
            idx += 1
    neg_faces = np.stack(neg_samples, axis=0)
    integrals = np.stack(integral_imgs, axis=0)
    return neg_faces, integrals
        
"""a = get_negative_samples(dir_path='Caltech_WebFaces', txt_path='WebFaces_GroundThruth.txt')
print(a.shape)
cv.imshow('example', a[0])
print(a[0].shape)
cv.waitKey(0)
cv.destroyAllWindows()"""

# get_negative_samples(dir_path='Caltech_WebFaces', txt_path='WebFaces_GroundThruth.txt')