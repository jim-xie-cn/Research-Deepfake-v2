import cv2
import pandas as pd
import os,sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from ImageConstruct import CImageUtils,CImageAlpha,CImageMFS
from skimage.metrics import structural_similarity as ssim
from joblib import Parallel, delayed
import hashlib

def detect_image(detector,source_file):
    node = None
    face_image = None
    image = CImageUtils.read_image(source_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:  
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = detector.detect_faces(image,fit_to_image= False)
    max_confidence = 0
    for item in result:
        x,y,w,h = item['box']
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > image.shape[1]:
            x = image.shape[1]
        if y > image.shape[0]:
            y = image.shape[0]
        item['box'] = [x,y,w,h]
        if min(w,h) > 64: #图像不能太小
            if item['confidence'] > max_confidence:
                max_confidence = item['confidence']
                node = item
                x, y, w, h = node['box']
                x_end = min(x + w, image.shape[1]) 
                y_end = min(y + h, image.shape[0])  
                cropped_img = image[y:y_end, x:x_end]
                face_image = cropped_img

    return node, face_image

def extract_center_crop(image, cx, cy, width, height):
    """
    从 image 中以中心 (cx, cy) 为中心裁剪出 (width x height) 尺寸的图像区域，
    如果边界不足，则截取边界内的最大可能区域。
    """
    h_img, w_img = image.shape[:2]
    x1 = max(cx - width // 2, 0)
    y1 = max(cy - height // 2, 0)
    x2 = min(cx + width // 2, w_img)
    y2 = min(cy + height // 2, h_img)
    
    if x2 - x1 < width:
        if x1 == 0:
            x2 = min(width, w_img)
        elif x2 == w_img:
            x1 = max(w_img - width, 0)

    if y2 - y1 < height:
        if y1 == 0:
            y2 = min(height, h_img)
        elif y2 == h_img:
            y1 = max(h_img - height, 0)

    cropped = image[y1:y2, x1:x2]
    return cropped

def get_face_image(detector,file_name):
    try:
        image = CImageUtils.read_image(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        node,face_image = detect_image(detector,file_name)
        if node != None:
            return face_image
        #if node:
        #    x, y, w, h = node['box']
        #    cx, cy = x + w // 2, y + h // 2
        #    target_width = max(2 * w, 256)
        #    target_height = max(2 * h, 256)
        #    face_crop = extract_center_crop(image, cx, cy, target_width, target_height)
        #   return face_crop
    except:
        print("failed to get face image",file_name)

    return np.array([[]])

def main():
    file_name = "./images/real-mean.exr"
    detector = MTCNN(device="cpu")
    face = get_face_image(detector,file_name)
    print(face)

if __name__ == "__main__":
    main()
