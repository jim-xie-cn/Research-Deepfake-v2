import os
import cv2
import argparse
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from ImageConstruct import CImageUtils,CImageAlpha,CImageMFS
from mtcnn import MTCNN
from face_tool import get_face_image
np.set_printoptions(suppress=True, precision=8)

import threading
detector_lock = threading.Lock()
detector = MTCNN(device="/GPU:1")

class CDatasetConstruct:

    def __init__(self,item,dest_root,dataset='train',img_size=256, count = 32):
        self.m_item = item
        self.m_count = count
        self.m_img_size = (img_size,img_size)
        self.m_dataset = dataset
        self.m_dest_root = dest_root
        dest_path = self.get_dest_folder()
        Path(dest_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_face_image(src_file):
        with detector_lock:
            face_image = get_face_image(detector, src_file)
            if len(face_image) > 0:
                return face_image.astype(np.float32)
            else:
                return face_image

    @staticmethod
    def compute_mean_std_image(image_paths):
        sum_img = None
        sq_sum_img = None
        count = 0
        for path in tqdm(image_paths):
            img = CDatasetConstruct.read_face_image(path)
            if len(img) == 0:
                continue
            img = CImageUtils.resize(img, (256, 256))
            img = img.astype(np.float64)
            
            if sum_img is None:
                sum_img = np.zeros_like(img, dtype=np.float64)
                sq_sum_img = np.zeros_like(img, dtype=np.float64)
            sum_img += img
            sq_sum_img += img ** 2
            count += 1
        mean_img = sum_img / count
        std_img = np.sqrt(sq_sum_img / count - mean_img ** 2)
        return mean_img.astype(np.float32), std_img.astype(np.float32)

    def get_dest_folder(self):
        raw_file = self.m_item['Image Path']
        dest_file = raw_file.replace("/AI_Face_imagesV2/",f"{self.m_dest_root}/AI_Face_imagesV2/{self.m_dataset}/")
        path_name = Path(dest_file).parent
        base_name = Path(dest_file).stem
        dest_path = f"{path_name}/{base_name}"
        return dest_path

    def processed(self,file_list):
        return all(os.path.exists(f) for f in file_list)

    def create_raw(self):
        dest_path = self.get_dest_folder()
        dest_file = f"{dest_path}/raw.exr"
        src_file = self.m_item['Image Path']
        
        if self.processed([dest_file]):
            return [dest_file]

        img = CDatasetConstruct.read_face_image(src_file)
        if len(img) == 0:
            print(f"Failed to read image: {src_file}")
            return []
        
        img = CImageUtils.resize(img,self.m_img_size)
        CImageUtils.write_image(dest_file,img)
        
        return [dest_file]

    def create_local_alpha(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']
        dest_file = f"{dest_path}/alpha.exr"
        
        if self.processed([dest_file]):
            return [dest_file]

        img = CDatasetConstruct.read_face_image(src_file)
        if len(img) == 0:
            print(f"Failed to read image: {src_file}")
            return []

        img = CImageUtils.resize(img,self.m_img_size)
        alpha = CImageAlpha(img).get_raw_alpha()
        CImageUtils.write_image(dest_file, alpha)  

        return [dest_file]

    def create_svd(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']
        
        dest_file_list = []
        for i in range(0,self.m_count):
            dest_file = f"{dest_path}/svd-{i}.exr"
            dest_file_list.append(dest_file)

        if self.processed(dest_file_list):
            return dest_file_list

        img = CDatasetConstruct.read_face_image(src_file)
        if len(img) == 0:
            print(f"Failed to read image: {src_file}")
            return []
        img = CImageUtils.resize(img,self.m_img_size)

        alpha = CImageAlpha(img,svd_count=self.m_count, max_scales=32).get_svd()
        for file_name,alpha_image in zip(dest_file_list,alpha):
            CImageUtils.write_image(file_name, alpha_image)

        return dest_file_list

    def create_gray_alpha(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']

        dest_file_list = []
        for i in range(0,self.m_count):
            dest_file = f"{dest_path}/svd-gray-{i}.exr"
            dest_file_list.append(dest_file)

        if self.processed(dest_file_list):
            return dest_file_list

        img = CDatasetConstruct.read_face_image(src_file)
        if len(img) == 0:
            print(f"Failed to read image: {src_file}")
            return []
        img = CImageUtils.resize(img,self.m_img_size)

        alpha = CImageAlpha(img,svd_count=self.m_count, max_scales=32).get_gray_alpha()
        for file_name,alpha_image in zip(dest_file_list,alpha):
            CImageUtils.write_image(file_name, alpha_image)

        return dest_file_list

    def create_binary_alpha(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']

        dest_file_list = []
        for i in range(0,self.m_count):
            dest_file = f"{dest_path}/svd-bin-{i}.exr"
            dest_file_list.append(dest_file)

        if self.processed(dest_file_list):
            return dest_file_list

        img = CDatasetConstruct.read_face_image(src_file)
        if len(img) == 0:
            print(f"Failed to read image: {src_file}")
            return []
        img = CImageUtils.resize(img,self.m_img_size)

        alpha = CImageAlpha(img,svd_count=self.m_count, max_scales=32).get_bin_alpha()
        for file_name,alpha_image in zip(dest_file_list,alpha):
            CImageUtils.write_image(file_name, alpha_image)

        return dest_file_list
    
    def create_rgb_alpha(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']

        dest_file_list = []
        for i in range(0,self.m_count):
            dest_file = f"{dest_path}/svd-rgb-{i}.exr"
            dest_file_list.append(dest_file)

        if self.processed(dest_file_list):
            return dest_file_list

        img = CDatasetConstruct.read_face_image(src_file)
        if len(img) == 0:
            print(f"Failed to read image: {src_file}")
            return []
        img = CImageUtils.resize(img,self.m_img_size)

        alpha = CImageAlpha(img,svd_count=self.m_count, max_scales=32).get_rgb_alpha()
        for file_name,alpha_image in zip(dest_file_list,alpha):
            CImageUtils.write_image(file_name, alpha_image)
        
        return dest_file_list

    def create_mfs_image(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']

        dest_file_list = []
        img_names = ["gray", "binary", "R", "G", "B"]
        for i in img_names:
            dest_file = f"{dest_path}/mfs-image-{i}.exr"
            dest_file_list.append(dest_file)
        
        if self.processed(dest_file_list):
            return dest_file_list

        img = CDatasetConstruct.read_face_image(src_file)
        if len(img) == 0:
            print(f"Failed to read image: {src_file}")
            return []
        img = CImageUtils.resize(img,self.m_img_size)

        MFS = CImageMFS(img,q_count=self.m_count)
        MFS.parse()
        mfs_images = MFS.get_mfs_images()
        for name,file_name in zip(img_names,dest_file_list):
            CImageUtils.write_image(file_name, mfs_images[name])   
        
        return dest_file_list

def cala_mean():
    df_train = pd.read_csv("../AI-Face-FairnessBench/dataset/train.csv")
    df_real = df_train[df_train['Target'] == 0]
    df_fake = df_train[df_train['Target'] == 1]

    real_file = df_real.iloc[0]['Image Path']
    real,_ = CDatasetConstruct.compute_mean_std_image([real_file])
    CImageUtils.write_image("./images/real", real)

    fake_file = df_fake.iloc[0]['Image Path']
    fake,_ = CDatasetConstruct.compute_mean_std_image([fake_file])
    CImageUtils.write_image("./images/fake", fake)

    mean_img,std_img = CDatasetConstruct.compute_mean_std_image(df_real['Image Path'])
    CImageUtils.write_image("./images/real-mean", mean_img)
    CImageUtils.write_image("./images/real-std", std_img)
  
    mean_img,std_img = CDatasetConstruct.compute_mean_std_image(df_fake['Image Path'])
    CImageUtils.write_image("./images/fake-mean", mean_img)                    
    CImageUtils.write_image("./images/fake-std", std_img)

def main():
    #return cala_mean()
    df_train = pd.read_csv("../AI-Face-FairnessBench/dataset/train.csv")
    item = df_train.iloc[0]
    test = CDatasetConstruct(item,"/disk/b", 'train', img_size=256, count=256)
    t0 = time.time()
    test.create_raw()
    test.create_local_alpha()
    test.create_svd()
    test.create_gray_alpha()
    test.create_binary_alpha()
    test.create_rgb_alpha()
    test.create_mfs_image()
    print("Time used:",time.time() - t0)

if __name__ == "__main__":
    main()
