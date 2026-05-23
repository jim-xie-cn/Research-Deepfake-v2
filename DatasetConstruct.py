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
np.set_printoptions(suppress=True, precision=8)

class CDatasetConstruct:

    def __init__(self,item, dataset='train',isRes=False,img_size=256, count = 32):
        self.m_item = item
        self.m_count = count
        self.m_isRes = isRes
        self.m_img_size = (img_size,img_size)
        self.m_dataset = dataset
        dest_path = self.get_dest_folder()
        Path(dest_path).mkdir(parents=True, exist_ok=True)

    def get_dest_folder(self):
        raw_file = self.m_item['Image Path']
        dest_file = raw_file.replace("/AI_Face_imagesV2/",f"/AI_Face_imagesV2/mfs/{self.m_dataset}/")
        path_name = Path(dest_file).parent
        base_name = Path(dest_file).stem
        dest_path = f"{path_name}/{base_name}"
        return dest_path

    def processed(self,file_name):
        return os.path.exists(file_name)

    def create_raw(self):
        dest_path = self.get_dest_folder()
        dest_file = f"{dest_path}/raw.exr"
        src_file = self.m_item['Image Path']
        if self.processed(dest_file):
            return [dest_file]

        img = CImageUtils.read_image(src_file)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {src_file}")
        
        img = img.astype(np.float32)
        img = CImageUtils.resize(img,self.m_img_size)
        CImageUtils.write_image(dest_file,img)
        
        return [dest_file]

    def create_local_alpha(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']
        dest_file = f"{dest_path}/alpha.exr"
        
        if self.processed(dest_file):
            return [dest_file]

        img = CImageUtils.read_image(src_file)
        img = CImageUtils.resize(img,self.m_img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        alpha = CImageAlpha( img, isRes = self.m_isRes ).get_raw_alpha()
        CImageUtils.write_image(dest_file, alpha)  

        return [dest_file]

    def create_svd_alpha(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']
        img = CImageUtils.read_image(src_file)
        img = CImageUtils.resize(img,self.m_img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dest_file_list = []
        for i in range(0,self.m_count):
            dest_file = f"{dest_path}/svd-alpha-{i}.exr"
            dest_file_list.append(dest_file)

        if self.processed(dest_file):
            return dest_file_list

        alpha = CImageAlpha(img,isRes=self.m_isRes).get_svd_alpha(count=self.m_count)
        for file_name,alpha in zip(dest_file_list,alpha):
            CImageUtils.write_image(file_name, alpha)
        
        return dest_file_list

    def create_image_mfs(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']
        
        img = CImageUtils.read_image(src_file)
        img = CImageUtils.resize(img,self.m_img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dest_file_list = []
        for i in range(0,3):
            dest_file = f"{dest_path}/image-mfs-{i}.exr"
            dest_file_list.append(dest_file)

        if self.processed(dest_file):
            return dest_file_list

        A_mfs,D_mfs,F_mfs = CImageMFS(img,isRes=self.m_isRes,count=self.m_count).get_image_mfs()
        for file_name,mfs in zip(dest_file_list,[A_mfs,D_mfs,F_mfs]):
            CImageUtils.write_image(file_name, mfs)

        return dest_file_list

    def create_mfs_image(self):
        dest_path = self.get_dest_folder()
        src_file = self.m_item['Image Path']
        
        img = CImageUtils.read_image(src_file)
        img = CImageUtils.resize(img,self.m_img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dest_file_list = []
        for i in range(0,3):
            dest_file = f"{dest_path}/mfs-image-{i}.exr"
            dest_file_list.append(dest_file)
        
        if self.processed(dest_file):
            return dest_file_list

        R_mfs,G_mfs,B_mfs = CImageMFS(img,isRes=self.m_isRes,count=self.m_count).get_mfs_image()
                                                
        for file_name,mfs in zip(dest_file_list,[R_mfs,G_mfs,B_mfs]):
            CImageUtils.write_image(file_name, mfs)   
        
        return dest_file_list

def main():
    df_train = pd.read_csv("../AI-Face-FairnessBench/dataset/train.csv")
    item = df_train.iloc[0]
    test = CDatasetConstruct(item, 'train', isRes=True, img_size=256, count=256)
    t0 = time.time()
    test.create_raw()
    test.create_local_alpha()
    test.create_svd_alpha()
    test.create_image_mfs()
    test.create_mfs_image()
    print("Time used:",time.time() - t0)

if __name__ == "__main__":
    main()
