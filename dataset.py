#!/usr/bin/env python
# coding: utf-8
import cv2,logging,os
import argparse,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import glob
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
import torch
from pathlib import Path
from ImageConstruct import CImageUtils,CImageAlpha,CImageMFS
from DatasetConstruct import CDatasetConstruct
np.set_printoptions(suppress=True, precision=8)

def if_processed(folder,file_name):
    if os.path.exists(f"{folder}{file_name}.png"):
        return True
    return False

def process_one_batch(params,indexes):
    action = params['action']
    dataset = params['dataset']
    df_data = params['data']

    try:
        for index in indexes:
            item = df_data.iloc[index]
            test = CDatasetConstruct(item,dataset,isRes = True, count = 256)
            #test.create_raw()
            #test.create_local_alpha()
            #test.create_svd_alpha()
            #test.create_image_mfs()
            #test.create_mfs_image()
            
            if action == 'raw':
                test.create_raw()
            elif action == 'local-alpha':
                test.create_local_alpha()
            elif action == 'svd-alpha':
                test.create_svd_alpha()
            elif action == 'mfs-image':
                test.create_image_mfs()
            elif action == 'image-mfs':
                test.create_mfs_image()
            else:
                print("unknown action",action)

    except Exception:
        logging.exception("failed %d", index)

def main(dataset,worker,action):
    if dataset == 'train':
        df_data = pd.read_csv("../AI-Face-FairnessBench/dataset/train.csv")
    elif dataset == 'test':
        df_data = pd.read_csv("../AI-Face-FairnessBench/dataset/test.csv")  
    
    params = {}
    params['action'] = action
    params['dataset'] = dataset
    params['data'] = df_data

    batch_count = len(df_data)   #总共有多少个batch
    batch_size = 50     #每个batch有多少个记录

    indexes_list = [list(range(start, min(start + batch_size, batch_count))) for start in range(0, batch_count, batch_size)]
    #with ThreadPoolExecutor(max_workers=worker) as executor:
    with ProcessPoolExecutor(max_workers=worker) as executor:
        futures = [executor.submit(process_one_batch, params, batch) for batch in indexes_list]
        for future in tqdm(as_completed(futures), total=len(futures),desc=f"{dataset} {action}"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch fractal or multifractal calculate.")
    parser.add_argument("--dataset", choices=['train','test'], default='train', help="train or test")
    parser.add_argument("--worker", type=int, default=16, help="int 0 to 64.")
    parser.add_argument("--action", choices=['raw','local-alpha','svd-alpha','mfs-image','image-mfs'], default='raw', help="raw/local-alpha/svd-alpha/mfs-image/image-mfs")
    args = parser.parse_args()
    
    print(f"Root selected: {args.dataset}")
    print(f"Worker selected : {args.worker}")
    print(f"Worker selected : {args.action}")

    main(args.dataset.strip(),args.worker,args.action.strip())
