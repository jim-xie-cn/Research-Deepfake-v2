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
import multiprocessing as mp

np.set_printoptions(suppress=True, precision=8)

def if_processed(folder,file_name):
    if os.path.exists(f"{folder}{file_name}.png"):
        return True
    return False

def process_one_batch(params,indexes):
    action = params['action']
    dest_root = params['dest_root']
    dataset = params['dataset']
    df_data = params['data']
    all_result = []   
    try:
        for index in indexes:
            item = df_data.iloc[index].to_dict()
            test = CDatasetConstruct(item,dest_root,dataset, count = 64)
            if action == 'raw':
                dest_file_list = test.create_raw()
            elif action == 'local-alpha':
                dest_file_list = test.create_local_alpha()
            elif action == 'rgb-alpha':
                dest_file_list = test.create_rgb_alpha()
            elif action == 'mfs':
                dest_file_list = test.create_mfs_image()
            else:
                dest_file_list = []
                print("unknown action",action)
            for f in dest_file_list:
                item['new_file'] = f
                all_result.append(item.copy())

    except Exception:
        logging.exception("failed %d", index)

    return all_result

def main(dest_root,dataset,worker,action):
    if action == 'raw':
        file_name = f"../AI-Face-FairnessBench/dataset/{dataset}.csv"
    elif action in ['local-alpha','rgb-alpha','mfs']:
        file_name = f"./dataset/raw-{dataset}.csv"
    else:
        print(f"Action is not supported {action},{dataset}")
    
    df_data = pd.read_csv(file_name)
    print(df_data['Image Path'])
    print(file_name)
    #df_data = df_data.head(5)   
    
    params = {}
    params['action'] = action
    params['dataset'] = dataset
    params['dest_root'] = dest_root
    params['data'] = df_data
    
    #df_data = df_data.tail(50)
    total_count = len(df_data)   #总共有多少个记录
    batch_size = 1               #每个batch有多少个记录

    results = []
    ctx = mp.get_context("spawn")
    indexes_list = [list(range(start, min(start + batch_size, total_count))) for start in range(0, total_count, batch_size)]
    with ThreadPoolExecutor(max_workers=worker) as executor:
    #with ProcessPoolExecutor(max_workers=worker,mp_context=ctx) as executor:
    #with ProcessPoolExecutor(max_workers=worker) as executor:
        futures = [executor.submit(process_one_batch, params, batch) for batch in indexes_list]
        for future in tqdm(as_completed(futures), total=len(futures),desc=f"{dataset} {action}"):
            results.extend(future.result())
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"./dataset/{action}-{dataset}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch fractal or multifractal calculate.")
    parser.add_argument("--root", type=str, default="/disk/b", help="dest images root")
    parser.add_argument("--dataset", choices=['train','test'], default='test', help="train or test")
    parser.add_argument("--worker", type=int, default=8, help="int 0 to 64.")
    parser.add_argument("--action", choices=['raw','local-alpha','rgb-alpha','mfs'], default='local-alpha', help="raw/local-alpha/rgb-alpha/mfs")
    args = parser.parse_args()
    
    print(f"Root selected: {args.root}")
    print(f"Dataset selected: {args.dataset}")
    print(f"Worker selected : {args.worker}")
    print(f"Worker selected : {args.action}")

    main(args.root.strip(),args.dataset.strip(),args.worker,args.action.strip())
