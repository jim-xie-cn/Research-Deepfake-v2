#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_videoswin_svd_autofeature.py
Compatible with Transformers 4.39.3
- CSV dataset
- Real-time SVD frame generation (32 frames)
- AutoFeatureExtractor + AutoModelForVideoClassification
- Train/Validation split from train.csv
- Multi-GPU (DataParallel)
- Accuracy, F1, AUC metrics
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from transformers import AutoFeatureExtractor, AutoModelForVideoClassification

# --------------------------
# SVD Decomposer
# --------------------------
class CImageDecompose:
    def __init__(self, img: np.ndarray):
        self.m_img = np.asarray(img, dtype=np.float32)

    @staticmethod
    def _check_count(count):
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("count must be integer >= 1")

    def get_svd_auto(self, count=32, mode='fade_energy', curve='smooth'):
        self._check_count(count)
        x = self.m_img
        frames = []
        def smoothstep(t): return t*t*(3-2*t)
        if x.ndim == 2:
            x = x[..., None]
        H, W, C = x.shape
        for c in range(C):
            U, S, Vt = np.linalg.svd(x[..., c], full_matrices=False)
            r = len(S)
            gamma = 1.2
            ts = np.linspace(0,1,count)
            per_ch_frames = []
            for t in ts:
                a = smoothstep(t) if curve=='smooth' else t
                p = S/S[0]
                base = np.clip(1 - a * p, 0, 1)
                w = (1 - a) * base
                rec = (U*(S*w))@Vt
                per_ch_frames.append(rec.astype(np.float32))
            if c==0:
                ch_frames = per_ch_frames
            else:
                for i in range(count):
                    ch_frames[i] = np.stack([ch_frames[i], per_ch_frames[i]], axis=2)
        return ch_frames

# --------------------------
# Dataset
# --------------------------
class SVDFakeDataset(Dataset):
    def __init__(self, csv_file, img_size=256, count=32, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.count = count
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['Image Path']
        label = int(row['Target'])
        if img_path in self.cache:
            frames = self.cache[img_path]
        else:
            img = np.array(Image.open(img_path).convert('RGB').resize((self.img_size,self.img_size)))
            decomposer = CImageDecompose(img)
            frames = decomposer.get_svd_auto(count=self.count)
            frames = np.stack(frames, axis=0) # (T,H,W,C)
            self.cache[img_path] = frames
        # normalize to [0,1] and convert to PIL Images for AutoFeatureExtractor
        frames = [Image.fromarray((f*255).astype(np.uint8)) for f in frames]
        return frames, torch.tensor(label, dtype=torch.long)

# --------------------------
# Training / Evaluation
# --------------------------
def train_one_epoch(model, feature_extractor, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for frames, labels in tqdm(loader, desc="Train"):
        # frames: list of list of PIL Images (batch_size, T)
        batch_inputs = [frames_per_sample for frames_per_sample in frames]
        # Convert using feature extractor
        inputs = feature_extractor(batch_inputs, return_tensors="pt").to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*len(labels)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0
    return total_loss/len(loader.dataset), acc, f1, auc

def validate(model, feature_extractor, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Val"):
            batch_inputs = [frames_per_sample for frames_per_sample in frames]
            inputs = feature_extractor(batch_inputs, return_tensors="pt").to(device)
            labels = labels.to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()*len(labels)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0
    return total_loss/len(loader.dataset), acc, f1, auc

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Dataset
    full_train = SVDFakeDataset(args.train_csv, img_size=args.img_size, count=args.frames)
    val_len = int(len(full_train)*args.val_ratio)
    train_len = len(full_train)-val_len
    train_dataset, val_dataset = random_split(full_train, [train_len,val_len])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)
    test_dataset = SVDFakeDataset(args.test_csv, img_size=args.img_size, count=args.frames)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, persistent_workers=True)

    # Model
    model_name = "MCG-NJU/videoswin-tiny-patch244-window877-kinetics400-pt"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForVideoClassification.from_pretrained(model_name, num_labels=2)
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc, train_f1, train_auc = train_one_epoch(model, feature_extractor, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} AUC: {train_auc:.4f}")

        val_loss, val_acc, val_f1, val_auc = validate(model, feature_extractor, val_loader, criterion, device)
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    # Test
    test_loss, test_acc, test_f1, test_auc = validate(model, feature_extractor, test_loader, criterion, device)
    print(f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f} AUC: {test_auc:.4f}")


if __name__=="__main__":
    main()
