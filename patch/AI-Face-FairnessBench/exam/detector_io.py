#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
detector_io.py
==============
Shared helpers for the *evaluation-only* reproduction scripts
(``eval_table5_generalization.py`` and ``eval_figure6_postprocessing.py``).

It hides the three different ways a detector is built in this repository:

  * 10 registry detectors  ->  training/detectors/__init__.py::DETECTOR[name]
      forward:  model({'image': x}, inference=True)['cls']
      output :  (N, 1) raw logits  (sigmoid)     -- except srm/core -> (N, 2) softmax
  * 'vit'      ->  torchvision vit_b_16, heads[0] = Linear(768, 1)   (train_test_vit.py)
  * 'UnivFD'   ->  CLIP ViT-L/14 backbone + Linear(768, 1)           (train_test_clip.py)

All of them are wrapped in a uniform :class:`Detector` object exposing::

    det.logits(image_batch)          -> np.ndarray  (raw logits)
    det.make_transform(methods=[])   -> albumentations.Compose  (test-time)
    det.activation                   -> 'sigmoid' | 'softmax'

IMPORTANT: the registry detectors hard-code ``./pretrained/xception-b5690688.pth``
in their ``build_backbone``.  We therefore ``chdir`` into ``training/`` on import
so those relative paths resolve.  Run the eval scripts from anywhere; paths you
pass on the command line should be absolute or relative to ``training/``.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Locate the repo and make `training/` importable + the working directory.
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
os.chdir(TRAINING_DIR)  # so detectors find ./pretrained/xception-b5690688.pth

SOFTMAX_MODELS = {"srm", "core"}
REGISTRY_MODELS = {
    "xception", "efficientnet", "f3net", "spsl", "srm", "core",
    "ucf", "daw_fdd", "dag_fdd", "fair_df_detector",
}
SPECIAL_MODELS = {"vit", "UnivFD"}
ALL_MODELS = sorted(REGISTRY_MODELS | SPECIAL_MODELS)


def transform_kind(model: str) -> str:
    """Which normalisation / resize family a model expects."""
    if model == "vit":
        return "vit"
    if model == "UnivFD":
        return "clip"
    return "xception"  # 256x256, mean/std = 0.5


class Detector:
    def __init__(self, model_name: str, module, activation: str, kind: str):
        self.name = model_name
        self.module = module          # the nn.Module
        self.activation = activation  # 'sigmoid' | 'softmax'
        self.kind = kind              # 'xception' | 'vit' | 'clip'

    # -- forward ----------------------------------------------------------- #
    @torch.no_grad()
    def logits(self, imgs: torch.Tensor) -> np.ndarray:
        """Return raw logits as a numpy array; shape (N,) or (N, 2)."""
        if self.name in REGISTRY_MODELS:
            out = self.module({"image": imgs}, inference=True)["cls"]
        else:  # vit / UnivFD take the raw image tensor
            out = self.module(imgs)
        out = out.detach().cpu().numpy()
        if out.ndim == 2 and out.shape[1] == 1:
            out = out[:, 0]
        return out

    # -- transforms -------------------------------------------------------- #
    def make_transform(self, methods=None):
        from transform import (get_albumentations_transforms,
                               get_albumentations_transforms_vit_clip)
        methods = methods or [""]
        if self.kind == "xception":
            return get_albumentations_transforms(methods)
        return get_albumentations_transforms_vit_clip(methods, model_type=self.kind)


def _load_state(module: nn.Module, ckpt_path: str, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and not any(
            k.startswith(("backbone", "encoder", "heads", "model", "fc"))
            for k in state):
        state = state["state_dict"]
    # strip possible DataParallel 'module.' prefix
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"  [detector_io] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [detector_io] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")


def build_detector(model_name: str, checkpoint: str, device=None) -> Detector:
    """Instantiate `model_name`, load `checkpoint`, return a ready Detector."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kind = transform_kind(model_name)
    activation = "softmax" if model_name in SOFTMAX_MODELS else "sigmoid"

    if model_name in REGISTRY_MODELS:
        from detectors import DETECTOR
        module = DETECTOR[model_name]()
    elif model_name == "vit":
        from torchvision import models
        module = models.vit_b_16(weights=None)
        module.heads[0] = nn.Linear(in_features=768, out_features=1)
    elif model_name == "UnivFD":
        from models.clip import clip

        class CLIPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model, _ = clip.load("ViT-L/14", device="cpu")
                self.fc = nn.Linear(768, 1)

            def forward(self, x):
                return self.fc(self.model.encode_image(x))
        module = CLIPModel()
    else:
        raise ValueError(f"unknown model '{model_name}'. Options: {ALL_MODELS}")

    if checkpoint:
        _load_state(module, checkpoint, device)
    module.to(device).eval()
    return Detector(model_name, module, activation, kind)
