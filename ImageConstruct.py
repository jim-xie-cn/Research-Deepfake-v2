import argparse
import numpy as np
import time,os
from tqdm import tqdm
import pandas as pd
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import matplotlib.pyplot as plt
from FreeAeonFractal.FAImageFourier import CFAImageFourier
from FreeAeonFractal.FAImage import CFAImage
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS
from sklearn.decomposition import PCA
from ImageDecompose import CImageDecompose

#GPU version
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU as CFAImageLAC
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

#CPU version
#from FreeAeonFractal.FAImageLAC import CFAImageLAC
#from FreeAeonFractal.FAImageFD import CFAImageFD
#from FreeAeonFractal.FAImageMFS import CFAImageMFS

np.set_printoptions(suppress=True, precision=8)

class CImageUtils:

    @staticmethod
    def resize(img,size_shape=(256,256)):
         return cv2.resize(img, size_shape, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def write_image(file_name, img):
        if not file_name.lower().endswith(".exr"):
            file_name = file_name + ".exr"
        params = [
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_ZIP  #无损压缩
        ]
        cv2.imwrite(file_name, img.astype(np.float32), params)

    @staticmethod
    def read_image(file_name):
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read EXR: {file_name}")
        return img.astype(np.float32)

    @staticmethod
    def get_svd_images(img, count=64,isRes=False):
        return CImageDecompose(img).get_svd_auto(count=count)
    
    @staticmethod
    def split_image(img):
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        return R, G, B

    @staticmethod
    def get_gray(img):
        R, G, B = CImageUtils.split_image(img)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        return gray

    @staticmethod
    def get_binary(gray):
        thresh = gray.mean()
        bin_img = gray > thresh
        return bin_img

    @staticmethod
    def merge_image(R, G, B):
        R = np.asarray(R)
        G = np.asarray(G)
        B = np.asarray(B)
        img = np.stack([R, G, B], axis=2)
        return img
        
    @staticmethod
    def normalize_by_channel(imgs, dtype=np.float32, eps=1e-8):
        if isinstance(imgs, np.ndarray) and imgs.ndim >= 2:
            single_image = True
            imgs_list = [imgs]
        else:
            single_image = False
            imgs_list = list(imgs)
        # -----------------------------
        # 2. NaN / Inf 处理
        # -----------------------------
        processed_imgs = []
        for img in imgs_list:
            img = np.asarray(img)
            finite_mask = np.isfinite(img)
            pos_inf_val = np.nanmax(img[finite_mask]) if np.any(finite_mask) else 1.0
            img = np.nan_to_num(
                img,
                nan=0.0,
                posinf=pos_inf_val,
                neginf=0.0)
            processed_imgs.append(img)
        if single_image:
            img = processed_imgs[0]
            if img.ndim == 3:
                if img.shape[-1] <= 10:
                    min_val = img.min(axis=(0, 1), keepdims=True)
                    max_val = img.max(axis=(0, 1), keepdims=True)
                else:
                    # CHW
                    min_val = img.min(axis=(1, 2), keepdims=True)
                    max_val = img.max(axis=(1, 2), keepdims=True)
            else:
                min_val = img.min()
                max_val = img.max()

            denom = max_val - min_val
            denom = np.where(denom < eps, 1.0, denom)

            norm_img = (img - min_val) / denom
            return norm_img.astype(dtype)
            
        stack = np.stack(processed_imgs, axis=0)
        global_min = stack.min(axis=(0, 1, 2), keepdims=True)
        global_max = stack.max(axis=(0, 1, 2), keepdims=True)
        denom = global_max - global_min
        denom = np.where(denom < eps, 1.0, denom)
        norm_stack = (stack - global_min) / denom
        
        return [img.astype(dtype) for img in norm_stack]
        
    @staticmethod
    def normalize(imgs, dtype=np.float32):
        if isinstance(imgs, np.ndarray) and imgs.ndim >= 2:
            single_image = True
            imgs_list = [imgs]
        else:
            single_image = False
            imgs_list = list(imgs)
            
        processed_imgs = []
        for img in imgs_list:
            finite_mask = np.isfinite(img)
            pos_inf_val = np.nanmax(img[finite_mask]) if np.any(finite_mask) else 1.0
            img = np.nan_to_num(img, nan=0.0, posinf=pos_inf_val, neginf=0.0)
            processed_imgs.append(img)

        if single_image:
            img = processed_imgs[0]
            min_val = img.min()
            max_val = img.max()
            if max_val == min_val:
                return np.zeros_like(img, dtype=dtype)
            norm_img = (img - min_val) / (max_val - min_val)
            return norm_img.astype(dtype)
        else:
            global_min = min(img.min() for img in processed_imgs)
            global_max = max(img.max() for img in processed_imgs)
            if global_max == global_min:
                return [np.zeros_like(img, dtype=dtype) for img in processed_imgs]
            norm_imgs = [(img - global_min) / (global_max - global_min) for img in processed_imgs]
            return [img.astype(dtype) for img in norm_imgs]
    
    @staticmethod
    def enhance_contrast(img, alpha=1.5, dtype=np.float32):
        img = np.asarray(img, dtype=np.float32)
        img = (img - 128.0) * alpha + 128.0
        img = np.clip(img, 0, 255)
        return img.astype(dtype)
    
    @staticmethod
    def display(imgs, auto_normalize=True, cols=8, cell_size=2.0, title_prefix=""):
        if isinstance(imgs, dict):
            items = [(f"{title_prefix}{k}", v) for k, v in imgs.items()]
        elif isinstance(imgs, (list, tuple)):
            items = [(f"{title_prefix}{i}", img) for i, img in enumerate(imgs)]
        else:
            raise TypeError("imgs 必须是 list/tuple 或 dict")
    
        if auto_normalize and len(items) > 0:
            try:
                if any(np.size(im) > 0 and np.nanmax(np.asarray(im)) > 1 for _, im in items):
                    normed = CImageUtils.normalize_by_channel([im for _, im in items])
                    items = [(t, im) for (t, _), im in zip(items, normed)]
            except Exception:
                pass
    
        n = len(items)
        if n == 0:
            return
    
        rows = int(np.ceil(n / cols))
        figsize = (cols * cell_size, rows * cell_size)
        plt.figure(figsize=figsize)
    
        for i, (title, img) in enumerate(items):
            ax = plt.subplot(rows, cols, i + 1)
            vmax = np.max(img) if np.size(img) > 0 else 1.0
            if img.ndim == 3:
                ax.imshow(img, vmin=0, vmax=vmax)
            else:
                ax.imshow(img, vmin=0, vmax=vmax, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            ax.set_aspect("equal")
    
        plt.tight_layout()
        plt.show()

class CImageMFS:
    
    def __init__(self, image, q_count=128):
        self.m_image = image
        self.m_q_count = q_count
        self.m_list_svd = []
        self.m_df_mfs = pd.DataFrame()

    @staticmethod
    def df_to_image(df: pd.DataFrame, q_count: int = None, n_count: int = None):
        required_cols = ["q", "n", "a(q)", "d(q)", "f(a)"]
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            raise ValueError(f"缺少列: {miss}")
    
        # 固定 q 轴
        if q_count is None:
            q_sorted = np.sort(df["q"].dropna().unique())
        else:
            # 若你q是linspace(-10,10,q_count)，建议直接固定这个轴
            q_sorted = np.linspace(-10, 10, q_count)
    
        # 固定 n 轴
        if n_count is None:
            n_sorted = np.sort(df["n"].dropna().unique())
        else:
            n_sorted = np.arange(n_count)
    
        R_df = df.pivot(index="q", columns="n", values="a(q)").reindex(index=q_sorted, columns=n_sorted)
        G_df = df.pivot(index="q", columns="n", values="d(q)").reindex(index=q_sorted, columns=n_sorted)
        B_df = df.pivot(index="q", columns="n", values="f(a)").reindex(index=q_sorted, columns=n_sorted)
    
        R = R_df.to_numpy(dtype=np.float32)
        G = G_df.to_numpy(dtype=np.float32)
        B = B_df.to_numpy(dtype=np.float32)
    
        rgb = np.dstack([R, G, B]).astype(np.float32)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        return rgb

    @staticmethod
    def get_batch_mfs(img_list, q_count=128, batch_size=64, with_progress=False):
        """
        返回: mfs_list[list[pd.DataFrame]]
        每个 df 期望包含列: q, a(q), d(q), f(a)
        """
        mfs_list = []
        total = len(img_list)
        q_list = np.linspace(-10, 10, q_count)
        start_idx = 0

        while start_idx < total:
            end_idx = min(start_idx + batch_size, total)
            batch_imgs = img_list[start_idx:end_idx]

            result_batch = CFAImageMFS.get_batch_mfs(
                batch_imgs,
                q_list=q_list,
                with_progress=with_progress
            )

            for item in result_batch:
                df_mass, df_fit, df_spec = item[0], item[1], item[2]
                if isinstance(df_spec, pd.DataFrame) and (not df_spec.empty):
                    df_spec = (df_spec.rename(columns={
                            "tau": "t(q)",
                            "Dq": "d(q)",
                            "alpha": "a(q)",
                            "f_alpha": "f(a)"
                        })
                        .drop(columns=["D1"], errors="ignore")
                    )
                    df_spec = df_spec.iloc[:q_count].copy()
                    mfs_list.append(df_spec)
                else:
                    mfs_list.append(pd.DataFrame())

            start_idx = end_idx

        return mfs_list

    def parse_svd(self, auto_normalize=True):
        self.m_list_svd = CImageDecompose(self.m_image).get_svd_auto(count=self.m_q_count)
        if auto_normalize:
            self.m_list_svd = CImageUtils.normalize_by_channel(self.m_list_svd)

    def parse_mfs(self):
        total_img_list = []
        for image in self.m_list_svd:
            R, G, B = CImageUtils.split_image(image)
            gray = CImageUtils.get_gray(image)
            binary = CImageUtils.get_binary(gray)
            total_img_list.extend([gray, binary, R, G, B])

        total_mfs_list = CImageMFS.get_batch_mfs(
            total_img_list, q_count=self.m_q_count, batch_size=64
        )
        
        if len(total_mfs_list) < len(self.m_list_svd) * 5:
            raise ValueError("total_mfs_list mismatch (CFAImageMFS output unstable)")
    
        result = []
        for i in range(len(self.m_list_svd)):
            base = i * 5
            names = ["gray", "binary", "R", "G", "B"]
            dfs = []

            for j, name in enumerate(names):
                df_tmp = total_mfs_list[base + j]
                if not isinstance(df_tmp, pd.DataFrame):
                    df_tmp = pd.DataFrame()
                df_tmp = df_tmp.copy()
                df_tmp["img"] = name
                dfs.append(df_tmp)

            df_mfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
            df_mfs["n"] = i
            result.append(df_mfs)

        self.m_df_mfs = pd.concat(result, ignore_index=True).reset_index(drop=True)

    def parse(self):
        self.parse_svd()
        self.parse_mfs()

    def get_svd(self):
        return self.m_list_svd

    def get_mfs(self):
        return self.m_df_mfs

    def get_mfs_images(self, size_shape=None):
        result = {}
        df_mfs = self.get_mfs()
        n_count = len(self.m_list_svd)
        for img, df_tmp in df_mfs.groupby("img"):
            result[img] = CImageMFS.df_to_image(df_tmp, q_count=self.m_q_count, n_count=n_count)
            if size_shape:
                result[img] = CImageUtils.resize(result[img], size_shape)
        return result

class CImageFD:
    def __init__(self,image,svd_count=128,max_scales=32):
        self.m_image=image
        self.m_svd_count=svd_count
        self.m_max_scales=max_scales
        self.m_list_svd=[]
        self.m_df_fd=pd.DataFrame()
        
    @staticmethod
    def df_to_image(df, n_count=None):
        required=["n","kind","gray","R","G","B"]
        miss=[c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"缺少列: {miss}")
    
        kind_map={"bc":0,"dbc":1,"sdbc":2}
        if n_count is None:
            n_sorted=np.sort(df["n"].dropna().unique())
        else:
            n_sorted=np.arange(n_count)
        img=np.zeros((4,len(n_sorted),3),dtype=np.float32)
        for _,row in df.iterrows():
            n=int(row["n"])
            c=kind_map.get(row["kind"],-1)
            if c<0 or n>=len(n_sorted):
                continue
            img[0,n,c]=row["gray"]
            img[1,n,c]=row["R"]
            img[2,n,c]=row["G"]
            img[3,n,c]=row["B"]
        return img
        
    @staticmethod
    def _safe_fd_value(item):
        return item.get("fd",np.nan) if isinstance(item,dict) else np.nan

    @staticmethod
    def _batch_call(func,img_list,max_scales=32,batch_size=64,with_progress=False):
        result=[];total=len(img_list);start_idx=0
        while start_idx<total:
            end_idx=min(start_idx+batch_size,total)
            result.extend(func(img_list[start_idx:end_idx],max_scales=max_scales,with_progress=with_progress))
            start_idx=end_idx
        return result

    @staticmethod
    def get_batch_fd(img_list,max_scales=32,batch_size=64,with_progress=False):
        gray_rgb_list=[];gray_bin_list=[]
        for image in img_list:
            R,G,B=CImageUtils.split_image(image)
            gray=CImageUtils.get_gray(image)
            gray_rgb_list.extend([gray,R,G,B])
            gray_bin_list.extend([
                CImageUtils.get_binary(gray),
                CImageUtils.get_binary(R),
                CImageUtils.get_binary(G),
                CImageUtils.get_binary(B)
            ])
        fd_bc_all=CImageFD._batch_call(CFAImageFD.get_batch_bc,gray_bin_list,max_scales=max_scales,batch_size=batch_size,with_progress=with_progress)
        fd_dbc_all=CImageFD._batch_call(CFAImageFD.get_batch_dbc,gray_rgb_list,max_scales=max_scales,batch_size=batch_size,with_progress=with_progress)
        fd_sdbc_all=CImageFD._batch_call(CFAImageFD.get_batch_sdbc,gray_rgb_list,max_scales=max_scales,batch_size=batch_size,with_progress=with_progress)

        result=[]
        names=["gray","R","G","B"]

        for i in range(len(img_list)):
            base=i*4

            row_bc={
                "n":i,
                "kind":"bc",
                "gray":CImageFD._safe_fd_value(fd_bc_all[base+0]) if base+0<len(fd_bc_all) else np.nan,
                "R":CImageFD._safe_fd_value(fd_bc_all[base+1]) if base+1<len(fd_bc_all) else np.nan,
                "G":CImageFD._safe_fd_value(fd_bc_all[base+2]) if base+2<len(fd_bc_all) else np.nan,
                "B":CImageFD._safe_fd_value(fd_bc_all[base+3]) if base+3<len(fd_bc_all) else np.nan
            }

            row_dbc={"n":i,"kind":"dbc"}
            row_sdbc={"n":i,"kind":"sdbc"}

            for j,nm in enumerate(names):
                idx=base+j
                row_dbc[nm]=CImageFD._safe_fd_value(fd_dbc_all[idx]) if idx<len(fd_dbc_all) else np.nan
                row_sdbc[nm]=CImageFD._safe_fd_value(fd_sdbc_all[idx]) if idx<len(fd_sdbc_all) else np.nan

            result.append(pd.DataFrame([row_bc,row_dbc,row_sdbc]))

        return result

    def parse_svd(self,auto_normalize=True):
        self.m_list_svd=CImageDecompose(self.m_image).get_svd_auto(count=self.m_svd_count)
        if auto_normalize:
            self.m_list_svd=CImageUtils.normalize_by_channel(self.m_list_svd)

    def parse_fd(self,batch_size=64,with_progress=False):
        total_fd_list=CImageFD.get_batch_fd(self.m_list_svd,max_scales=self.m_max_scales,batch_size=batch_size,with_progress=with_progress)
        result=[]
        for df_fd in total_fd_list:
            if isinstance(df_fd,pd.DataFrame) and not df_fd.empty:
                result.append(df_fd)
        self.m_df_fd=pd.concat(result,ignore_index=True) if result else pd.DataFrame(columns=["n","kind","gray","R","G","B"])

    def parse(self,batch_size=64,with_progress=False):
        self.parse_svd()
        self.parse_fd(batch_size=batch_size,with_progress=with_progress)

    def get_svd(self):
        return self.m_list_svd

    def get_fd(self):
        return self.m_df_fd

    '''
    H: 4 (Gray,R,G,B)
    W: n 
    C: 3 (bc,dbc,sdbc)
    '''
    def get_fd_image(self, size_shape=None):
        img = CImageFD.df_to_image(self.m_df_fd)
        if size_shape:
            img=CImageUtils.resize(img,size_shape)
        return img

class CImageAlpha:
    def __init__(self, img, svd_count=128, max_scales=32):
        self.m_img = img
        self.svd_count = svd_count + 1
        self.max_scales = max_scales
        
    def _get_scales(self):
        H, W = self.m_img.shape[:2]
        max_scale = max(4, min(H, W) // 4)
        scales = np.linspace(2, max_scale, self.max_scales)
        scales = np.unique(scales.astype(np.int32))
        if len(scales) < 4:
            scales = np.array([2, 4, 8, 16], dtype=np.int32)
        return scales

    def _get_svd_images(self):
        imgs = CImageUtils.get_svd_images(self.m_img, count=self.svd_count)
        if isinstance(imgs, tuple):
            imgs = imgs[0]
        safe_imgs = []
        for im in imgs:
            im = np.asarray(im)
            if np.std(im) > 1e-6:
                safe_imgs.append(im)
        return safe_imgs[:self.svd_count - 1]

    @staticmethod
    def compute_alpha_map_batch(img_list, scales_list, batch_size=64, with_progress=False):
        alpha_all = []
        info_all = []
        start = 0
        total = len(img_list)
        while start < total:
            end = min(start + batch_size, total)
            alpha, info = CFAImageMFS.compute_alpha_map_batch(img_list[start:end],
                                                              scales=scales_list,
                                                              with_progress=with_progress)
            alpha_all.extend(alpha)
            info_all.extend(info)
            start = end
        return alpha_all, info_all
        
    @staticmethod
    def _clean_alpha(alpha):
        alpha = np.asarray(alpha, dtype=np.float32)
        if np.isnan(alpha).mean() > 0.75:
            return None
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=np.max(alpha), neginf=0.0)
        return alpha

    def get_raw_alpha(self):
        R, G, B = CImageUtils.split_image(self.m_img)
        scales = self._get_scales()
        alpha, _ = CImageAlpha.compute_alpha_map_batch([R, G, B],
                                                       scales_list=scales,
                                                       batch_size=3)
        alpha = [self._clean_alpha(a) for a in alpha]
        return CImageUtils.merge_image(alpha[0], alpha[1], alpha[2])

    def get_gray_alpha(self):
        svd = self._get_svd_images()
        scales = self._get_scales()
        img_list = []
        valid_index = []
        for i, im in enumerate(svd):
            gray = CImageUtils.get_gray(im)
            if np.std(gray) > 1e-6:
                img_list.append(gray)
                valid_index.append(i)
        if len(img_list) == 0:
            return []
        alpha, _ = CImageAlpha.compute_alpha_map_batch(img_list,scales_list=scales)

        alpha = [self._clean_alpha(a) for a in alpha]
        return alpha

    def get_bin_alpha(self):
        svd = self._get_svd_images()
        scales = self._get_scales()
        img_list = []
        for im in svd:
            gray = CImageUtils.get_gray(im)
            binary = CImageUtils.get_binary(gray)
            img_list.append(binary)
        if len(img_list) == 0:
            return []
        alpha, _ = CImageAlpha.compute_alpha_map_batch(img_list,scales_list=scales)
        return [self._clean_alpha(a) for a in alpha]

    def get_rgb_alpha(self):
        svd = self._get_svd_images()
        scales = self._get_scales()
        img_list = []
        svd_index = []
        for i, im in enumerate(svd):
            R, G, B = CImageUtils.split_image(im)
            img_list.extend([R, G, B])
            svd_index.extend([i, i, i])
        if len(img_list) == 0:
            return []
        alpha, _ = CImageAlpha.compute_alpha_map_batch(img_list,scales_list=scales)
        alpha = [self._clean_alpha(a) for a in alpha]
        if len(alpha) < len(svd) * 3:
            raise ValueError("Alpha length mismatch (CFAImageMFS output unstable)")
        merged = []
        for i in range(len(svd)):
            base = 3 * i
            merged.append(CImageUtils.merge_image(alpha[base],alpha[base+1],alpha[base+2]))
        return merged

    def get_svd(self):
        return self._get_svd_images()
