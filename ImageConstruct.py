import argparse
import numpy as np
import time,os
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import matplotlib.pyplot as plt
from FreeAeonFractal.FAImageFourier import CFAImageFourier
from FreeAeonFractal.FAImage import CFAImage
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS
from sklearn.decomposition import PCA

#GPU version
#from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU as CFAImageLAC
#from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD
#from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

#CPU version
from FreeAeonFractal.FAImageLAC import CFAImageLAC
from FreeAeonFractal.FAImageFD import CFAImageFD
from FreeAeonFractal.FAImageMFS import CFAImageMFS

np.set_printoptions(suppress=True, precision=8)

class CImageUtils:
    
    @staticmethod    
    def write_image(file_name, img):
        if not file_name.lower().endswith(".exr"):
            cv2.imwrite(file_name + ".exr", img)
        else:
            cv2.imwrite(file_name, img)
            
    @staticmethod
    def read_image(file_name):
        cv2_img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        if cv2_img is None:
            raise FileNotFoundError(f"Failed to read EXR: {file_name}")
            
        return cv2_img.astype(np.float32)
    
    @staticmethod
    def get_one_svd_image(img, tau):
        if img.ndim == 3:
            channels = []
            for c in range(img.shape[2]):
                channel = img[..., c]
                low_rank = CImageUtils.get_one_svd_image(channel, tau)
                channels.append(low_rank)
            return np.stack(channels, axis=2)
        else:
            U, S, Vt = np.linalg.svd(img, full_matrices=False)
            S_norm = S / S.max()
            weights = np.exp(-S_norm / tau)
            S_soft = S * weights
            low_rank_soft = U @ np.diag(S_soft) @ Vt
            return low_rank_soft

    @staticmethod
    def get_svd_images(img, isRes = False, count=128):
        all_results = []
        if count == 1:
            return [img]
        for tau in np.linspace(0.01, 1, count):
            svd_img = CImageUtils.get_one_svd_image(img, tau=tau)
            if isRes:
                all_results.append(img - svd_img)
            else:
                all_results.append(svd_img)
        return all_results

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
    def display(imgs, cols=8, cell_size=2.0):
        n = len(imgs)
        rows = int(np.ceil(n / cols))
        figsize = (cols * cell_size, rows * cell_size)
        plt.figure(figsize=figsize)
        for i, img in enumerate(imgs):
            ax = plt.subplot(rows, cols, i + 1)
            if img.ndim == 3:
                ax.imshow(img, vmin=0, vmax=1)
            else:
                ax.imshow(img, vmin=0, vmax=1, cmap='gray')
            ax.set_title(f"{i}", fontsize=8)
            ax.axis("off")
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

class CImageAlpha:
    
    def __init__(self, img,isRes = False, scales = 32):
        self.m_img = CImageUtils.normalize(img)
        self.m_scales = scales
        self.m_isRes = isRes
        
    def get_raw_alpha(self):
        R,G,B = CImageUtils.split_image(self.m_img)
        H, W = self.m_img.shape[:2]
        scales_list = np.linspace(1, min(H, W), self.m_scales)
        alpha_maps,info = CFAImageMFS.compute_alpha_map_batch([R,G,B],scales=scales_list,with_progress=False)
        result = CImageUtils.merge_image(alpha_maps[0],alpha_maps[1],alpha_maps[2])
        return CImageUtils.normalize(result)
        
    def get_gray_alpha(self,count = 128):
        H, W = self.m_img.shape[:2]
        scales_list = np.linspace(1, min(H, W), self.m_scales)
        img_list = []
        for im in CImageUtils.get_svd_images(self.m_img,isRes=self.m_isRes,count = count):
            gray = CImageUtils.get_gray(im)
            img_list.append(gray)
        alpha_map, info = CFAImageMFS.compute_alpha_map_batch(img_list,scales=scales_list,with_progress=False)
        return CImageUtils.normalize(alpha_map)
        
    def get_bin_alpha(self,count = 128):
        H, W = self.m_img.shape[:2]
        scales_list = np.linspace(1, min(H, W), self.m_scales)
        img_list = []
        for im in CImageUtils.get_svd_images(self.m_img,isRes=self.m_isRes,count = count):
            gray = CImageUtils.get_gray(im)
            bin = CImageUtils.get_binary(gray)
            img_list.append(bin)
        alpha_map, info = CFAImageMFS.compute_alpha_map_batch(img_list,scales=scales_list,with_progress=False)
        return CImageUtils.normalize(alpha_map)

    @staticmethod
    def compute_alpha_map_batch_batched(img_list, scales_list, batch_size=64, with_progress=False):
            alpha_maps_all = []
            info_list_all = []
            total = len(img_list)
            start_idx = 0
            while start_idx < total:
                end_idx = min(start_idx + batch_size, total)
                batch_imgs = img_list[start_idx:end_idx]
                alpha_maps_batch, info_list_batch = CFAImageMFS.compute_alpha_map_batch(batch_imgs,scales=scales_list,with_progress=with_progress)

                alpha_maps_all.extend(alpha_maps_batch)
                info_list_all.extend(info_list_batch)
                start_idx = end_idx
            
                #torch.cuda.empty_cache()
            
            return alpha_maps_all, info_list_all

    def get_svd_alpha(self, count = 128 ):
        H, W = self.m_img.shape[:2]
        scales_list = np.linspace(1, min(H, W),self.m_scales)
        img_list = []
        for im in CImageUtils.get_svd_images(self.m_img,isRes=self.m_isRes,count = count):
            #gray = CImageUtils.get_gray(img)
            #bin = CImageUtils.get_binary(gray)
            R,G,B = CImageUtils.split_image(im)
            img_list.append(R)
            img_list.append(G)
            img_list.append(B)
        
        alpha_maps, info_list = CImageAlpha.compute_alpha_map_batch_batched(img_list,scales_list=scales_list,batch_size=64,with_progress=False)
        merged_images = []
        for i in range(0, len(alpha_maps), 3):
            R = alpha_maps[i]
            G = alpha_maps[i + 1]
            B = alpha_maps[i + 2]
            merged = CImageUtils.merge_image(R, G, B)
            merged_images.append(merged)
            
        return  CImageUtils.normalize(merged_images)

class CImageMFS:
    
    def __init__(self, img,isRes = False, count = 32):
        self.m_img = CImageUtils.normalize(img)
        self.m_count = count
        self.m_isRes = isRes

    def get_q_list(self):
        return np.linspace(-10, 10, self.m_count)
    
    @staticmethod
    def get_batch_mfs_batched(img_list, q_list=None, batch_size=64, with_progress=False):
        result_all = []
        total = len(img_list)
        start_idx = 0
        while start_idx < total:
            end_idx = min(start_idx + batch_size, total)
            batch_imgs = img_list[start_idx:end_idx]
            result_batch = CFAImageMFS.get_batch_mfs(batch_imgs,q_list=q_list,with_progress=with_progress)
            result_all.extend(result_batch)
            start_idx = end_idx

            #torch.cuda.empty_cache()
        return result_all

    #Image list : R,G,B
    #Image Channel is: a(q),d(q),f(a)
    def get_mfs_image(self):
        img_list = []
        for im in CImageUtils.get_svd_images(self.m_img,isRes=self.m_isRes,count=self.m_count):
            #gray = CImageUtils.get_gray(img)
            #bin = CImageUtils.get_binary(gray)
            R,G,B = CImageUtils.split_image(im)
            img_list.append(R)
            img_list.append(G)
            img_list.append(B)

        q_list = self.get_q_list()
        result = CImageMFS.get_batch_mfs_batched(img_list,q_list = q_list, batch_size = 64, with_progress=False )
        values_R = []
        values_G = []
        values_B = []
        
        for i in range(0, len(result), 3):
            item = result[i]
            df_R_mass, df_R_fit, df_R_spec = item[0],item[1],item[2]
            df_R_spec = df_R_spec.rename(columns={"tau":"t(q)","Dq":"d(q)","alpha":"a(q)","f_alpha":"f(a)"}).drop(columns="D1").iloc[:128]
            values_R.append(df_R_spec[['a(q)','d(q)','f(a)']].values[0:self.m_count])
            
            item = result[i + 1]
            df_G_mass, df_G_fit, df_G_spec = item[0],item[1],item[2]
            df_G_spec = df_G_spec.rename(columns={"tau":"t(q)","Dq":"d(q)","alpha":"a(q)","f_alpha":"f(a)"}).drop(columns="D1").iloc[:128]
            values_G.append(df_G_spec[['a(q)','d(q)','f(a)']].values[0:self.m_count])
        
            item = result[i + 2]
            df_B_mass, df_B_fit, df_B_spec = item[0],item[1],item[2]
            df_B_spec = df_B_spec.rename(columns={"tau":"t(q)","Dq":"d(q)","alpha":"a(q)","f_alpha":"f(a)"}).drop(columns="D1").iloc[:128]
            values_B.append(df_B_spec[['a(q)','d(q)','f(a)']].values[0:self.m_count])
        
        R_mfs = CImageUtils.normalize(np.array(values_R))
        G_mfs = CImageUtils.normalize(np.array(values_G))
        B_mfs = CImageUtils.normalize(np.array(values_B))
        return R_mfs,G_mfs,B_mfs

    #Image list : a(q),d(q),f(a)
    #Image Channel is: R,G,B
    def get_image_mfs(self):
        img_list = []
        for im in CImageUtils.get_svd_images(self.m_img,isRes=self.m_isRes,count=self.m_count):
            #gray = CImageUtils.get_gray(img)
            #bin = CImageUtils.get_binary(gray)
            R,G,B = CImageUtils.split_image(im)
            img_list.append(R)
            img_list.append(G)
            img_list.append(B)

        q_list = self.get_q_list()
        result = CImageMFS.get_batch_mfs_batched(img_list,q_list = q_list, batch_size = 64, with_progress=False )
        values_A = []
        values_D = []
        values_F = []
        
        for i in range(0, len(result), 3):
            item = result[i]
            df_R_mass, df_R_fit, df_R_spec = item[0],item[1],item[2]
            df_R_spec = df_R_spec.rename(columns={"tau":"t(q)","Dq":"d(q)","alpha":"a(q)","f_alpha":"f(a)"}).drop(columns="D1").iloc[:128]
            
            item = result[i + 1]
            df_G_mass, df_G_fit, df_G_spec = item[0],item[1],item[2]
            df_G_spec = df_G_spec.rename(columns={"tau":"t(q)","Dq":"d(q)","alpha":"a(q)","f_alpha":"f(a)"}).drop(columns="D1").iloc[:128]
        
            item = result[i + 2]
            df_B_mass, df_B_fit, df_B_spec = item[0],item[1],item[2]
            df_B_spec = df_B_spec.rename(columns={"tau":"t(q)","Dq":"d(q)","alpha":"a(q)","f_alpha":"f(a)"}).drop(columns="D1").iloc[:128]

            a = np.stack([df_R_spec['a(q)'].values,
                 df_G_spec['a(q)'].values,
                 df_B_spec['a(q)'].values], axis=-1) 
            d = np.stack([df_R_spec['d(q)'].values,
                 df_G_spec['d(q)'].values,
                 df_B_spec['d(q)'].values], axis=-1)  
            f = np.stack([df_R_spec['f(a)'].values,
                 df_G_spec['f(a)'].values,
                 df_B_spec['f(a)'].values], axis=-1)
            
            values_A.append(a[0:self.m_count])
            values_D.append(d[0:self.m_count])
            values_F.append(f[0:self.m_count])

        A_mfs = CImageUtils.normalize(np.array(values_A))
        D_mfs = CImageUtils.normalize(np.array(values_D))
        F_mfs = CImageUtils.normalize(np.array(values_F))
        return A_mfs,D_mfs,F_mfs
