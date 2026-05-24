import cv2
import argparse
import numpy as np
from tqdm import tqdm
import time
from FreeAeonFractal.FAImageFourier import CFAImageFourier
from FreeAeonFractal.FAImage import CFAImage
from FreeAeonFractal.FASeriesMFS import CFASeriesMFS

#CPU version
#from FreeAeonFractal.FAImageLAC import CFAImageLAC
#from FreeAeonFractal.FAImageFD import CFAImageFD
#from FreeAeonFractal.FAImageMFS import CFAImageMFS

#GPU version
from FreeAeonFractal.FAImageLACGPU import CFAImageLACGPU as CFAImageLAC
from FreeAeonFractal.FAImageFDGPU import CFAImageFDGPU as CFAImageFD
from FreeAeonFractal.FAImageMFSGPU import CFAImageMFSGPU as CFAImageMFS

def demo_fd(image_path):
    rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    bin_image,threshold = CFAImage.otsu_binarize(gray_image)

    max_scales = 32
    #  ---- single ----
    t0 = time.time()
    fd_bc = CFAImageFD(bin_image,max_scales=max_scales,with_progress=False).get_bc_fd(corp_type=-1)
    fd_dbc = CFAImageFD(gray_image,max_scales=max_scales,with_progress=False).get_dbc_fd(corp_type=-1)
    fd_sdbc = CFAImageFD(gray_image,max_scales=max_scales,with_progress=False).get_sdbc_fd(corp_type=-1)
    # ---- batch ----
    bin_imgs = [bin_image] * 100
    gray_imgs = [gray_image] * 100
    t0 = time.time()
    bc_list = CFAImageFD.get_batch_bc(bin_imgs, max_scales=max_scales,with_progress=False)
    dbc_list = CFAImageFD.get_batch_dbc(gray_imgs, max_scales=max_scales,with_progress=False)
    sdbc_list = CFAImageFD.get_batch_sdbc(gray_imgs, max_scales=max_scales,with_progress=False)
 
def demo_mfs(image_path):
    rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    q_list = np.linspace(-10, 10, 101)
    # --- single ----
    t0 = time.time()
    MFS = CFAImageMFS(gray_image,q_list = q_list ,with_progress=False)
    df_mass, df_fit, df_spec = MFS.get_mfs()
    
    # ---- batch ----
    t0 = time.time()
    imgs = [gray_image] * 20
    batch_results = CFAImageMFS.get_batch_mfs( imgs, 
            with_progress=False, q_list=q_list, corp_type=-1,
            bg_reverse=False, bg_threshold=0.01, bg_otsu=False, max_scales=80,
            min_points=6, use_middle_scales=False, if_auto_line_fit=False,
            fit_scale_frac=(0.3, 0.7), auto_fit_min_len_ratio=0.6,
                                                    cap_d0_at_2=False)
    df_mass1, df_fit1, df_spec1 = batch_results[0]

def demo_alpha(image_path):
    rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    q_list = np.linspace(-5, 5, 51)
    # --- single ----
    t0 = time.time()
    MFS = CFAImageMFS(gray_image,q_list = q_list ,with_progress=False)
    scales = list(range(1, 100)) 
    alpha_map, info = MFS.compute_alpha_map(scales=scales)
    CFAImageMFS.plot_alpha_map(alpha_map)
    # ---- batch ----
    t0 = time.time()
    imgs = [gray_image] * 20
    t0 = time.time()
    batch_alpha_map = CFAImageMFS.compute_alpha_map_batch(imgs,with_progress=False, scales=scales)

def main(image_path, mode):
    t0 = time.time()
    for i in tqdm(range(100),desc="FD"):
        demo_fd(image_path)
    print("FD",time.time() - t0)

    t0 = time.time()
    for i in tqdm(range(100),desc="MFS"):
        demo_mfs(image_path)
    print("MFS",time.time() - t0)

    t0 = time.time()
    for i in tqdm(range(100),desc="Alpha"):
        demo_alpha(image_path)
    print("alpha",time.time() - t0)

if __name__ == "__main__":
    main("./images/63008.png","")

