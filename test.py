import cv2
import os,sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

img = cv2.imread(
    "/disk/a/AI_Face_imagesV2/test/GANs/AttGAN/7990/raw.exr",
    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
)

print(img)
