import os
import shutil
import glob
import math
import argparse
import warnings
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import random

import read_hdr

##########################################################################################
##########################################################################################

folders = [
    "/hdd1/works/datasets/ssd3/hdr/RAISE/tif",
    "/hdd1/works/datasets/ssd3/hdr/HDR+/tif"
    ]

# 꼭 sorted 사용해주기.
imagePaths_total = []

for folder in folders:
    imgPath = sorted(glob.glob(f"{folder}/*.tif"))
    imagePaths_total += imgPath

imagePaths_total = imagePaths_total  # [2204:2205]


print(f"cpu count : {cpu_count()}")
print('imagePaths_total :', len(imagePaths_total))

def write_yuv420_10bit(img, name):
    cropped_yuv, w, h = read_hdr.bgr2yuv420(img, 'yuv')

    # 0~1 to 10bit
    cropped_yuv *= 1023
    cropped_yuv = np.around(cropped_yuv)
    cropped_yuv = np.clip(cropped_yuv, 0, 1023)

    # save yuv420
    with open(f'{name}_{w}x{h}_10bit.yuv', "wb") as f_yuv:
        f_yuv.write(cropped_yuv.astype('uint16').tobytes())

# img_ori = read_hdr.my_imread('/hdd1/works/datasets/ssd3/hdr/HDR+/tif/5a9e_20141005_144419_100.tif')
# cv2.imwrite('ori.png', img_ori * 255)
#
# h = img_ori.shape[0]
# w = img_ori.shape[1]
#
# scale_factor = 0.05
# img = cv2.resize(img_ori, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_AREA)
# write_yuv420_10bit(img, f'inter.png')
#
# img = cv2.resize(img_ori, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_NEAREST)
# write_yuv420_10bit(img, f'inter.png')
#
# img = cv2.resize(img_ori, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_CUBIC)
# write_yuv420_10bit(img, f'inter.png')



#########################################

resized_dir = f'results/resized'
os.makedirs(f'{resized_dir}', exist_ok=True)


def imwrite_with_patch_info(patch_idxs_per_img):
    # 읽고
    img = read_hdr.my_imread(f'{patch_idxs_per_img}')
    n = os.path.splitext(os.path.basename(patch_idxs_per_img))[0]

    mu, sigma = 0.16, 0.02  # mean and standard deviation
    scale_factor = np.random.normal(mu, sigma)
    h = img.shape[0]
    w = img.shape[1]
    interpolation_methods = [cv2.INTER_AREA, cv2.INTER_AREA, cv2.INTER_AREA, cv2.INTER_AREA, cv2.INTER_CUBIC]
    interpolation_method = random.choice(interpolation_methods)
    img = cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)), interpolation=interpolation_method)

    cropped_yuv, w, h = read_hdr.bgr2yuv420(img, 'yuv')

    # 0~1 to 10bit
    cropped_yuv *= 1023
    cropped_yuv = np.around(cropped_yuv)
    cropped_yuv = np.clip(cropped_yuv, 0, 1023)
    center_name = f'resized__{n}_{w}x{h}'

    # save yuv420
    with open(f'{resized_dir}/{center_name}_10bit.yuv', "wb") as f_yuv:
        f_yuv.write(cropped_yuv.astype('uint16').tobytes())

    progressive_bar.update(1)


pool = ThreadPool(cpu_count())
progressive_bar = tqdm(total=len(imagePaths_total))
result = pool.map(imwrite_with_patch_info, imagePaths_total)
pool.close()
pool.join()
progressive_bar.close()

#########################################

# mu, sigma = 0.15, 0.02 # mean and standard deviation
# s = np.random.normal(mu, sigma, 1000)
#
# import matplotlib.pyplot as plt
#
# plt.plot(s)
# plt.ylabel('y-label')
# plt.show()