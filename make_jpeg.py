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
from PIL import Image

"""
jpg 로 압축하는 코드.
"""



# datas_path = '/hdd1/works/datasets/ssd1/DIV2K_train'
# datatype = 'DIV2K_train_HR'
# ori_folder = f'{datas_path}/original/{datatype}'
# jpg_folder = f'{datas_path}/jpeg/{datatype}'
# os.makedirs(f'{jpg_folder}', exist_ok=True)
#
#
# imagePaths = sorted(glob.glob(f"{ori_folder}/*.png"))
# print(imagePaths)
# for imagePath in imagePaths:
#     with Image.open(imagePath) as im:
#         d = os.path.splitext(os.path.basename(imagePath))[0]
#         os.makedirs(f'{jpg_folder}', exist_ok=True)
#         im.save(f"{jpg_folder}/{d}.jpg", quality=40)

datas_path = '/hdd1/works/projects/data_reduction/results/'
datatype = 'color100_lbp100_ems100_centor'
ori_folder = f'{datas_path}/{datatype}'
jpg_folder = f'{datas_path}/{datatype}_jpg'
os.makedirs(f'{jpg_folder}', exist_ok=True)



imagePaths = sorted(glob.glob(f"{ori_folder}/*.png"))

for imagePath in tqdm(imagePaths):
    with Image.open(imagePath) as im:
        d = os.path.splitext(os.path.basename(imagePath))[0]
        im.save(f"{jpg_folder}/{d}.jpg", quality=40)