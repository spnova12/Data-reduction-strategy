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

import time


def sixteen2eight_bit(img):
    img = img / (2**16 - 1) * 255
    return img


def get_yuv_histogram(yuv, uv_down_scale, bins):
    yuv = sixteen2eight_bit(yuv)

    y_his = np.histogram(yuv[:, :, 0], bins=bins, range=(0, 255), density=False)[0]

    u_his = np.histogram(yuv[:, :, 1], bins=int(bins / uv_down_scale), range=(0, 255), density=False)[0] / uv_down_scale
    v_his = np.histogram(yuv[:, :, 2], bins=int(bins / uv_down_scale), range=(0, 255), density=False)[0] / uv_down_scale

    # 3가지 histogram 을 이어 붙여준다.
    yuv_histogram = np.concatenate((y_his, u_his, v_his))

    # 영상의 사이즈대로 정규화를 해준다.
    yuv_histogram = yuv_histogram / (yuv.shape[0] * yuv.shape[1])

    weight = 1
    return yuv_histogram * weight


def get_lbp_histogram(y, radius=1, n_points=8):
    # get local_binary_pattern
    # settings for LBP
    METHOD = 'uniform'

    # get lbp
    lbp = local_binary_pattern(y, n_points, radius, METHOD)

    # get lbp histogram
    bins = 2**n_points
    lbp_histogram = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=False)[0]

    # 영상의 사이즈대로 정규화를 해준다.
    lbp_histogram = lbp_histogram / (y.shape[0] * y.shape[1])

    weight = 5
    return lbp_histogram * weight


def get_edge_magnitude_sum(y):
    y = sixteen2eight_bit(y)

    def my_sobel(img, d):
        if d == 0:
            sobel = np.array(
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]]
            )
        else:  # 1
            sobel = np.array(
                [[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]]
            )
        return cv2.filter2D(img, cv2.CV_32F, sobel)

    dx = my_sobel(y, 0)
    dy = my_sobel(y, 1)

    my_magnitude = np.sqrt(dx * dx + dy * dy)

    # np.sqrt(((255*4)**2)*2) 은 1500 정도.
    my_magnitude = my_magnitude / 1500  # 0~1 정도 사이즈로 normalize 해준다.

    edge_magnitude_sum = np.sum(my_magnitude) / (my_magnitude.shape[0] * my_magnitude.shape[1])

    weight = 50
    return edge_magnitude_sum * weight


def image2patches(img, img_name):
    """
    img 를 patch 단위로 잘라서 반환해준다.
    """
    # 이미지의 길이.
    ori_h_length = img.shape[0]
    ori_w_length = img.shape[1]

    # crop 할 사이즈.
    wl = 640
    hl = 640

    # 영상 썸네일 영상 사이즈.
    thumbnail_size = 8

    # color_histogram, lbp 에 사용할 영상 사이즈.
    img_size_lbp = 64

    # 예상되는 patch 개수
    patches_count = int(np.ceil(ori_w_length/wl) * np.ceil(ori_h_length/hl))

    # 처리한 영상의 index 를 저장 할 list
    patch_idxs = []

    mini_patches = np.zeros((patches_count, thumbnail_size, thumbnail_size, 3))

    # yuv histograms
    bins = 12
    uv_down_scale = 4
    yuv_histograms = np.zeros((patches_count, (bins + int(bins/uv_down_scale) * 2)))  # 3 channel 이니까 3 곱해줌.

    # lbp histograms
    radius = 1
    n_points = 8
    lbp_bins = 2**n_points
    lbp_histograms = np.zeros((patches_count, lbp_bins))

    # edge magnitude sum
    edge_magnitude_sums = np.zeros((patches_count, 1))  # scalar 이기 때문에 다음과 같이 x, 1 모양으로 만들어줘야 vstack 이 됨.

    # crop 해주기.
    w_pos = 0
    h_pos = 0
    total_count = 0
    while ori_h_length > h_pos:
        while ori_w_length > w_pos:

            if w_pos + wl > ori_w_length:
                w_pos = ori_w_length - wl
            if h_pos + hl > ori_h_length:
                h_pos = ori_h_length - hl

            j, i, w, h = w_pos, h_pos, wl, hl
            cropped_img = img[i:(i+h), j:(j + w), :]

            # cropped 된 영상의 정보.
            # None 은 label 을 담을 자리, False 는 대표가 아님을 True 면 대표임을 의미.
            patch_idxs.append([img_name, i, j, h, w, None, False])

            # 영상의 썸네일.
            mini_patches[total_count] = cv2.resize(cropped_img, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA)

            # histogram 구할 때 빠른 속도 등을 위해 down scale 해준다.
            cropped_resized_img = cv2.resize(cropped_img, (img_size_lbp, img_size_lbp), interpolation=cv2.INTER_AREA)

            # RGB to YUV
            cropped_resized_img_yuv = cv2.cvtColor(cropped_resized_img, cv2.COLOR_BGR2YUV)
            cropped_resized_img_y = cropped_resized_img_yuv[:, :, 0]

            # 영상의 color histogram.
            yuv_histograms[total_count] = get_yuv_histogram(cropped_resized_img_yuv, uv_down_scale=uv_down_scale, bins=bins)

            # 영상의 lbp histogram. (gray 영상 사용)
            lbp_histograms[total_count] = get_lbp_histogram(cropped_resized_img_y, radius=radius, n_points=n_points)

            # 영상의 전역 기술자 histogram. (gray 영상 사용)
            edge_magnitude_sums[total_count] = get_edge_magnitude_sum(cropped_resized_img_y)

            total_count += 1

            w_pos += wl

        w_pos = 0
        h_pos += hl
    return patch_idxs, mini_patches, yuv_histograms, lbp_histograms, edge_magnitude_sums


##########################################################################################
##########################################################################################

folder = "/hdd1/works/datasets/ssd1/HDR_DB/kdw_small"
group_count = 500

# 꼭 sorted 사용해주기.
imagePaths = sorted(glob.glob(f"{folder}/*.tif"))[0:30]

print(f"db len : {len(imagePaths)}")
print(f"cpu count : {cpu_count()}")


def read_image_and_get_patches(img_dir):
    # 읽고
    img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img_name = os.path.basename(img_dir)
    # crop 하고
    patches = image2patches(img, img_name)
    # tqdm update.
    progressive_bar.update(1)
    return patches


# CPU 를 병렬적으로 사용해서 영상을 빠르게 읽어준다.
pool = ThreadPool(cpu_count())
progressive_bar = tqdm(total=len(imagePaths))
result = pool.map(read_image_and_get_patches, imagePaths)
pool.close()
pool.join()
progressive_bar.close()

# 결과를 뽑아서 따로따로 묶어준다.
patch_idxs_per_imgs = [r[0] for r in result]
mini_patches = [r[1] for r in result]
yuv_histograms = [r[2] for r in result]
lbp_histograms = [r[3] for r in result]
edge_magnitude_sums = [r[4] for r in result]


# list 에 담긴 여러장의 img 들을 np array 로 묶어(stack) 준다.
target_to_group1 = np.vstack(mini_patches)
target_to_group1 = np.reshape(target_to_group1, (target_to_group1.shape[0], -1))
target_to_group2 = np.vstack(yuv_histograms)
target_to_group3 = np.vstack(lbp_histograms)
target_to_group4 = np.vstack(edge_magnitude_sums)


# k-medoids 알고리즘에 사용할 feature 들을 concat 해준다.
print(f'\nk-medoids is running...')
target_to_group = np.concatenate((target_to_group2, target_to_group3, target_to_group4), axis=1)

print(f'\nTotal patches len : {target_to_group.shape[0]}')
print(f'group1 : {target_to_group1.shape}')
print(f'group2 : {target_to_group2.shape}, min : {np.min(target_to_group2)}, max : {np.max(target_to_group2)}')
print(f'group3 : {target_to_group3.shape}, min : {np.min(target_to_group3)}, max : {np.max(target_to_group3)}')
print(f'group4 : {target_to_group4.shape}, min : {np.min(target_to_group4)}, max : {np.max(target_to_group4)}')
print(f'group : {target_to_group.shape}')

# k-medoids 알고리즘을 적용해준다.
target_to_group_reshaped = target_to_group
kmedoids = KMedoids(n_clusters=group_count,  max_iter=1000, method='pam', init='k-medoids++').fit(target_to_group_reshaped)


##########################################################################################
##########################################################################################

result_name = "temp3w5_4w50"

# 어떤식으로 label 이 나눠졌는지 가시화 해보자.
result_dir = f'results/{result_name}'
os.makedirs(f'{result_dir}', exist_ok=True)

# centor 를 따로 저장한다.
centor_dir = f'results/{result_name}_centor'
os.makedirs(f'{centor_dir}', exist_ok=True)


# patch_idxs_per_imgs 에 구해진 label 정보를 넣어준다.
label_idx = 0
for i in range(len(patch_idxs_per_imgs)):
    for j in range(len(patch_idxs_per_imgs[i])):
        label = kmedoids.labels_[label_idx]

        patch_idxs_per_imgs[i][j][-2] = label

        if label_idx in kmedoids.medoid_indices_:
            patch_idxs_per_imgs[i][j][-1] = True

        label_idx += 1


def imwrite_with_patch_info(patch_idxs_per_img):
    # 읽고
    img = cv2.imread(f'{folder}/{patch_idxs_per_img[0][0]}', cv2.IMREAD_UNCHANGED).astype(np.float32)

    for patch_info in patch_idxs_per_img:
        # 정보대로 crop 하고
        n, i, j, h, w, label, centor = patch_info
        cropped_img = img[i:(i + h), j:(j + w), :]
        n = os.path.splitext(n)[0]

        # 영상 write 하기.
        cv2.imwrite(f'{result_dir}/{label}__{n}__{i}_{j}_{h}_{w}.png', sixteen2eight_bit(cropped_img))

        # centor 영상 write 하기.
        if centor:
            cv2.imwrite(f'{centor_dir}/{label}__{n}__{i}_{j}_{h}_{w}.png', sixteen2eight_bit(cropped_img))

    progressive_bar.update(1)


pool = ThreadPool(cpu_count())
progressive_bar = tqdm(total=len(patch_idxs_per_imgs))
result = pool.map(imwrite_with_patch_info, patch_idxs_per_imgs)
pool.close()
pool.join()
progressive_bar.close()


for label in range(group_count):
    black = np.zeros((30,30))
    cv2.imwrite(f'{result_dir}/{label}.png', black)


