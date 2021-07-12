import cv2
import numpy as np
import random
import os
import tqdm

"""
아래 링크에 HDR 설명이 잘 되어있음. 
https://learnopencv.com/high-dynamic-range-hdr-imaging-using-opencv-cpp-python/
"""


def drago_tone_map(hdr):
    # Tonemap using Drago's method to obtain 24-bit color image
    tonmapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonmapDrago.process(hdr)
    ldrDrago = 3 * ldrDrago
    ldrDrago = np.clip(ldrDrago, 0, 1)
    return ldrDrago


def reinhard_tone_map(hdr):
    # Tonemap using Reinhard's method to obtain 24-bit color image
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdr)
    ldrReinhard = np.clip(ldrReinhard, 0, 1)
    return ldrReinhard


def mantiuk_tone_map(hdr):
    # Tonemap using Mantiuk's method to obtain 24-bit color image
    # 너무 느려서 사용 안함..
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdr)
    ldrMantiuk = 3 * ldrMantiuk
    ldrMantiuk = np.clip(ldrMantiuk, 0, 1)
    return ldrMantiuk


def bgr2yuv420(bgr, out):
    # bgr 의 range : 0~1 이어야 한다.

    bgr = img_to_multipleof16(bgr)
    cvt_M = np.array(
        [[0.2126, 0.7152, 0.0722],
         [-0.09991, -0.33609, 0.436],
         [0.615, -0.55861, -0.05639]]
    )

    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    y = cvt_M[0][0]*r + cvt_M[0][1]*g + cvt_M[0][2]*b
    u = cvt_M[1][0] * r + cvt_M[1][1] * g + cvt_M[1][2] * b + 0.5
    v = cvt_M[2][0] * r + cvt_M[2][1] * g + cvt_M[2][2] * b + 0.5

    # resize with antialias.
    u = cv2.resize(u, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    v = cv2.resize(v, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    h, w = y.shape

    if out == 'yuv':
        yuv = np.hstack([y.flatten(), u.flatten(), v.flatten()])
    else:
        yuv = y

    return yuv, w, h


def img_to_multipleof16(img):
    h, w = img.shape[:2]
    h = (h//16)*16
    w = (w//16)*16
    return img[:h, :w]


def my_imread(my_img_dir):
    """
    read image.
    range : 0~1
    """
    my_img = None
    if os.path.splitext(os.path.basename(my_img_dir))[-1] == '.hdr':
        # .hdr 영상은 대부분 24-bits 라고 함.
        my_hdr = cv2.imread(my_img_dir, cv2.IMREAD_UNCHANGED)
        # gama correction
        tone_map_func = [drago_tone_map, reinhard_tone_map]
        my_tone_map = reinhard_tone_map # random.choice(tone_map_func)
        my_img = my_tone_map(my_hdr)
        my_img = np.nan_to_num(my_img)

    elif os.path.splitext(os.path.basename(my_img_dir))[-1] == '.tif':
        # tif 는 만들 당시 16bit 로 저장하였음.
        my_bit = 16
        my_img = cv2.imread(my_img_dir, cv2.IMREAD_UNCHANGED).astype(np.float32)/(2**my_bit - 1)

    return my_img.astype(np.float32)


if __name__ == "__main__":
    # read my hdr image
    my_dir = '/hdd1/works/datasets/ssd1/HDR_DB/HDR-Eye/00000/gt.hdr'
    my_dir = '/hdd1/works/datasets/ssd3/hdr/HDR+/tif/0155_20160810_212319_430.tif'
    my_dir = '/hdd1/works/datasets/ssd1/HDR_DB/Rockies3b.hdr'

    # import glob
    # folder = "/hdd1/works/datasets/ssd1/HDR_DB"
    # # 꼭 sorted 사용해주기.
    # imagePaths = sorted(glob.glob(f"{folder}/*.hdr"))[0:1000]  # [0:30]  #[2204:2205]
    # print(imagePaths)
    #
    #
    # for i, imgPath in tqdm.tqdm(enumerate(imagePaths)):
    #     my_img = my_imread(imgPath)
    #     cv2.imwrite(f'temp/{i}.jpg', my_img* 255)


    # 아래 코드 잠시 주석처리.
    # BGR to YUV420
    my_img = my_imread(my_dir)
    my_img, w, h = bgr2yuv420(my_img, 'yuv')
    print(w, h)

    # 0~1 to 10bit
    my_img *= 1023
    my_img = np.around(my_img)
    my_img = np.clip(my_img, 0, 1023)


    # save yuv420
    with open(f'hdr_temp1_{w}x{h}_10bit.yuv', "wb") as f_yuv:
        f_yuv.write(my_img.astype('uint16').tobytes())


