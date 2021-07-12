import cv2
import numpy as np
import math
from glob import glob
import os
import tqdm
from random import *

from os import listdir
from os.path import join
import os
import shutil

if __name__ == '__main__':
    ###
    my_path = '/ssd1/HDR_DB/SingleHDR_training_data/HDR-Real/HDR_gt'
    contents = [join(my_path, x) for x in sorted(listdir(my_path))]

    for content in contents:
        new_name =  f'/ssd1/HDR_DB/new/SingleHDR_training_real{os.path.basename(content)}'
        print(new_name)
        shutil.move(content, f"{new_name}")





        #
    # # YUV dirs
    # new_yuv_list = []
    #
    #
    # def check_yuv_and_append_to_list(my_dir):
    #     contents = [join(my_dir, x) for x in sorted(listdir(my_dir))]
    #
    #     if len(contents) != 0 and os.path.splitext(contents[0])[-1] == '.yuv':
    #         png_datas = glob(os.path.join(my_dir, '*.yuv'))
    #         for yuv in png_datas:
    #             new_yuv_list.append(yuv)
    #     else:
    #         for content in contents:
    #             if os.path.isdir(content) and '_hm' not in os.path.basename(content):
    #                 check_yuv_and_append_to_list(content)
    #
    #
    # # 재귀함수를 이용하여 모든 yuv 파일을 list 에 넣어준다.
    # check_yuv_and_append_to_list(my_path)
