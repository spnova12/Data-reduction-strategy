import cv2
import numpy as np
import math
from glob import glob
import os
import tqdm
from random import *
import random
from os import listdir
from os.path import join
import os
import shutil

if __name__ == '__main__':

    new_db_dir = 'results/TF1'
    new_db2_dir = 'results/TF2'

    ###
    db_dirs = [
        'results/D',
        'results/H',
        'results/P'
    ]

    for db_dir in db_dirs:
        header = db_dir.split('/')[1]

        contents = [join(db_dir, x) for x in sorted(listdir(db_dir))]
        for content in tqdm.tqdm(contents):
            targets = [new_db_dir, new_db2_dir]
            target = random.choice(targets)
            new_name = f'{target}/{header}_{os.path.basename(content)}'
            shutil.move(content, f"{new_name}")



