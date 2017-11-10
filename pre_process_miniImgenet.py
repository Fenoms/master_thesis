import os
import numpy as np
import pandas as pd
from collections import defaultdict
import tarfile
import gc
import warnings
import six
import argparse
from PIL import Image

input_dir = '/home/fenoms/meta_data/miniImagenet/csv'

data_dir = '/home/fenoms/meta_data/miniImagenet/'

def _read_image_as_array(path, dtype='int32'):
    f = Image.open(path)
    
    try:
        image = np.asarray(f, dtype=dtype)
        
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image


def download_imgs():
    train = pd.read_csv(input_dir + '/train.csv', sep=',')
    test = pd.read_csv(input_dir + '/test.csv', sep=',')
    val = pd.read_csv(input_dir + '/val.csv', sep=',')
    
    labels = train.label.unique().tolist() + test.label.unique().tolist() + val.label.unique().tolist()

    for label in labels:
        os.system('wget "http://image-net.org/download/synset?wnid=' + label + '&username=fenoms&accesskey=acdcf71fbeafbdbc2cb0036e89b1c614346c733e&release=latest&src=stanford" -O /home/fenoms/meta_data/miniImagenet/' + label + '.tar')


if __name__ == "__main__":
    
    download_imgs()