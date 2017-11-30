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
import scipy.misc
import cv2
from PIL import Image

input_dir = '/home/fenoms/meta_learning/master_thesis/miniImagenet/csv/'

data_dir = '/home/fenoms/meta_learning/master_thesis/miniImagenet/'

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
        os.system('wget "http://image-net.org/download/synset?wnid=' + label + '&username=fenoms&accesskey=acdcf71fbeafbdbc2cb0036e89b1c614346c733e&release=latest&src=stanford" -O /home/fenoms/meta_learning/master_thesis/miniImagenet/' + label + '.tar')

def pre_process_data(data_path, out_to_file):
    dataset = pd.read_csv(data_path, sep=',')
    labels = dataset.label.unique().tolist()
    tmp_dict = defaultdict(list)
    file_names = []
    lable_names = []
    for label in labels:
        print(label)
        tar = tarfile.open(data_dir + label + '.tar')
        imgs = tar.getmembers()
        print(len(imgs))
        np.random.shuffle(imgs)
        
        c = 0
        
        for img in imgs:
            k = np.random.randint(0, 4)
            f = tar.extractfile(img)
            f = Image.open(f)
            f.rotate(k*90)
            try:
                img_array = _read_image_as_array(f)
                img_array = scipy.misc.imresize(img_array, (84, 84))
                img_array = img_array.astype('float32')
                img_array *= (1.0/255.0)
                tmp_dict[label].append(img_array.reshape((1, 84, 84, 3)))
                file_names.append(img.name)
                lable_names.append(label)
                c += 1
            except Exception as e:
                print("skipping image, because " + str(e))
            
            if c == 600:
                break
        print(c)
    results = {key : np.concatenate(value) for key, value in tmp_dict.items()}
    np.savez(data_dir + out_to_file + ".npz", **results)
    sub = pd.DataFrame({'img_file': file_names, 'label':lable_names})
    sub.to_csv(data_dir + out_to_file + '.csv', index=False)


if __name__ == '__main__':
    #download_imgs()
    print("trian...")
    pre_process_data(input_dir + 'train.csv', 'train')
    print("val...")
    pre_process_data(input_dir + 'val.csv', 'val')
    print("test...")
    pre_process_data(input_dir + 'test.csv', 'test')