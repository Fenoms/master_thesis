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
import pickle


input_dir = '/home/fenoms/master_thesis/miniImagenet/csv/'

data_dir = '/home/fenoms/master_thesis/miniImagenet/'

def _read_image_as_array(image, dtype='int32'):
    f = Image.open(image)
    k = np.random.randint(0, 4)
    f.rotate(k*90)
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
        os.system('wget "http://image-net.org/download/synset?wnid=' + label + '&username=fenoms&accesskey=acdcf71fbeafbdbc2cb0036e89b1c614346c733e&release=latest&src=stanford" -O /home/fenoms/master_thesis/miniImagenet/' + label + '.tar')


def arrange_images(data_path, out_to_file):
    csv = pd.read_csv(data_path, sep =',')
    labels = csv.label.unique().tolist()
    folder_1 = data_dir + out_to_file
    if os.path.exists(folder_1):
        print("folder already exists")
    else:
        os.mkdir(folder_1)
        for i, label in enumerate(labels):
            k = i
            if out_to_file == 'val':
                #different label space
                k = k +  64
            elif out_to_file == 'test':
                k = k +  80
            folder_2 = folder_1 + '/' + str(k)
            os.mkdir(folder_2)
            tar = tarfile.open(data_dir + label + '.tar')
            imgs = tar.getmembers()
            data = np.zeros((2000, 224, 224, 3), dtype = np.float32)
            for j, img in enumerate(imgs):
                f = tar.extractfile(img)
                try:
                    f = Image.open(f)
                    f = f.resize((224, 224))
                    f = np.asarray(f, dtype = np.float32)
                    f = np.reshape(f, (224, 224, 3))
                    data[j] = f
                except Exception as e:
                     print("skipping image, because " + str(e))
                # f.save(folder_2 + '/' + str(k) + '_' + str(j) + '.JPEG', 'JPEG')
            data = data[:j]
            np.save(folder_2 + '/' + out_to_file + '.npy', data)
        print("finish extracting image files")


def data_process(data_path, out_to_file):
    dataset = pd.read_csv(data_path, sep = ',')
    labels = dataset.label.unique().tolist()

    data = np.zeros((64, 600, 244, 224, 3), dtype = np.float32)
    label = np.zeros((64, 600,), dtype = np.float32)
    i = 0
    for i, label in enumerate(labels):

        tar = tarfile.open(data_dir + label + '.tar')
        imgs = tar.getmembers()

        c = 0

        for j, img in enumerate(imgs):
            f = tar.extractfile(img)

            try:
                f = Image.open(f)
                f = f.resize((224, 224))
                f = np.asarray(f, dtype = np.float32)
                f = np.reshape(f, (224, 224, 3))
                data[i][j] = f
                c = c + 1
            except Exception as e:
                print("skipping image, because " + str(e)) 

            if c == 600:
                break


    np.save(data_dir + out_to_file + '.npy', data)


def pre_process_data(data_path, out_to_file):
    dataset = pd.read_csv(data_path, sep=',')
    labels = dataset.label.unique().tolist()
    tmp_dict = defaultdict(list)

    for label in labels:
        
        tar = tarfile.open(data_dir + label + '.tar')
        imgs = tar.getmembers()
        np.random.shuffle(imgs)
        
        c = 0
        
        for img in imgs:
            
            f = tar.extractfile(img)
            
            try:
                img_array = _read_image_as_array(f)
                img_array = scipy.misc.imresize(img_array, (84, 84))
                img_array = img_array.astype('float32')
                tmp_dict[label].append(img_array.reshape((1, 84, 84, 3)))
                c += 1
            except Exception as e:
                print("skipping image, because " + str(e))
            
            if c == 600:
                break
        print(c)

    results = {key: np.concatenate(value) for key, value in tmp_dict.items()}
    np.savez(data_dir + out_to_file + ".npz", **results)



def process_data(data_path, out_to_file):
    csv = pd.read_csv(data_path, sep = ',')
    labels = csv.label.unique().tolist()
    tra_data = np.zeros((33280, 224, 224, 3), dtype=np.float32)
    val_data = np.zeros((5120, 224, 224, 3), dtype=np.float32)
    tra_labels = np.zeros((33280,), dtype=np.uint8)
    val_labels = np.zeros((5210,), dtype=np.uint8)
    nb_train_images = 0
    nb_val_images = 0
    for k, label in enumerate(labels):
        tar = tarfile.open(data_dir + label + '.tar')
        imgs = tar.getmembers()
        c = 0

        for img in imgs:
            f = tar.extractfile(img)
            if c < 80:
                try:
                    img_array = _read_image_as_array(f)
                    img_array = scipy.misc.imresize(img_array, (224, 224))
                    img_array = img_array.astype('float32')
                    img_array = np.reshape(img_array, (1,224, 224, 3))
                    val_data[nb_val_images] = img_array
                    val_labels[nb_val_images] = k
                    c += 1
                    nb_val_images += 1
                except Exception as e:
                    print("skipping image, because " + str(e))
            elif c >= 80 and c < 600:
                try:
                    img_array = _read_image_as_array(f)
                    img_array = scipy.misc.imresize(img_array, (224, 224))
                    img_array = img_array.astype('float32')
                    img_array = np.reshape(img_array, (1,224, 224, 3))
                    tra_data[nb_train_images] = img_array
                    tra_labels[nb_train_images] = k
                    c += 1
                    nb_train_images += 1
                except Exception as e:
                    print("skipping image, because " + str(e))

            else:
                print(c)
                break
        print(nb_train_images)
        print(nb_val_images)

    tra_data = tra_data[:nb_train_images]
    tra_labels = tra_labels[:nb_train_images]
    val_data = val_data[:nb_val_images]
    val_labels = val_labels[:nb_val_images]

    training_data = {"training_data" : tra_data, "training_label": tra_labels}
    val_data = {"val_data": val_data, "val_label": val_labels}

    pickle_tra_in = open("pickle_tra_224", "wb")
    pickle_val_in = open("pickle_val_224", "wb")
    pickle.dump(training_data, pickle_tra_in)
    pickle.dump(val_data, pickle_val_in)
    print("saved successfully")
	   
 
if __name__ == '__main__':
    #download_imgs()
    # print("trian...")
    # process_data(input_dir + 'train.csv', 'training_data')
    print("arrange training images")
    arrange_images(input_dir + 'train.csv', 'train')
    print("arrange validation images")
    arrange_images(input_dir + 'val.csv', 'val')
    print("arrange testing images")
    arrange_images(input_dir + 'test.csv', 'test')
    # print("val...")
    # pre_process_data(input_dir + 'val.csv', 'val')
    # print("test...")
    # pre_process_data(input_dir + 'test.csv', 'test')