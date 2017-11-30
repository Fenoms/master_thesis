import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import PIL
import os
import random

path = os.getcwd()

class MiniImageNet():
    def __init__(self, batch_size, classes_per_set, samples_per_class):
        self.x_train = np.load(path + '/miniImagenet/' + 'train.npz')
        self.x_val = np.load(path + '/miniImagenet/' + 'val.npz')
        self.x_test = np.load(path + '/miniImagenet/' + 'test.npz')
        self.batch_size = batch_size
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        
    

    def get_batch(self, dataset_name):
        data = self.datasets[dataset_name]
        
        data = {key: val for (key, val) in data.items()}
        shape = list(data.values())[0].shape[1:]
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, shape[0], shape[1], shape[2]), dtype=np.float32)
        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), dtype=np.float32)
        target_x = np.zeros((self.batch_size, shape[0], shape[1], shape[2]), dtype=np.float32)
        target_y = np.zeros((self.batch_size), dtype=np.float32)
        
        for i in xrange(self.batch_size):
            
            sampled_classes = random.sample(data.keys(), self.classes_per_set)
            for k, class_ in enumerate(sampled_classes):
                #the number of total images, in most case: 600
                samples_per_class_idx = np.arange(data[class_].shape[0])
                #an array of chosed images
                choose_samples_per_class = np.random.choice(samples_per_class_idx, size=self.samples_per_class, replace=False)
                support_set_x[i][k] = data[class_][choose_samples_per_class]
                support_set_y[i][k] = class_
            target_x[i] = support_set_x[-1,-1,:,:,:]
            target_y[i] = support_set_y[-1, -1, 0:1]
            
            support_set_x = support_set_x[:,:-1,:,:,:]
            support_set_y = support_set_y[:,:,:-1]
        
        return support_set_x, support_set_y, target_x, target_y