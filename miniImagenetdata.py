import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import PIL
import os
import random

path = os.getcwd()

class MiniImageNet():
    def __init__(self, batch_size, ways=10, shots=1):
        self.x_train = _load_data('train.npz')
        self.x_val = _load_data('val.npz')
        self.x_test = _load_data('test.npz')
        self.batch_size = batch_size
        self.ways = ways
        self.shots = shots
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        
    
    
    def _load_data(self, data_file):
        data_dict = np.load(path + '/miniImagenet' + data_file)
        
        return {key: val for (key, val) in data_dict.items()}


    def get_batch(self, dataset_name):
        data = self.datasets[dataset_name]
        
        shape = list(data.values())[0].shape[1:]
        support_set_x = np.zeros((self.batch_size, self.ways, self.shots, shape[0], shape[1], shape[2]), dtype=np.float32)
        support_set_y = np.zeros((self.batch_size, self.ways, self.shots), dtype=np.float32)
        target_x = np.zeros((self.batch_size, shape[0], shape[1], shape[2]), dtype=np.float32)
        target_y = np.zeros((self.batch_size), dtype=np.float32)
        
        for i in xrange(self.batch_size):
            
            sampled_classes = random.sample(data.keys(), self.ways)
            for k, class_ in enumerate(sampled_classes):
                #the number of total images, in most case: 600
                shots_idx = np.arange(data[class_].shape[0])
                #an array of chosed images
                choose_shots = np.random.choice(shots_idx, size=self.shots, replace=False)
                support_set_x[i][k] = data[class_][choose_shots]
                support_set_y[i][k] = class_
            target_x[i] = support_set_x[-1,-1,:,:,:]
            target_y[i] = support_set_y[-1, -1, 0:1]
            
            support_set_x = support_set_x[:,:-1,:,:,:]
            support_set_y = support_set_y[:,:,:-1]
        
        return support_set_x, support_set_y, target_x, target_y