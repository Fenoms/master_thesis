import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
import os
import random




class MiniImagenetData():

    def __init__(self, data_dir, image_shape, batch_size, ways = 5, shots = 5, query_size = 15):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.ways = ways
        self.shots = shots
        self.num_iter = 0
        # self.max_iter = max_iter
        self.query_size = query_size
        #image info
        self.data_shape = image_shape
        self.data = {'train': self._load_data('train')}  
        # self.data = {'tarin': self._load_data('train'), 'val': self._load_data('val'), 'test': self._load_data('test')}

    def _load_data(self, mode):
            data = np.load(self.data_dir + mode + '.npz')
            return {key: value for (key, value) in data.items()}
            # return np.load(self.data_dir + '/' + mode + '.npy')
            

    def _sample_classes(self, num_classes):

        return np.random.choice(num_classes, self.ways, replace = False)


    def get_batch(self, mode = 'train'):
        
        """

        """  
        support_set_x = np.zeros((self.batch_size, self.ways, self.shots, self.data_shape[0], 
                                    self.data_shape[1], self.data_shape[2]), dtype=np.float32)

        support_set_y = np.zeros((self.batch_size, self.ways, self.shots), dtype=np.float32)

        query_x = np.zeros((self.batch_size, self.query_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]), dtype=np.float32)
        query_y = np.zeros((self.batch_size, self.query_size), dtype=np.float32)

        data = self.data[mode]
        

        for i in range(self.batch_size):
            
            # if mode == 'train':
            #     c = np.arange(0, 64)
            # elif mode == 'val':
            #     c = np.arange(64, 80)
            # else:
            #     c = np.arange(80, 100)

            # sampled_classes = np.random.choice(c, self.ways, replace = False)

            # t = np.random.randint(0, self.ways)

            sampled_classes = random.sample(data.keys(), self.ways)
            # sampled_classes = _sample_classes(data.shape[0])

            for k, class_ in enumerate(sampled_classes):
                #the number of total images, in most case: 600
                # data_dir = path + '/miniImagenet' + '/' + mode + '/' + str(class_) + '/' + mode + '.npy'
                # data = np.load(data_dir)
                # shots_idx = data[class_].shape[0]

                #average num_samples for query image set
                q = int(self.query_size / self.ways)
                #for support_set + query_size
                num_imgs = data[class_].shape[0]
                

                index = np.random.choice(num_imgs, size=self.shots + q, replace=False)
                support_set_x[i][k] = data[class_][index[:self.shots]]
                support_set_y[i][k] = k

                query_x[i][k*q: (k+1)*q] = data[class_][index[self.shots:]]
                query_y[i][k*q: (k+1)*q] = k
                
        
        print(support_set_x.shape, support_set_y.shape, query_x.shape, query_y.shape)
        return support_set_x, support_set_y, query_x, query_y



