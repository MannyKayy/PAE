from __future__ import division
import numpy as np
import pickle
import os
import tarfile
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 
font = {'family' : "serif",
        'size'   : 15}
matplotlib.rc('font', **font)
import scipy.misc
import time

# Training data
class UniformNoise:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, N):
        samples = np.random.uniform(-1.0, 1.0, (self.dim, N))
        return samples

class NormalNoise:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, N):
        samples = np.random.normal(0.0, 1.0, (self.dim, N))
        return samples

class Grid:
    def __init__(self):
        self.modes = 25

    def sample(self, N):
        cov = [[0.1, 0.0],[0.0, 0.1]]
        num_modes_per_axis = int(np.sqrt(self.modes))
        mode_range = np.linspace(-21,21,num_modes_per_axis)
        x, y = np.meshgrid(mode_range,mode_range)
        x = x.flatten()
        y = y.flatten()
        for i in range(self.modes):
            mean = [x[i], y[i]]
            sample = np.random.multivariate_normal(mean, cov, int(N/self.modes)).T
            if i == 0:
                samples = sample
            else:
                samples = np.hstack((samples,sample))
        return samples, None
    
    def test_sample(self, N):
        return self.sample(N)
        
    def shuffle(self, samples):
        _, N = np.shape(samples)
        id = np.random.permutation(N)
        samples = samples[:,id]
        return samples

    def visualize(self, samples, save_path=None):
        #real_samples,_ = self.sample(gen_samples.shape[1])
        #plt.plot(real_samples[0,:], real_samples[1,:], '.r', label='Real')
        plt.plot(samples[0,:], samples[1,:], '.b', label='Generated')
        plt.xlim((-30, 30))
        plt.ylim((-30, 30))
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #   ncol=2, mode="expand", borderaxespad=0.)
        if save_path:
            plt.savefig(save_path,bbox_inches='tight',dpi=800)
            plt.close()
        else:
            plt.show()
            plt.close()

class LowDimEmbed:
    def __init__(self):
        self.hd = 1000
        self.ld = 10
        self.modes = 10
        self.mode_ranges = np.linspace(-5,5,self.modes)
        self.dim_offsets = np.floor(np.linspace(0,self.hd-self.ld-1,self.modes)).astype(int)
    
    def sample(self, batch_size):
        '''
            This function generates a mixture of 10 isotropic Gaussians in 10-d space
            and embeds into 1000-d space
        '''
        for i in range(self.modes):
            mean = np.ones(self.ld)*self.mode_ranges[i]
            cov = np.eye(self.ld)
            sample = np.zeros((self.hd,int(batch_size/self.modes)))#np.random.normal(0,1,(self.hd,int(batch_size/self.modes)))
            offset = self.dim_offsets[i]
            sample[offset:offset+self.ld,:] = np.random.multivariate_normal(mean, cov, int(batch_size/self.modes)).T
            if i == 0:
                samples = sample
                labels = np.ones(int(batch_size/self.modes),dtype=np.int32)*i
            else:
                samples = np.hstack((samples,sample))
                labels = np.concatenate((labels,np.ones(int(batch_size/self.modes),dtype=np.int32)*i))
        return samples, labels

    def test_sample(self, batch_size):
        return self.sample(batch_size)

    def val_sample(self, batch_size):
        return self.sample(batch_size)

class MNIST:
    '''
    This class reads samples from grayscale MNIST
    Usage: 
        mnist = MNIST('data/mnist')
        x_train,y_train = mnist.sample(128)     # x_train.shape = [d,n]
        x_test,y_test = mnist.test_samples()    # x_test.shape = [d,n]
    '''
    def __init__(self, db_path):
        self.mnist = input_data.read_data_sets(db_path, one_hot=True)
        self.num_train_samples = self.mnist.train.num_examples
    def sample(self, batch_size):
        x_batch, y_batch = self.mnist.train.next_batch(batch_size)
        return x_batch.T, y_batch
    def test_sample(self):
        x, y = self.mnist.test.images, self.mnist.test.labels
        return x.T, y
    def val_sample(self):
        x, y = self.mnist.validation.images, self.mnist.validation.labels
    

class CMNIST:
    '''
    This class generate stacked MNIST of 1000 classes
    Usage: 
        cmnist = CMNIST('data/mnist')
        x_train,y_train = cmnist.sample(128)     # x_train.shape = [d,n]
        x_test,y_test = cmnist.test_samples()    # x_test.shape = [d,n]
    '''
    def __init__(self, db_path):
        self.mnist = input_data.read_data_sets(db_path, one_hot=False)
        self.num_train_samples = self.mnist.train.num_examples
        self.num_test_samples = self.mnist.test.num_examples
        self.num_val_samples = self.mnist.validation.num_examples

    def sample(self, batch_size):
        x_batch_color = np.zeros((28*28*3,batch_size),dtype=np.float32)
        y_batch = np.zeros(batch_size,dtype=np.int32)
        for i in range(3):
            x_batch, labels_ = self.mnist.train.next_batch(batch_size)
            x_batch = x_batch.T
            y_batch += labels_.astype(np.int32) * (10**i)
            x_batch_color[i*28*28:(i+1)*28*28,:] = x_batch
        return x_batch_color, y_batch

    def test_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.num_test_samples
        x_color = np.zeros((28*28*3,batch_size),dtype=np.float32)
        y = np.zeros(batch_size,dtype=np.int32)
        x, t = self.mnist.test.images, self.mnist.test.labels
        x = x.T
        rand_inds = np.arange(batch_size)
        for i in range(3):
            np.random.shuffle(rand_inds)
            x_color[i*28*28:(i+1)*28*28,:] = x[:,rand_inds]
            y += t[rand_inds].astype(np.int32) * (10**i)
        return x_color, y

    def val_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.num_val_samples
        x_color = np.zeros((28*28*3,batch_size),dtype=np.float32)
        y = np.zeros(batch_size,dtype=np.int32)
        x, t = self.mnist.validation.images, self.mnist.validation.labels
        x = x.T
        rand_inds = np.arange(batch_size)
        for i in range(3):
            np.random.shuffle(rand_inds)
            x_color[i*28*28:(i+1)*28*28,:] = x[:,rand_inds]
            y += t[rand_inds].astype(np.int32) * (10**i)
        return x_color, y
    
    def visualize(self, gen_data, real_data=None, save_path=None):
        assert gen_data.shape[1] >= 90, 'At least 90 generated images are needed'
        if real_data is None:
            real_data,_ = self.sample(10)
        data = np.zeros((28*10,28*10,3))
        for r in range(10):
            for c in range(10):
                for i in range(3):
                    if c == 0:
                        data[r*28:(r+1)*28,c*28:(c+1)*28,i] = real_data[i*28*28:(i+1)*28*28,r].reshape((28,28))
                    else:
                        data[r*28:(r+1)*28,c*28:(c+1)*28,i] = gen_data[i*28*28:(i+1)*28*28,r*9+c-1].reshape((28,28))
        if save_path:
            plt.imsave(save_path, data, dpi=800)
            plt.close()
        else:
            plt.imshow(data)
            plt.close()
            