import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class read_imgs():
    def __init__(self):
        self.length = 0
        self.now_size = 0
        self.imgs = []
        self.tags = False
    
    def read_source1(self, dirs):

        from tqdm import tqdm
        print("read data from: ", dirs)
        trainlist = os.popen('ls '+dirs).read().split()

        for index, name in tqdm(enumerate(trainlist)):
            self.imgs.append(plt.imread(dirs+name))


        ## define some variables
        self.length = self.length + len(trainlist)

    
    def next_batch(self, batch_size, normal=False): 
        
        next_size = self.now_size + batch_size
        if next_size > self.length:
            next_imgs0 = np.array(self.imgs[self.now_size:])
            
            next_size = next_size - self.length
            next_imgs1 = np.array(self.imgs[:next_size])
            next_imgs = np.concatenate((next_imgs0, next_imgs1), axis=0)
        
        elif next_size == self.length:
            next_imgs = np.array(self.imgs[self.now_size:])
            next_size = 0

        else:
            next_imgs = np.array(self.imgs[self.now_size:next_size])

        
        self.now_size = next_size
        
        # if normal=True, normalize data from -1 to 1
        if normal:
            next_imgs = (next_imgs / 255.) * 2 - 1
        
        return next_imgs

    def read_source2(self, dirs, resize = False):
        
        from tqdm import tqdm
        print("read data from: ", dirs)
        trainlist = os.popen('ls '+dirs).read().split()
        imgs = []

        for index, name in tqdm(enumerate(trainlist)):
            if resize:
                img0 = Image.open(dirs+name)
                img0.thumbnail((64, 64), Image.ANTIALIAS)
                self.imgs.append(np.array(img0))
            else:
                self.imgs.append(plt.imread(dirs+name))
        
        # define
        self.length = self.length + len(trainlist)
    
    def shuffle_data(self):
        import random
        random.shuffle(self.imgs)
        

if __name__ == '__main__':
    from tqdm import tqdm
    datasets = read_imgs()
    datasets.read_source1('./dataset/extra_data/images/')
    datasets.shuffle_data()
    
    next_imgs = datasets.next_batch(128)

