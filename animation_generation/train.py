import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tqdm import tqdm
from models_tips import DCGAN
from read_data import read_imgs
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import yaml


## Read config yaml
try:
    with open('./model_config.yml', 'r') as f:
        config = yaml.safe_load(f)
except:
    print('Error reading the config file')
    sys.exit()

epochs = config['epochs']
batch_size = config['batchsize']
z_dimension = config['z_dims']

## Init the model
model = DCGAN()
z_batch0 = tf.random.normal([batch_size, z_dimension])

## read dataset
print('Read dataset1')
img_dir1 = config['files']['img_dir1']
datasets = read_imgs()
datasets.read_source1(img_dir1)
print('Datase1 length:', datasets.length)

## Define loss_object
train_dloss = tf.keras.metrics.Mean(name='dloss')
train_gloss = tf.keras.metrics.Mean(name='gloss')

print('Start to pre-train discriminator')
for i in range(config['d_pres']):
    real_image_batch = datasets.next_batch(batch_size, True)
    real_image_batch = real_image_batch.astype('float32')

    model.train_dis([batch_size, z_dimension], real_image_batch)

print('Start training')
nn = 0
## Reset Train Loss
train_dloss.reset_states()
train_gloss.reset_states()

## Train generator and discriminator together
for epoch in tqdm(range(epochs)):


    iter_nums = int(np.ceil(datasets.length / batch_size))
    datasets.shuffle_data()

    for itera in range(iter_nums):
        nn += 1
        real_image_batch = datasets.next_batch(batch_size, True)
        real_image_batch = real_image_batch.astype('float32')

        # Train discriminator
        for ii in range(1):
            loss_ = model.train_dis([batch_size, z_dimension], real_image_batch)
            train_dloss(loss_)

        # Train generator
        for ii in range(2):
            loss_ = model.train_gen([batch_size, z_dimension])
            train_gloss(loss_)

        if nn % 1000 == 0 or nn == 1:
            print('Epoch: %d/%d, Iter: %d/%d, GLoss loss: %.2f, DLoss: %.2f'%(epoch, epochs, itera+1, iter_nums, train_gloss.result(), train_dloss.result()))
            ## Reset Train Loss
            train_dloss.reset_states()
            train_gloss.reset_states()
            
            ## Generator generates some images
            testImage = model.Gz([z_batch0]).numpy()
            testImage = (testImage + 1) / 2.
            for jj, im in enumerate(testImage):
                plt.subplot(8, 8, jj+1)
                plt.axis('off')
                plt.imshow(im)

            plt.savefig(os.path.join(config['files']['out_dir'], 'test_%d.png'%(nn)))

