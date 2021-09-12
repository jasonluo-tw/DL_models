import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import pandas as pd
import numpy as np
import random
## custom
from DANN import DANN
from data_process import process
from functions import shuffle_data, val_split

## Fix random seed
tf.random.set_seed(100)
np.random.seed(100)
random.seed(100)

def process_sta_info(x):
    items = x.rstrip('\n').split(',')
    one = {items[0]: {'lon': float(items[1]), 'lat': float(items[2])}}
    return one

##
batch = 32
epochs = 80

## read dicts
sta_dicts = {}
with open('./sta_info.csv', 'r') as f:
    sta_infos = f.readlines()
    [sta_dicts.update(i) for i in list(map(process_sta_info, sta_infos[1:]))]

## source domain data
sta = 'AMEDAS-44132'
x_source, y_source = process(sta, sta_dicts[sta]['lon'], sta_dicts[sta]['lat'])
#x_source, y_source, _ = shuffle_data(x_source, y_source)
xs_train, ys_train, xs_val, ys_val = val_split(x_source, y_source, 0.2)

## target domain data
sta = 'AMEDAS-43256'
x_target, y_target = process(sta, sta_dicts[sta]['lon'], sta_dicts[sta]['lat'])
#x_target, y_target, _ = shuffle_data(x_target, y_target)
xt_train, yt_train, xt_val, yt_val = val_split(x_target, y_target, 0.2)
###
train_mse_loss = tf.keras.metrics.Mean(name='mse_train_loss')
train_dc_loss = tf.keras.metrics.Mean(name='bc_train_loss')

###
#print(x_target.shape)
model = DANN((x_source.shape[1]))
for ep in range(epochs):
    print(f'Start to train {ep+1} epoch ----------')
    xs_train, ys_train, _ = shuffle_data(xs_train, ys_train)
    xt_train, yt_train, _ = shuffle_data(xt_train, yt_train)
    bs = 0
    while (bs+1)*batch <= len(xs_train) and (bs+1)*batch <= len(xt_train):
        ll = bs*batch
        rr = (bs+1)*batch
        ##
        xbatch = xs_train[ll:rr]
        ybatch = ys_train[ll:rr]
        xbatch_t = xt_train[ll:rr]
        mse_loss, dc_loss = model.train(xbatch, xbatch_t, ybatch)
        #mse_loss = model.train_source(xbatch, ybatch)
        #dc_loss = 0
        train_mse_loss(mse_loss)
        train_dc_loss(dc_loss)
        ##
        bs += 1
    ## Evaluate the validation set
    source_rmse = model.evaluate(xs_train, ys_train)
    target_rmse = model.evaluate(xt_val, yt_val)
    target_train_rmse = model.evaluate(xt_train, yt_train)

    ## Show the results
    print(f'Epoch:{ep+1}/{epochs}')
    print('train results: mse_loss:%.3f, dc_loss:%.3f, target_rmse:%.3f'%(
        train_mse_loss.result(),
        train_dc_loss.result(),
        target_train_rmse
    ))
    print('val results: source_rmse:%.3f, target_rmse:%.3f'%(source_rmse, target_rmse))
    ## Reset loss
    train_mse_loss.reset_states()
    train_dc_loss.reset_states()




