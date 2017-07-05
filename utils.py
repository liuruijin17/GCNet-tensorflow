from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')

def conv3d(x, W, stride=1):
    return tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME')

def conv3dt(x, W, outputshape, stride=1):
    return tf.nn.conv3d_transpose(x, W, output_shape=outputshape,strides=[1, stride, stride, stride, 1], padding='SAME')

def batchnorm(Ylogits, is_test, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2, 3])

    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)

    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)

    return Ybn, update_moving_everages

def cropdata(trainimg_left, trainimg_right, traindpt_left, traindpt_right, H_in, W_in, H_out, W_out, batchsize):
    H_ori = H_in
    W_ori = W_in
    H = H_out
    W = W_out
    H_delta = H_ori - H
    W_delta = W_ori - W
    batch_size = batchsize
    train_img_left = trainimg_left
    train_img_right = trainimg_right
    train_dpt_left = traindpt_left
    train_dpt_right = traindpt_right

    images_left = np.zeros((batch_size, H, W), dtype=np.float32)
    images_right = np.zeros((batch_size, H, W), dtype=np.float32)
    disparity_left = np.zeros((batch_size, H, W))
    disparity_right = np.zeros((batch_size, H, W))
    index_ori = np.random.random_integers(len(train_img_left) - 1, size=(batch_size,))
    loc_y = (np.random.random_sample((batch_size, 1)) * H_delta).astype(int)
    loc_x = (np.random.random_sample((batch_size, 1)) * W_delta).astype(int)
    loc = np.append(loc_y, loc_x, axis=1)

    for p in range(batch_size):
        images_left[p, :, :] = train_img_left[index_ori[p]][loc[p, 0]:loc[p, 0] + H, loc[p, 1]:loc[p, 1] + W]
        images_right[p, :, :] = train_img_right[index_ori[p]][loc[p, 0]:loc[p, 0] + H, loc[p, 1]:loc[p, 1] + W]
        disparity_left[p, :, :] = train_dpt_left[index_ori[p]][loc[p, 0]:loc[p, 0] + H, loc[p, 1]:loc[p, 1] + W]
        disparity_right[p, :, :] = train_dpt_right[index_ori[p]][loc[p, 0]:loc[p, 0] + H, loc[p, 1]:loc[p, 1] + W]

    return images_left, images_right, disparity_left, disparity_right

def getdataset(root):

    train_img_left_dir = root + 'scene_forwards_img_left.pkl'
    train_img_right_dir = root + 'scene_forwards_img_right.pkl'
    train_disparity_left_dir = root + 'scene_forwards_disparity_left.pkl'
    train_disparity_right_dir = root + 'scene_forwards_disparity_right.pkl'

    print('Extracting training_data...')
    read_img_left = open(train_img_left_dir, 'rb')
    read_img_right = open(train_img_right_dir, 'rb')
    train_img_left_all = pickle.load(read_img_left)
    train_img_right_all = pickle.load(read_img_right)
    print('Type:', type(train_img_left_all), type(train_img_right_all))
    print('Train image Length:', len(train_img_left_all), len(train_img_right_all))
    print('Train image Shape:', train_img_left_all[0].shape, train_img_right_all[0].shape)
    print('Train image Type:', train_img_left_all[0].dtype, train_img_right_all[0].dtype)


    read_disparity_left = open(train_disparity_left_dir, 'rb')
    read_disparity_right = open(train_disparity_right_dir, 'rb')
    train_disparity_left_all = pickle.load(read_disparity_left)
    train_disparity_right_all = pickle.load(read_disparity_right)
    print('Type:', type(train_disparity_left_all), type(train_disparity_right_all))
    print('Train disparity Length:', len(train_disparity_left_all), len(train_disparity_right_all))
    print('Train disparity Shape:', train_disparity_left_all[0].shape, train_disparity_right_all[0].shape)
    print('Train disparity Type:', train_disparity_left_all[0].dtype, train_disparity_right_all[0].dtype)

    return train_img_left_all, train_img_right_all, train_disparity_left_all, train_disparity_right_all

def creat_dir(root, section):

    direction = os.path.join(root,section)
    if not os.path.exists(direction):
        os.makedirs(direction)

    return direction

def returnAccuracy(prediction, groundTruth, delta=10. ,H=256, W=512, batchSize=1):
    tmp = np.zeros((batchSize, H, W))
    bool = np.abs(prediction - groundTruth) <= delta ### correct acc must <=
    tmp[bool == True] = 1
    acc = np.mean(tmp)
    return acc

def textdata(data):
    return np.min(data), np.max(data), np.mean(data)

def watchdata(img, cImg, gt, cGt):
    l, lg, r, rg = img, gt, cImg, cGt
    print(l.shape, textdata(l))
    print(lg.shape, textdata(lg))
    print(r.shape, textdata(r))
    print(rg.shape, textdata(rg))
    # num = l.shape[0]
    # set = np.concatenate((l, r, lg, rg), axis=0)
    # plt.figure()
    # for i in range(num):
    #     plt.subplot(2, 2, i + 1)
    #     plt.axis('off')
    #     plt.imshow(set[i,:,:], 'gray')
    #
    #     plt.show()



