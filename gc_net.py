# encoding: utf-8

# Standard libraries
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import logging

# 3rd-part libraries
import tensorflow as tf

# Self-define libraries
from unary_features import UnaryFeatures
from cost_volume import CostVolume
from learning_regularization import LearningRegularization
from critic_network import SoftArgmin, CriticNet, TrainingOp
from utils import creat_dir, returnAccuracy

from sceneflow_dataset import getBatchData, getIndexLists, Dataset

logging.getLogger().setLevel(logging.INFO)


def main():

    output_dir = '/home/ricklrj/GcNet/GcNetBeta3/output/'
    data_dir = '/media/home_bak/share/Dataset/SceneFlow-dataset/'
    model_dir = creat_dir(output_dir, 'model')
    summary_dir = creat_dir(output_dir, 'summary')
    sfDataSet = Dataset(data_dir)   # For random index again

    # TODO: Get training data
    leftImgPath, rightImgPath, leftDptPath, rightDptPath, sceneDirSizeNumberList, \
    indexList, sceneIndexList, randomIndexArray = getIndexLists(isTraining=True)

    # TODO: Initialize parameters
    H = 256
    W = 512
    C = 1
    batch_size = 1
    trainDataSetSize = len(indexList)
    train_iters = 100000

    # TODO: Make placeholder
    leftimg_ph = tf.placeholder(tf.float32, [batch_size, H, W])  ### batch_size img_size
    leftdpt_ph = tf.placeholder(tf.float32, [batch_size, H, W])
    rightimg_ph = tf.placeholder(tf.float32, [batch_size, H, W])
    rightdpt_ph = tf.placeholder(tf.float32, [batch_size, H, W])
    tst = tf.placeholder(tf.bool)

    # TODO: Reshape input for summary
    img = tf.reshape(leftimg_ph, [-1, H, W, C])
    cImg = tf.reshape(rightimg_ph, [-1, H, W, C])
    gt = tf.reshape(leftdpt_ph, [-1, H, W, C])
    cGt = tf.reshape(rightdpt_ph, [-1, H, W, C])
    tf.summary.image('input/left_image', img, batch_size)
    tf.summary.image('input/left_disparity', gt, batch_size)
    tf.summary.image('input/right_image', cImg, batch_size)
    tf.summary.image('input/right_disparity_', cGt, batch_size)

    # TODO: Build graph
    SHARE = True
    uf_left, uf_right, ema_uf_left, ema_uf_right = UnaryFeatures(img, cImg, tst, SHARE=SHARE)
    cv_left, cv_right = CostVolume(uf_left, uf_right)
    lr_left, lr_right, ema_lr_left, ema_lr_right = LearningRegularization(cv_left, cv_right, tst, SHARE=SHARE)
    rs_left, rs_right = SoftArgmin(lr_left, lr_right)
    loss = CriticNet(rs_left, rs_right, leftdpt_ph, rightdpt_ph)
    train_op, learning_rate = TrainingOp(loss, trainDataSetSize, batch_size=batch_size, max_grad_norm=5)
    update_ema_left = tf.group(ema_uf_left, ema_lr_left)
    update_ema_right = tf.group(ema_uf_right, ema_lr_right)

    # TODO: Reshape output for summary
    rsLeft = tf.reshape(rs_left, [-1, H, W, C])
    rsRight = tf.reshape(rs_right, [-1, H, W, C])
    tf.summary.image('output/left_disparity', rsLeft, batch_size)
    tf.summary.image('output/right_disparity', rsRight, batch_size)

    sessConf = tf.ConfigProto(allow_soft_placement=False)  # , log_device_placement=True
    sessConf.gpu_options.allow_growth = True

    with tf.Session(config=sessConf) as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(train_iters):

            # TODO: After one epoch, random again
            if i and i % trainDataSetSize == 0:

                randomIndexArray = sfDataSet.getRandomIndex()

            images_left, images_right, disparity_left, disparity_right = getBatchData(
                sceneIndexList, indexList, randomIndexArray, leftImgPath, rightImgPath, leftDptPath, rightDptPath,
                sceneDirSizeNumberList, i
            )

            feed_dict = {leftimg_ph: images_left, leftdpt_ph: disparity_left,
                         rightimg_ph: images_right, rightdpt_ph: disparity_right, tst: False}

            sess.run([train_op, update_ema_left, update_ema_right], feed_dict)

            if i and i % 10 == 0:

                summary = sess.run(merged, feed_dict)
                train_writer.add_summary(summary, i)

                predict_left, predict_right = sess.run([rs_left, rs_right], feed_dict)
                loss_val, lr_val = sess.run([loss, learning_rate], feed_dict)

                acc_left_10 = returnAccuracy(predict_left, disparity_left)
                acc_left_5 = returnAccuracy(predict_left, disparity_left, delta=5)
                acc_left_3 = returnAccuracy(predict_left, disparity_left, delta=3)
                acc_left_1 = returnAccuracy(predict_left, disparity_left, delta=1)


                acc_right_10 = returnAccuracy(predict_right, disparity_right)
                acc_right_5 = returnAccuracy(predict_right, disparity_right, delta=5)
                acc_right_3 = returnAccuracy(predict_right, disparity_right, delta=3)
                acc_right_1 = returnAccuracy(predict_right,disparity_right, delta=1)

                logging.info(
                    'Step {}\tlearning_rate: {:1.6f}\tloss: {:3.1f}\t|pre-gt|<=10: LeftAcc {:1.3f}\tRightAcc {:1.3f}\t<=5: LeftAcc {:1.3f}\tRightAcc {:1.3f}\t'
                             '<=3: LeftAcc {:1.3f}\tRightAcc {:1.3f}\t<=1: LeftAcc {:1.3f}\tRightAcc {:1.3f}'.format(
                        i, lr_val, loss_val, acc_left_10, acc_right_10, acc_left_5, acc_right_5, acc_left_3, acc_right_3, acc_left_1, acc_right_1)
                )
                # logging.info('\t|pre-gt|<=10: LeftAcc {:1.3f}\tRightAcc {:1.3f}\t<=5: LeftAcc {:1.3f}\tRightAcc {:1.3f}\t'
                #              '<=3: LeftAcc {:1.3f}\tRightAcc {:1.3f}\t<=1: LeftAcc {:1.3f}\tRightAcc {:1.3f}'.format(
                #     acc_left_10, acc_right_10, acc_left_5, acc_right_5, acc_left_3, acc_right_3, acc_left_1, acc_right_1
                # )
                # )

            if i and i % 5000 == 0:

                modelSubDir = creat_dir(model_dir, 'GcNetAlpha' + str(i) + '/')
                model_name = 'GcNetAlpha' + str(i) + '.ckpt'
                ckpt_path = modelSubDir + model_name
                tf.train.Saver().save(sess, ckpt_path)
                print('Model has been saved in %s direction' % ckpt_path)

        sess.close()

if __name__ == '__main__':
    main()

