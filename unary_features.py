import tensorflow as tf
from utils import weight_variable, bias_variable, conv2d, batchnorm






def UnaryFeatures(left_image, right_image, tst, F=32, SHARE=None):

    with tf.name_scope('Conv1'):
        with tf.variable_scope('params',reuse=SHARE):
            W1 = weight_variable((5, 5, 1, F))
            B1 = bias_variable((F,))
        Y1_left = conv2d(left_image, W1, stride=2)
        Y1_right = conv2d(right_image, W1, stride=2)
        Y1bn_left, update_ema1_left = batchnorm(Y1_left, tst, B1, convolutional=True)
        Y1bn_right, update_ema1_right = batchnorm(Y1_right, tst, B1, convolutional=True)
        # Y1bn_left, _= batchnorm(Y1_left, tst, B1, convolutional=True)
        # Y1bn_right, _= batchnorm(Y1_right, tst, B1, convolutional=True)
        Y1relu_left = tf.nn.relu(Y1bn_left)
        Y1relu_right = tf.nn.relu(Y1bn_right)
        # print('Output/Conv1->',Y1r_left.shape, Y1r_right.shape)

    with tf.name_scope('Conv2'):
        with tf.variable_scope('params',reuse=SHARE):
            W2 = weight_variable((3, 3, F, F))
            B2 = bias_variable((F,))
        Y2_left = conv2d(Y1relu_left, W2, stride=1)
        Y2_right = conv2d(Y1relu_right, W2, stride=1)
        Y2bn_left, update_ema2_left = batchnorm(Y2_left, tst, B2, convolutional=True)
        Y2bn_right, update_ema2_right = batchnorm(Y2_right, tst, B2, convolutional=True)
        # Y2bn_left, _= batchnorm(Y2_left, tst, B2, convolutional=True)
        # Y2bn_right, _= batchnorm(Y2_right, tst, B2, convolutional=True)
        Y2relu_left = tf.nn.relu(Y2bn_left)
        Y2relu_right = tf.nn.relu(Y2bn_right)
        # print('Output/Conv2->',Y2r_left.shape, Y2r_right.shape)

    with tf.name_scope('Conv3'):
        with tf.variable_scope('params',reuse=SHARE):
            W3 = weight_variable((3, 3, F, F))
            B3 = bias_variable((F,))
        Y3_left = conv2d(Y2relu_left, W3, stride=1)
        Y3_right = conv2d(Y2relu_right, W3, stride=1)
        Y3bn_left, update_ema3_left = batchnorm(Y3_left + Y1_left, tst, B3, convolutional=True)
        Y3bn_right, update_ema3_right = batchnorm(Y3_right + Y1_right, tst, B3, convolutional=True)
        # Y3bn_left, _= batchnorm(Y3_left + Y1_left, tst, B3, convolutional=True)
        # Y3bn_right, _= batchnorm(Y3_right + Y1_right, tst, B3, convolutional=True)
        Y3relu_left = tf.nn.relu(Y3bn_left)
        Y3relu_right = tf.nn.relu(Y3bn_right)
        # print('Output/Conv3->', Y3r_left.shape, Y3r_right.shape)

    with tf.name_scope('Conv4'):
        with tf.variable_scope('params', reuse=SHARE):
            W4 = weight_variable((3, 3, F, F))
            B4 = bias_variable((F,))
        Y4_left = conv2d(Y3relu_left, W4, stride=1)
        Y4_right = conv2d(Y3relu_right, W4, stride=1)
        Y4bn_left, update_ema4_left = batchnorm(Y4_left, tst, B4, convolutional=True)
        Y4bn_right, update_ema4_right = batchnorm(Y4_right, tst, B4, convolutional=True)
        # Y4bn_left, _= batchnorm(Y4_left, tst, B4, convolutional=True)
        # Y4bn_right, _= batchnorm(Y4_right, tst, B4, convolutional=True)
        Y4relu_left = tf.nn.relu(Y4bn_left)
        Y4relu_right = tf.nn.relu(Y4bn_right)

    with tf.name_scope('Conv5'):
        with tf.variable_scope('params', reuse=SHARE):
            W5 = weight_variable((3, 3, F, F))
            B5 = bias_variable((F,))
        Y5_left = conv2d(Y4relu_left, W5, stride=1)
        Y5_right = conv2d(Y4relu_right, W5, stride=1)
        Y5bn_left, update_ema5_left = batchnorm(Y5_left + Y3_left + Y1_left, tst, B5, convolutional=True)
        Y5bn_right, update_ema5_right = batchnorm(Y5_right + Y3_right + Y1_right, tst, B5, convolutional=True)
        # Y5bn_left, _= batchnorm(Y5_left + Y3_left + Y1_left, tst, B5, convolutional=True)
        # Y5bn_right, _= batchnorm(Y5_right + Y3_right + Y1_right, tst, B5, convolutional=True)
        Y5relu_left = tf.nn.relu(Y5bn_left)
        Y5relu_right = tf.nn.relu(Y5bn_right)


    with tf.name_scope('Conv6'):
        with tf.variable_scope('params', reuse=SHARE):
            W6 = weight_variable((3, 3, F, F))
            B6 = bias_variable((F,))
        Y6_left = conv2d(Y5relu_left, W6, stride=1)
        Y6_right = conv2d(Y5relu_right, W6, stride=1)
        Y6bn_left, update_ema6_left = batchnorm(Y6_left, tst, B6, convolutional=True)
        Y6bn_right, update_ema6_right = batchnorm(Y6_right, tst, B6, convolutional=True)
        # Y6bn_left, _= batchnorm(Y6_left, tst, B6, convolutional=True)
        # Y6bn_right, _= batchnorm(Y6_right, tst, B6, convolutional=True)
        Y6relu_left = tf.nn.relu(Y6bn_left)
        Y6relu_right = tf.nn.relu(Y6bn_right)

    with tf.name_scope('Conv7'):
        with tf.variable_scope('params', reuse=SHARE):
            W7 = weight_variable((3, 3, F, F))
            B7 = bias_variable((F,))
        Y7_left = conv2d(Y6relu_left, W7, stride=1)
        Y7_right = conv2d(Y6relu_right, W7, stride=1)
        Y7bn_left , update_ema7_left = batchnorm(Y7_left + Y5_left + Y3_left + Y1_left, tst, B7, convolutional=True)
        Y7bn_right, update_ema7_right = batchnorm(Y7_right + Y5_right + Y3_right + Y1_right, tst, B7, convolutional=True)
        # Y7bn_left, _= batchnorm(Y7_left + Y5_left + Y3_left + Y1_left, tst, B7, convolutional=True)
        # Y7bn_right, _= batchnorm(Y7_right + Y5_right + Y3_right + Y1_right, tst, B7,
        #                                           convolutional=True)

        Y7relu_left = tf.nn.relu(Y7bn_left)
        Y7relu_right = tf.nn.relu(Y7bn_right)

    with tf.name_scope('Conv8'):
        with tf.variable_scope('params', reuse=SHARE):
            W8 = weight_variable((3, 3, F, F))
            B8 = bias_variable((F,))
        Y8_left = conv2d(Y7relu_left, W7, stride=1)
        Y8_right = conv2d(Y7relu_right, W7, stride=1)
        Y8bn_left, update_ema8_left = batchnorm(Y8_left, tst, B8, convolutional=True)
        Y8bn_right, update_ema8_right = batchnorm(Y8_right, tst, B8, convolutional=True)
        # Y8bn_left, _= batchnorm(Y8_left, tst, B8, convolutional=True)
        # Y8bn_right, _= batchnorm(Y8_right, tst, B8, convolutional=True)
        Y8relu_left = tf.nn.relu(Y8bn_left)
        Y8relu_right = tf.nn.relu(Y8bn_right)

    with tf.name_scope('Conv9'):
        with tf.variable_scope('params', reuse=SHARE):
            W9 = weight_variable((3, 3, F, F))
            B9 = bias_variable((F,))
        Y9_left = conv2d(Y8relu_left, W9, stride=1)
        Y9_right = conv2d(Y8relu_right, W9, stride=1)
        Y9bn_left, update_ema9_left = batchnorm(Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
                                                tst, B9, convolutional=True)
        Y9bn_right, update_ema9_right = batchnorm(Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
                                                  tst, B9, convolutional=True)
        # Y9bn_left, _= batchnorm(Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
        #                                         tst, B9, convolutional=True)
        # Y9bn_right, _= batchnorm(Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
        #                                           tst, B9, convolutional=True)
        Y9relu_left = tf.nn.relu(Y9bn_left)
        Y9relu_right = tf.nn.relu(Y9bn_right)

    with tf.name_scope('Conv10'):
        with tf.variable_scope('params', reuse=SHARE):
            W10 = weight_variable((3, 3, F, F))
            B10 = bias_variable((F,))
        Y10_left = conv2d(Y9relu_left, W10, stride=1)
        Y10_right = conv2d(Y9relu_right, W10, stride=1)
        Y10bn_left, update_ema10_left = batchnorm(Y10_left, tst, B10, convolutional=True)
        Y10bn_right, update_ema10_right = batchnorm(Y10_right, tst, B10, convolutional=True)
        # Y10bn_left, _= batchnorm(Y10_left, tst, B10, convolutional=True)
        # Y10bn_right, _= batchnorm(Y10_right, tst, B10, convolutional=True)
        Y10relu_left = tf.nn.relu(Y10bn_left)
        Y10relu_right = tf.nn.relu(Y10bn_right)

    with tf.name_scope('Conv11'):
        with tf.variable_scope('params', reuse=SHARE):
            W11 = weight_variable((3, 3, F, F))
            B11 = bias_variable((F,))
        Y11_left = conv2d(Y10relu_left, W11, stride=1)
        Y11_right = conv2d(Y10relu_right, W11, stride=1)
        Y11bn_left ,update_ema11_left = batchnorm(Y11_left + Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
                                                  tst, B11, convolutional=True)
        Y11bn_right, update_ema11_right = batchnorm(Y11_right + Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
                                                    tst, B11, convolutional=True)
        # Y11bn_left, _= batchnorm(Y11_left + Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
        #                                           tst, B11, convolutional=True)
        # Y11bn_right, _= batchnorm(Y11_right + Y9_right + Y7_right + Y5_right + Y3_right + Y1_right, tst, B11, convolutional=True)
        Y11relu_left = tf.nn.relu(Y11bn_left)
        Y11relu_right = tf.nn.relu(Y11bn_right)

    with tf.name_scope('Conv12'):
        with tf.variable_scope('params', reuse=SHARE):
            W12 = weight_variable((3, 3, F, F))
            B12 = bias_variable((F,))
        Y12_left = conv2d(Y11relu_left, W12, stride=1)
        Y12_right = conv2d(Y11relu_right, W12, stride=1)
        Y12bn_left, update_ema12_left = batchnorm(Y12_left, tst, B12, convolutional=True)
        Y12bn_right, update_ema12_right = batchnorm(Y12_right, tst, B12, convolutional=True)
        # Y12bn_left, _= batchnorm(Y12_left, tst, B12, convolutional=True)
        # Y12bn_right, _= batchnorm(Y12_right, tst, B12, convolutional=True)
        Y12relu_left = tf.nn.relu(Y12bn_left)
        Y12relu_right = tf.nn.relu(Y12bn_right)

    with tf.name_scope('Conv13'):
        with tf.variable_scope('params', reuse=SHARE):
            W13 = weight_variable((3, 3, F, F))
            B13 = bias_variable((F,))
        Y13_left = conv2d(Y12relu_left, W13, stride=1)
        Y13_right = conv2d(Y12relu_right, W13, stride=1)
        Y13bn_left, update_ema13_left = batchnorm(
            Y13_left + Y11_left + Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
            tst, B13, convolutional=True)
        Y13bn_right, update_ema13_right = batchnorm(
            Y13_right + Y11_right + Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
            tst, B13, convolutional=True)
        # Y13bn_left, _= batchnorm(
        #     Y13_left + Y11_left + Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
        #     tst, B13, convolutional=True)
        # Y13bn_right, _= batchnorm(
        #     Y13_right + Y11_right + Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
        #     tst, B13, convolutional=True)
        Y13relu_left = tf.nn.relu(Y13bn_left)
        Y13relu_right = tf.nn.relu(Y13bn_right)

    with tf.name_scope('Conv14'):
        with tf.variable_scope('params', reuse=SHARE):
            W14 = weight_variable((3, 3, F, F))
            B14 = bias_variable((F,))
        Y14_left = conv2d(Y13relu_left, W14, stride=1)
        Y14_right = conv2d(Y13relu_right, W14, stride=1)
        Y14bn_left, update_ema14_left = batchnorm(Y14_left, tst, B14, convolutional=True)
        Y14bn_right, update_ema14_right = batchnorm(Y14_right, tst, B14, convolutional=True)
        # Y14bn_left, _= batchnorm(Y14_left, tst, B14, convolutional=True)
        # Y14bn_right, _= batchnorm(Y14_right, tst, B14, convolutional=True)
        Y14relu_left = tf.nn.relu(Y14bn_left)
        Y14relu_right = tf.nn.relu(Y14bn_right)

    with tf.name_scope('Conv15'):
        with tf.variable_scope('params', reuse=SHARE):
            W15 = weight_variable((3, 3, F, F))
            B15 = bias_variable((F,))
        Y15_left = conv2d(Y14relu_left, W15, stride=1)
        Y15_right = conv2d(Y14relu_right, W15, stride=1)
        Y15bn_left, update_ema15_left = batchnorm(
            Y15_left + Y13_left + Y11_left + Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
            tst, B15, convolutional=True)
        Y15bn_right, update_ema15_right = batchnorm(
            Y15_right+ Y13_right + Y11_right + Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
            tst, B15, convolutional=True)
        # Y15bn_left, _= batchnorm(
        #     Y15_left + Y13_left + Y11_left + Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
        #     tst, B15, convolutional=True)
        # Y15bn_right, _= batchnorm(
        #     Y15_right + Y13_right + Y11_right + Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
        #     tst, B15, convolutional=True)
        Y15relu_left = tf.nn.relu(Y15bn_left)
        Y15relu_right = tf.nn.relu(Y15bn_right)

    with tf.name_scope('Conv16'):
        with tf.variable_scope('params', reuse=SHARE):
            W16 = weight_variable((3, 3, F, F))
            B16 = bias_variable((F,))
        Y16_left = conv2d(Y15relu_left, W16, stride=1)
        Y16_right = conv2d(Y15relu_right, W16, stride=1)
        Y16bn_left, update_ema16_left = batchnorm(Y16_left, tst, B16, convolutional=True)
        Y16bn_right, update_ema16_right = batchnorm(Y16_right, tst, B16, convolutional=True)
        # Y16bn_left, _= batchnorm(Y16_left, tst, B16, convolutional=True)
        # Y16bn_right, _= batchnorm(Y16_right, tst, B16, convolutional=True)
        Y16relu_left = tf.nn.relu(Y16bn_left)
        Y16relu_right = tf.nn.relu(Y16bn_right)

    with tf.name_scope('Conv17'):
        with tf.variable_scope('params', reuse=SHARE):
            W17 = weight_variable((3, 3, F, F))
            B17 = bias_variable((F,))
        Y17_left = conv2d(Y16relu_left, W17, stride=1)
        Y17_right = conv2d(Y16relu_right, W17, stride=1)
        Y17bn_left, update_ema17_left = batchnorm(
            Y17_left + Y15_left + Y13_left + Y11_left + Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
            tst, B17, convolutional=True
        )
        Y17bn_right, update_ema17_right = batchnorm(
            Y17_right + Y15_right + Y13_right + Y11_right + Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
            tst, B17, convolutional=True
        )
        # Y17bn_left, _= batchnorm(
        #     Y17_left + Y15_left + Y13_left + Y11_left + Y9_left + Y7_left + Y5_left + Y3_left + Y1_left,
        #     tst, B17, convolutional=True
        # )
        # Y17bn_right, _= batchnorm(
        #     Y17_right + Y15_right + Y13_right + Y11_right + Y9_right + Y7_right + Y5_right + Y3_right + Y1_right,
        #     tst, B17, convolutional=True
        # )
        Y17relu_left = tf.nn.relu(Y17bn_left)
        Y17relu_right = tf.nn.relu(Y17bn_right)

    with tf.name_scope('Conv18'):
        with tf.variable_scope('params', reuse=SHARE):
            W18 = weight_variable((3, 3, F, F))
            B18 = bias_variable((F,))
        Y18_left = conv2d(Y17relu_left, W18, stride=1)
        Y18_right = conv2d(Y17relu_right, W18, stride=1)

    update_uf_left = tf.group(update_ema1_left, update_ema2_left, update_ema3_left, update_ema4_left, update_ema5_left,
                              update_ema6_left, update_ema7_left, update_ema8_left, update_ema9_left, update_ema10_left,
                              update_ema11_left, update_ema12_left, update_ema13_left, update_ema14_left,
                              update_ema15_left, update_ema16_left, update_ema17_left)

    update_uf_right = tf.group(update_ema1_right, update_ema2_right, update_ema3_right, update_ema4_right, update_ema5_right,
                               update_ema6_right, update_ema7_right, update_ema8_right, update_ema9_right, update_ema10_right,
                               update_ema11_right, update_ema12_right, update_ema13_right, update_ema14_right, update_ema15_right,
                               update_ema16_right, update_ema17_right)

    return Y18_left, Y18_right, update_uf_left, update_uf_right
