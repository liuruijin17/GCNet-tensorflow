import tensorflow as tf
from utils import weight_variable, bias_variable, conv3d, conv3dt, batchnorm






def LearningRegularization(cv_left, cv_right, tst,
                           batch_size=1, F=32, D=192, H=256, W=512, SHARE=None):


    with tf.name_scope('Conv3d19'):
        with tf.variable_scope('params', reuse=SHARE):
            W19 = weight_variable((3, 3, 3, 2 * F, F))
            B19 = bias_variable((F,))
        Y19_left = conv3d(cv_left, W19, stride=1)            ###batch D/2 H/2 W/2 F
        Y19_right = conv3d(cv_right, W19, stride=1)
        Y19bn_left, update_ema19_left = batchnorm(Y19_left, tst, B19)
        Y19bn_right, update_ema19_right = batchnorm(Y19_right, tst, B19)
        # Y19bn_left, _= batchnorm(Y19_left, tst, B19)
        # Y19bn_right, _= batchnorm(Y19_right, tst, B19)
        Y19relu_left = tf.nn.relu(Y19bn_left)
        Y19relu_right = tf.nn.relu(Y19bn_right)

    with tf.name_scope('Conv3d20'):
        with tf.variable_scope('params', reuse=SHARE):
            W20 = weight_variable((3, 3, 3, F, F))
            B20 = bias_variable((F,))
        Y20_left = conv3d(Y19relu_left, W20, stride=1)                   ###batch D/2 H/2 W/2 F
        Y20_right = conv3d(Y19relu_right, W20, stride=1)
        Y20bn_left, update_ema20_left = batchnorm(Y20_left, tst, B20)
        Y20bn_right, update_ema20_right = batchnorm(Y20_right, tst, B20)
        # Y20bn_left, _= batchnorm(Y20_left, tst, B20)
        # Y20bn_right, _= batchnorm(Y20_right, tst, B20)
        Y20relu_left = tf.nn.relu(Y20bn_left)
        Y20relu_right = tf.nn.relu(Y20bn_right)


    with tf.name_scope('Conv3d21'):
        with tf.variable_scope('params', reuse=SHARE):
            W21 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B21 = bias_variable((2 * F,))
        Y21_left = conv3d(cv_left, W21, stride=2)            ###batch D/4 H/4 W/4 2F
        Y21_right = conv3d(cv_right, W21, stride=2)
        Y21bn_left, update_ema21_left = batchnorm(Y21_left, tst, B21)
        Y21bn_right, update_ema21_right = batchnorm(Y21_right, tst, B21)
        # Y21bn_left, _= batchnorm(Y21_left, tst, B21)
        # Y21bn_right, _= batchnorm(Y21_right, tst, B21)
        Y21relu_left = tf.nn.relu(Y21bn_left)
        Y21relu_right = tf.nn.relu(Y21bn_right)

    with tf.name_scope('Conv3d22'):
        with tf.variable_scope('params', reuse=SHARE):
            W22 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B22 = bias_variable((2 * F,))
        Y22_left = conv3d(Y21relu_left, W22, stride=1)                   ###batch D/4 H/4 W/4 2F
        Y22_right = conv3d(Y21relu_right, W22, stride=1)
        Y22bn_left, update_ema22_left = batchnorm(Y22_left, tst, B22)
        Y22bn_right, update_ema22_right = batchnorm(Y22_right, tst, B22)
        # Y22bn_left, _= batchnorm(Y22_left, tst, B22)
        # Y22bn_right, _= batchnorm(Y22_right, tst, B22)
        Y22relu_left = tf.nn.relu(Y22bn_left)
        Y22relu_right = tf.nn.relu(Y22bn_right)

    with tf.name_scope('Conv3d23'):
        with tf.variable_scope('params', reuse=SHARE):
            W23 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B23 = bias_variable((2 * F,))
        Y23_left = conv3d(Y22relu_left, W23, stride=1)                   ###batch D/4 H/4 W/4 2F
        Y23_right = conv3d(Y22relu_right, W23, stride=1)
        Y23bn_left, update_ema23_left = batchnorm(Y23_left, tst, B23)
        Y23bn_right, update_ema23_right = batchnorm(Y23_right, tst, B23)
        # Y23bn_left, _= batchnorm(Y23_left, tst, B23)
        # Y23bn_right, _= batchnorm(Y23_right, tst, B23)
        Y23relu_left = tf.nn.relu(Y23bn_left)
        Y23relu_right = tf.nn.relu(Y23bn_right)


    with tf.name_scope('Conv3d24'):
        with tf.variable_scope('params', reuse=SHARE):
            W24 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B24 = bias_variable((2 * F,))
        Y24_left = conv3d(Y21relu_left, W24, stride=2)                   ###batch D/8 H/8 W/8 2F
        Y24_right = conv3d(Y21relu_right, W24, stride=2)
        Y24bn_left, update_ema24_left = batchnorm(Y24_left, tst, B24)
        Y24bn_right, update_ema24_right = batchnorm(Y24_right, tst, B24)
        # Y24bn_left, _= batchnorm(Y24_left, tst, B24)
        # Y24bn_right, _= batchnorm(Y24_right, tst, B24)
        Y24relu_left = tf.nn.relu(Y24bn_left)
        Y24relu_right = tf.nn.relu(Y24bn_right)

    with tf.name_scope('Conv3d25'):
        with tf.variable_scope('params', reuse=SHARE):
            W25 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B25 = bias_variable((2 * F,))
        Y25_left = conv3d(Y24relu_left, W25, stride=1)                   ###batch D/8 H/8 W/8 2F
        Y25_right = conv3d(Y24relu_right, W25, stride=1)
        Y25bn_left, update_ema25_left = batchnorm(Y25_left, tst, B25)
        Y25bn_right, update_ema25_right = batchnorm(Y25_right, tst, B25)
        # Y25bn_left, _= batchnorm(Y25_left, tst, B25)
        # Y25bn_right, _= batchnorm(Y25_right, tst, B25)
        Y25relu_left = tf.nn.relu(Y25bn_left)
        Y25relu_right = tf.nn.relu(Y25bn_right)

    with tf.name_scope('Conv3d26'):
        with tf.variable_scope('params', reuse=SHARE):
            W26 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B26 = bias_variable((2 * F,))
        Y26_left = conv3d(Y25relu_left, W26, stride=1)                   ###batch D/8 H/8 W/8 2F
        Y26_right = conv3d(Y25relu_right, W26, stride=1)
        Y26bn_left, update_ema26_left = batchnorm(Y26_left, tst, B26)
        Y26bn_right, update_ema26_right = batchnorm(Y26_right, tst, B26)
        # Y26bn_left, _= batchnorm(Y26_left, tst, B26)
        # Y26bn_right, _= batchnorm(Y26_right, tst, B26)
        Y26relu_left = tf.nn.relu(Y26bn_left)
        Y26relu_right = tf.nn.relu(Y26bn_right)

    with tf.name_scope('Conv3d27'):
        with tf.variable_scope('params', reuse=SHARE):
            W27 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B27 = bias_variable((2 * F,))

        Y27_left = conv3d(Y24relu_left, W27, stride=2)                   ###batch D/16 H/16 W/16 2F
        Y27bn_left, update_ema27_left = batchnorm(Y27_left, tst, B27)
        # Y27bn_left, _= batchnorm(Y27_left, tst, B27)
        Y27relu_left = tf.nn.relu(Y27bn_left)

        Y27_right = conv3d(Y24relu_right, W27, stride=2)
        Y27bn_right, update_ema27_right = batchnorm(Y27_right, tst, B27)
        # Y27bn_right, _= batchnorm(Y27_right, tst, B27)
        Y27relu_right = tf.nn.relu(Y27bn_right)

    with tf.name_scope('Conv3d28'):
        with tf.variable_scope('params', reuse=SHARE):
            W28 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B28 = bias_variable((2 * F,))
        Y28_left = conv3d(Y27relu_left, W28, stride=1)                   ###batch D/16 H/16 W/16 2F
        Y28bn_left, update_ema28_left = batchnorm(Y28_left, tst, B28)
        # Y28bn_left, _= batchnorm(Y28_left, tst, B28)
        Y28relu_left = tf.nn.relu(Y28bn_left)

        Y28_right = conv3d(Y27relu_right, W28, stride=1)
        Y28bn_right, update_ema28_right = batchnorm(Y28_right, tst, B28)
        # Y28bn_right, _= batchnorm(Y28_right, tst, B28)
        Y28relu_right = tf.nn.relu(Y28bn_right)

    with tf.name_scope('Conv3d29'):
        with tf.variable_scope('params', reuse=SHARE):
            W29 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B29 = bias_variable((2 * F,))
        Y29_left = conv3d(Y28relu_left, W29, stride=1)                   ###batch D/16 H/16 W/16 2F
        Y29bn_left, update_ema29_left = batchnorm(Y29_left, tst, B29)
        # Y29bn_left, _= batchnorm(Y29_left, tst, B29)
        Y29relu_left = tf.nn.relu(Y29bn_left)

        Y29_right = conv3d(Y28relu_right, W29, stride=1)
        Y29bn_right, update_ema29_right = batchnorm(Y29_right, tst, B29)
        # Y29bn_right, _= batchnorm(Y29_right, tst, B29)
        Y29relu_right = tf.nn.relu(Y29bn_right)

    with tf.name_scope('Conv3d30'):
        with tf.variable_scope('params', reuse=SHARE):
            W30 = weight_variable((3, 3, 3, 2 * F, 4 * F))
            B30 = bias_variable((4 * F,))
        Y30_left = conv3d(Y27relu_left, W30, stride=2)                   ###batch D/32 H/32 W/32 4F
        Y30bn_left, update_ema30_left = batchnorm(Y30_left, tst, B30)
        # Y30bn_left, _= batchnorm(Y30_left, tst, B30)
        Y30relu_left = tf.nn.relu(Y30bn_left)

        Y30_right = conv3d(Y27relu_right, W30, stride=2)
        Y30bn_right, update_ema30_right = batchnorm(Y30_left, tst, B30)
        # Y30bn_right, _= batchnorm(Y30_left, tst, B30)
        Y30relu_right = tf.nn.relu(Y30bn_right)

    with tf.name_scope('Conv3d31'):
        with tf.variable_scope('params', reuse=SHARE):
            W31 = weight_variable((3, 3, 3, 4 * F, 4 * F))
            B31 = bias_variable((4 * F,))
        Y31_left = conv3d(Y30relu_left, W31, stride=1)                   ###batch D/32 H/32 W/32 4F
        Y31bn_left, update_ema31_left = batchnorm(Y31_left, tst, B31)
        # Y31bn_left, _= batchnorm(Y31_left, tst, B31)
        Y31relu_left = tf.nn.relu(Y31bn_left)

        Y31_right = conv3d(Y30relu_right, W31, stride=1)
        Y31bn_right, update_ema31_right = batchnorm(Y31_right, tst, B31)
        # Y31bn_right, _= batchnorm(Y31_right, tst, B31)
        Y31relu_right = tf.nn.relu(Y31bn_right)

    with tf.name_scope('Conv3d32'):
        with tf.variable_scope('params', reuse=SHARE):
            W32 = weight_variable((3, 3, 3, 4 * F, 4 * F))
            B32 = bias_variable((4 * F,))
        Y32_left = conv3d(Y31relu_left, W32, stride=1)                   ###batch D/32 H/32 W/32 4F
        Y32bn_left, update_ema32_left = batchnorm(Y32_left, tst, B32)
        # Y32bn_left, _= batchnorm(Y32_left, tst, B32)
        Y32relu_left = tf.nn.relu(Y32bn_left)

        Y32_right = conv3d(Y31relu_right, W32, stride=1)
        Y32bn_right, update_ema32_right = batchnorm(Y32_right, tst, B32)
        # Y32bn_right, _= batchnorm(Y32_right, tst, B32)
        Y32relu_right = tf.nn.relu(Y32bn_right)

    with tf.name_scope('Conv3d33'):
        with tf.variable_scope('params', reuse=SHARE):
            W33 = weight_variable((3, 3, 3, 2 * F, 4 * F))
            B33 = bias_variable((2 * F,))
        output33shape = [batch_size, D // 16, H // 16, W // 16, 2 * F]
        _Y33_left = conv3dt(Y32relu_left, W33, outputshape=output33shape, stride=2)        ###batch D/16 H/16 W/16 2F
        _Y33bn_left, update_ema33_left = batchnorm(_Y33_left, tst, B33)
        # _Y33bn_left, _= batchnorm(_Y33_left, tst, B33)
        _Y33relu_left = tf.nn.relu(_Y33bn_left)

        _Y33_right = conv3dt(Y32relu_right, W33, outputshape=output33shape, stride=2)
        _Y33bn_right, update_ema33_right = batchnorm(_Y33_right, tst, B33)
        # _Y33bn_right, _= batchnorm(_Y33_right, tst, B33)
        _Y33relu_right = tf.nn.relu(_Y33bn_right)

        Y33relu_left = _Y33relu_left + Y29relu_left
        Y33relu_right = _Y33relu_right + Y29relu_right

    with tf.name_scope('Conv3d34'):
        with tf.variable_scope('params', reuse=SHARE):
            W34 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B34 = bias_variable((2 * F,))
        output34shape = [batch_size, D // 8, H // 8, W // 8, 2 * F]
        _Y34_left = conv3dt(Y33relu_left, W34, outputshape=output34shape, stride=2)        ###batch D/8 H/8 W/8 2F
        _Y34bn_left, update_ema34_left = batchnorm(_Y34_left, tst, B34)
        # _Y34bn_left, _= batchnorm(_Y34_left, tst, B34)
        _Y34relu_left = tf.nn.relu(_Y34bn_left)

        _Y34_right = conv3dt(Y33relu_right, W34, outputshape=output34shape, stride=2)
        _Y34bn_right, update_ema34_right = batchnorm(_Y34_right, tst, B34)
        # _Y34bn_right, _= batchnorm(_Y34_right, tst, B34)
        _Y34relu_right = tf.nn.relu(_Y34bn_right)

        Y34relu_left = _Y34relu_left + Y26relu_left
        Y34relu_right = _Y34relu_right + Y26relu_right

    with tf.name_scope('Conv3d35'):
        with tf.variable_scope('params', reuse=SHARE):
            W35 = weight_variable((3, 3, 3, 2 * F, 2 * F))
            B35 = bias_variable((2 * F,))
        output35shape = [batch_size, D // 4, H // 4, W // 4, 2 * F]
        _Y35_left = conv3dt(Y34relu_left, W35, outputshape=output35shape, stride=2)        ###batch D/4 H/4 W/4 2F
        _Y35bn_left, update_ema35_left = batchnorm(_Y35_left, tst, B35)
        # _Y35bn_left, _= batchnorm(_Y35_left, tst, B35)
        _Y35relu_left = tf.nn.relu(_Y35bn_left)

        _Y35_right = conv3dt(Y34relu_right, W35, outputshape=output35shape, stride=2)
        _Y35bn_right, update_ema35_right = batchnorm(_Y35_right, tst, B35)
        # _Y35bn_right, _= batchnorm(_Y35_right, tst, B35)
        _Y35relu_right = tf.nn.relu(_Y35bn_right)

        Y35relu_left = _Y35relu_left + Y23relu_left
        Y35relu_right = _Y35relu_right + Y23relu_right

    with tf.name_scope('Conv3d36'):
        with tf.variable_scope('params', reuse=SHARE):
            W36 = weight_variable((3, 3, 3, F, 2 * F))
            B36 = bias_variable((F,))
        output36shape = [batch_size, D // 2, H // 2, W // 2, F]
        _Y36_left = conv3dt(Y35relu_left, W36, outputshape=output36shape, stride=2)        ###batch D/2 H/2 W/2 F
        _Y36bn_left, update_ema36_left = batchnorm(_Y36_left, tst, B36)
        # _Y36bn_left, _= batchnorm(_Y36_left, tst, B36)
        _Y36relu_left = tf.nn.relu(_Y36bn_left)

        _Y36_right = conv3dt(Y35relu_right, W36, outputshape=output36shape, stride=2)
        _Y36bn_right, update_ema36_right = batchnorm(_Y36_right, tst, B36)
        # _Y36bn_right, _= batchnorm(_Y36_right, tst, B36)
        _Y36relu_right = tf.nn.relu(_Y36bn_right)

        Y36relu_left = _Y36relu_left + Y20relu_left
        Y36relu_right = _Y36relu_right + Y20relu_right


    update_lr_left = tf.group(update_ema19_left, update_ema20_left, update_ema21_left, update_ema22_left,
                              update_ema23_left, update_ema24_left, update_ema25_left, update_ema26_left,
                              update_ema27_left, update_ema28_left, update_ema29_left, update_ema30_left,
                              update_ema31_left, update_ema32_left, update_ema33_left, update_ema34_left,
                              update_ema35_left, update_ema36_left)

    update_lr_right = tf.group(update_ema19_right, update_ema20_right, update_ema21_right, update_ema22_right,
                               update_ema23_right, update_ema24_right, update_ema25_right, update_ema26_right,
                               update_ema27_right, update_ema28_right, update_ema29_right, update_ema30_right,
                               update_ema31_right, update_ema32_right, update_ema33_right, update_ema34_right,
                               update_ema35_right, update_ema36_right)


    with tf.name_scope('Conv3d37'):
        with tf.variable_scope('params', reuse=SHARE):
            W37 = weight_variable((3, 3, 3, 1, F))
        output37shape = [batch_size, D, H, W, 1]
        Y37_left = conv3dt(Y36relu_left, W37, outputshape=output37shape, stride=2)
        Y37_right = conv3dt(Y36relu_right, W37, outputshape=output37shape, stride=2)


    return Y37_left, Y37_right, update_lr_left, update_lr_right

def LearningRegularizationOmitting(cv_left, cv_right,
                           batch_size=1, F=32, D=192, H=256, W=512, SHARE=None):

    Y36relu_left = cv_left
    Y36relu_right = cv_right
    with tf.name_scope('Conv3d37'):
        with tf.variable_scope('params', reuse=SHARE):
            W37 = weight_variable((3, 3, 3, 1, 2 * F))
        output37shape = [batch_size, D, H, W, 1]
        Y37_left = conv3dt(Y36relu_left, W37, outputshape=output37shape, stride=2)
        Y37_right = conv3dt(Y36relu_right, W37, outputshape=output37shape, stride=2)

    return Y37_left, Y37_right
