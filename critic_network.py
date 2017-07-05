import tensorflow as tf






def SoftArgmin(outputLeft, outputRight, D=192):

    left_result_D = outputLeft
    right_result_D = outputRight
    left_result_D_squeeze = tf.squeeze(left_result_D, axis=[0, 4])
    right_result_D_squeeze = tf.squeeze(right_result_D, axis=[0, 4])  # 192 256 512
    left_result_softmax = tf.nn.softmax(left_result_D_squeeze, dim=0)
    right_result_softmax = tf.nn.softmax(right_result_D_squeeze, dim=0)  # 192 256 512

    d_grid = tf.cast(tf.range(D), tf.float32)
    d_grid = tf.reshape(d_grid, (-1, 1, 1))
    d_grid = tf.tile(d_grid, [1, 256, 512])

    left_softargmin = tf.reduce_sum(tf.multiply(left_result_softmax, d_grid), axis=0, keep_dims=True)
    right_softargmin = tf.reduce_sum(tf.multiply(right_result_softmax, d_grid), axis=0, keep_dims=True)

    return left_softargmin, right_softargmin

def CriticNet(prediction, cPrediction, gt, cGt):

    with tf.name_scope('Loss'):
        loss_left = tf.reduce_mean(tf.abs(gt - prediction))
        loss_right = tf.reduce_mean(tf.abs(cGt - cPrediction))
        loss = loss_left + loss_right
        tf.summary.scalar('Loss/left', loss_left)
        tf.summary.scalar('Loss/right', loss_right)
        tf.summary.scalar('Loss/l1_loss', loss)

    return loss

def TrainingOp(loss, dataSetSize, batch_size, max_grad_norm):

    var_list = tf.trainable_variables()
    grads = tf.gradients(loss, var_list)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), trainable=False
    )
    training_steps_per_epoch = dataSetSize // batch_size
    learning_rate = tf.train.exponential_decay(
        1e-3, global_step, training_steps_per_epoch, 0.999,staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, var_list), global_step=global_step)

    return train_op, learning_rate