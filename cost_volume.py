import tensorflow as tf






def CostVolume(output_left, output_right, batch_size=1, F=32, D=192, H=256, W=512):
    LEFT = output_left   # 1, H//2, W//2, F
    RIGHT = output_right # 1, H//2, W//2, F
    leftPlusrightmove_list = []
    rightPlusleftmove_list = []

    for dis in range(D // 2):
        if dis == 0:
            leftPlusrightmove = tf.concat([LEFT, RIGHT], axis = 3)
            rightPlusleftmove = tf.concat([RIGHT, LEFT], axis = 3)
            leftPlusrightmove_list.append(leftPlusrightmove)
            rightPlusleftmove_list.append(rightPlusleftmove)
        else:
            zerotmp = tf.zeros((batch_size, H // 2, dis, F))

            rightmove = tf.concat([zerotmp, RIGHT], axis = 2) # dis + W // 2
            leftmove = tf.concat([LEFT, zerotmp], axis = 2)# W // 2 + dis

            leftPlusrightmove = tf.concat([LEFT, rightmove[:, :, :W // 2, :]], axis = 3)
            rightPlusleftmove = tf.concat([RIGHT, leftmove[:, :, dis:, :]], axis = 3)

            leftPlusrightmove_list.append(leftPlusrightmove)
            rightPlusleftmove_list.append(rightPlusleftmove)


    left_costvolume = tf.stack(leftPlusrightmove_list, axis = 1)
    right_costvolume  = tf.stack(rightPlusleftmove_list, axis = 1)

    return left_costvolume, right_costvolume
