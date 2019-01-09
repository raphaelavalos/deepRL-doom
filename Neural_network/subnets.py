import tensorflow as tf

number_action = 256
number_measurement = 3
horizon = 6

"""PERCEPTION BLOCK"""
def perception_conv(number, input, filter, strides, padding, activation=tf.nn.leaky_relu):

    output_conv = tf.nn.conv2d(input, filter, strides, padding, name="perception_conv_{}".format(number))
    b = tf.get_variable('b_conv_'.format(number), filter[-1], initializer=tf.constant_initializer(0.0))
    output = activation(output_conv, b)

    return output


def perception_full(number, input, output_dim):

    flat = tf.contrib.layers.flatten(input)
    output = tf.layers.dense(flat, output_dim, name="perception_full_{}".format(number))

    return output


def perception_net(input):
    """"The shape are as in the paper for Basic model"""
    with tf.variable_scope("perception"):
        conv1 = perception_conv('1',
                                input,
                                [8, 8, 1, 32],
                                [1, 4, 4, 1],
                                padding="SAME")
        conv2 = perception_conv('2',
                                conv1,
                                [4, 4, 32, 64],
                                [1, 2, 2, 1],
                                padding="SAME")
        conv3 = perception_conv('3',
                                conv2,
                                [3, 3, 64, 64],
                                [1, 1, 1, 1],
                                padding="SAME")
        output = perception_full('1', conv3, 512)

    return output


"""MEASUREMENT BLOCK"""
def measurement_full(number, input, output_dim):

    output = tf.layers.dense(input, output_dim, name="measurement_full_{}".format(number))

    return output


def measurement_net(input):
    with tf.variable_scope("measurement"):
        fc1 = measurement_full('1', input, 128)

        fc2 = measurement_full('2', fc1, 128)

        output = measurement_full('3', fc2, 128)

    return output


"""GOAL BLOCK"""
def goal_full(number, input, output_dim):

    output = tf.layers.dense(input, output_dim, name="goal_full_{}".format(number))

    return output


def goal_net(input):
    with tf.variable_scope("goal"):
        input_flat = tf.contrib.layers.flatten(input)

        fc1 = goal_full('1', input_flat, 128)

        fc2 = goal_full('2', fc1, 128)

        output = goal_full('3', fc2, 128)

    return output


"""EXPECTATION BLOCK"""
def expectation_full(number, input, output_dim):

    output = tf.layers.dense(input, output_dim, name="expectation_full_{}".format(number))

    return output


def expectation_net(perception_out, measurement_out, goal_out):
    with tf.variable_scope("expectation"):
        input = tf.concat([perception_out, measurement_out, goal_out], 1)

        fc1 = expectation_full('1', input, 512)

        output = expectation_full('2', fc1, number_measurement * horizon)

    return output


"""ACTION BLOCK"""
def action_full(number, input, output_dim):
    output = tf.layers.dense(input, output_dim, name="action_full_{}".format(number))

    return output


def action_net(perception_out, measurement_out, goal_out):
    with tf.variable_scope("action"):
        input = tf.concat([perception_out, measurement_out, goal_out], 1)

        fc1 = expectation_full('1', input, 512)

        output = expectation_full('2', fc1, number_action * number_measurement * horizon)

    return output