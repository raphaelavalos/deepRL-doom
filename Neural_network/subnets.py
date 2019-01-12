import tensorflow as tf

number_action = 256
number_measurement = 3
horizon = 6

"""PERCEPTION BLOCK"""
def perception_conv(number, input, filters, kernel_size, strides, padding, activation=tf.nn.leaky_relu):

    output_conv = tf.layers.conv2d(input, filters, kernel_size, strides, padding, name="perception_conv_{}".format(number))
    b = tf.get_variable('b_conv_{}'.format(number), filters, initializer=tf.constant_initializer(0.0))
    output = activation(output_conv, b)

    return output


def perception_full(number, input, output_dim):

    flat = tf.contrib.layers.flatten(input)
    output = tf.layers.dense(flat, output_dim, name="perception_full_{}".format(number))

    return output


def perception_net(input,parameters):
    """"The shape are as in the paper for Basic model"""
    with tf.variable_scope("perception"):
        conv1 = perception_conv('1',
                                input,
                                parameters.get('filters')[0],
                                parameters.get('kernel_size')[0],
                                parameters.get('strides')[0],
                                padding="SAME")
        conv2 = perception_conv('2',
                                conv1,
                                parameters.get('filters')[1],
                                parameters.get('kernel_size')[1],
                                parameters.get('strides')[1],
                                padding="SAME")
        conv3 = perception_conv('3',
                                conv2,
                                parameters.get('filters')[2],
                                parameters.get('kernel_size')[2],
                                parameters.get('strides')[2],
                                padding="SAME")
        output = perception_full('1', conv3, parameters.get('fc_out')[0])

    return output


"""MEASUREMENT BLOCK"""
def measurement_full(number, input, output_dim , activation=tf.nn.leaky_relu):

    output = tf.layers.dense(input, output_dim, name="measurement_full_{}".format(number), activation=activation)

    return output


def measurement_net(input, parameters):
    with tf.variable_scope("measurement"):
        fc1 = measurement_full('1',
                               input,
                               parameters.get('fc_out')[0])
        fc2 = measurement_full('2',
                               fc1,
                               parameters.get('fc_out')[1])
        output = measurement_full('3',
                                  fc2,
                                  parameters.get('fc_out')[2],
                                  activation=None)

    return output


"""GOAL BLOCK"""
def goal_full(number, input, output_dim, activation=tf.nn.leaky_relu):

    output = tf.layers.dense(input, output_dim, name="goal_full_{}".format(number), activation=activation)

    return output


def goal_net(input, parameters):
    with tf.variable_scope("goal"):
        input_flat = tf.contrib.layers.flatten(input)
        fc1 = measurement_full('1',
                               input_flat,
                               parameters.get('fc_out')[0])
        fc2 = measurement_full('2',
                               fc1,
                               parameters.get('fc_out')[1])
        output = measurement_full('3',
                                  fc2,
                                  parameters.get('fc_out')[2],
                                  activation=None)

    return output


"""EXPECTATION BLOCK"""
def expectation_full(number, input, output_dim, activation=tf.nn.leaky_relu):

    output = tf.layers.dense(input, output_dim, name="expectation_full_{}".format(number), activation=activation)

    return output


def expectation_net(perception_out, measurement_out, goal_out, parameters, arch):
    with tf.variable_scope("expectation"):
        input = tf.concat([perception_out, measurement_out, goal_out], 1)

        fc1 = expectation_full('1',
                               input,
                               parameters.get('fc_out')[0])

        output = expectation_full('2',
                                  fc1,
                                  arch.get('meas_size') * arch.get('num_time'),
                                  activation=None)

    return output


"""ACTION BLOCK"""
def action_full(number, input, output_dim, activation=tf.nn.leaky_relu):
    output = tf.layers.dense(input, output_dim, name="action_full_{}".format(number), activation=activation)

    return output


def action_net(perception_out, measurement_out, goal_out, parameters, arch):
    with tf.variable_scope("action"):
        input = tf.concat([perception_out, measurement_out, goal_out], 1)

        fc1 = expectation_full('1',
                               input,
                               parameters.get('fc_out')[0])

        output = expectation_full('2',
                                  fc1,
                                  arch.get('num_action') * arch.get('meas_size') * arch.get('num_time'),
                                  activation=None)

    return output