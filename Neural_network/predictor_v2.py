import tensorflow as tf
import numpy as np


class DOOM_Predictor():
    """
    DOOM Predictor implenting the paper 'Learning To Act By Predicting The Future'
    """

    def __init__(self, conf, visual, measurement, goal, true_action, true_future):
        self.conf = conf

        # For now we are saving the groups and intermediate outputs to facilitate debugging
        # After that will just need to call DOOM_Predictor._build_net to get prediction

        # Perception
        conv_group, perception_output = DOOM_Predictor._build_perception(conf['perception'], visual)
        self._perception_conv_group = conv_group
        self._perception_output = perception_output

        # Measurement
        measurement_group, measurement_output = DOOM_Predictor._build_dense(conf['measurement'],
                                                                            measurement,
                                                                            'measurement')
        self._measurement_group = measurement_group
        self._measurement_output = measurement_output

        # Goal
        goal_group, goal_output = DOOM_Predictor._build_dense(conf['goal'], goal, 'goal')
        self._goal_group = goal_group
        self._goal_output = goal_output

        # Concatenation of outputs
        representation_j = tf.concat([perception_output, measurement_output, goal_output],
                                     axis=-1,
                                     name='representation_j')
        self._representation_j = representation_j

        # Expectation
        expectation_group, expection_output = DOOM_Predictor._build_dense(conf['expectation'],
                                                                          representation_j,
                                                                          'expectation')
        self._expectation_group = expectation_group
        self._expection_output = expection_output

        # Action
        action_group, action_output, action_normalized = DOOM_Predictor._build_action(conf['action'], representation_j)
        self._action_group = action_group
        self._action_output = action_output
        self._action_normalized = action_normalized

        # Prediction
        prediction = DOOM_Predictor._build_prediction(conf['action'], expection_output, action_normalized)
        self._prediction = prediction

        # Choose action
        self._goal_for_action_selection = tf.constant(0)  # TODO: create tf.constant with value from conf
        action_chooser = DOOM_Predictor._choose_action(conf['choose_action'], prediction,
                                                       self._goal_for_action_selection)
        self.action_chooser = action_chooser

        # TODO: need to check batch_gather output
        # Loss
        loss = tf.losses.mean_squared_error(true_future, tf.batch_gather(prediction, tf.expand_dims(true_action, -1)))

        self.loss = loss
        # Optimizer
        learning_rate, optimizer, learning_step = DOOM_Predictor._build_optimizer(conf['optimizer'], loss)
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self.learning_step = learning_step

    @staticmethod
    def _build_perception(conf, perception_input, name="perception"):
        """

        Args:
            conf (dict): Dictionary containing the perception module configuration
            perception_input (tf.Tensor): Input tensor of the perception module

        Returns:
            Tuple containing the list of the convolution layers and the output Tensor.

        """
        assert conf['conv_nbr'] > 0, "Need at least one convolution.\nCheck perception configuration."
        with tf.name_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            conv_group = []
            conv = None
            for i in range(conf['conv_nbr']):
                conv_conf = conf['conv_%i' % i]
                _input = perception_input if i == 0 else conv_group[-1]
                activation = tf.nn.leaky_relu if i != (conf['conv_nbr'] - 1) else None  # TODO: Check that
                conv = tf.layers.conv2d(inputs=_input,
                                        filters=conv_conf['filters'],
                                        kernel_size=conv_conf['kernel_size'],
                                        strides=conv_conf['strides'],
                                        activation=activation,
                                        kernel_initializer=xavier_init,
                                        bias_initializer=xavier_init,
                                        padding="SAME",
                                        name="conv_%i" % i)
                conv_group.append(conv)
            flatten_conv_output = tf.layers.flatten(conv)
            dense_conf = conf['dense']
            dense_layer = tf.layers.dense(inputs=flatten_conv_output,
                                          units=dense_conf['units'],
                                          use_bias=False,  # TODO: Check that
                                          kernel_initializer=xavier_init,
                                          name="dense")
        return conv_group, dense_layer

    @staticmethod
    def _build_dense(conf, inputs, name):
        assert conf['dense_nbr'] > 0, "Need at least one dense layer.\nCheck configuration."
        with tf.name_scope(name):
            dense_group = []
            xavier_init = tf.contrib.layers.xavier_initializer()
            dense_layer = None
            for i in range(conf['dense_nbr']):
                dense_conf = conf['dense_%i' % i]
                _input = inputs if i == 0 else dense_group[-1]
                # No activation for the last layer
                activation = tf.nn.leaky_relu if i != (conf['dense_nbr'] - 1) else None
                dense_layer = tf.layers.dense(inputs=_input,
                                              units=dense_conf['units'],
                                              kernel_initializer=xavier_init,
                                              activation=activation,
                                              name="dense_%i" % i)
                dense_group.append(dense_layer)
        return dense_group, dense_layer

    @staticmethod
    def _build_action(conf, inputs, name="action"):
        with tf.name_scope(name):
            dense_group, dense_layer = DOOM_Predictor._build_dense(conf['dense'], inputs, "dense" % name)
            action_reshaped = tf.reshape(dense_layer,
                                         shape=(-1, conf['action_nbr'], conf['offsets_dim'] * conf['measurement_dim']))
            action_normalized = action_reshaped - tf.reduce_mean(action_reshaped, axis=1, keepdims=True)
            action_normalized = tf.reshape(action_normalized,
                                           shape=(
                                               -1, conf['action_nbr'] * conf['offsets_dim'] * conf['measurement_dim']))
        return dense_group, dense_layer, action_normalized

    @staticmethod
    def _build_prediction(conf, expectation, action, name="prediction"):
        with tf.name_scope(name):
            tiled_expectation = tf.tile(expectation, [1, conf['action_nbr']], name="tile_expectation")
            prediction = tiled_expectation + action
        return prediction

    @staticmethod
    def _build_net(conf, image, measurement, goal):
        _, perception_output = DOOM_Predictor._build_perception(conf['perception'], image)
        _, measurement_output = DOOM_Predictor._build_dense(conf['measurement'], measurement, 'measurement')
        _, goal_output = DOOM_Predictor._build_dense(conf['goal'], goal, 'goal')
        representation_j = tf.concat([perception_output, measurement_output, goal_output],
                                     axis=-1,
                                     name='representation_j')
        _, expection_output = DOOM_Predictor._build_dense(conf['expectation'], representation_j, 'expectation')
        _, _, action_normalized = DOOM_Predictor._build_action(conf['action'], representation_j)
        prediction = DOOM_Predictor._build_prediction(conf['action'], expection_output, action_normalized)
        return prediction

    @staticmethod
    def _build_optimizer(conf, loss):
        with tf.name_scope('optimizer'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate=np.array(conf['learning_rate'], dtype=np.float32),
                                                       global_step=global_step,
                                                       decay_steps=conf['decay_steps'],
                                                       decay_rate=conf['decay_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate)
            learning_step = optimizer.minimize(loss, global_step)
        return learning_rate, optimizer, learning_step

    @staticmethod
    def _choose_action(conf, prediction, goal):
        """
        Choose action

        Args:
            prediction (tf.Tensor): Tensor containing the prediction, shape=[batch, action_nbr*offsets_dim*measurement_dim]
            goal (tf.Tensor): Tensor of the shape [1,1,offsets_dim*measure_dim] defining the weight of the prediction measures

        Returns:
            tf.Tensor: Tensor containing the index of the chosen actions, shape=[batch,]

        """
        with tf.name_scope('choose_action'):
            reshaped_prediction = tf.reshape(prediction,
                                             (-1, conf['action_nbr'], conf['offsets_dim'] * conf['measurement_dim']))
            weighted_actions = tf.reduce_sum(reshaped_prediction * goal - 1)
            action = tf.argmax(weighted_actions, axis=-1)
        return action
