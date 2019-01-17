import tensorflow as tf
import numpy as np
import math


class DOOM_Predictor():
    """
    DOOM Predictor implenting the paper 'Learning To Act By Predicting The Future'
    """

    def __init__(self, conf, visual, measurement, goal, true_action, true_future):
        self.conf = conf

        # For now we are saving the groups and intermediate outputs to facilitate debugging
        # After that will just need to call DOOM_Predictor._build_net to get prediction

        prediction = DOOM_Predictor._build_net(conf, visual, measurement, goal)
        print(prediction)
        self._prediction = prediction

        # Choose action
        self._goal_for_action_selection = tf.constant([[[0, 0, 0, 0.5, 0.5, 1]]], dtype=tf.float32)
        action_chooser = DOOM_Predictor._choose_action(conf['action'], prediction, self._goal_for_action_selection,
                                                       goal)
        self.action_chooser = action_chooser

        # Loss
        loss = tf.losses.mean_squared_error(true_future,
                                            tf.squeeze(tf.batch_gather(prediction, tf.expand_dims(true_action, -1)), 1))
        loss_summary = tf.summary.scalar("Loss", loss)
        self.loss = loss
        # Optimizer
        learning_rate, optimizer, learning_step, detailed_summary, param_summary = DOOM_Predictor._build_optimizer(
            conf['optimizer'], loss)

        self.param_summary = param_summary
        self.detailed_summary = tf.summary.merge([loss_summary] + detailed_summary, name="Detailed_summary")
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
        with tf.variable_scope(name):
            conv_group = []
            conv = None
            channel = 1
            for i in range(conf['conv_nbr']):
                conv_conf = conf['conv_%i' % i]
                _input = perception_input if i == 0 else conv_group[-1]
                mrsa = 0.9 / math.sqrt(0.5 * (conv_conf['kernel_size']**2) * channel)
                conv = tf.layers.conv2d(inputs=_input,
                                        filters=conv_conf['filters'],
                                        kernel_size=conv_conf['kernel_size'],
                                        strides=conv_conf['stride'],
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=mrsa),
                                        bias_initializer=tf.constant_initializer(0.),
                                        padding="SAME",
                                        name="conv_%i" % i)
                conv_group.append(conv)
                channel = conv_conf['filters']
            flatten_conv_output = tf.layers.flatten(conv)
            dense_conf = conf['dense']
            mrsa = 0.9 / math.sqrt(0.5 * flatten_conv_output.get_shape().as_list()[-1])
            dense_layer = tf.layers.dense(inputs=flatten_conv_output,
                                          units=dense_conf['units'],
                                          use_bias=True,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=mrsa),
                                          bias_initializer=tf.constant_initializer(0.),
                                          name="dense")
        return conv_group, dense_layer

    @staticmethod
    def _build_dense(conf, inputs, name, last_linear=False):
        assert conf['dense_nbr'] > 0, "Need at least one dense layer.\nCheck configuration."
        with tf.variable_scope(name):
            dense_group = []
            xavier_init = tf.contrib.layers.xavier_initializer()
            dense_layer = None
            for i in range(conf['dense_nbr']):
                dense_conf = conf['dense_%i' % i]
                _input = inputs if i == 0 else dense_group[-1]
                mrsa = 0.9 / math.sqrt(0.5 * _input.get_shape().as_list()[-1])
                activation = None if i == (conf['dense_nbr'] - 1) and last_linear else tf.nn.leaky_relu
                dense_layer = tf.layers.dense(inputs=_input,
                                              units=dense_conf['units'],
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=mrsa),
                                              bias_initializer=tf.constant_initializer(0.),
                                              activation=activation,
                                              name="dense_%i" % i)
                dense_group.append(dense_layer)
        return dense_group, dense_layer

    @staticmethod
    def _build_action(conf, inputs, name="action"):
        with tf.variable_scope(name):
            dense_group, dense_layer = DOOM_Predictor._build_dense(conf['dense'], inputs, "dense_%s" % name,
                                                                   last_linear=True)
            action_reshaped = tf.reshape(dense_layer,
                                         shape=(-1, conf['action_nbr'], conf['offsets_dim'] * conf['measurement_dim']))
            action_normalized = action_reshaped - tf.reduce_mean(action_reshaped, axis=1, keepdims=True)
            action_normalized = tf.reshape(action_normalized,
                                           shape=(
                                               -1, conf['action_nbr'] * conf['offsets_dim'] * conf['measurement_dim']))
        return dense_group, dense_layer, action_normalized

    @staticmethod
    def _build_prediction(conf, expectation, action, name="prediction"):
        with tf.variable_scope(name):
            tiled_expectation = tf.tile(expectation, [1, conf['action_nbr']], name="tile_expectation")
            prediction = tiled_expectation + action
            prediction = tf.reshape(prediction,
                                    (-1, conf['action_nbr'], conf['offsets_dim'], conf['measurement_dim']))
        return prediction

    @staticmethod
    def _build_net(conf, image, measurement, goal):
        _, perception_output = DOOM_Predictor._build_perception(conf['perception'], image)
        _, measurement_output = DOOM_Predictor._build_dense(conf['measurement'], measurement, 'measurement')
        if conf['use_goal']:
            goal_tiled = tf.tile(goal, [1, conf['offsets_dim']])
            _, goal_output = DOOM_Predictor._build_dense(conf['goal'], goal_tiled, 'goal')
            representation_j = tf.concat([perception_output, measurement_output, goal_output],
                                         axis=-1,
                                         name='representation_j')
        else:
            representation_j = tf.concat([perception_output, measurement_output],
                                         axis=-1,
                                         name='representation_j')
        _, expection_output = DOOM_Predictor._build_dense(conf['expectation'], representation_j, 'expectation')
        _, _, action_normalized = DOOM_Predictor._build_action(conf['action'], representation_j)
        prediction = DOOM_Predictor._build_prediction(conf['action'], expection_output, action_normalized)
        return prediction

    @staticmethod
    def _build_optimizer(conf, loss):
        with tf.variable_scope('optimizer'):
            detailed_summary = []
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate=np.array(conf['learning_rate'], dtype=np.float32),
                                                       global_step=global_step,
                                                       decay_steps=conf['decay_steps'],
                                                       decay_rate=conf['decay_rate'],
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.95, beta2=0.999, epsilon=1e-4)
            t_vars = tf.trainable_variables()
            if True:
                grads, grad_norm = tf.clip_by_global_norm(tf.gradients(loss, t_vars), 1.0)
                detailed_summary += [tf.summary.scalar("gradient norm", grad_norm)]
                grads_and_vars = list(zip(grads, t_vars))
            else:
                grads_and_vars = optimizer.compute_gradients(loss, var_list=t_vars)

            learning_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            param_hists = [tf.summary.histogram(gv[1].name, gv[1]) for gv in grads_and_vars]
            grad_hists = [tf.summary.histogram(gv[1].name + '/gradients', gv[0]) for gv in grads_and_vars]

            detailed_summary += [tf.summary.scalar("learning rate", learning_rate)]
            param_summary = tf.summary.merge(param_hists + grad_hists, name="Param_summary")

        return learning_rate, optimizer, learning_step, detailed_summary, param_summary

    @staticmethod
    def _choose_action(conf, prediction, goal_weigh, goal, multinomial=False):
        with tf.variable_scope('choose_action'):
            if goal is None:
                assert conf['measurement_dim'] in [1, 3], 'Measurement dim different from 1 and 3'
                if conf['measurement_dim'] == 1:
                    goal = tf.constant([1], tf.float32)
                else:
                    goal = tf.constant([0.5, 0.5, 1.], tf.float32)
            weighted_actions = tf.reduce_sum(
                goal_weigh * tf.reduce_sum(prediction * tf.reshape(goal, (-1, 1, 1, conf['measurement_dim'])), -1),
                -1)
            if multinomial:
                action = tf.reshape(tf.multinomial(weighted_actions, 1), (-1,))
            else:
                action = tf.argmax(weighted_actions, axis=-1)
        return action
