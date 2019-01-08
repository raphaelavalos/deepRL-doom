import tensorflow as tf


class DOOM_Predictor():
    """
    DOOM Predictor implenting the paper 'Learning To Act By Predicting The Future'
    """

    def __init__(self, conf):
        self._visual_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(-1, conf.arch['vis_size'], conf.arch['vis_size'], 1),
                                                  name='visual_input')
        self._measurement_placeholder = tf.placeholder(dtype=tf.float32,
                                                       shape=(-1, conf.arch['meas_size']),
                                                       name='meas_input')
        self._goal_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(-1, conf.arch['num_time'], conf.arch['goal_size']))



    def predict(self, data):
        # data is (batch,s=image,m=measurement,g=goal)
        pass

    def optimize(self, data):
        # data is ((batch,s=image,m=measurement,g=goal), (batch, target))
        pass

    def chose_action(self, data):
        # data is (batch,s=image,m=measurement,g=goal)
        # calls predict
        pass


