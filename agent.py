import tensorflow as tf
from Neural_network.predictor_v2 import DOOM_Predictor


class Agent:

    def __init__(self, conf):
        self._visual_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(-1,) + conf['image_resolution'],
                                                  name='_visual_placeholder')
        self._measurement_placeholder = tf.placeholder(dtype=tf.float32,
                                                       shape=(-1, conf['measurement_dim']),
                                                       name='_measurement_placeholder')
        self._goal_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(-1, conf['measurement_dim'] * conf['offsets_dim']),
                                                name="_goal_placeholder")
        self._true_action_placeholder = tf.placeholder(dtype=tf.int32,
                                                       shape=(-1,),
                                                       name="_true_action_placeholder")
        self._true_future_placeholder = tf.placeholder(dtype=tf.float32,
                                                       shape=(-1, conf['measurement_dim'] * conf['offsets_dim']),
                                                       name='_true_future_placeholder')
        doom_predictor = DOOM_Predictor(conf,
                                        self._visual_placeholder,
                                        self._measurement_placeholder,
                                        self._goal_placeholder,
                                        self._true_action_placeholder,
                                        self._true_future_placeholder)

        # To get predictions, learning_step... doom_predictor._predictions ..., do not forget to feed!
