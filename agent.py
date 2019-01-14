import tensorflow as tf
from Neural_network.predictor_v2 import DOOM_Predictor
from simulator.multi_doom_simulator import  MultiDoomSimulator
from data.memory import Memory

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
        self.doom_predictor = DOOM_Predictor(conf,
                                        self._visual_placeholder,
                                        self._measurement_placeholder,
                                        self._goal_placeholder,
                                        self._true_action_placeholder,
                                        self._true_future_placeholder)

        self.memory = Memory(conf) # TODO : See how handle args with train
        self.doom_simulator = MultiDoomSimulator(conf, self.memory) # TODO See how handle args with train

    def fill_in_memory(self):
        pass

    def get_learning_step(self):
        "Return "
        return self.doom_predictor._learning_step


        # To get predictions, learning_step... doom_predictor._predictions ..., do not forget to feed!
        # TODO : Method to fill memory by calling last predictor version to choose action
        # TODO : Method to train predictor
