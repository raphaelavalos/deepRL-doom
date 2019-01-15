import tensorflow as tf
from Neural_network.predictor_v2 import DOOM_Predictor
from simulator.multi_doom_simulator import MultiDoomSimulator
from data.memory import Memory
import numpy as np


class Agent:

    def __init__(self, conf):
        self.conf = conf
        self.graph = tf.Graph()
        self.saver = tf.train.Saver()


        # Configuration for session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = False

        with self.graph.device(self.conf.device):
            # Session creation
            self.sess = tf.Session(config=config)

            # Placeholder creation
            self._visual_placeholder = tf.placeholder(dtype=tf.float32,
                                                      shape=(-1,) + conf['image_resolution'],
                                                      name='visual_placeholder')
            self._measurement_placeholder = tf.placeholder(dtype=tf.float32,
                                                           shape=(-1, conf['measurement_dim']),
                                                           name='measurement_placeholder')
            self._goal_placeholder = tf.placeholder(dtype=tf.float32,
                                                    shape=(-1, conf['measurement_dim'] * conf['offsets_dim']),
                                                    name="goal_placeholder")
            self._true_action_placeholder = tf.placeholder(dtype=tf.int32,
                                                           shape=(-1,),
                                                           name="true_action_placeholder")
            self._true_future_placeholder = tf.placeholder(dtype=tf.float32,
                                                           shape=(-1, conf['measurement_dim'] * conf['offsets_dim']),
                                                           name='true_future_placeholder')
            self.doom_predictor = DOOM_Predictor(conf,
                                                 self._visual_placeholder,
                                                 self._measurement_placeholder,
                                                 self._goal_placeholder,
                                                 self._true_action_placeholder,
                                                 self._true_future_placeholder)

            self.learning_step = self.doom_predictor.learning_step

            # Initialise all variables
            init = tf.initialize_all_variables
            self.sess.run([init])

        self.memory = Memory(conf.memory)  # TODO : See how handle args with train
        self.doom_simulator = MultiDoomSimulator(conf.simulator, self.memory)  # TODO See how handle args with train
        self.doom_simulator.init_simulators()

    def run_episode(self, epsilon):  # TODO : pass epsilon as arg
        running_simulators = list(range(self.doom_simulator.nbr_of_simulators))
        self.doom_simulator.new_episodes()
        goal = np.random.rand(self.doom_simulator.nbr_of_simulators, 3)  # TODO: replace 3
        images, measures = self.doom_simulator.get_state()
        while len(running_simulators) != 0:
            p = np.random.random()  # TODO: need to replace with a vector of len running_simulators
            if p > epsilon:
                feed_dict = {self._visual_placeholder: images,
                             self._measurement_placeholder: measures,
                             self._goal_placeholder: goal[running_simulators].reshape(-1, 1)}
                next_actions = self.sess.run(self.doom_predictor.action_chooser, feed_dict=feed_dict)
                images, measures, _, _, running_simulators = self.doom_simulator.step(next_actions, running_simulators)
            else:
                images, measures, _, _, running_simulators = self.doom_simulator.step(None, running_simulators)

    def get_learning_step(self, batch_size):
        batch =  self.memory.get_batch(batch_size)
        feed_dict = { self._visual_placeholder: batch[0],
                      self._measurement_placeholder: batch[1],
                      self._goal_placeholder: batch[2], # TODO: Make sure we reintroduce goal in get_batch
                      self._true_action_placeholder: batch[3],
                      self._true_future_placeholder: batch[4]}


        self.sess.run(self.learning_step,
                      feed_dict)

    def save_pred(self, path, epoch, step):
        self.saver.save(self.sess, path + "epoch_%s_step_%s.tf" % (epoch, step))

        # To get predictions, learning_step... doom_predictor._predictions ..., do not forget to feed!
