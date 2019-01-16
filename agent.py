import tensorflow as tf
from model.predictor import DOOM_Predictor
from simulator.multi_doom_simulator import MultiDoomSimulator
from data.memory import Memory
import numpy as np
import os


class Agent:

    def __init__(self, conf):
        self.conf = conf
        self.graph = tf.Graph()

        # Configuration for session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = False

        with self.graph.device(self.conf['device']):
            # Session creation
            self.sess = tf.Session(config=config)
            # Placeholder creation
            self._visual_placeholder = tf.placeholder(dtype=tf.float32,
                                                      shape=(None,) + conf['image_resolution'],
                                                      name='visual_placeholder')
            self._measurement_placeholder = tf.placeholder(dtype=tf.float32,
                                                           shape=(None, conf['measurement_dim']),
                                                           name='measurement_placeholder')
            self._goal_placeholder = tf.placeholder(dtype=tf.float32,
                                                    shape=(None, conf['measurement_dim']),
                                                    name="goal_placeholder")
            self._true_action_placeholder = tf.placeholder(dtype=tf.int32,
                                                           shape=(None,),
                                                           name="true_action_placeholder")
            self._true_future_placeholder = tf.placeholder(dtype=tf.float32,
                                                           shape=(None, conf['offsets_dim'], conf['measurement_dim']),
                                                           name='true_future_placeholder')
            self.doom_predictor = DOOM_Predictor(conf,
                                                 self._visual_placeholder,
                                                 self._measurement_placeholder,
                                                 self._goal_placeholder,
                                                 self._true_action_placeholder,
                                                 self._true_future_placeholder)

            self.learning_step = self.doom_predictor.learning_step
            self.loss_summary = tf.summary.scalar("Loss", self.doom_predictor.loss)
            self.counter = 0

            # Initialise all variables
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.writer = tf.summary.FileWriter("log/" + conf['experiment_name'], self.sess.graph)
            self.sess.run([init])

        self.memory = Memory(conf)
        self.doom_simulator = MultiDoomSimulator(conf, self.memory)
        self.doom_simulator.init_simulators()

    def run_episode(self, epsilon):
        running_simulators = list(range(self.doom_simulator.nbr_of_simulators))
        self.doom_simulator.new_episodes()
        goal = np.random.rand(self.doom_simulator.nbr_of_simulators,
                              self.conf['measurement_dim'])
        images, measures = self.doom_simulator.get_state()
        while len(running_simulators) != 0:
            p = np.random.random()  # TODO: need to replace with a vector of len running_simulators
            if p > epsilon:
                feed_dict = {self._visual_placeholder: images,
                             self._measurement_placeholder: measures,
                             self._goal_placeholder: goal[running_simulators]}
                next_actions = self.sess.run(self.doom_predictor.action_chooser, feed_dict=feed_dict)

                images, measures, _, _, running_simulators, _ = self.doom_simulator.step(next_actions, goal,
                                                                                         running_simulators)
            else:
                images, measures, _, _, running_simulators, _ = self.doom_simulator.step(None, goal,
                                                                                         running_simulators)

    def validate(self):
        running_simulators = list(range(self.doom_simulator.nbr_of_simulators))
        self.doom_simulator.new_episodes()
        goal = np.random.rand(self.doom_simulator.nbr_of_simulators,
                              self.conf['measurement_dim'])
        images, measures = self.doom_simulator.get_state()
        f_measures = []
        while len(running_simulators) != 0:
            feed_dict = {self._visual_placeholder: images,
                         self._measurement_placeholder: measures,
                         self._goal_placeholder: goal[running_simulators]}
            next_actions = self.sess.run(self.doom_predictor.action_chooser, feed_dict=feed_dict)

            images, measures, _, _, running_simulators, f_measure = self.doom_simulator.step(next_actions, goal,
                                                                                             running_simulators)

            f_measures.append(f_measure)
        f_measures = np.concatenate(f_measures)
        return f_measures
        # print('Medium measure at the end ', f_measures.mean())

    def get_learning_step(self, batch_size):
        images, measures, actions, targets, goals = self.memory.get_batch(batch_size)
        feed_dict = {self._visual_placeholder: images,
                     self._measurement_placeholder: measures,
                     self._goal_placeholder: goals,
                     self._true_action_placeholder: actions,
                     self._true_future_placeholder: targets}

        _, loss_summary_ = self.sess.run([self.learning_step, self.loss_summary], feed_dict)

        self.writer.add_summary(loss_summary_, self.counter)
        self.counter += 1

    def save_pred(self, path, epoch):
        self.saver.save(self.sess, os.path.join(path + "epoch_%s.tf" % epoch))

        # To get predictions, learning_step... doom_predictor._predictions ..., do not forget to feed!

    def random_exploration_prob(self,epoch):
        #Here a epoch is a train on one batch plus adding the new experiences to memory
        return lambda step: (0.02 + 145000. / (float(epoch) + 150000.))