import tensorflow as tf
from model.predictor import DOOM_Predictor
from simulator.multi_doom_simulator import MultiDoomSimulator
from data.memory import Memory
from tqdm import trange
import numpy as np
import os


class Agent:

    def __init__(self, conf):
        self.conf = conf
        self.graph = tf.Graph()
        self.use_goal = conf['use_goal']

        # Configuration for session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = False

        tf.random.set_random_seed(124)

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

            if self.use_goal:
                self._goal_placeholder = tf.placeholder(dtype=tf.float32,
                                                        shape=(None, conf['measurement_dim']),
                                                        name="goal_placeholder")
            else:
                self._goal_placeholder = None

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

            self.dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(self._visual_placeholder),
                                                tf.data.Dataset.from_tensor_slices(self._measurement_placeholder),
                                                tf.data.Dataset.from_tensor_slices(self._true_action_placeholder),
                                                tf.data.Dataset.from_tensor_slices(self._true_future_placeholder)))
            # self.dataset = self.dataset.batch(64)
            # self.dataset = self.dataset.repeat()
            # self.iterator = self.dataset.make_initializable_iterator()

            self.learning_step = self.doom_predictor.learning_step
            self.counter = 0
            self.validate_counter = 0

            # Duration log
            self._duration_placeholder = tf.placeholder(dtype=tf.float32,
                                                        shape=(None,),
                                                        name='duration_placeholder')
            self._fmeasure_placeholder = tf.placeholder(dtype=tf.float32,
                                                        shape=(None, conf['measurement_dim']),
                                                        name='fmeasure_placeholder')
            self._actions_placeholder = tf.placeholder(dtype=tf.float32,
                                                       shape=(None,),
                                                       name='actions_placeholder')
            duration_mean, duration_std = tf.nn.moments(self._duration_placeholder, 0)
            duration_sum = tf.summary.tensor_summary("duration_tensor", self._duration_placeholder)
            duration_hist_sum = tf.summary.histogram("duration_tensor_hist", self._duration_placeholder)
            duration_mean_sum = tf.summary.scalar("duration_mean", duration_mean)
            duration_std_sum = tf.summary.scalar("duration_std", duration_std)
            fmeasure_hist_sum = tf.summary.histogram("fmeasures_hist", self._fmeasure_placeholder)
            fmeasure_sum = tf.summary.tensor_summary("fmeasures_tensor", self._fmeasure_placeholder)
            actions_hist_sum = tf.summary.histogram("actions_hist", self._actions_placeholder)
            actions_sum = tf.summary.tensor_summary("actions_tensor", self._actions_placeholder)
            self.validation_summary = tf.summary.merge([duration_mean_sum,
                                                        duration_std_sum,
                                                        duration_sum,
                                                        duration_hist_sum,
                                                        fmeasure_hist_sum,
                                                        fmeasure_sum,
                                                        actions_hist_sum,
                                                        actions_sum],
                                                       name="Validation")
            self.detailed_summary = self.doom_predictor.detailed_summary
            self.param_summary = self.doom_predictor.param_summary

            # Initialise all variables
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.writer = tf.summary.FileWriter("log/" + conf['experiment_name'], self.sess.graph)
            self.sess.run([init])

        self.memory = Memory(conf)
        self.doom_simulator = MultiDoomSimulator(conf, self.memory)
        self.doom_simulator.init_simulators()

    def run_episode(self, epsilon, max_steps=None):
        running_simulators = list(range(self.doom_simulator.nbr_of_simulators))
        self.doom_simulator.new_episodes()
        if self.use_goal:
            goal = np.random.rand(self.doom_simulator.nbr_of_simulators,
                                  self.conf['measurement_dim'])
        else:
            goal = np.ones((self.doom_simulator.nbr_of_simulators, self.conf['measurement_dim']))
        images, measures = self.doom_simulator.get_state()
        while len(running_simulators) != 0:
            p = np.random.random()  # TODO: need to replace with a vector of len running_simulators
            if p > epsilon:
                feed_dict = {self._visual_placeholder: images,
                             self._measurement_placeholder: measures}
                if self.use_goal:
                    feed_dict[self._goal_placeholder] = goal[running_simulators]
                next_actions = self.sess.run(self.doom_predictor.action_chooser, feed_dict=feed_dict)

                images, measures, _, _, running_simulators, _ = self.doom_simulator.step(next_actions, goal,
                                                                                         running_simulators)
            else:
                images, measures, _, _, running_simulators, _ = self.doom_simulator.step(None, goal,
                                                                                         running_simulators)
        self.doom_simulator.build_commit_reset(max_steps)

    def validate(self):
        f_measures = []
        durations = []
        actions = []
        for _ in trange(20, desc="Validation", leave=False):
            running_simulators = list(range(self.doom_simulator.nbr_of_simulators))
            self.doom_simulator.new_episodes()
            if self.use_goal:
                goal = np.random.rand(self.doom_simulator.nbr_of_simulators, self.conf['measurement_dim'])
            else:
                goal = np.ones((self.doom_simulator.nbr_of_simulators, self.conf['measurement_dim']))
            images, measures = self.doom_simulator.get_state()
            while len(running_simulators) != 0:
                feed_dict = {self._visual_placeholder: images,
                             self._measurement_placeholder: measures}
                if self.use_goal:
                    feed_dict[self._goal_placeholder] = goal[running_simulators]
                next_actions = self.sess.run(self.doom_predictor.action_chooser, feed_dict=feed_dict)
                images, measures, _, _, running_simulators, f_measure = self.doom_simulator.step(next_actions, goal,
                                                                                                 running_simulators)

                actions.append(next_actions)
            duration, f_measure = self.doom_simulator.get_duration_and_measurements()
            durations.append(duration)
            f_measures.append(f_measure)
        f_measures = np.concatenate(f_measures, 0)
        durations = np.concatenate(durations)
        actions = np.concatenate(actions, 0)
        validation_summary = self.sess.run(self.validation_summary,
                                           feed_dict={self._duration_placeholder: durations,
                                                      self._fmeasure_placeholder: f_measures,
                                                      self._actions_placeholder: actions})
        self.writer.add_summary(validation_summary, self.validate_counter)
        self.validate_counter += 1
        self.doom_simulator.reset_tmp_memory()
        return f_measures, durations

    def training_step(self, batch_size):
        images, measures, actions, targets, goals = self.memory.get_batch(batch_size)
        feed_dict = {self._visual_placeholder: images,
                     self._measurement_placeholder: measures,
                     self._true_action_placeholder: actions,
                     self._true_future_placeholder: targets}
        if self.use_goal:
            feed_dict[self._goal_placeholder] = goals

        _, detailed_summary, param_summary = self.sess.run(
            [self.learning_step, self.detailed_summary, self.param_summary], feed_dict)

        self.writer.add_summary(detailed_summary, self.counter)
        self.writer.add_summary(param_summary, self.counter)
        self.counter += 1

    def save_pred(self, path, epoch):
        self.saver.save(self.sess, os.path.join(path, "epoch_%s.tf" % epoch))

        # To get predictions, learning_step... doom_predictor._predictions ..., do not forget to feed!

    @staticmethod
    def random_exploration_prob(epoch):
        # Here a epoch is a train on one batch plus adding the new experiences to memory
        return 0.02 + 145000. / (8 * float(epoch) + 150000.)

    def print_memory(self):
        self.memory.print_values()

