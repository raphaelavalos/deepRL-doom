import numpy as np
from .doom_simulator import DoomSimulator


class MultiDoomSimulator:

    def __init__(self, conf, memory):
        self.conf = conf
        self.use_goal = conf['use_goal']
        self.nbr_of_simulators = conf['nbr_of_simulators']
        self.memory = memory
        self.simulators = [DoomSimulator(conf, memory, _id=i) for i in range(conf['nbr_of_simulators'])]

    def init_simulators(self):
        for simulator in self.simulators:
            simulator.init_game()

    def close_simulators(self):
        for simulator in self.simulators:
            simulator.close_game()

    def new_episodes(self):
        for simulator in self.simulators:
            simulator.new_episode()

    def reset_tmp_memory(self):
        for simulator in self.simulators:
            simulator.reset_tmp_memory()

    def build_commit_reset(self, max_steps=None):
        for simulator in self.simulators:
            simulator.build_commit_reset(max_steps)

    def get_duration_and_measurements(self):
        duration = np.zeros((self.nbr_of_simulators,), dtype=np.int64)
        measurements = np.zeros((self.nbr_of_simulators, self.conf['measurement_dim']), dtype=np.float32)
        for i, simulator in enumerate(self.simulators):
            duration[i] = simulator.duration
            measurements[i] = np.array(simulator.last_measurements).reshape((self.conf['measurement_dim'],))
        return duration, measurements

    def step(self, actions, goals, ids):
        ids = range(self.nbr_of_simulators) if ids is None else ids

        if actions is None:
            actions = [None] * len(ids)

        assert len(actions) == len(ids), "The nbr of actions and simulators don't match"

        images = []
        measures = []
        rewards = []
        terms = []

        for i, sim in enumerate(ids):
            img, measure, reward, term = self.simulators[sim].step(actions[i], goals[i])
            images.append(img)
            measures.append(measure)
            rewards.append(reward)
            terms.append(term)

        terms = np.array(terms)
        future_ids = np.fromiter((sim for i, sim in enumerate(ids) if not terms[i]), dtype=np.int64)

        images = np.stack(images, 0)[terms == False]
        measures = np.stack(measures, 0)
        f_measures = measures[terms]
        measures = measures[terms == False]
        rewards = np.stack(rewards, 0)[terms == False]
        terms = np.stack(terms, 0)[terms == False]

        return images, measures, rewards, terms, future_ids, f_measures

    def get_state(self):
        images = []
        measures = []

        for simulator in self.simulators:
            img, measure = simulator.get_state()
            images.append(img)
            measures.append(measure)

        images = np.stack(images, 0)
        measures = np.stack(measures, 0)

        return images, measures
