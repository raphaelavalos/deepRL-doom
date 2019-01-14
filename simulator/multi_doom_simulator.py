import numpy as np
from .doom_simulator import DoomSimulator


class MultiDoomSimulator:

    def __init__(self, args, memory):
        self.args = args
        self.nbr_of_simulators = args.nbr_of_simulators
        self.memory = memory
        self.simulators = [DoomSimulator(args['simulator'], memory, _id=i) for i in range(args.nbr_of_simulators)]

    def step(self, actions, goals):
        # TODO: might not need goal
        if actions is None:
            actions = [None] * self.nbr_of_simulators

        assert len(actions) == self.nbr_of_simulators, "The nbr of actions and simulators don't match"

        images = []
        measures = []
        rewards = []
        terms = []

        for simulator, action, goal in zip(self.simulators, actions, goals):
            img, measure, reward, term = simulator.step(action, goal)
            images.append(img)
            measures.append(measure)
            rewards.append(reward)
            terms.append(term)

        images = np.stack(images, 0)
        measures = np.stack(measures, 0)
        rewards = np.stack(rewards, 0)
        terms = np.stack(terms, 0)

        return images, measures, rewards, terms

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





