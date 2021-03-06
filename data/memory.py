import numpy as np
import collections


class Memory:

    def __init__(self, conf):
        self.conf = conf
        self.use_goal = conf['use_goal']
        self.capacity = conf['memory']['capacity']
        self.nbr_simulators = conf['nbr_of_simulators']
        self.image_resolution = (84, 84, 1)  # TODO: pass in arg
        self.measure_dim = 1  # TODO: pass in arg
        self.counter = 0
        self._images = np.zeros((self.capacity,) + self.image_resolution, dtype=np.float32)
        self._measures = np.zeros((self.capacity, self.measure_dim), dtype=np.float32)
        self._targets = np.zeros((self.capacity, conf['offsets_dim'], self.measure_dim), dtype=np.float32)
        if self.use_goal:
            self._goals = np.zeros((self.capacity, self.measure_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self.full_once = False

    # TODO: might require a lock
    def add_experience(self, images, measures, actions, targets, goals):
        size = len(images)
        if self.counter + size < self.capacity:
            self._images[self.counter: self.counter + size] = images
            self._measures[self.counter: self.counter + size] = measures
            self._targets[self.counter: self.counter + size] = targets
            self._actions[self.counter: self.counter + size] = actions
            if self.use_goal:
                self._goals[self.counter: self.counter + size] = goals
            self.counter += size
        else:
            split = self.capacity - self.counter
            rest = size - split
            self._images[self.counter:], self._images[:rest] = images[:split], images[split:]
            self._measures[self.counter:], self._measures[:rest] = measures[:split], measures[split:]
            self._targets[self.counter:], self._targets[:rest] = targets[:split], targets[split:]
            self._actions[self.counter:], self._actions[:rest] = actions[:split], actions[split:]
            if self.use_goal:
                self._goals[self.counter:], self._goals[:rest] = goals[:split], goals[split:]
            self.counter = rest
            self.full_once = True

    def get_batch(self, batch_size):
        upper = self.capacity if self.full_once else self.counter
        index = np.random.randint(upper, size=batch_size)
        images = self._images[index]
        actions = self._actions[index]
        targets = self._targets[index]
        measures = self._measures[index]
        if self.use_goal:
            goals = self._goals[index]
        else:
            goals = None
        return images, measures, actions, targets, goals

    def print_values(self):
        print(collections.Counter(self._actions))

    def reduce(self, size):
        self.full_once = False
        self.capacity = size
        self.counter = 0
        self._images = np.zeros((self.capacity,) + self.image_resolution, dtype=np.float32)
        self._measures = np.zeros((self.capacity, self.measure_dim), dtype=np.float32)
        self._targets = np.zeros((self.capacity, self.conf['offsets_dim'], self.measure_dim), dtype=np.float32)
        if self.use_goal:
            self._goals = np.zeros((self.capacity, self.measure_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
