import numpy as np


class Memory:

    def __init__(self, args):
        self.args = args
        self.capacity = args.memory_capacity
        self.nbr_simulators = args.nbr_simulators
        self.image_resolution = (1, 84, 84)  # TODO: pass in arg
        self.measure_dim = 1  # TODO: pass in arg
        self.action_dim = 8  # TODO: pass in arg
        self.counter = 0
        self._images = np.zeros((self.capacity,) + self.image_resolution, dtype=np.float32)
        self._measures = np.zeros((self.capacity, self.measure_dim), dtype=np.float32)
        self._targets = np.zeros((self.capacity, len(args.time_offsets) * self.measure_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.int64)
        self._goals = np.zeros((self.capacity, self.measure_dim), dtype=np.int64)
        self.full_once = False

    def add_experience(self, images, measures, goals, actions, targets, _id):
        size = len(images)
        if self.counter + size < self.capacity:
            self._images[self.counter: self.counter + size] = images
            self._measures[self.counter: self.counter + size] = measures
            self._targets[self.counter: self.counter + size] = targets
            self._goals[self.counter: self.counter + size] = goals
            self._actions[self.counter: self.counter + size] = actions
            self.counter += size
        else:
            split = self.capacity - self.counter
            rest = size - split
            self._images[self.counter:], self._images[:, rest] = images[:split], images[split:]
            self._measures[self.counter:], self._measures[:, rest] = measures[:split], measures[split:]
            self._targets[self.counter:], self._targets[:, rest] = targets[:split], targets[split:]
            self._goals[self.counter:], self._goals[:, rest] = goals[:split], goals[split:]
            self._actions[self.counter:], self._actions[:, rest] = actions[:split], actions[split:]
            self.counter = rest
            self.full_once = True

    def get_batch(self, batch_size):
        upper = self.capacity if self.full_once else self.counter
        index = np.random.randint(upper, size=batch_size)
        images = self._images[index]
        actions = self._actions[index]
        targets = self._targets[index]
        goals = self._goals[index]
        measures = self._measures[index]
        return images, measures, goals, actions, targets
