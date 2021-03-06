import numpy as np


class TmpMemory:

    def __init__(self, conf, memory, _id):
        self.id = _id
        self.use_goal = conf['use_goal']
        self.memory = memory
        self.conf = conf
        self.time_offset = np.sort(conf['offsets'])
        self._images = []
        self._measures = []
        self._actions = []
        if self.use_goal:
            self._goals = []
        self._targets = None
        self.built = False

    def reset(self):
        self._images = []
        self._measures = []
        self._actions = []
        if self.use_goal:
            self._goals = []
        self._targets = None
        self.built = False

    def add(self, image, measure, action, goal):
        self._images.append(image)
        self._measures.append(measure)
        self._actions.append(action)
        if self.use_goal:
            self._goals.append(goal)

    def build(self):
        if len(self._images) <= self.time_offset[-1]:
            self.reset()
        else:
            self._images = np.stack(self._images[:-self.time_offset[-1]], 0)
            self._actions = np.stack(self._actions[:-self.time_offset[-1]], 0)
            measures = np.stack(self._measures, 0)
            targets = [np.roll(measures, -offset, 0)[:-self.time_offset[-1]] for offset in self.time_offset]
            targets = np.stack(targets, 1)
            self._measures = measures[:-self.time_offset[-1]]
            self._targets = targets - np.expand_dims(self._measures, 1)
            if self.use_goal:
                self._goals = np.stack(self._goals[:-self.time_offset[-1]], 0)
        self.built = True

    def commit(self, max_steps=None):
        assert self.built, "Memory not built can't commit!"
        if len(self._images) != 0:
            if max_steps is None or max_steps > len(self._images):
                goals = self._goals if self.use_goal else None
                self.memory.add_experience(self._images, self._measures, self._actions, self._targets, goals)
            else:
                index = np.random.choice(len(self._images), max_steps, replace=False)
                goals = self._goals[index] if self.use_goal else None
                self.memory.add_experience(self._images[index],
                                           self._measures[index],
                                           self._actions[index],
                                           self._targets[index],
                                           goals)

    def build_commit_reset(self, max_steps=None):
        self.build()
        self.commit(max_steps)
        self.reset()
