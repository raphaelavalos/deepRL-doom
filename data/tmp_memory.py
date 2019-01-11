import numpy as np


class TmpMemory:

    def __init__(self, args, memory, _id):
        self.id = _id
        self.memory = memory
        self.args = args
        self.time_offset = np.sort(args.time_offsets)
        self._images = []
        self._measures = []
        self._goals = []
        self._actions = []
        self._target = None
        self.built = False

    def reset(self):
        self._images = []
        self._measures = []
        self._goals = []
        self._actions = []
        self._target = None
        self.built = False

    def add(self, image, measure, goal, action):
        self._images.append(image)
        self._measures.append(measure)
        self._goals.append(goal)
        self._actions.append(action)

    def build(self):
        if len(self._images) <= self.time_offset[-1]:
            self.reset()
        else:
            self._images = np.stack(self._images[:-self.time_offset[-1]], 0)
            measures = np.stack(self._measures, 0)
            self._goals = np.stack(self._goals[:-self.time_offset[-1]], 0)
            self._actions = np.stack(self._actions[:-self.time_offset[-1]], 0)
            targets = [np.roll(measures, -offset, 0)[:-self.time_offset[-1]] for offset in self.time_offset]
            targets = np.concatenate(targets, -1)
            self._measures = measures[:-self.time_offset[-1]]
            self._target = targets
        self.built = True

    def commit(self):
        assert self.built, "Memory not built can't commit!"
        if len(self._images) != 0:
            self.memory.add_experience(self._images, self._measures, self._goals, self._actions, self._target, self.id)

    def build_commit_reset(self):
        self.build()
        self.commit()
        self.reset()