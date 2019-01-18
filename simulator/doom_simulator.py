import vizdoom
import numpy as np
import cv2
from data.tmp_memory import TmpMemory


class DoomSimulator:

    def __init__(self, conf, memory, _id):
        self.conf = conf
        self.use_goal = conf['use_goal']
        self.memory = memory
        self.id = _id
        self._game = vizdoom.DoomGame()
        self._game.load_config(conf['mode_path']['cfg'])
        self.resolution = (84, 84, 1)  # TODO: pass in arg
        self.num_measure = self._game.get_available_game_variables_size()
        self.available_buttons = self._game.get_available_buttons()
        self.episode_count = -1
        self.duration = 0
        self.game_initialized = False
        self.tmp_memory = TmpMemory(conf, memory, _id)
        self.term = False  # TODO: might be removed in favor of closing/opening the game however need to check speed
        self.last_measurements = None
        # self.num_action_to_bool = {
        #     0: np.array([False, False, False]),
        #     1: np.array([False, False, True]),
        #     2: np.array([False, True, False]),
        #     3: np.array([False, True, True]),
        #     4: np.array([True, False, False]),
        #     5: np.array([True, False, True]),
        # }
        self.num_action_to_bool = {
            0: np.array([False, False, True]),
            1: np.array([False, True, False]),
            2: np.array([False, True, True]),
            3: np.array([True, False, False]),
            4: np.array([True, False, True]),
        }

    # def num_action_to_bool(self, action):
    #     if len(self.available_buttons) == 3:
    #         b = np.array([bool(int(i)) for i in np.binary_repr(action, len(self.available_buttons))])
    #         if b[0] == b[1]:
    #             b[0] = False
    #             b[1] = False
    #     return np.array([bool(int(i)) for i in np.binary_repr(action, len(self.available_buttons))])

    def init_game(self):
        if not self.game_initialized:
            self._game.init()
            self.game_initialized = True
        self.term = False

    def close_game(self):
        if self.game_initialized:
            self._game.close()
            self.game_initialized = False
        self.term = True

    def new_episode(self):
        self.tmp_memory.reset()
        self.term = False
        self.last_measurements = None
        self.duration = 0
        self._game.new_episode()
        self.episode_count += 1

    def build_commit_reset(self, max_steps=None):
        self.tmp_memory.build_commit_reset(max_steps)

    def reset_tmp_memory(self):
        self.tmp_memory.reset()

    def step(self, action, goal):
        '''
        Perform step
        Args:
            action (int): an integer between 0 and the number of actions - 1
            goal (np.array): goal vector

        Returns:
        '''
        if self.term is True:
            return np.zeros(self.resolution, dtype=np.float32), \
                   np.full((self.num_measure,), -0.5, dtype=np.float32), 0, True

        if action is None:
            action = self.get_random_action()

        assert 0 <= action <= (2 ** len(self.available_buttons) - 1), 'Unknown action!'

        bool_action = self.num_action_to_bool[action]
        reward = self._game.make_action(list(bool_action), self.conf['skip_tic'])
        term = self._game.is_episode_finished() or self._game.is_player_dead()

        if term:
            self.term = True
            img = np.zeros(self.resolution, dtype=np.float32)
            measure = np.full((self.num_measure,), -0.5, dtype=np.float32)  # TODO: modify this
            if self._game.get_episode_time() < (self._game.get_episode_timeout() - 2):
                self.last_measurements = np.full_like(self.last_measurements, 0)  # TODO: only works for mode 1 and 2

        else:
            img, measure = self.get_state()
            self.last_measurements = (measure + 0.5) * 100

        if not self.use_goal:
            goal = None

        self.tmp_memory.add(img, measure, action, goal)
        self.duration += 1

        return img, measure, reward, term

    def get_state(self):
        state = self._game.get_state()
        measure = state.game_variables
        img = state.screen_buffer
        img = cv2.resize(img, self.resolution[:-1])  # TODO: Check image shape
        img = np.expand_dims(img, -1)  # channel at the end for convolutions
        img = img / 255. - 0.5
        measure = measure / 100. - 0.5
        return img, measure

    def get_random_action(self):
        return np.random.randint(0, len(self.num_action_to_bool), dtype=np.int64)
