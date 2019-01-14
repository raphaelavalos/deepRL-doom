import vizdoom
import numpy as np
import cv2
from data.tmp_memory import TmpMemory


class DoomSimulator:

    def __init__(self, args, memory, _id):
        self.args = args
        self.memory = memory
        self.id = _id
        self._game = vizdoom.DoomGame()
        self._game.load_config(args.mode_path['conf'])
        self.resolution = (84, 84, 1)  # TODO: pass in arg
        self.num_measure = self._game.get_available_game_variables_size()
        self.available_buttons = self._game.get_available_buttons()
        self.episode_count = 0
        self.game_initialized = False
        self.tmp_memory = TmpMemory(args, memory, _id)

    def num_action_to_bool(self, action):
        return np.array([bool(int(i)) for i in np.binary_repr(action, self.available_buttons)])

    def init_game(self):
        if not self.game_initialized:
            self._game.init()
            self.game_initialized = True

    def close_game(self):
        if self.game_initialized:
            self._game.close()
            self.game_initialized = False

    def new_episode(self):
        self.tmp_memory.build_commit_reset()
        self._game.new_episode()
        self.episode_count += 1

    def step(self, action, goal):
        '''
        Perform step
        Args:
            action (int): an integer between 0 and the number of actions - 1
            goal (np.array): goal vector

        Returns:
        '''

        # TODO: see if goal is really needed
        if action is None:
            action = self.get_random_action()

        assert 0 <= action <= (2**self.available_buttons - 1), 'Unknown action!'

        bool_action = self.num_action_to_bool(action)
        reward = self._game.make_action(bool_action, self.args.skip_tic)
        term = self._game.is_episode_finished() or self._game.is_player_dead()

        if term:
            self.new_episode()  # Restart new episode to keep with the flow
            img = np.zeros(self.resolution, dtype=np.uint8)
            measure = np.zeros((self.num_measure,), dtype=np.uint32)

        else:
            img, measure = self.get_state()

        self.tmp_memory.add(img, measure, goal, action)

        return img, measure, reward, term

    def get_state(self):
        state = self._game.get_state()
        measure = state.game_variables
        img = state.screen_buffer
        img = cv2.resize(img, self.resolution[:-1])  # TODO: Check image shape
        img = np.expand_dims(img, -1)  # channel at the end for convolutions
        return img, measure

    def get_random_action(self):
        return np.random.randint(0, 2**self.available_buttons - 1, dtype=np.int64)










