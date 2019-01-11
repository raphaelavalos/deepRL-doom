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
        self.resolution = (1, 84, 84)  # TODO: pass in arg
        self.num_measure = self._game.get_available_game_variables_size()
        self.available_buttons = self._game.get_available_buttons()
        self.episode_count = 0
        self.game_initialized = False
        self.tmp_memory = TmpMemory(args, memory, _id)

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

        reward = self._game.make_action(action, self.args.skip_tic)
        term = self._game.is_episode_finished() or self._game.is_player_dead()

        if term:
            self.new_episode()  # Restart new episode to keep with the flow
            img = np.zeros(self.resolution, dtype=np.uint8)
            measure = np.zeros((self.num_measure,), dtype=np.uint32)

        else:
            state = self._game.get_state()
            measure = state.game_variables
            img = state.screen_buffer
            img = cv2.resize(img, self.resolution[1:])  # TODO: Check image shape
            img = np.expand_dims(img, 0)

        self.tmp_memory.add(img, measure, goal, action)

        return img, measure, reward, term









