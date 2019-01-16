import numpy as np
from config.options import get_options
from agent import Agent
from tqdm import trange
from model.config import build_conf


# TODO : Create training loop calling agent with succession of fill memory/train predictor


if __name__ == '__main__':
    args = get_options()
    epochs = args.epoch
    steps = args.step
    save_freq = args.save_frequency
    save_dir = args.full_saving_path
    batch_size = args.batch_size
    agent_conf = build_conf(args)
    agent = Agent(agent_conf)
    epsilon = 1
    # We fill the memory in the while loop
    while not agent.memory.full_once:
        agent.run_episode(epsilon)
    for epoch in trange(epochs, desc="Epoch"):
        for k in trange(8, desc="Playing"):
            agent.run_episode(epsilon, 8)
        epsilon *= 0.99
        # Train predictor and save every save_freq epochs
        for step in trange(args.step, desc="Step", leave=False):
            agent.get_learning_step(batch_size)

        f_measures = []
        durations = []
        for _ in trange(1, desc="Validation", leave=False):
            f_measure, duration = agent.validate()
            f_measures.append(f_measure)
            durations.append(duration)

        f_measures = np.concatenate(f_measures)
        durations = np.concatenate(durations)
        print('Health: %.3f +/- %.3f' % (f_measures.mean(), f_measures.std()))  # TODO: change it for modes 3 & 4
        # print('Duration: %.3f +/- %.3f' % (durations.mean(), durations.std()))  # TODO: change it for modes 3 & 4

        if (epoch % save_freq) == 0:
            agent.save_pred(save_dir, epoch)
