import numpy as np
from config.options import get_options
from agent import Agent
from tqdm import trange
from model.config import build_conf


if __name__ == '__main__':
    args = get_options()
    epochs = args.epoch
    steps = args.step
    save_freq = args.save_frequency
    save_dir = args.full_saving_path
    batch_size = args.batch_size
    agent_conf = build_conf(args)
    agent_conf['memory']['capacity'] = 100000
    agent = Agent(agent_conf)
    epsilon = 1
    # We fill the memory in the while loop

    for w in trange(6, desc="Warmup"):
        agent.memory.reduce(100000)
        while not agent.memory.full_once:
            agent.run_episode(epsilon)
        for step in trange(600, desc="Step", leave=False):
            agent.training_step(1024)
        agent.validate()

    agent.memory.reduce(20000)
    while not agent.memory.full_once:
        agent.run_episode(epsilon)
    for epoch in trange(epochs, desc="Epoch"):
        epsilon = agent.random_exploration_prob(epoch * args.step)
        playing_nbr, max_steps = 1, None
        for k in trange(playing_nbr, desc="Playing"):
            agent.run_episode(epsilon, max_steps=max_steps)
        # agent.print_memory()
        # Train predictor and save every save_freq epochs
        for step in trange(2, desc="Step", leave=False):
            agent.training_step(512)

        f_measures = []
        durations = []
        if epoch % 20 == 0:
            agent.validate()

        if (epoch % save_freq) == 0:
            agent.save_pred(save_dir, epoch)
