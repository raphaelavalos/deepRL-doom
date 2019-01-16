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
            agent.run_episode(epsilon)
        epsilon = agent.random_exploration_prob(epoch)
        # Train predictor and save every save_freq epochs
        for step in trange(args.step, desc="Step", leave=False):
            agent.get_learning_step(batch_size)

        val = []
        for _ in trange(4, desc="Validation", leave=False):
            val.append(agent.validate())
        val = np.concatenate(val)
        print('Validation: %.3f +/- %.3f' % (val.mean(), val.std()))  # TODO: change it for modes 3 & 4

        if (epoch % save_freq) == 0:
            agent.save_pred(save_dir, epoch)
