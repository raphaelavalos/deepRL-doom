import tensorflow as tf
from config.options import get_options
from agent import Agent
from tqdm import trange
from Neural_network.config_v2 import build_conf

# TODO : Create training loop calling agent with succession of fill memory/train predictor


def get_device(args):
    if not args.cuda or args.gpu == -1:
        device = "/cpu:0"
    else:
        device = "/device:GPU:%i" % args.gpu
    return device


if __name__ == '__main__':
    args = get_options()
    epochs = args.epoch
    steps = args.step
    save_freq = args.save_frequency
    save_dir = args.full_saving_path
    batch_size = args.batch_size
    agent = Agent(build_conf(args.mode, args.mode_path, args.nbr_of_simulators,get_device(args),args.skip_tic))
    epsilon = 1
    # We fill the memory in the while loop
    while not agent.memory.full_once:
        agent.run_episode(epsilon)
    for epoch in trange(epochs, desc="Epoch"):
        for k in trange(8, desc="Playing"):
            agent.run_episode(epsilon)
        epsilon *= 0.99
        # Train predictor and save every save_freq epochs
        for step in trange(args.step, desc="Step", leave=False):
            agent.get_learning_step(batch_size)

        if (epoch % save_freq) == 0:
            agent.save_pred(save_dir, epoch, step)
