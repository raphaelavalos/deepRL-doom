import tensorflow as tf
import numpy as np
from options import get_options


def get_device(args):
    if not args.cuda or args.gpu == -1:
        device = "/cpu:0"
    else:
        device = "/device:GPU:%i" % args.gpu
    return device


if __name__ == '__main__':
    args = get_options()

    # Get device
    device = get_device(args)

    # Create Graph & Session config
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False

    with graph.device(device):
        sess = tf.Session(config=config)
