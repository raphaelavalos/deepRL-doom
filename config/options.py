import argparse
import pprint
import json
import os
import time


def get_device(args):
    if not args.cuda or args.gpu == -1:
        device = "/cpu:0"
    else:
        device = "/device:GPU:%i" % args.gpu
    return device


def get_options():
    parser = argparse.ArgumentParser(description="Learning To Act By Predicting The Future")
    parser.add_argument('--epoch', default=102500, type=int, help="The number of epoch to train", required=False)
    parser.add_argument('--step', default=8, type=int, help="The number of steps per epoch", required=False)
    parser.add_argument('--architecture', default="basic", type=str, help="The type of architecture, (basic or large",
                        required=False)
    parser.add_argument('--save_dir', default="experiments", type=str, help="The saving directory", required=False)
    parser.add_argument('--name', default=time.strftime("%Y%m%d_%H%M%S"), type=str, help="Experiment name",
                        required=False)
    parser.add_argument('--cuda', default=True, type=bool, help="Use cuda (default True)", required=False)
    parser.add_argument('--gpu', type=int, default=0, help="The id of the GPU, -1 for CPU (default: 0).",
                        required=False)
    parser.add_argument('--mode', type=int, default=2, choices=[1, 2, 3, 4], help="Scenario number 1,2,3 or 4",
                        required=False)
    parser.add_argument('--skip_tic', type=int, default=4, help="Number of frames skipped (default:4)", required=False)
    parser.add_argument('--memory', type=int, default=20000, help="Memory capacity", required=False)
    parser.add_argument('--nbr_of_simulators', default=8, type=int, help='Number of simulators to run in parallel',
                        required=False)
    parser.add_argument('--save_frequency', default=10,
                        type=int,
                        help='Frequency in terms of step at which we save the model',
                        required=False)
    parser.add_argument('--batch_size', default=64,
                        type=int,
                        help='Size of the batch processed per step',
                        required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-2, help="Learning rate default 0.0001",
                        required=False)
    parser.add_argument('--decay_rate', type=float, default=0.3, help="Decay rate default: 0.3",
                        required=False)
    parser.add_argument('--decay_steps', type=int, default=250000, help="Decay step size",
                        required=False)

    args = parser.parse_args()

    assert args.skip_tic > 0, 'The number of skip frames must be at least 1'

    full_saving_path = os.path.join(args.save_dir, args.name)
    setattr(args, 'full_saving_path', full_saving_path)

    available_modes = {
        1: {'cfg': 'maps/D1_basic.cfg', 'wad': 'maps/D1_basic.cfg'},
        2: {'cfg': 'maps/D2_navigation.cfg', 'wad': 'maps/D2_navigation.cfg'},
        3: {'cfg': 'maps/D3_battle.cfg', 'wad': 'maps/D3_battle.cfg'},
        4: {'cfg': 'maps/D4_battle2.cfg', 'wad': 'maps/D4_battle2.cfg'}}

    setattr(args, 'mode_path', available_modes[args.mode])
    setattr(args, 'device', get_device(args))

    pprint.pprint(vars(args))

    os.makedirs(full_saving_path, exist_ok=True)

    with open(os.path.join(full_saving_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    return args
