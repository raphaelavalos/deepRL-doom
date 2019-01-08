import argparse
import pprint
import json
import os
import time


def get_options():
    parser = argparse.ArgumentParser(description="Learning To Act By Predicting The Future")
    parser.add_argument('--epoch', default=100, type=int, help="The number of epoch to train", required=False)
    parser.add_argument('--step', default=100, type=int, help="The number of steps per epoch", required=False)
    parser.add_argument('--step', default=100, type=int, help="The number of steps per epoch", required=False)
    parser.add_argument('--architecture', default="basic", type=str, help="The type of architecture, (basic or large",
                        required=False)
    parser.add_argument('--save_dir', default="experiments", type=str, help="The saving directory", required=False)
    parser.add_argument('--name', default=time.strftime("%Y%m%d_%H%M%S"), type=str, help="Experiment name",
                        required=False)
    parser.add_argument('--cuda', default=True, type=bool, help="Use cuda (default True)", required=False)
    parser.add_argument('--gpu', type=int, default=0,
                        help="The id of the GPU, -1 for CPU (default: 0).",
                        required=False)


    args = parser.parse_args()
    pprint.pprint(vars(args))

    full_saving_path = os.path.join(args.save_dir, args.name)

    with open(os.path.join(full_saving_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    return args

