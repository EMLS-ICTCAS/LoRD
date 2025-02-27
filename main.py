# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import importlib
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser

from ofa.ofa_resnets import OFAResNets
from utils.args import add_management_args, add_gcil_args
from utils.LoRD_utils import get_net
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    # parser.add_argument('--weight_dist',
    #                     help='paraments for gcil-cifar100 ')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)
    if args.load_best_args:
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
        print(args)

    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()
        print(args)
    return args
def main(args = None):
    if args is None:
        args = parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer':
        setattr(args, 'batch_size', 1)

    dataset = get_dataset(args)
    if args.model == 'lord':
        backbone = get_net(args)
    else:
        backbone = dataset.get_backbone()

    # summary(backbone, input_shape=(3, 224, 224))
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task')
        ctrain(args)


if __name__ == '__main__':
    args = parse_args()
    args.n_epochs = 1
    for i in range(0, 1):
        args.seed = i
        main(args)
