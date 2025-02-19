# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import os
import random
import math
import numpy as np
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class loRD(ContinualModel):
    NAME = 'loRD'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(loRD, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.supernet_index = 0
        self.sub_buffer = [0]*5
        self.seen_so_far = torch.tensor([]).long().to(self.device)


    def observe(self, inputs, labels, not_aug_inputs, sub_net_dict=None, config_list=None):
        # lo=dict()
        # self.net.set_max_net()
        self.net.set_active_subnet(maxnet=True)
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels.long())
        loss.backward()
        occupy = 0
        if sub_net_dict:
            # self.net.re_organize_middle_weights()
            index = random.choice(list(range(0, len(sub_net_dict["config"]))))
            sub_config = sub_net_dict["config"][index]
            subnet = sub_net_dict["net"][index]
            occupy = sub_net_dict["occupy"][index]
            self.net.set_active_subnet(**sub_config)
            sub_logits=subnet(inputs)
            sub_outputs=self.net(inputs)
            sub_loss = F.mse_loss(sub_outputs, sub_logits)
            loss_sub = 0.05*sub_loss
            loss_sub.backward()

        #####################################################################



        if not self.buffer.is_empty():
            # self.net.set_max_net()
            self.net.set_active_subnet(maxnet=True)
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_ce += self.args.beta * self.loss(buf_outputs, buf_labels)
            # loss_ce = self.loss(buf_outputs, buf_labels)
            loss_ce.backward()

        # self.net.set_max_net()
        self.net.set_active_subnet(maxnet=True)
        self.opt.step()
        # self.net.set_max_net()


        ran = random.random()
        if ran<(math.exp(-occupy*0.75)):
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs.data)
        # print(lo)

        return loss.item()

    # def observe(self, inputs, labels, not_aug_inputs, sub_net_dict=None, config_list=None):
    #
    #     present = labels.unique().long()
    #     self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
    #     self.net.set_max_net()
    #
    #     logits = self.net(inputs)
    #     mask = torch.zeros_like(logits)
    #     mask[:, present] = 1
    #
    #     self.opt.zero_grad()
    #     if self.seen_so_far.max() < (self.num_classes - 1):
    #         mask[:, self.seen_so_far.max():] = 1
    #
    #     if self.current_task > 0:
    #         logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
    #
    #     loss = self.loss(logits, labels)
    #     loss_re = torch.tensor(0.)
    #     occupy = 0
    #
    #     if self.current_task > 0:
    #         # sample from buffer
    #         buf_inputs, buf_labels = self.buffer.get_data(
    #             self.args.minibatch_size, transform=self.transform)
    #         loss_re = self.loss(self.net(buf_inputs), buf_labels)
    #
    #     loss += loss_re
    #     if sub_net_dict:
    #         # self.net.re_organize_middle_weights()
    #         index = random.choice(list(range(0, len(sub_net_dict["config"]))))
    #         sub_config = sub_net_dict["config"][index]
    #         subnet = sub_net_dict["net"][index]
    #         occupy = sub_net_dict["occupy"][index]
    #         self.net.set_active_subnet(**sub_config)
    #         sub_logits=subnet(inputs)
    #         sub_outputs=self.net(inputs)
    #         sub_loss = F.mse_loss(sub_outputs, sub_logits)
    #         loss += 0.02*sub_loss
    #
    #     self.net.set_max_net()
    #     loss.backward()
    #     self.opt.step()
    #     ran = random.random()
    #     if ran<(math.exp(-occupy*0.75)):
    #         self.buffer.add_data(examples=not_aug_inputs,
    #                          labels=labels)
    #
    #     return loss.item()


    def end_task(self, dataset) -> None:
        self.current_task += 1
        model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))



