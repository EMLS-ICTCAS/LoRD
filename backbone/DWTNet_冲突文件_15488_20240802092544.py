# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List, Tuple
import numpy as np
from DWT_IDWT.DWT_IDWT_layer import *



def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )

        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        # out = self.linear(out)
        return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        feat = self._features(x)
        out = avg_pool2d(feat, feat.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return feat, out

    def extract_features(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = relu(self.bn1(self.conv1(x)))
        feat1 = self.layer1(out)  # 64, 32, 32
        feat2 = self.layer2(feat1)  # 128, 16, 16
        feat3 = self.layer3(feat2)  # 256, 8, 8
        feat4 = self.layer4(feat3)  # 512, 4, 4
        out = avg_pool2d(feat4, feat4.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        return (feat1, feat2, feat3, feat4), out

    def get_features_only(self, x: torch.Tensor, feat_level: int) -> torch.Tensor:
        """
        Returns the non-activated output of the last convolutional.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """

        feat = relu(self.bn1(self.conv1(x)))

        if feat_level > 0:
            feat = self.layer1(feat)  # 64, 32, 32
        if feat_level > 1:
            feat = self.layer2(feat)  # 128, 16, 16
        if feat_level > 2:
            feat = self.layer3(feat)  # 256, 8, 8
        if feat_level > 3:
            feat = self.layer4(feat)  # 512, 4, 4
        return feat

    def predict_from_features(self, feats: torch.Tensor, feat_level: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns the non-activated output of the last convolutional.
        :param feats: input tensor (batch_size, *input_shape)
        :param feat_level: resnet block
        :return: output tensor (??)
        """

        out = feats

        if feat_level < 1:
            out = self.layer1(out)  # 64, 32, 32
        if feat_level < 2:
            out = self.layer2(out)  # 128, 16, 16
        if feat_level < 3:
            out = self.layer3(out)  # 256, 8, 8
        if feat_level < 4:
            out = self.layer4(out)  # 512, 4, 4

        out = avg_pool2d(out, out.shape[2])  # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)

        return out

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

class feature_extractor(nn.Module):

    def __init__(self,  wavename = 'haar'):

        super(feature_extractor, self).__init__()
        self.Downsample = DWT_2D(wavename=wavename)
        self.Upsample = IDWT_2D_ll(wavename=wavename)

        # self.conv1 = nn.Conv2d(12, 12, kernel_size=1, stride=1, bias=False)
        # self.conv2 = nn.Conv2d(48, 3, kernel_size=1, stride=1, bias=False)
        # self.conv1 = nn.Conv2d(12, 3, kernel_size=1, stride=1, bias=False)
        self.conv1_l = nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False)
        self.conv1_m = nn.Conv2d(12, 1, kernel_size=1, stride=1, bias=False)
        self.conv1_h = nn.Conv2d(9, 1, kernel_size=1, stride=1, bias=False)

    # def forward(self, x: torch.Tensor):
    #     """
    #     Compute a forward pass.
    #     :param x: input tensor (batch_size, *input_shape)
    #     :return: output tensor (output_classes)
    #     """
    #
    #     ll,lh,hl,hh = self.Downsample(x)
    #     x = torch.cat((ll, lh, hl, hh), 1)
    #     x = self.conv1(x)
    #     ll_s,lh_s,hl_s,hh_s = self.Downsample(x)
    #     x = torch.cat((ll_s, lh_s, hl_s, hh_s), 1)
    #     out = self.conv2(x)
    #
    #     return out
    def forward(self, x: torch.Tensor):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """

        # ll,lh,hl,hh = self.Downsample(x)
        # x = torch.cat((ll, lh, hl, hh), 1)
        # out = self.conv1(x)
        ll,lh,hl,hh = self.Downsample(x)
        h = torch.cat((lh, hl, hh), 1)
        m = torch.cat((ll, lh, hl, hh), 1)
        x_l = self.conv1_l(ll)
        x_m = self.conv1_m(m)
        x_h = self.conv1_h(h)
        out = torch.cat((x_l, x_m, x_h), 1)

        return hh,ll

class DWT_ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, wavename='haar') -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(DWT_ResNet, self).__init__()
        self.feature_extractor = feature_extractor(wavename=wavename)
        self.num_classes = num_classes
        self.model_l = ResNet(block, num_blocks, num_classes, nf)
        self.nf = 512
        self.classwise_select_counts = torch.zeros(num_classes, self.nf)
        self.select_probs = torch.zeros(self.nf)
        self.dropout_st = 0.6
        self.select_probs[:min(int(self.nf*1.1 * self.dropout_st),self.nf)] = 1
        # self.select_probs[:] = self.dropout_st
        self.classwise_select_probs = torch.zeros(num_classes, self.nf)
        self.classifier = nn.Linear(self.nf, num_classes)



    def forward(self, x: torch.Tensor, y = None, retufull = False):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        feature, _ = self.feature_extractor(x)
        x = self.model_l(feature)

        # if self.training and y is not None:
        #     classwise_mask = (torch.rand(x.shape[1]) < self.classwise_select_probs[y.long()]).to(x.device)
        #     assert all(self.classwise_select_probs.sum(dim = 1)[y.long()] > 0), "mask error"
        #     x *= classwise_mask
        # x = self.get_kvalue(x, y, self.dropout_st)

        out = self.classifier(x)

        if retufull:
            return feature.detach(), out

        return out


    def construct(self, x: torch.Tensor, y = None):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        x = self.model_l(x)
        x = self.get_kvalue(x, y, self.dropout_st)
        out = self.classifier(x)

        return out


    def freeze_layers(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def get_kvalue(self, x, y, k_percent_on):
        num_filters = x.shape[1]
        k = int(k_percent_on * num_filters)
        score = torch.abs(x)
        threshold = score.kthvalue(num_filters - k, dim=1, keepdim=True)[0]
        mask = score > threshold
        x = x * mask
        if y is not None:
            for class_idx in range(self.num_classes):
                sel_idx = y == class_idx
                self.classwise_select_counts[class_idx] += (mask[sel_idx]).sum(
                    dim=0).cpu()
        return x


def DWT_resnet18(nclasses: int, nf: int=64, wavename = "haar") -> DWT_ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return DWT_ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, wavename = wavename)

if __name__ == '__main__':
    z = torch.randn((32, 3, 8, 8))
    model = DWT_resnet18(10, wavename = "haar")
    model.to("cuda")
    z = z.to("cuda")
    max_output = model(z)

    print(model)