import copy
import random

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .Fed import FedAvg


def _safe_prob(probs, small_value=1e-30):
    return np.maximum(probs, small_value)


def uncertainty(probability, n_classes):
    uncert = []
    for i in range(len(probability)):
        unc = (-1 / np.log(n_classes)) * np.sum(probability[i] * np.log(_safe_prob(probability[i])))
        uncert.append(unc)
    return uncert


def entropy_modified(probability, target):
    entr_modi = []
    for i in range(len(probability)):
        ent_mod_1 = (-1) * (1 - probability[i][int(target[i])]) * np.log(_safe_prob(probability[i][int(target[i])]))
        probability_rest = np.delete(probability[i], int(target[i]))
        ent_mod_2 = -np.sum(probability_rest * np.log(_safe_prob(1 - probability_rest)))
        ent_mod = ent_mod_1 + ent_mod_2
        entr_modi.append(ent_mod)
    return entr_modi


def cosine_similarity(vec1, vec2):  # 两个模型参数的余弦相似度
    if vec1.dim() > 1:
        vec1 = vec1.squeeze()
    if vec2.dim() > 1:
        vec2 = vec2.squeeze()
    dot_product = torch.dot(vec1, vec2)
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def calculate_l2_distance(vec1, vec2):
    return torch.norm(vec1 - vec2, p=2).item()


def reverse_SGD(self, net, data, target):
    net.load_state_dict(self.w_glob)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    for _ in range(self.epochs):
        optimizer.zero_grad()
        output = net(data)
        loss = self.loss_func(output, target)
        loss.backward()
        # 反转梯度
        for param in net.parameters():
            param.grad.data = -param.grad.data
        optimizer.step()  # 更新参数
    return net.state_dict()


def poisoning_attack(self, net, w_local, data, target):
    # poisoning data
    keys = list(self.dict_mia_users.keys())
    # 移除当前的键
    keys.remove(target)
    # 随机选择一个新的键
    target_changed = random.choice(keys)

    if self.args.gpu != -1:
        data, target_changed = data.cuda(), target_changed.cuda()

    net.load_state_dict(w_local)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
    for _ in range(self.epochs):
        optimizer.zero_grad()
        output = net(data)
        loss = self.loss_func(output, target_changed)
        loss.backward()
        optimizer.step()  # 更新参数
    return net.state_dict()


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class SIA(object):
    def __init__(self, args, w_locals=None, dataset=None, dict_mia_users=None, size_weight=None):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_mia_users = dict_mia_users
        self.size_weight = size_weight
        self.w_glob = None

    def attack(self, net=None, data=None, target=None, w_locals=None):
        self.w_locals = w_locals
        keys = list(self.dict_mia_users.keys())
        keys.remove(target)
        target_changed = random.choice(keys)
        target_changed = torch.tensor(target_changed).unsqueeze(0)

        y_losses = list()

        if self.args.gpu != -1:
            data, target, target_changed = data.cuda(), target.cuda(), target_changed.cuda()

        # evaluate each party's training data on each party's model
        for local in self.dict_mia_users:
            net.load_state_dict(self.w_locals[local])
            net.eval()
            log_prob = net(data)
            loss_func = nn.CrossEntropyLoss(reduction='none')
            y_loss = loss_func(log_prob, target)
            y_losses.append(y_loss.cpu().detach().numpy())
        minv = min(y_losses)
        minidx = y_losses.index(minv)



        # update global weights
        self.w_glob = FedAvg(self.w_locals, self.size_weight)

        return minidx, self.w_glob
