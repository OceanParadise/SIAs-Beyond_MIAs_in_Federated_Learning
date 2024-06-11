import copy
import random

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .Fed import train_local_model


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
    optimizer = torch.optim.SGD(net.parameters(), lr=-self.args.lr, momentum=self.args.momentum)

    for _ in range(self.epochs):
        optimizer.zero_grad()
        output = net(data)
        loss = self.loss_func(output, target)
        loss.backward()
        optimizer.step()  # 更新参数
    return net.state_dict()


def poisoning_attack(self, net, w_local, data, target):
    # poisoning data
    keys = list(self.dict_mia_users.keys())
    # 移除当前的键
    keys.remove(target)
    # 随机选择一个新的键
    target_changed = random.choice(keys)
    while target_changed == target:
        target_changed = random.choice([i for i in self.dict_mia_users.k])

    if self.args.gpu != -1:
        data, target_changed = data.cuda(), target_changed.cuda()

    net.load_state_dict(w_local)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=-self.args.lr, momentum=self.args.momentum)
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
    def __init__(self, args, w_locals, dataset, dict_mia_users, w_global):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_mia_users = dict_mia_users
        self.w_glob = w_global
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.epochs = 5



    '''self.epochs = 10
    self.loss_func = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.empty_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
    self.similarities = dict()
    
    self.net = copy.deepcopy(self.empty_net)
    self.net.to(self.args.device)
    self.net.load_state_dict(self.w_glob)
    self.net.eval()'''

    '''data, target = data.to(self.args.device), target.to(self.args.device)
    self.net.load_state_dict(self.w_glob)
    self.net.eval()
    log_prob_compare = self.net(data)
    y_loss_compare = self.loss_func(log_prob_compare, target)
    
    net_reverse_SGD = reverse_SGD(self.args, self.index_users, data, target,
                                  copy.deepcopy(self.net).to(self.args.device), self.w_glob)
    w_before = torch.cat([p.data.view(-1) for p in net_reverse_SGD.parameters()])
    similarity_list = list()
    
    
    for w in self.w_locals:
        self.net.load_state_dict(w)
        self.net.eval()
        log_prob = self.net(data.to(self.args.device))
        y_loss = self.loss_func(log_prob, target.to(self.args.device))
        #w = torch.cat([p.view(-1) for p in w.values()])
        #similarity = calculate_l2_distance(w_before.unsqueeze(0), w.unsqueeze(0))
        similarity_list.append(y_loss)
    
    min_value = min(similarity_list)
    min_index = similarity_list.index(min_value)
    return min_index'''

    def attack(self, data, target, net):
        if self.args.gpu != -1:
            data, target = data.cuda(), target.cuda()

        y_loss_all = []
        for idx in self.dict_mia_users:
            for local in self.dict_mia_users:
                net.load_state_dict(self.w_locals[local])
                net.eval()
                log_prob = net(data)
                y_loss = self.loss_func(log_prob, target)
                y_loss_all.append(y_loss)
        min_value = min(y_loss_all)
        index = y_loss_all.index(min_value)

        rSGD_dict = reverse_SGD(net, data, target)
        poisoned_dict = poisoning_attack(net, rSGD_dict, data, target)

        return index, poisoned_dict







