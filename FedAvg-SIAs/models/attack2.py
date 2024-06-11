import copy
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .Fed import train_local_model
from .Update import LocalUpdate


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


def reverse_SGD(args, data, target, net, w_avg):  # 向SGD相反的方向更新参数
    net.load_state_dict(w_avg)
    net.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.local_ep * 2):
        net.zero_grad()
        # 前向传播
        output = net(data)
        loss = loss_func(output, target)

        loss.backward()  # 反向传播

        for param in net.parameters():  # 对每个参数的梯度取反
            if param.grad is not None:
                param.grad.data.neg_()  # inplace取反

        optimizer.step()  # 更新参数

    return net


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class SIA2(object):
    def __init__(self, args, net, w_locals, index_users, dataset, dict_party_user):
        self.args = args
        self.empty_net = net
        self.w_locals = w_locals
        self.index_users = index_users
        self.dataset = dataset
        self.dict_mia_users = dict_party_user

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.empty_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.similarities = dict()

    def attack2(self, data, target):
        similarity_list = list()
        data, target = data.to(self.args.device), target.to(self.args.device)
        for idx in self.index_users:
            w = self.w_locals[idx]
            net = reverse_SGD(self.args, data, target, copy.deepcopy(self.empty_net).to(self.args.device), self.w_locals[idx])
            w_reverse = net.state_dict()
            net.eval()
            log_prob1 = net(data)
            y_loss1 = self.loss_func(log_prob1, target)
            #w1 = torch.cat([p.data.view(-1) for p in net.parameters()])

            list_temp = list()
            list_temp.append(idx)
            #w_loacl = train_local_model(self.args, self.empty_net, self.w_locals[idx], list_temp, self.dataset, self.dict_mia_users)
            net.load_state_dict(w_reverse)
            net.eval()
            log_prob2 = net(data)
            y_loss2 = self.loss_func(log_prob2, target)
            # w2 = torch.cat([p.data.view(-1) for p in net.parameters()])
            similarity_list.append(y_loss2 - y_loss1)

        max_value = max(similarity_list)
        max_index = similarity_list.index(max_value)
        return max_index

    def cluster_attack(self, dict_sample_user):
        net = copy.deepcopy(self.empty_net).to(self.args.device)
        for i, w in enumerate(self.w_locals):
            net.load_state_dict(w)
            for index in self.index_users:
                for data, target in dict_sample_user[index]:
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    net.eval()
                    log_prob = net(data)
                    y_loss = self.loss_func(log_prob, target)
                    loss[index][i] = y_loss

