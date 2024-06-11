import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics


# we use prediction loss to conduct our attacks prediction loss: for a given sample (x, y), every local model will
# has a prediction loss on it. we consider the party who has the smallest prediction loss owns the sample.


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


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class poisoned_Dataset(Dataset):
    def __init__(self, dataset, idxs, no_classed):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.num_classes = no_classed

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]

        # 这里修改标签，比如进行某种转换
        modified_label =  random.choice([x for x in range(self.num_classes) if x != label])

        return image, modified_label


class SIA(object):
    def __init__(self, args, w_locals=None, dataset=None, dict_mia_users=None, w_global=None):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_mia_users = dict_mia_users
        self.w_glob = w_global
        self.epochs = 5

    def attack(self, net):
        correct_loss = 0
        len_set = 0
        net_train = copy.deepcopy(net).to(self.args.device)
        net_train.load_state_dict(self.w_glob)

        for idx in self.dict_mia_users:

            dataset_local = DataLoader(DatasetSplit(self.dataset, self.dict_mia_users[idx]),
                                       batch_size=self.args.local_bs, shuffle=False)
            y_loss_all = []

            # evaluate each party's training data on each party's model
            for local in self.dict_mia_users:

                y_losse = []

                idx_tensor = torch.tensor(idx)
                net.load_state_dict(self.w_locals[local])
                net.eval()
                for id, (data, target) in enumerate(dataset_local):
                    if self.args.gpu != -1:
                        data, target = data.cuda(), target.cuda()
                        idx_tensor = idx_tensor.cuda()
                    log_prob = net(data)
                    # prediction loss based attack: get the prediction loss of the test sample
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss_func(log_prob, target)
                    y_losse.append(y_loss.cpu().detach().numpy())

                y_losse = np.concatenate(y_losse).reshape(-1)
                y_loss_all.append(y_losse)

            y_loss_all = torch.tensor(y_loss_all).to(self.args.gpu)

            # test if the owner party has the largest prediction probability
            # get the parties' index of the largest probability of each sample
            index_of_party_loss = y_loss_all.min(0, keepdim=True)[1]
            correct_local_loss = index_of_party_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum()

            correct_loss += correct_local_loss
            len_set += len(dataset_local.dataset)

            # revsere_SGD
            optimizer = torch.optim.SGD(net_train.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            for _ in (range(self.epochs)):
                for batch_idx, (images, labels) in enumerate(dataset_local):
                    if self.args.gpu != -1:
                        images, labels = images.cuda(), labels.cuda()
                    net_train.zero_grad()
                    out = net_train(images)
                    loss_func = nn.CrossEntropyLoss()
                    loss = loss_func(out, labels)
                    loss.backward()
                    # 反转梯度
                    for param in net_train.parameters():
                        param.grad.data = -param.grad.data
                    optimizer.step()

            # poisoning_attack
            poisoned_dataset = DataLoader(poisoned_Dataset(self.dataset, self.dict_mia_users[idx], self.args.num_classes),
                                       batch_size=1, shuffle=False)
            optimizer = torch.optim.SGD(net_train.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            for _ in range(self.epochs):
                for batch_idx, (images, labels) in enumerate(poisoned_dataset):
                    if self.args.gpu != -1:
                        images, labels = images.cuda(), labels.cuda()
                    net_train.zero_grad()
                    out = net_train(images)
                    loss_func = nn.CrossEntropyLoss()
                    loss = loss_func(out, labels)
                    loss.backward()
                    optimizer.step()

        # calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set

        print('\nTotal attack accuracy of prediction loss based attack: {}/{} ({:.2f}%)\n'.format(correct_loss, len_set,
                                                                                                  accuracy_loss))

        return accuracy_loss, net_train.state_dict()
