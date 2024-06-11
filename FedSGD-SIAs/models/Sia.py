import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics


# we use prediction loss to conduct our attacks
# prediction loss: for a given sample (x, y), every local model will has a prediction loss on it. we consider the party who has the smallest prediction loss owns the sample.


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
        ent_mod_2 = -np.sum(probability_rest* np.log(_safe_prob(1 - probability_rest)))
        ent_mod = ent_mod_1 + ent_mod_2
        entr_modi.append(ent_mod)
    return entr_modi

'''
************
'''
def cosine_similarity(vec1, vec2): #两个模型参数的余弦相似度（百分数）
    if vec1.dim() > 1:
        vec1 = vec1.squeeze()
    if vec2.dim() > 1:
        vec2 = vec2.squeeze()
    dot_product = torch.dot(vec1, vec2)
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return 1 - ((similarity+1)/2)
"""
************
"""

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
    def __init__(self, args, w_locals=None, dataset=None, dict_mia_users=None):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_mia_users = dict_mia_users

    def attack(self, net):
        correct_loss = 0
        len_set = 0
        for idx in self.dict_mia_users:

            dataset_local = DataLoader(DatasetSplit(self.dataset, self.dict_mia_users[idx]),
                                       batch_size=1, shuffle=False)

            y_loss_all = []

            # evaluate each party's training data on each party's model
            for local in self.dict_mia_users:

                y_losse = []

                idx_tensor = torch.tensor(idx)

                '''
************
                '''
                # net.load_state_dict(self.w_locals[local])
                # net.eval()
                net.train()
                '''
************
                '''

                for id, (data, target) in enumerate(dataset_local):
                    if self.args.gpu != -1:
                        data, target = data.cuda(), target.cuda()
                        idx_tensor = idx_tensor.cuda()

                    '''
************
                    '''
                    epochs = 5  # 定义训练的轮次
                    net.load_state_dict(self.w_locals[local])
                    optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
                    loss_func = nn.CrossEntropyLoss()

                    weights_before = torch.cat([p.data.view(-1) for p in net.parameters()])

                    for epoch in range(epochs):
                        net.zero_grad()
                        # 前向传播
                        output = net(data)
                        loss = loss_func(output, target)

                        loss.backward()  # 反向传播
                        optimizer.step()  # 更新参数

                    weights_after = torch.cat([p.data.view(-1) for p in net.parameters()])

                    similarity = cosine_similarity(weights_before.unsqueeze(0), weights_after.unsqueeze(0))

                    net.load_state_dict(self.w_locals[local])
                    net.eval()
                    '''
************
                    '''
                    log_prob = net(data)
                    # prediction loss based attack: get the prediction loss of the test sample
                    loss = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss(log_prob, target)

                    '''
************
                    '''
                    y_losse.append(y_loss.cpu().detach().numpy()* 0  + similarity.cpu().numpy())
                    # y_losse.append(y_loss.cpu().detach().numpy())
                    '''
************
                    '''



                y_losse = np.concatenate(y_losse).reshape(-1)
                y_loss_all.append(y_losse)

            y_loss_all = torch.tensor(y_loss_all).to(self.args.gpu)

            # test if the owner party has the largest prediction probability
            # get the parties' index of the largest probability of each sample
            # y_loss_all.min(0, keepdim=True)的作用是沿着第一个维度（通常是行）找到最小值，并返回最小值和对应的索引。而[1]则表示取最小值的索引部分,[0]保存的是最小值的值。
            index_of_party_loss = y_loss_all.min(0, keepdim=True)[1]
            correct_local_loss = index_of_party_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum()

            correct_loss += correct_local_loss
            len_set += len(dataset_local.dataset)

        # calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set

        print('\nTotal attack accuracy of prediction loss based attack: {}/{} ({:.2f}%)\n'.format(correct_loss, len_set,
                                                                                                  accuracy_loss))

        return accuracy_loss