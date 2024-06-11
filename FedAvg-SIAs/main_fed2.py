import copy
import random
from collections import Counter

import numpy as np
import torch
import os
from models.Fed import FedAvg
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from models.Update import LocalUpdate, DatasetSplit
from models.Sia4_single import SIA
from models.Nets import MLP, Mnistcnn, CifarCnn
from utils.dataset import get_dataset
from utils.options import args_parser
from utils.logger import Logger, mkdir_p


def build_membership_dataset(args, model, state_dicts, train_loader, test_loader, device='cpu'):
    model_loss_trajectory = []
    member_status = []
    for epoch in range(args.epochs):
        model.load_state_dict(state_dicts[epoch])
        model.eval()

        epoch_losses = []
        with torch.no_grad():
            for loader_idx, data_loader in enumerate([train_loader, test_loader]):
                for data_idx, (data, target) in enumerate(data_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = F.cross_entropy(output, target, reduction='none')
                    epoch_losses.append(loss.cpu().numpy())
                    member_label = np.ones(len(data)) if loader_idx == 0 else np.zeros(len(data))
                    member_status.extend(member_label)
        model_loss_trajectory.append(np.concatenate(epoch_losses, axis=0))
    model_loss_trajectory = np.stack(model_loss_trajectory, axis=1)
    member_status = np.array(member_status).astype(int)
    return model_loss_trajectory, member_status


def build_mia_model(model, model_loss_trajectory, member_status):
    for loader_idx, data_loader in enumerate([train_loader, test_loader]):
        for data_idx, (data, target) in enumerate(data_loader):





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
        modified_label = random.choice([x for x in range(self.num_classes) if x != label])

        return image, modified_label


if __name__ == '__main__':
    args = args_parser()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # record the experimental results
    logger = Logger(os.path.join(args.checkpoint, 'log_seed{}.txt'.format(args.manualseed)))
    logger.set_names(['alpha', 'comm. round', 'ASR'])

    # parse args
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dataset_train, dataset_test, dict_party_user, dict_sample_user, data_classes = get_dataset(args)

    # build model
    if args.model == 'cnn' and args.dataset == 'CIFAR10':
        net_glob = CifarCnn(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = Mnistcnn(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        dataset_train = dataset_train.dataset
        dataset_test = dataset_test.dataset
        data_size = dataset_train[0][0].shape

        for x in data_size:
            len_in *= x

        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    empty_net = net_glob
    net_glob.train()

    print(net_glob)

    size_per_client = []
    for i in range(args.num_users):
        size = len(dict_party_user[i])
        size_per_client.append(size)

    total_size = sum(size_per_client)
    size_weight = np.array(np.array(size_per_client) / total_size)

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    acc_loss_attack = []

    S_attack = SIA(args=args, dataset=dataset_train, dict_mia_users=dict_sample_user,
                   size_weight=size_weight)

    # Aggregation local models
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    for iter in range(args.epochs * 2):
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            w_glob = FedAvg(w_locals, size_weight)
            net_glob.load_state_dict(w_glob)

    sample_classes = {}
    sample_index = {}
    for user in dict_sample_user:
        for item in dict_sample_user[user]:
            _, label = dataset_train[item]
            if torch.is_tensor(label):
                label = label.item()
            else:
                label = label
            if label in sample_classes:
                sample_classes[label].append(item)
                sample_index[label].append(user)
            else:
                sample_classes[label] = list()
                sample_classes[label].append(item)
                sample_index[label] = list()
                sample_index[label].append(user)

    net_glob_copy = net_glob
    correct = 0
    counter = 0
    idx_list = list()

    for label in sample_classes:
        dataset_local = DataLoader(DatasetSplit(dataset_train, sample_classes[label]),
                                   batch_size=args.local_bs, shuffle=False)

        for ID, (data, target) in enumerate(dataset_local):
            net_glob = net_glob_copy
            net_glob.train()
            optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr / 2, momentum=args.momentum)
            loss_func = nn.CrossEntropyLoss(reduction="mean")
            target_p = (label + 1) % 10
            target_p = torch.tensor(target_p).unsqueeze(0)
            target_p.repeat(args.local_bs)
            for _ in range(2 * args.local_ep):
                if args.gpu != -1:
                    data, target_p = data.cuda(), target_p.cuda()
                optimizer.zero_grad()
                output = net_glob(data)
                loss = loss_func(output, target_p)
                loss.backward()
                optimizer.step()  # 更新参数

        dataset_local = DataLoader(DatasetSplit(dataset_train, sample_classes[label]),
                                   batch_size=1, shuffle=False)
        for ID, (data, target) in enumerate(dataset_local):
            for iter in range(args.epochs):

                loss_locals = []
                if not args.all_clients:
                    w_locals = []
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                for idx in idxs_users:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx])
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    if args.all_clients:
                        w_locals[idx] = copy.deepcopy(w)
                    else:
                        w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))

                # implement source inference attack

                i, w_glob = S_attack.attack(net=copy.deepcopy(empty_net).to(args.device), data=data, target=target,
                                            w_locals=w_locals)

                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)

                idx_list.append(i)

            count = Counter(idx_list)
            most_common_idx = count.most_common(1)[0][0]
            counter = counter + 1
            if most_common_idx == sample_index[label][ID]:q
                correct = correct + 1
                print(f"{counter} correct, list:{idx_list}, index:{sample_index[label][ID]}\n")
            else:
                print(f"{counter} worry, list:{idx_list}, index:{sample_index[label][ID]}\n")
            idx_list.clear()
    print(f"{correct / counter}")

    # implement source inference attack
    # S_attack = SIA(args=args, w_locals=w_locals, dataset=dataset_train, dict_mia_users=dict_sample_user)
    # attack_acc_loss = S_attack.attack(net=empty_net.to(args.device))
