#!usr/bin/env/python
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .initialization import truncated_normal_


class Highway(nn.Module):
    '''Custom highway layer'''
    def __init__(self, input_size, output_size, num_layers=1, bias=0):
        super(Highway, self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(input_size, output_size) for _ in range(num_layers)
        ])
        self.bias = bias

    def forward(self, x):
        for linear in self.linears:
            g = F.relu(linear(x))
            t = F.sigmoid(linear(x) + self.bias)
            out = t * g + (1.0 - t) * x
            x = out

        return out


class Discriminator(nn.Module):
    '''Discriminator'''
    def __init__(self, vocab_size, emb_size, num_classes, filter_sizes, num_filters, dropout, use_cuda=False):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_size)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = Highway(sum(num_filters), sum(num_filters))
        self.linear = nn.Linear(sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.criterion = nn.NLLLoss().cuda()
        else:
            self.criterion = nn.NLLLoss()

        self.reset_parameters()

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pools = torch.cat(pools, 1)
        highway = self.highway(pools)
        out = F.log_softmax(self.linear(self.dropout(highway)), dim=1)
        return out

    def set_optim(self, lr, l2_reg=0.0, optimizer='Adagrad'):
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=l2_reg)
        elif optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr, weight_decay=l2_reg)

    def reset_parameters(self):
        '''
        Resets parameters
        '''
        for name, param in self.named_parameters():
            if name.startswith('emb'):
                nn.init.uniform_(param, -1.0, 1.0)
            elif name.startswith('convs'):
                if name.endswith('weight'):
                    truncated_normal_(param, 0.1)
                elif name.endswith('bias'):
                    nn.init.constant_(param, 0.1)
            elif name.startswith('linear'):
                if name.endswith('weight'):
                    truncated_normal_(param, 0.1)
                elif name.endswith('bias'):
                    nn.init.constant_(param, 0.1)


    def dtrain(self, dataloader):
        self.train()

        total_loss = 0.0
        for X, y in dataloader:
            if self.use_cuda:
                X, y = X.cuda(), y.cuda()
        
            y = y.contiguous().view(-1)
            pred = self(X)
            loss = self.criterion(pred, y)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss / math.ceil(len(dataloader.dataset) / dataloader.batch_size)

    def get_reward(self, dataloader):
        self.eval()
        reward = 0.0
        for X, _ in dataloader:
            if self.use_cuda:
                X = X.cuda()

            out = self(X)
            pred = torch.exp(out)
            reward += np.sum(pred.cpu().data[:, 1].numpy())

        return reward

