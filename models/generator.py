#!usr/bin/env/python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .losses import AdversarialLoss


class Generator(nn.Module):
    '''Generator'''
    def __init__(self, vocab_size, emb_size, hidden_size, use_cuda=False):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        if self.use_cuda:
            self.mle_criterion = nn.CrossEntropyLoss().cuda()
            self.adv_criterion = AdversarialLoss().cuda()
        else:
            self.mle_criterion = nn.CrossEntropyLoss()
            self.adv_criterion = AdversarialLoss()

        self.reset_parameters()

    def forward(self, x):
        emb = self.emb(x)
        h_0, c_0 = self.init_hidden(x.size(0))
        out, _ = self.lstm(emb, (h_0, c_0))
        pred = self.linear(out.contiguous().view(-1, self.hidden_size))
        return pred 

    def set_optim(self, lr, optimizer='Adam'):
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)

    def step(self, x, h, c):
        '''
        Args:
            x: (batch_size, 1); sequence of generated tokens
            h: (1, batch_size, hidden_dim); lstm hidden state
            c: (1, batch_size, hidden_dim); lstm cell state
        Returns:
            pred: (batch_size, vocab_size); predicted prob for next tokens
            h: (1, batch_size, hidden_dim); lstm new hidden state
            c: (1, batch_size, hidden_dim); lstm new cell state
        '''
        self.lstm.flatten_parameters()
        emb = self.emb(x)
        out, (h, c) = self.lstm(emb, (h, c))
        pred = self.linear(out.view(-1, self.hidden_size))

        return pred, h, c
            
    def sample(self, batch_size, seq_len, x=None):
        '''
        Creates sequence of length seq_len to be added to training set
        '''
        samples = []
        if x is None:
            x = torch.zeros(batch_size, 1, dtype=torch.int64)
            h, c = self.init_hidden(batch_size)
            if self.use_cuda:
                x = x.cuda()
            for _ in range(seq_len):
                out, h, c = self.step(x, h, c)
                x = torch.multinomial(out, 1)
                samples.append(x)
        else:
            h, c = self.init_hidden(x.size(0))
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                out, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = torch.multinomial(out, 1)
            for _ in range(given_len, seq_len):
                samples.append(x)
                out, h, c = self.step(x, h, c)
                x = torch.multinomial(out, 1)

        return torch.cat(samples, dim=1)

    def init_hidden(self, batch_size):
        '''
        Initializes hidden and cell states
        '''
        h = torch.zeros((1, batch_size, self.hidden_size))
        c = torch.zeros((1, batch_size, self.hidden_size))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()

        return h, c

    def reset_parameters(self):
        '''
        Resets parameters to be drawn from the standard normal distribution
        '''
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def pretrain(self, dataloader):
        self.train()

        total_loss = 0.0
        for X, y in dataloader:
            if self.use_cuda:
                X, y = X.cuda(), y.cuda()
        
            y = y.contiguous().view(-1)
            pred = self(X)
            loss = self.mle_criterion(pred, y)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total_loss / math.ceil(len(dataloader.dataset) / dataloader.batch_size)

    def pgtrain(self, batch_size, seq_len, rollout, netD):
        self.train()

        samples = self.sample(batch_size, seq_len)
        zeros = torch.zeros(batch_size, 1, dtype=torch.int64)
        if self.use_cuda:
            zeros = zeros.cuda()
        inputs = torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous()
        targets = samples.data.contiguous().view(-1)

        # calculate reward
        rewards = torch.tensor(rollout.get_reward(samples, netD))
        if self.use_cuda:
            rewards = rewards.cuda()
            
        prob = F.log_softmax(self(inputs), dim=1)
        loss = self.adv_criterion(prob, targets, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        