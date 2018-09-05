#!usr/bin/env/python
import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    '''Reward-refined NLLLoss for adversarial training of generator'''
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, pred, target, reward):
        '''
        Args:
            pred: (batch_size, seq_len), 
            target : (batch_size, seq_len), 
            reward : (batch_size, ); reward of each whole sentence
        '''
        one_hot = torch.zeros(pred.size(), dtype=torch.uint8)
        if pred.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view(-1, 1), 1)
        loss = torch.masked_select(pred, one_hot)
        loss = loss * reward.contiguous().view(-1)
        loss = -torch.sum(loss)
        return loss