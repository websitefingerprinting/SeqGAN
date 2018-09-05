#!usr/bin/env python3
import argparse
import copy
import math
import random
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from models.oracle import Oracle

from data_utils import GeneratorDataset, DiscriminatorDataset


BATCH_SIZE = 64
TOTAL_EPOCHS = 200 
GENERATED_NUM = 6400 # change to 50000
VOCAB_SIZE = 10 # change to 1000
SEQUENCE_LEN = 20

REAL_FILE = 'data/real.data'
FAKE_FILE = 'data/fake.data'
EVAL_FILE = 'data/eval.data'

# generator params
PRE_G_EPOCHS = 120
G_EMB_SIZE = 32
G_HIDDEN_SIZE = 32
G_LR = 1e-3

# discriminator params
PRE_D_EPOCHS = 50
D_EMB_SIZE = 64
D_NUM_CLASSES = 2
D_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
D_NUM_FILTERS = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
DROPOUT = 0.5
D_LR = 1e-3
D_L2_REG = 0.0 

G_STEPS = 5 # needs to be tuned
D_STEPS = 5
K_STEPS = 3

NOISE_STD = 0.01
POPULATION_SIZE = 50 # needs to be tuned
PARENTS_COUNT = 10

EVAL_NUM = GENERATED_NUM // POPULATION_SIZE # needs to be tuned


def generate_samples(net, batch_size, generated_num, output_file):
    samples = []
    for _ in range(generated_num // batch_size):
        sample = net.sample(batch_size, SEQUENCE_LEN).cpu().data.numpy().tolist()
        samples.extend(sample)
    
    with open(output_file, 'w') as f:
        for sample in samples:
            string = ''.join([str(s) for s in sample])
            f.write('{}\n'.format(string))


def mutate_net(net, use_cuda=False, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        if use_cuda:
            noise_t = noise_t.cuda()
        p.data += NOISE_STD * noise_t
    return new_net


def evaluate(netG, netD):
    generate_samples(netG, BATCH_SIZE, EVAL_NUM, FAKE_FILE)
    dis_set = GeneratorDataset(FAKE_FILE)
    disloader = DataLoader(dataset=dis_set, 
                           batch_size=BATCH_SIZE, 
                           shuffle=True)

    reward = netD.get_reward(disloader)
    return reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    args = parser.parse_args()
    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    netG = Generator(VOCAB_SIZE, G_EMB_SIZE, G_HIDDEN_SIZE, use_cuda)
    netD = Discriminator(VOCAB_SIZE, D_EMB_SIZE, D_NUM_CLASSES, D_FILTER_SIZES, D_NUM_FILTERS, DROPOUT, use_cuda)
    oracle = Oracle(VOCAB_SIZE, G_EMB_SIZE, G_HIDDEN_SIZE, use_cuda)

    if use_cuda:
        netG, netD, oracle = netG.cuda(), netD.cuda(), oracle.cuda()

    netG.create_optim(G_LR)
    netD.create_optim(D_LR, D_L2_REG)

    # generating synthetic data
    print('Generating data...')
    generate_samples(oracle, BATCH_SIZE, GENERATED_NUM, REAL_FILE)

    # pretrain generator
    gen_set = GeneratorDataset(REAL_FILE)
    genloader = DataLoader(dataset=gen_set, 
                           batch_size=BATCH_SIZE, 
                           shuffle=True)

    print('\nPretraining generator...\n')
    for epoch in range(PRE_G_EPOCHS):
        loss = netG.pretrain(genloader)
        print('Epoch {} pretrain generator training loss: {}'.format(epoch + 1, loss))

        generate_samples(netG, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        val_set = GeneratorDataset(EVAL_FILE)
        valloader = DataLoader(dataset=val_set,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
        loss = oracle.val(valloader)
        print('Epoch {} pretrain generator val loss: {}'.format(epoch + 1, loss))

    # pretrain discriminator
    print('\nPretraining discriminator...\n')
    for epoch in range(PRE_D_EPOCHS):
        generate_samples(netG, BATCH_SIZE, GENERATED_NUM, FAKE_FILE)
        dis_set = DiscriminatorDataset(REAL_FILE, FAKE_FILE)
        disloader = DataLoader(dataset=dis_set,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
        
        for k_step in range(K_STEPS):
            loss = netD.dtrain(disloader)
            print('Epoch {} K-step {} pretrain discriminator training loss: {}'.format(epoch + 1, k_step + 1, loss))

    print('\nStarting adversarial training...')
    for epoch in range(TOTAL_EPOCHS):
    
        nets = [
            copy.deepcopy(netG) for _ in range(POPULATION_SIZE)
        ]
        population = [
            (net, evaluate(net, netD)) for net in nets
        ]
        for g_step in range(G_STEPS):
            t_start = time.time()
            population.sort(key=lambda p: p[1], reverse=True)
            rewards = [p[1] for p in population[:PARENTS_COUNT]]
            reward_mean = np.mean(rewards)
            reward_max = np.max(rewards)
            reward_std = np.std(rewards)
            print("Epoch %d step %d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, time=%.2f s" % (
                epoch, g_step, reward_mean, reward_max, reward_std, time.time() - t_start))

            elite = population[0]
            # generate next population
            prev_population = population
            population = [elite]
            for _ in range(POPULATION_SIZE - 1):
                parent_idx = np.random.randint(0, PARENTS_COUNT)
                parent = prev_population[parent_idx][0]
                net = mutate_net(parent, use_cuda)
                fitness = evaluate(parent, netD)
                population.append((net, fitness))
            
        netG = elite[0]

        for d_step in range(D_STEPS):
            # train discriminator
            generate_samples(netG, BATCH_SIZE, GENERATED_NUM, FAKE_FILE)
            dis_set = DiscriminatorDataset(REAL_FILE, FAKE_FILE)
            disloader = DataLoader(dataset=dis_set,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)
        
            for k_step in range(K_STEPS):
                loss = netD.dtrain(disloader)
                print('D_step {}, K-step {} adversarial discriminator training loss: {}'.format(d_step + 1, k_step + 1, loss))   

        generate_samples(netG, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        val_set = GeneratorDataset(EVAL_FILE)
        valloader = DataLoader(dataset=val_set,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
        loss = oracle.val(valloader)
        print('Epoch {} adversarial generator val loss: {}'.format(epoch + 1, loss))


if __name__ == '__main__':
    main()