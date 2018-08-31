# seqGAN
PyTorch implementation of seqGAN and other modifications

## Requirements
* Python 3.6.x
* PyTorch 0.4.0
* CUDA

## Description
Various PyTorch implementations of seqGAN and its variants.  I am interested in this model because it combines a few different ideas: natural language processing, generative adversarial networks, and reinforcement learning.  Creativity is something I am particularly interested in.

* [`seqgan.py`](https://github.com/rhshi/seqGAN/blob/master/seqgan.py) contains the basic implentation of [SeqGAN](https://arxiv.org/pdf/1609.05473.pdf).  I use PyTorch's DataLoader to iterate through data (something other PyTorch implementations fail to use) and implement my own Highway architecture based on the author's Tensorflow implementation.

## To-do
* Try to implement evolutionary algorithms in place of policy gradient
* Implement [ORGAN](https://arxiv.org/pdf/1705.10843.pdf)
* Implement PPO in place of policy gradient
