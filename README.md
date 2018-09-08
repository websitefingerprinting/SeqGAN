# SeqGAN
PyTorch implementation of SeqGAN and other architectures

## Requirements
* Python 3.6.x
* PyTorch 0.4.0
* CUDA

## Description
Various PyTorch implementations of seqGAN and its variants.  I am interested in this model because it combines a few different ideas: natural language processing, generative adversarial networks, and reinforcement learning.  Creativity is something I am particularly interested in.

* [`seqgan.py`](https://github.com/rhshi/seqGAN/blob/master/seqgan.py) contains the basic implentation of [SeqGAN](https://arxiv.org/pdf/1609.05473.pdf).  I use PyTorch's DataLoader to iterate through data (something other PyTorch implementations fail to use) and implement my own Highway architecture based on the author's Tensorflow implementation.
* [`seqgan_ga.py`](https://github.com/rhshi/SeqGAN/blob/master/seqgan_ga.py) contains my implementation of SeqGan with genetic algorithm in place of polic gradient.  There are some issues with pretraining the discriminator, but overall, the implementation works okay.  It shows improvement in the generator, though it improves slower than the policy gradient; it also only trains on full sequences, so it does not take advantange of the Monte Carlo search in the original SeqGAN.  Whereas in the original SeqGAN the generator trains once for each epoch, I implemented 5 mutations for each epoch; this could definitely be tuned.  Additionally, one of the main benefits of evolutionary algorithms as a whole is the ease in which they are parallelized; this is something to be considered if I decide to come back to this.

## To-do
* Implement [ORGAN](https://arxiv.org/pdf/1705.10843.pdf)
