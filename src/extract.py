import numpy as np
import os
from os.path import join
import argparse
import logging
import common
import glob
import multiprocessing as mp
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def parallel(flist, n_jobs=20):
    pool = mp.Pool(n_jobs)
    data_dict = pool.map(extractfeature, flist)
    return data_dict


def extractfeature(f):
    global n, w
    fname = f.split('/')[-1]
    with open(f, 'r') as f:
        tcp_dump = f.readlines()

    trace = np.array(pd.Series(tcp_dump).str.slice(0, -1).str.split('\t', expand=True).astype("float"))
    # # remove the first few incoming ones if it is the case
    # start = 0
    # for pkt in trace:
    #     if pkt > 0:
    #         break
    #     start += 1
    # trace = trace[start:]

    # new_x = []
    # sign = np.sign(trace[0])
    # cnt = 0
    # for e in trace:
    #     if np.sign(e) == sign:
    #         cnt += 1
    #     else:
    #         new_x.append(int(cnt))
    #         cnt = 1
    #         sign = np.sign(e)
    # new_x.append(int(cnt))
    features = []
    for i in range(n):
        start, stop = np.around(i*w,1), np.around((i+1)*w,1)
        tw = trace[(trace[:,0] >= start) & (trace[:,0] < stop)]
        if len(tw) == 0:
            features.extend([0,0])
        else:
            tw_out = sum(tw[:,1]>0)
            tw_in = sum(tw[:,1]<0)
            features.extend([tw_out, tw_in])
    return features

    return new_x


def build_dict(features):
    vocab, inverse = np.unique(features, return_inverse=True)
    return vocab, inverse.reshape(features.shape[0], features.shape[1])


if __name__ == '__main__':
    global n, w
    '''initialize logger'''
    logger = common.init_logger("extract")

    '''read config file'''
    cf = common.read_conf(common.confdir)
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    num_class = MON_SITE_NUM
    if cf['open_world'] == '1':
        UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
        num_class += 1
    else:
        UNMON_SITE_NUM = 0

    '''read in arg'''
    parser = argparse.ArgumentParser(description='DF feature extraction')
    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('-n',
                        type=int,
                        default=500,
                        help='number of chunks.')
    parser.add_argument('-w',
                        type=float,
                        default=0.1,
                        help='Time window in seconds.')
    parser.add_argument('--format',
                        metavar='<suffix of files>',
                        help='The suffix of files',
                        default=".cell")
    args = parser.parse_args()
    n, w = args.n, args.w

    outputdir = common.outputdir + args.dir.rstrip('/').split('/')[-1]
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    flist = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            if os.path.exists(os.path.join(args.dir, str(i) + "-" + str(j) + args.format)):
                flist.append(os.path.join(args.dir, str(i) + '-' + str(j) + args.format))
    for i in range(UNMON_SITE_NUM):
        if os.path.exists(os.path.join(args.dir, str(i) + args.format)):
            flist.append(os.path.join(args.dir, str(i) + args.format))
    logger.info("Total file number:{}".format(len(flist)))

    res = parallel(flist, n_jobs=20)
    raw_features = np.array(res)
    # raw_features = pad_sequences(raw_features, padding='post', truncating='post', value=0, dtype='int')

    seq_len = raw_features.shape[1]
    vocab, features = build_dict(raw_features)
    logger.info("data shape:{}, vocab:{}".format(raw_features.shape,vocab.shape))
    np.save(join(outputdir,"vocab.npy"), vocab)
    with open(join(outputdir, "raw_feature.data"),"w") as f:
        for feature in raw_features:
            for burst in feature[:-1]:
                f.write("{} ".format(burst))
            f.write("{}\n".format(feature[-1]))
    with open(join(outputdir, "real.data"),"w") as f:
        for feature in features[::2]:
            for burst in feature[:-1]:
                f.write("{} ".format(burst))
            f.write("{}\n".format(feature[-1]))
    with open(join(outputdir, "eval.data"),"w") as f:
        for feature in features[1::2]:
            for burst in feature[:-1]:
                f.write("{} ".format(burst))
            f.write("{}\n".format(feature[-1]))



