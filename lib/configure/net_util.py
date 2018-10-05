from __future__ import print_function

import os
import time
import csv
import torch
import numpy as np
from random import randint
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler


def set_parameters(opts):
    '''
    This function is called before training/testing to set parameters
    :param opts:
    :return opts:
    '''

    if not opts.__contains__('train_losses'):
        opts.train_losses=[]

    if not opts.__contains__('train_accuracies'):
        opts.train_accuracies = []

    if not opts.__contains__('valid_losses'):
        opts.valid_losses = []
    if not opts.__contains__('valid_accuracies'):
        opts.valid_accuracies = []

    if not opts.__contains__('test_losses'):
        opts.test_loss=[]

    if not opts.__contains__('test_accuracies'):
        opts.test_accuracies = []

    if not opts.__contains__('best_acc'):
        opts.best_acc = 0.0

    if not opts.__contains__('lowest_loss'):
        opts.lowest_loss = 1e4

    if not opts.__contains__('checkpoint_path'):
        opts.checkpoint_path = 'checkpoint'

    if not os.path.exists(opts.checkpoint_path):
        os.mkdir(opts.checkpoint_path)

    if not opts.__contains__('checkpoint_epoch'):
        opts.checkpoint_epoch = 5

    if not opts.__contains__('valid_pearson_r'):
        opts.valid_pearson_r = []

    if not opts.__contains__('test_pearson_r'):
        opts.test_pearson_r = []


class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


