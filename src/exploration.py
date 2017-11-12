"""Exploration strategy"""

import numpy as np

class Explorer(object):

    def __init__(self, eps_start, eps_end, eps_decay):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay


    def calc_eps_threshold(self, step, mode='exp'):
        if mode == 'exp':
            threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                np.exp(-1 * (step/self.eps_decay))
        else:
            raise ValueError('Unknown mode for eps threshold')
        return threshold

