"""Exploration strategy"""

import numpy as np

class Explorer(object):

    def __init__(self, eps_start=None, eps_end=None, eps_decay=None):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay


    def calc_eps_threshold(self, step, mode='exp'):
        if mode == 'exp':
            threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                np.exp(-1 * (step/self.eps_decay))
        elif mode == 'exp_cutoff_100k':
            if step >= 100000:
                return self.eps_end
            else:
                return self.calc_eps_threshold(step, mode='exp')
        elif mode == 'linear_decay':
            if step <= self.eps_decay:
                return -(1/self.eps_decay)*step + 1
            else:
                return 0
        else:
            raise ValueError('Unknown mode for eps threshold')
        return threshold

