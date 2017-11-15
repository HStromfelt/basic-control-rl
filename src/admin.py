"""Manage admin related stuff e.g. checkpointing"""
import os

import torch


def save_checkpoint(self, params, basepath='checkpoints/', filename='checkpoint.pth.tar'):
    dirpath = basepath

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    filepath = os.path.join(dirpath, filename)

    checkpoint = {
            'dqn': params['dqn_state_dict'],
            'target': params['target_state_dict'],
            'optimizer': params['optimizer_state_dict'],
            'epoch': params['epoch']
            'step': params['step'],
            }
    torch.save(checkpoint, filepath)



def load_checkpoint(self, basepath='checkpoints/' filename='checkpoint.pth.tar', epsilon=None):
    filepath = os.path.join(basepath, filename)
    checkpoint = torch.load(filepath)
    data = {
            'dqn': checkpoint['dqn'],
            'target': checkpoint['target'],
            'optimizer': checkpoint['optimizer'],
            'epoch': checkpoint['epoch'],
            'step': checkpoint['step']
            }
    return data
