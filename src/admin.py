"""Manage admin related stuff e.g. checkpointing"""
import os

import torch


 def save_checkpoint(self, params, filename='checkpoints/checkpoint.pth.tar'):
        dirpath = os.path.dirname(filename)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        checkpoint = {
            'dqn': params['dqn_state_dict'],
            'target': params['target_state_dict'],
            'optimizer': params['optimizer_state_dict'],
            'epoch': params['epoch']
			'step': params['step'],
        }
        torch.save(checkpoint, filename)



def load_checkpoint(self, filename='dqn_checkpoints/checkpoint.pth.tar', epsilon=None):
        checkpoint = torch.load(filename)
        data = {
			'dqn': checkpoint['dqn'],
        	'target': checkpoint['target'],
       		'optimizer': checkpoint['optimizer'],
			'epoch': checkpoint['epoch'],
       		'step': checkpoint['step']
		}
