"""Model tests for generalisation capabilities"""
import os

#mode = 'linear'
mode = 'vanilla'
#sub_mode = 'prioritised_replay'
sub_mode = None

from src.admin import *
from src.env import Env
from src.action_select import *
from src.data_manager import DataManager
from src.model import vanilla_Linear_Net

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import random
import numpy as np
import os

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
# if gpu is to be used
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def get_checkpoints(basepath='checkpoints/', filename=None):
    dirname = os.path.dirname(basepath)
    if filename == None:
        return (f for f in os.listdir(dirname))
    else:
        return os.path.join(dirname, filename)


#################################################
## Hyperparams
##################################################
# Model and Environment
#NUM_TIME_STEPS = 6 * 24  # 10 min intervals in a day
NUM_TIME_STEPS = 4 * 2.5  # 4 @ 15 minutes in 2.5 hrs
assert NUM_TIME_STEPS % 1 == 0
NUM_TIME_STEPS = int(NUM_TIME_STEPS)
ACTION_SPACE = 21  # 5% speed intervals + 1 maintain speed action
#STATE_SPACE = 148  # Time one hot + 4 system variables
#STATE_SPACE = NUM_TIME_STEPS + 4
STATE_SPACE = NUM_TIME_STEPS + 20 + 2  # time and speed as one hot and q_so_far and old speed

env = Env(STATE_SPACE, ACTION_SPACE)

##################################################
# Optimiser and Replay
#optimizer = optim.SGD(model.parameters(), lr=0.01)
#macro_memory = ReplayMemory(BATCH_SIZE* 100)
##################################################
##################################################
##################################################
# Test
def test(model, run_name):
    # Data Storer/Manager
    if not os.path.exists('./tests/'):
        os.makedirs('./tests/')
    path = './tests/{}.h5'.format(run_name)
    dm = DataManager('./tests/{}.h5'.format(run_name), mode='swmr')
    datasets = dm.datasets
    rewards_dataset = datasets['rewards']
    speeds_dataset = datasets['speeds']
    quota_dataset = datasets['quota']
    quota_err_dataset = datasets['quota_err']
    overall_reward_dataset = datasets['overall_reward']

    dm.start_swmr()

    if use_cuda:
        model.cuda()

    # Initialize the environment and state
    net_loss = 0
    net_reward = 0
    env.reset()
    old = 0
    state = env.get_tensor_state(reset=True)
    for t in range(NUM_TIME_STEPS):
        # Select and perform an action
        action = select_action(
                model,
                state,
                ACTION_SPACE,
                0#explorer.calc_eps_threshold(j, decay_type)
            )#'exp_cutoff_100k'))
        reward = env.step_test(action[0, 0])
        reward = Tensor([reward])
        net_reward += reward[0]

        # Observe new state
        next_state = env.get_tensor_state(state)

        # Move to the next state
        state = next_state
        print('peak: {} \t Reward: {:.2f} \t State: {}'.format(
            env.is_peak, reward[0], env.print_env_state()))

        # VIZZZ
        # resize data
        new_shape = (t+1,)
        rewards_dataset.resize(new_shape)
        speeds_dataset.resize(new_shape)
        quota_dataset.resize(new_shape)
        #
        # add data
        rewards_dataset[t] = reward[0]
        speeds_dataset[t] = env.speed
        if env.q_so_far < old:
            import pdb
            pdb.set_trace()
        quota_dataset[t] = env.q_so_far
        old = env.q_so_far
        #
        # flush all
        rewards_dataset.flush()
        speeds_dataset.flush()
        quota_dataset.flush()

    overall_reward_dataset.resize((1,))
    overall_reward_dataset[0] = net_reward
    quota_err_dataset.resize((1,))
    quota_err_dataset[0] = 1-env.q_so_far
    overall_reward_dataset.flush()
    quota_err_dataset.flush()


    print('done - day: {}'.format(1))
    print('complete')
    return path





if __name__ == '__main__':
    checkpoints = get_checkpoints()
    model = vanilla_Linear_Net(STATE_SPACE, ACTION_SPACE)
    test_paths = []
    epochs = []
    for checkpoint in checkpoints:
        params = load_checkpoint(filename=checkpoint)
        run = 'model_epoch_{}'.format(params['epoch'])
        model.load_state_dict(params['dqn'])

        test_paths.append(test(model, run))
        epochs.append(params['epoch'])

    print('done testing')

    print('visualising')
    from src.test_visualise import *

    for i, path in enumerate(test_paths):
        visualise_test(path, epochs[i])

    print('Visualisations complete')

