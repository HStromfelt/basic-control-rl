"""
Simple model to test regression learning based on constraints
"""

#mode = 'linear'
mode = 'vanilla'
#sub_mode = 'prioritised_replay'
sub_mode = None

from src.replay import ReplayMemory
from src.env import Env
from src.action_select import *
from src.model import Linear_Net, vanilla_Linear_Net
from src.optimise import *
from src.exploration import Explorer
from src.data_manager import DataManager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import random
import numpy as np

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
#################################################
## Hyperparams
BATCH_SIZE = 32 * 300
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
explorer = Explorer(EPS_START, EPS_END, EPS_DECAY)

NUM_ATTEMPTS = 120000
##################################################
# Model and Environment
#NUM_TIME_STEPS = 6 * 24  # 10 min intervals in a day
NUM_TIME_STEPS = 2 * 16  # 30 mins 16 hour shift
ACTION_SPACE = 21  # 5% speed intervals + 1 maintain speed action
#STATE_SPACE = 148  # Time one hot + 4 system variables
#STATE_SPACE = NUM_TIME_STEPS + 4
STATE_SPACE = NUM_TIME_STEPS + 20 + 1  # time and speed as one hot and q_so_far

env = Env(STATE_SPACE, ACTION_SPACE)

#model = vanilla_Linear_Net(STATE_SPACE, ACTION_SPACE)
model = vanilla_Linear_Net(STATE_SPACE, ACTION_SPACE)
target = vanilla_Linear_Net(STATE_SPACE, ACTION_SPACE)
#target = None


if use_cuda:
	model.cuda()
	target.cuda()
##################################################
# Optimiser and Replay
#optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.1)
memory = ReplayMemory(BATCH_SIZE * 100)
##################################################
# Data Storer/Manager
dm = DataManager('./data.h5', mode='swmr')
datasets = dm.datasets
rewards_dataset = datasets['rewards']
speeds_dataset = datasets['speeds']
quota_dataset = datasets['quota']
quota_err_dataset = datasets['quota_err']
losses_dataset = datasets['losses']
overall_reward_dataset = datasets['overall_reward']

dm.start_swmr()

##################################################
##################################################
# Training
traceback_error = True
for j in range(NUM_ATTEMPTS):
	# Initialize the environment and state
	net_loss = 0
	net_reward = 0
	env.reset()
	old = 0
	state = env.get_tensor_state(reset=True)
	for t in range(NUM_TIME_STEPS):
		# Select and perform an action
		action = select_action(
				model, state, ACTION_SPACE, explorer.calc_eps_threshold(j))
		reward = env.step_test(action[0, 0])
		reward = Tensor([reward])
		net_reward += reward[0]

		# Observe new state
		next_state = env.get_tensor_state(state)
		# Store the transition in memory
		memory.push(state, action, next_state, reward)

		# Move to the next state
		state = next_state

		# Perform one step of the optimization (on the target network)
		#train_target = True if t % 100 == 0 else False
		#loss = optimize_model(optimizer, memory, model, BATCH_SIZE, GAMMA,
		#		target=target, train_target=train_target)
		#loss = 0 if loss is None else loss.data[0]
		#net_loss += loss
		#if done:
		#    break
		#print(
		#		'Loss: {:.2f} \t Reward: {:.2f} \t State: {}'.format(
		#			loss, reward[0], env.print_env_state())
		#		)
		print('peak: {} \t State: {}'.format(
			env.is_peak, env.print_env_state()))

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
		quota_dataset[t] = 0 #env.q_so_far
		old = env.q_so_far
		#
		# flush all
		rewards_dataset.flush()
		speeds_dataset.flush()
		quota_dataset.flush()

	# trace back reward for last episode
	if traceback_error:
		final_reward = reward[0]
		for i in range(NUM_TIME_STEPS):
			transition = memory.memory[memory.position - i - 1 % memory.capacity]
			memory.memory[memory.position - i - 1 % memory.capacity] = list(transition)
			memory.memory[memory.position - i - 1 % memory.capacity][3] += \
					final_reward*(1-i/NUM_TIME_STEPS)
			memory.memory[memory.position - i - 1 % memory.capacity] = \
					ReplayMemory.Transition(
							*memory.memory[memory.position - i - 1 % memory.capacity]
							)

	train_target = True if j % 5 == 0 else False
	loss = optimize_model(optimizer, memory, model, BATCH_SIZE, GAMMA,
			target=target, train_target=train_target)
	loss = 0 if loss is None else loss.data[0]
	net_loss += loss


	# Viz
	losses_dataset.resize((j+1,))
	losses_dataset[j] = net_loss
	losses_dataset.flush()

	overall_reward_dataset.resize((j+1,))
	overall_reward_dataset[j] = net_reward
	quota_err_dataset.resize((j+1,))
	quota_err_dataset[j] = 1-env.q_so_far
	overall_reward_dataset.flush()
	quota_err_dataset.flush()

	print('done - day: {} step: {} net_loss: {}'.format(j, t, net_loss))


print('complete')




