import numpy as np

import torch

## STATE
# current time: time
# quantity produced so far: q_t
# current speed: speed
class Env(object):

	def __init__(self, state_space, action_space):
		self.time = 0.
		self.q_so_far = 0.
		self.speed = 0.

		self.overspeed_penalty = 0.
		self.underspeed_penalty = 0.
		self.accum_system_err = 0.

		self.accum_energy_err = 0.

		self.state_space = state_space
		self.action_space = action_space


		# Constraints
		self.acc = 0.1
		self.deacc = 0.1
		self.max_speed = 10
		self.day_quota = 160
		self.total_time = 2 * 16

		# Details
		self.is_peak = False
		self.peak_time_start = self.total_time/24 * 4
		self.peak_time_end = self.total_time/24 * 10

	def reset(self):
		self.time = 0.
		self.q_so_far = 0.
		self.speed = 0.

	def get_tensor_state(self, old_state=None, reset=False):
		if reset:
			time_one_hot = torch.Tensor([
				0. for i in range(self.total_time)
				])
			speed_one_hot = torch.Tensor([
				0. for i in range(self.action_space - 1)  # -1 for maintain speed
				])
			rest_of_state = torch.Tensor([
				0.,  # q_so_far
				#0.,  # speed
				#0.,  # diff q_so_far
				#0.  # old_speed
				])
		else:
			time_one_hot = torch.Tensor([
				1. if i == int(self.time*self.total_time) else 0. for i in range(self.total_time)
				])
			speed_one_hot = torch.Tensor([
				1. if i == (self.speed * 100 / 5) else 0. for i in range(self.action_space - 1)
				])
			rest_of_state = torch.Tensor([
				self.q_so_far,  # q_so_far
				#0.,  # speed
				#0.,  # diff q_so_far
				#0.  # old_speed
				])

		tensor_state =  torch.cat([
			time_one_hot, speed_one_hot, rest_of_state
			]
			).unsqueeze(0)
		return tensor_state

	def print_env_state(self):
		return 'time: {:.4f} - q_so_far: {:.2f} - speed: {:.2f}'.format(
				self.time, self.q_so_far, self.speed)

	def step_test(self, action):
		old_speed = self.speed
		if action != 20:  #action 21(20) is speed stay same
			self.speed = action * 5 / 100
		# capture cost of changing speed
		if self.time != 0:
			self.overspeed_penalty = 10*np.abs(old_speed - self.speed)
			if self.overspeed_penalty == 0:
				self.overspeed_penalty = -1  # reward speed maintain
		#if self.overspeed_penalty == 0:
		#    self.overspeed_penalty = - 0.3 # reinforce speed constant
		self.calc_new_q()
		self.time += 1/self.total_time

		#return self.calc_energy_reward()[0] + self.calc_forecast_reward()[0] + self.overspeed_penalty
		return self.calc_forecast_reward()[0] + self.calc_system_reward() + self.calc_energy_reward()

	def system_acc(self, fast=False):
		if fast:
			self.speed = self.speed + (10*self.acc)/self.max_speed
		else:
			self.speed = self.speed + self.acc/self.max_speed

		if self.speed > 1.:
			self.speed = 1
			self.overspeed_penalty = 0.3
		else:
			self.overspeed_penalty = 0.

	def system_deacc(self, fast=False):
		if fast:
			self.speed = self.speed - (10*self.deacc)/self.max_speed
		else:
			self.speed = self.speed - self.deacc/self.max_speed
		if self.speed < 0.:
			self.speed = 0
			self.underspeed_penalty = 0.3
		else:
			self.underspeed_penalty = 0.

	def calc_new_q(self):
		old = self.q_so_far
		self.q_so_far += (self.speed * self.max_speed)/self.day_quota
		if old > self.q_so_far:
			import pdb
			pdb.set_trace()

	def calc_energy_reward(self):
		"""
		Given the time of day, speed and peak times for energy,
		calculates the reward such that higher speeds are penalised
		during peak hours and subsidised during off-peak times.
		"""
		if self.peak_time_start <= self.time*self.total_time <= self.peak_time_end:
			# inside peak time example
			#return -state['speed'], True
			self.is_peak = True
			if self.speed == 0:
				self.accum_energy_err += 1
			else:
				self.accum_energy_err += self.speed
		else:
			# outside peak time examples
			#av_exp_speed = cls.calc_req_speed(state['time'], state['q_so_far'])
			#err = np.abs(av_exp_speed - state['speed'])  # l1 from average, normalised
			self.is_peak = False
			if self.speed == 0:
				self.accum_energy_err += 1
		if self.time >= 0.9999:
			energy_t_err = self.accum_energy_err
			self.accum_energy_err = 0.
			return -energy_t_err
		else:
			return 0


	def calc_system_reward(self):
		"""If AI doesn't understand own constraints"""
		#return -(self.overspeed_penalty + self.underspeed_penalty)
		if self.time >= 0.9999:
			system_t_err = self.accum_system_err
			self.accum_system_err = 0.
			return system_t_err
		else:
			self.accum_system_err += -(self.overspeed_penalty + self.underspeed_penalty)
			return 0.

	def calc_forecast_reward(self):
		#x = self.calc_req_speed(self.time, self.q_so_far)
		#if x < 0.01:  # less than slowest speed
		#    err = 1
		#elif x > 0.99:
		#    err = 1
		#else:
		#    err = 0
		err = 0
		done = False
		#if x < 0:
			# Too fast
		#    done = True
			# stopping condition
		#    if self.q_so_far <= 1:
		#        err += np.abs(2*(1-self.time)) # later this happens the better
		#    else:
		#        err += 2*self.q_so_far
			#err += 10*self.q_so_far
		#    err += -(1 - np.exp(-5 * np.abs(1-self.q_so_far)))
		#if x > self.max_speed:
		#    done = True
			 # Gone too slow
		#    err += 10*(1-self.q_so_far) # worst when more left over
		#    err += -(1 - np.exp(-5 * np.abs(1-self.q_so_far)))
		if self.time >= 0.9999:  # end of game
			#err += -100*(1 - np.exp(-5 * np.abs(1-self.q_so_far)))  # error is just distance away from quota
			err += 100*(np.exp(-5 * np.abs(1-self.q_so_far)))
		return err, done

	def calc_req_speed(self, time, q_so_far):
		#num = 1 - q_so_far  # percentages
		#denom = 1 - time # percentages
		num = to_make = (1-q_so_far) * self.day_quota
		denom = time_steps_left = (1-time) * self.total_time
		return num/denom
