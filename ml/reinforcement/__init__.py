import gym
import operator
import random
import torch
import torchvision
import numpy                as np
import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

class RLModel:
	def train(self):
		criterion = self.criterion()
		optimizer = self.optimizer()

		for _ in range(self.epochs):
			self.model.train()
			self._train_epoch(criterion, optimizer)
			self.test()

	def _train_epoch(self, criterion, optimizer):
		"""
		Runs an epoch of training. All environments will be reset after this returns.
		"""
		raise NotImplementedError('Training not implemented for particular model.')


	def test(self):
		done = False
		rwrd = 0.
		self.model.eval()
		while not done:
			action = self.model(self.model.env.obsv).argmax().item()
			(r, done) = self.model.env.step(action)
			rwrd += 1
		print(f'Total Evaluation Reward: {rwrd}')

	@property
	def model(self):
		raise NotImplementedError('Subclass must define their model to be used.')
	
	@property
	def epochs(self):
		return 6

	@property
	def learning_rate(self):
		return 0.001

	@property
	def regularization_beta(self):
		return 1e-5
	
	def criterion(self):
		return torch.nn.MSELoss()
	
	def optimizer(self):
		return torch.optim.Adam(
			self.model.parameters(), 
			lr=self.learning_rate)
	

class CartpoleV0:
	def __init__(self, render=False):
		self.__env = gym.make('CartPole-v0')
		self.__obsv = self.__env.reset()
		self.__done = False
		
	def reset(self):
		self.__done = False
		self.__obsv = self.__env.reset()

	@property
	def env(self):
		return self

	@property
	def obsv(self):
		return torch.from_numpy(self.__obsv.astype('float32'))
	
	@property
	def num_act(self):
		return 2

	@property
	def obsv_shape(self):
		return (4,)
	
	@property
	def done(self):
		return self.__done
	
	def seed(self, val):
		self.__env.seed(val)

	def step(self, action):
		(self.__obsv, reward, self.__done, _) = self.__env.step(action)
		return (reward, self.__done)
	
	




