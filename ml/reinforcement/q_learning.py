import gym
import sys
import operator
import random
import torch
import torchvision
import numpy                as np
import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

import ml.reinforcement as rl

class QLearning(rl.RLModel):
	def __init__(self, model):
		self.__model = model

	def _train_epoch(self, crit, opt):
		self.model.env.reset()
		self.model.env.seed(random.randint(0, sys.maxsize))
		rewards = []

		for _ in range(self.iterations):
			if self.model.env.done:
				self.model.env.reset()
				self.model.env.seed(random.randint(0, sys.maxsize))
			qvalue = self.model(self.model.env.obsv	)			
			(r, done) = self.model.env.step(qvalue.argmax().item())
			rewards.append(r)

			qnext = self.model(self.model.env.obsv)

			loss = crit(r + qnext.max(), qvalue.max())
			opt.zero_grad()
			loss.backward()
			opt.step()
			
		return torch.FloatTensor(rewards)

	@property
	def iterations(self):
		return 500
	
	@property
	def model(self):
		return self.__model
	


class CartpoleModule(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.__env = rl.CartpoleV0()

		print(self.__env.obsv_shape)
		self.l1 = torch.nn.Linear(ftools.reduce(operator.mul, self.__env.obsv_shape), 64)
		self.r1 = torch.nn.Sigmoid()
		self.l2 = torch.nn.Linear(64, 30)
		self.r2 = torch.nn.Sigmoid()
		self.l3 = torch.nn.Linear(30, self.__env.num_act)
		self.r3 = torch.nn.Softmax(dim=0)

	def forward(self, x):
		out = self.l1(x)
		out = self.r1(out)
		out = self.l2(out)
		out = self.r2(out)
		out = self.l3(out)
		return self.r3(out)

	@property
	def env(self):
		return self.__env


def main(args):
	if len(args) == 0:
		QLearning(CartpoleModule()).train()