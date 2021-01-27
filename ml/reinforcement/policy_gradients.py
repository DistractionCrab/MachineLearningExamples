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

from collections import deque



class PolicyGradients(rl.RLModel):
	def __init__(self, model):
		self.__model = model

	def _train_epoch(self, crit, opt):
		self.model.env.reset()
		self.model.env.seed(random.randint(0, sys.maxsize))
		rewards = deque(maxlen=self.iterations)
		pis     = deque(maxlen=self.iterations)
		actions = deque(maxlen=self.iterations)

		for _ in range(self.iterations):
			if self.model.env.done:
				self.model.env.reset()
				self.model.env.seed(random.randint(0, sys.maxsize))
			pvalue = self.model(self.model.env.obsv)
			action = random.choices(list(range(self.model.env.num_act)), weights=pvalue.flatten())[0]
			(r, done) = self.model.env.step(action)
			rewards.append(r)
			pis.append(pvalue)
			actions.append(action)



		rewards      = torch.Tensor(rewards)
		pis          = torch.stack(tuple(pis)).squeeze(1)
		actions      = torch.LongTensor(actions)
		disc_rewards = self.__compute_discounted(rewards)
		actions      = torch.nn.functional.one_hot(actions, num_classes=self.model.env.num_act).float()

		pis = torch.log((pis*actions).sum(1))
		loss = torch.mean(disc_rewards * pis)

		#loss = crit(r + qnext.max(), qvalue.max())
		opt.zero_grad()
		loss.backward()
		opt.step()

		return rewards

	def __compute_discounted(self, l):
		for i in reversed(range(len(l))[:-1]):
			l[i] = l[i+1] * self.discount + l[i]
		return l

	@property
	def discount(self):
		"""
		Discount factor for normalizing rewards (Gamma).
		"""
		return 0.95
	

	@property
	def iterations(self):
		return 500
	
	@property
	def model(self):
		return self.__model
	


class MsPacmanModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.__env = rl.MsPacman()

		self.layer1 = nn.Sequential(
			nn.Conv2d(4, 32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(21 * 21 * 64, 1000)
		self.fc2 = nn.Linear(1000, self.env.num_act)
		self.act = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.reshape(x.size(0), -1)
		x = self.drop_out(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return self.act(x)

	@property
	def env(self):
		return self.__env


def main(args):
	if len(args) == 0:
		PolicyGradients(MsPacmanModel()).train()