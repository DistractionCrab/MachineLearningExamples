import numpy
import operator
import random
import torch
import torchvision

import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

import ml.image as img_nn
import ml.image.mnist_scaled as img_scaled


class Module(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(7 * 7 * 64, 1000)
		self.fc2 = nn.Linear(1000, img_nn.MNIST_NUM_CLASSES)
		self.act = nn.Softmax(dim=0)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.reshape(x.size(0), -1)
		x = self.drop_out(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return self.act(x)


class Model(img_scaled.Model):
	def __init__(self):
		super().__init__()
		self.__model = Module()

	@property
	def model(self):
		return self.__model

	@property
	def data_transform(self):
		return torchvision.transforms.ToTensor()
	

def main(args):
	Model().train()

