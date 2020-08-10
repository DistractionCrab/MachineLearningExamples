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
import ml.image.mnist as base





class Model(base.Model):
	@property
	def data_transform(self):
		t1 = torchvision.transforms.Lambda(lambda i: torch.tensor(i.getdata(), dtype=torch.float))
		t2 = torchvision.transforms.Lambda(lambda i: i/255.0)
		return torchvision.transforms.Compose([t1, t2])
		#return torchvision.transforms.Compose([	
		#	torchvision.transforms.Normalize((0.1307,), (0.3081,))
		#])

	
	

def main(args):
	i = Model()
	i.train()
