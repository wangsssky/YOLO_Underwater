import torch
import torch.nn as nn

import os
import numpy as np


def seed_torch(seed=22):
	os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True


def truncated_normal_(tensor, mean=0, std=1):
	size = tensor.shape
	tmp = tensor.new_empty(size + (4,)).normal_()
	valid = (tmp < 2) & (tmp > -2)
	ind = valid.max(-1, keepdim=True)[1]
	tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
	tensor.data.mul_(std).add_(mean)


def init_weights(m):
	if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
		nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
		# nn.init.normal_(m.weight, std=0.001)
		# nn.init.normal_(m.bias, std=0.001)
		if m.bias is not None:
			truncated_normal_(m.bias, mean=0, std=0.001)
	if type(m) == nn.Linear:
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	if type(m) == nn.BatchNorm2d:
		nn.init.uniform_(m.weight)
		nn.init.constant_(m.bias, 0)