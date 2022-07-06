import torch
import torch.nn as nn
from torchvision import transforms


class Preprocessing(nn.Module):
	def __init__(self, input_channels, pool_size=8, hidden_channels=64):
		super(Preprocessing, self).__init__()
		self.input_channels = input_channels
		self.pool_size = pool_size
		self.mean = torch.tensor([0.485, 0.456, 0.406])
		self.std = torch.tensor([0.229, 0.224, 0.225])

		self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
		self.fc = nn.Sequential(
			nn.Linear(input_channels * self.pool_size * self.pool_size, hidden_channels),
			nn.ReLU6(inplace=True),
			nn.Linear(hidden_channels, input_channels*input_channels + input_channels*2, bias=False),
		)
		self.act1 = nn.Sigmoid()

	def forward(self, x):
		b, c, h, w = x.size()
		param = self.pool(x).view(b, c * self.pool_size * self.pool_size)
		param = self.act1(self.fc(param))
		out = torch.bmm(param[:, :c*c].view(b, 3, 3), x.view(b, c, -1)).view(b, c, h, w) + param[:, c * c: c*(c+1)].view(b, c, 1, 1).expand([b, c, h, w])
		out = torch.pow(out, param[:, c*(c+1):].view(b, c, 1, 1).expand([b, c, h, w]))
		# out = x + out
		out = transforms.Normalize(self.mean, self.std)(out)
		return out
