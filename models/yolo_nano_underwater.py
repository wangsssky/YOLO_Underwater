import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_layers import conv1x1, conv3x3, EP, PEP, FCA, YOLOLayer


class YOLONano_Underwater(nn.Module):
	def __init__(self, num_classes, image_size):
		super(YOLONano_Underwater, self).__init__()
		self.num_classes = num_classes
		self.image_size = image_size
		self.num_anchors = 3
		self.yolo_channels = (self.num_classes + 5) * self.num_anchors

		# anchors52 = [[10, 13], [16, 30], [33, 23]]  # 52x52
		# anchors26 = [[30, 61], [62, 45], [59, 119]]  # 26x26
		# anchors13 = [[116, 90], [156, 198], [373, 326]]  # 13x13
		# UPRC @ 512
		anchors52 = [[17, 24], [24, 37], [28, 52]]  # 52x52
		anchors26 = [[40, 45], [39, 66], [55, 61]]  # 26x26
		anchors13 = [[50, 89], [71, 111], [120, 167]]  # 13x13

		# image:  416x416x3
		self.conv1 = conv3x3(3, 16, stride=1)  		# output: 416x416x12
		self.conv2 = conv3x3(16, 32, stride=2)  	# output: 208x208x24
		self.pep1 = PEP(32, 32, 8, stride=1)  		# output: 208x208x24
		self.ep1 = EP(32, 64, stride=2)  			# output: 104x104x70
		self.pep2 = PEP(64, 64, 16, stride=1)  		# output: 104x104x70
		self.pep3 = PEP(64, 64, 16, stride=1)  		# output: 104x104x70
		self.ep2 = EP(64, 128, stride=2)  			# output: 52x52x150
		self.pep4 = PEP(128, 128, 32, stride=1)  	# output: 52x52x150
		self.conv3 = conv1x1(128, 128, stride=1)  	# output: 52x52x150
		self.fca1 = FCA(128, 8)  					# output: 52x52x150
		self.pep5 = PEP(128, 128, 32, stride=1)  	# output: 52x52x150
		self.pep6 = PEP(128, 128, 32, stride=1)  	# output: 52x52x150

		# self.pep7 = PEP(150, 150, 75, stride=1)  	# output: 52x52x150
		self.ep3 = EP(128, 256, stride=2)  			# output: 26x26x325
		self.pep8 = PEP(256, 256, 64, stride=1)  	# output: 26x26x325
		self.pep9 = PEP(256, 256, 128, stride=1)  	# output: 26x26x325
		self.pep10 = PEP(256, 256, 256, stride=1)  	# output: 26x26x325
		self.pep11 = PEP(256, 256, 128, stride=1)  	# output: 26x26x325
		self.pep12 = PEP(256, 256, 64, stride=1)  	# output: 26x26x325
		# self.pep13 = PEP(325, 325, 135, stride=1)  	# output: 26x26x325
		# self.pep14 = PEP(325, 325, 133, stride=1)  	# output: 26x26x325
		# self.pep15 = PEP(325, 325, 140, stride=1)  	# output: 26x26x325

		self.ep4 = EP(256, 384, stride=2)  			# output: 13x13x545
		self.pep16 = PEP(384, 384, 192, stride=1)  	# output: 13x13x545
		self.conv4 = conv1x1(384, 192, stride=1)  	# output: 13x13x230
		self.ep5 = EP(192, 384, stride=1)  			# output: 13x13x489
		self.pep17 = PEP(384, 384, 192, stride=1)  	# output: 13x13x469
		self.conv5 = conv1x1(384, 256, stride=1)  	# output: 13x13x189

		self.conv6 = conv1x1(256, 128, stride=1)  	# output: 13x13x105

		# upsampling conv6 to 26x26x105
		# concatenating [conv6, pep15] -> pep18 (26x26x430)
		self.pep18 = PEP(384, 256, 128, stride=1)  	# output: 26x26x325
		self.pep19 = PEP(256, 256, 128, stride=1)  	# output: 26x26x325

		self.conv7 = conv1x1(256, 128, stride=1)  	# output: 26x26x98
		self.conv8 = conv1x1(128, 64, stride=1)  	# output: 26x26x47

		# upsampling conv8 to 52x52x47
		# concatenating [conv8, pep7] -> pep20 (52x52x197)
		self.pep20 = PEP(192, 128, 64, stride=1)  	# output: 52x52x122
		self.pep21 = PEP(128, 128, 64, stride=1)  	# output: 52x52x87
		self.pep22 = PEP(128, 128, 64, stride=1)  	# output: 52x52x93
		self.conv9 = conv1x1(128, self.yolo_channels, stride=1, bn=False)  # output: 52x52x yolo_channels
		self.yolo_layer52 = YOLOLayer(anchors52, num_classes, img_dim=image_size)

		# conv7 -> ep6
		self.ep6 = EP(128, 256, stride=1)  # output: 26x26x183
		self.conv10 = conv1x1(256, self.yolo_channels, stride=1, bn=False)  # output: 26x26x yolo_channels
		self.yolo_layer26 = YOLOLayer(anchors26, num_classes, img_dim=image_size)

		# conv5 -> ep7
		self.ep7 = EP(256, 512, stride=1)  # output: 13x13x462
		self.conv11 = conv1x1(512, self.yolo_channels, stride=1, bn=False)  # output: 13x13x yolo_channels
		self.yolo_layer13 = YOLOLayer(anchors13, num_classes, img_dim=image_size)

		self.yolo_layers = [self.yolo_layer52, self.yolo_layer26, self.yolo_layer13]

	def forward(self, x, targets=None, indexes=None):
		loss = 0
		yolo_outputs = []
		image_size = x.size(2)

		out = self.conv1(x)
		out = self.conv2(out)
		out = self.pep1(out)
		out = self.ep1(out)
		out = self.pep2(out)
		out = self.pep3(out)
		out = self.ep2(out)
		out = self.pep4(out)
		out = self.conv3(out)
		out = self.fca1(out)
		out = self.pep5(out)
		out_pep6 = self.pep6(out)

		out = self.ep3(out_pep6)
		out = self.pep8(out)
		out = self.pep9(out)
		out = self.pep10(out)
		out = self.pep11(out)
		out_pep12 = self.pep12(out)

		out = self.ep4(out_pep12)
		out = self.pep16(out)
		out = self.conv4(out)
		out = self.ep5(out)
		out = self.pep17(out)

		out_conv5 = self.conv5(out)
		out = F.interpolate(self.conv6(out_conv5), scale_factor=2)
		out = torch.cat([out, out_pep12], dim=1)
		out = self.pep18(out)
		out = self.pep19(out)

		out_conv7 = self.conv7(out)
		out = F.interpolate(self.conv8(out_conv7), scale_factor=2)
		out = torch.cat([out, out_pep6], dim=1)
		out = self.pep20(out)
		out = self.pep21(out)
		out = self.pep22(out)
		out_conv9 = self.conv9(out)
		temp, layer_loss = self.yolo_layer52(out_conv9, targets, indexes, image_size)
		loss += layer_loss
		yolo_outputs.append(temp)

		out = self.ep6(out_conv7)
		out_conv10 = self.conv10(out)
		temp, layer_loss = self.yolo_layer26(out_conv10, targets, indexes, image_size)
		loss += layer_loss
		yolo_outputs.append(temp)

		out = self.ep7(out_conv5)
		out_conv11 = self.conv11(out)
		temp, layer_loss = self.yolo_layer13(out_conv11, targets, indexes, image_size)
		loss += layer_loss
		yolo_outputs.append(temp)

		yolo_outputs = torch.cat(yolo_outputs, 1)

		return yolo_outputs if targets is None else (loss, yolo_outputs)

	def name(self):
		return "YOLO-Nano-Underwater"