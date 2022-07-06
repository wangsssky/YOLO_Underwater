import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_layers import conv1x1, conv3x3, EP, PEP, FCA, YOLOLayer
from .ghost_module import GhostModule, GhostBottleneck, CheapOps, Mix
from .preprocessing_module import Preprocessing


class YOLO_Underwater_Tiny(nn.Module):
	def __init__(self, num_classes, image_size, use_preprocessing=False):
		super(YOLO_Underwater_Tiny, self).__init__()
		self.num_classes = num_classes
		self.image_size = image_size
		self.num_anchors = 3
		self.yolo_channels = (self.num_classes + 5) * self.num_anchors

		# anchors52 = [[10, 13], [16, 30], [33, 23]]  # 52x52
		# anchors26 = [[30, 61], [62, 45], [59, 119]]  # 26x26
		# anchors13 = [[116, 90], [156, 198], [373, 326]]  # 13x13
		# UPRC @ 512
		anchors26 = [[17, 25], [24, 37], [32, 50]]  	# 26x26
		anchors13 = [[41, 64], [52, 84], [74, 113]]  	# 13x13

		self.use_preprocessing = use_preprocessing
		if self.use_preprocessing:
			self.preprocessing = Preprocessing(input_channels=3)

		# image:  416x416x3
		ratio = 1
		self.conv1 = conv3x3(3, int(16*ratio), stride=1)  															# output: 416x416

		self.conv2 = GhostModule(int(16*ratio), int(32*ratio), ratio=2, kernel_size=3, stride=2)  					# output: 208x208
		self.pep1 = GhostBottleneck(int(32*ratio), 32, int(32*ratio), se_ratio=0.25, ghost_ratio=2)  				# output: 208x208

		self.ep1 = GhostModule(int(32*ratio), int(64*ratio), ratio=2, stride=2)  									# output: 104x104
		self.pep3 = GhostBottleneck(int(64*ratio), 96, int(64*ratio), se_ratio=0.25, ghost_ratio=2)  				# output: 104x104

		self.ep2 = GhostBottleneck(int(64*ratio), 160, int(128*ratio), se_ratio=0, ghost_ratio=2, stride=2)  		# output: 52x52
		self.pep6 = GhostBottleneck(int(128*ratio), 192, int(128*ratio), se_ratio=0.25, ghost_ratio=2, stride=1)  	# output: 52x52

		self.ep3 = GhostBottleneck(int(128*ratio), 384, int(128*ratio), se_ratio=0, ghost_ratio=2, stride=2)  		# output: 26x26
		self.pep9 = GhostBottleneck(int(128*ratio), 512, int(128*ratio), se_ratio=0.25, ghost_ratio=2, stride=1)  	# output: 26x26
		self.pep12 = GhostBottleneck(int(128*ratio), 768, int(128*ratio), se_ratio=0, ghost_ratio=2, stride=1)  	# output: 26x26
		self.chp_op1 = CheapOps(int(128*ratio), int(128*ratio))
		self.mix1 = Mix(inp=int(256*ratio), oup=int(128*ratio))

		self.ep4 = GhostBottleneck(int(256*ratio), 768, int(128*ratio), se_ratio=0, ghost_ratio=2, stride=2)  		# output: 13x13
		self.pep17 = GhostBottleneck(int(128*ratio), 1024, int(128*ratio), se_ratio=0.25, ghost_ratio=2, stride=1)  # output: 13x13
		self.conv5 = GhostBottleneck(int(128*ratio), 1024, int(128*ratio), se_ratio=0, ghost_ratio=2, stride=1)  	# output: 13x13
		self.chp_op2 = CheapOps(int(128*ratio), int(128*ratio))
		self.mix2 = Mix(inp=int(256*ratio), oup=int(128*ratio))

		self.conv6 = conv1x1(int(256*ratio), int(128*ratio), stride=1)  											# output: 13x13

		# upsampling conv6 to 26x26x105
		# concatenating [conv6, pep15] -> pep18 (26x26x430)
		self.pep19 = GhostBottleneck(int(384*ratio), 768, int(256*ratio), se_ratio=0, ghost_ratio=2, stride=1)  	# output: 26x26

		self.conv7 = conv1x1(int(256*ratio), int(128*ratio), stride=1)  											# output: 26x26

		# conv7 -> ep6
		self.ep6_reg_iou = GhostBottleneck(int(128*ratio), 768, int(256*ratio), se_ratio=0, ghost_ratio=2, stride=1)# output: 26x26
		self.ep6_cls = GhostBottleneck(int(128*ratio), 768, int(256*ratio), se_ratio=0, ghost_ratio=2, stride=1)  	# output: 26x26
		self.conv10_reg = conv1x1(int(256*ratio), 4 * self.num_anchors, stride=1, bn=False)  						# output: 26x26
		self.conv10_iou = conv1x1(int(256*ratio), self.num_anchors, stride=1, bn=False)  							# output: 26x26
		self.conv10_cls = conv1x1(int(256*ratio), self.num_classes * self.num_anchors, stride=1, bn=False)  		# output: 26x26
		self.yolo_layer26 = YOLOLayer(anchors26, num_classes, img_dim=image_size)

		# conv5 -> ep7
		self.ep7_reg_iou = GhostBottleneck(int(256*ratio), 1024, int(512*ratio), se_ratio=0, ghost_ratio=2, stride=1)  	# output: 13x13
		self.ep7_cls = GhostBottleneck(int(256*ratio), 1024, int(512*ratio), se_ratio=0, ghost_ratio=2, stride=1)  	# output: 13x13

		self.conv11_reg = conv1x1(int(512*ratio), 4 * self.num_anchors, stride=1, bn=False)  						# output: 13x13
		self.conv11_iou = conv1x1(int(512*ratio), self.num_anchors, stride=1, bn=False)  							# output: 13x13
		self.conv11_cls = conv1x1(int(512*ratio), self.num_classes * self.num_anchors, stride=1, bn=False)  		# output: 13x13
		self.yolo_layer13 = YOLOLayer(anchors13, num_classes, img_dim=image_size)

		self.yolo_layers = [self.yolo_layer26, self.yolo_layer13]

	def forward(self, x, targets=None, indexes=None):
		loss = 0
		yolo_outputs = []
		image_size = x.size(2)

		if self.use_preprocessing:
			x = self.preprocessing(x)

		out = self.conv1(x)
		out = self.conv2(out)
		out = self.pep1(out)
		out = self.ep1(out)
		out = self.pep3(out)
		out = self.ep2(out)
		out_pep6 = self.pep6(out)

		out_ep3 = self.ep3(out_pep6)
		out_pep9 = self.pep9(out_ep3)
		out_pep12 = self.pep12(out_pep9)
		chp_op1 = self.chp_op1(out_ep3)
		mix1 = self.mix1(blocks=[out_pep9, out_pep12], target=chp_op1)
		cat_1 = torch.cat([out_pep12, mix1], dim=1)

		ep4 = self.ep4(cat_1)
		pep17 = self.pep17(ep4)
		out_conv5 = self.conv5(pep17)
		chp_op2 = self.chp_op1(ep4)
		mix2 = self.mix1(blocks=[pep17, out_conv5], target=chp_op2)
		cat_2 = torch.cat([out_conv5, mix2], dim=1)

		out = F.interpolate(self.conv6(cat_2), scale_factor=2)
		out = torch.cat([out, cat_1], dim=1)
		out = self.pep19(out)

		out_conv7 = self.conv7(out)

		out_reg_iou = self.ep6_reg_iou(out_conv7)
		out_cls = self.ep6_cls(out_conv7)
		out_conv10_reg = self.conv10_reg(out_reg_iou)
		out_conv10_iou = self.conv10_iou(out_reg_iou)
		out_conv10_cls = self.conv10_cls(out_cls)
		out_conv10 = torch.cat([out_conv10_reg, out_conv10_iou, out_conv10_cls], dim=1)
		temp, layer_loss = self.yolo_layer26(out_conv10, targets, indexes, image_size)
		loss += layer_loss
		yolo_outputs.append(temp)

		out_reg_iou = self.ep7_reg_iou(cat_2)
		out_cls = self.ep7_cls(cat_2)
		out_conv11_reg = self.conv11_reg(out_reg_iou)
		out_conv11_iou = self.conv11_iou(out_reg_iou)
		out_conv11_cls = self.conv11_cls(out_cls)
		out_conv11 = torch.cat([out_conv11_reg, out_conv11_iou, out_conv11_cls], dim=1)
		temp, layer_loss = self.yolo_layer13(out_conv11, targets, indexes, image_size)
		loss += layer_loss
		yolo_outputs.append(temp)

		yolo_outputs = torch.cat(yolo_outputs, 1)
		return yolo_outputs if targets is None else (loss, yolo_outputs)

	def name(self):
		return "YOLO-Underwater-Tiny"
