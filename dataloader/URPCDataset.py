import torch
import xml.etree.ElementTree as ET
import os
import cv2
import copy
import numpy as np
from torchvision import transforms

from tqdm import tqdm

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class URPCDataset(torch.utils.data.Dataset):
	CLASSES_NAME = (
		# "__background__ ",
		"holothurian",
		"echinus",
		"scallop",
		"starfish"
	)

	def __init__(
			self, root_dir, image_size=416, split='train',
			use_augmentation=False, box_type='yolo', cache=False, preprocessing=False):

		self.root = root_dir
		self.imgset = split
		self.image_size = image_size
		self.use_augmentation = use_augmentation
		self.box_type = box_type
		self.transform = self.get_augmentation(self.imgset, self.use_augmentation)
		self.cache = cache
		self.preprocessing = preprocessing

		self._annopath = os.path.join(self.root, "refined_box", "%s.xml")
		self._imgpath = os.path.join(self.root, "image", "%s.jpg")
		self._imgsetpath = os.path.join(self.root, "%s.txt")

		# read dataset
		with open(self._imgsetpath % self.imgset) as f:
			self.img_ids = f.readlines()
		self.img_ids = [x.strip() for x in self.img_ids]

		self.name2id = dict(zip(URPCDataset.CLASSES_NAME, range(len(URPCDataset.CLASSES_NAME))))

		self.mean = np.array([0.485, 0.456, 0.406])
		self.std = np.array([0.229, 0.224, 0.225])

		if self.cache:
			print('LOADING {} dataset...'.format(self.imgset))
			self.cached_images = []
			self.cached_boxes = []
			self.cached_classes = []
			for index in tqdm(range(len(self.img_ids))):
				img, boxes, classes = self.load_image_label(index)
				self.cached_images.append(img)
				self.cached_boxes.append(boxes)
				self.cached_classes.append(classes)
		print("INFO=====>URPC {} init finished !".format(split))

	def __len__(self):
		return len(self.img_ids)

	def _read_img_rgb(self, path):
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		assert img is not None, f"file named {path} not found"

		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	def load_image_label(self, index):
		img_id = self.img_ids[index]
		img = self._read_img_rgb(self._imgpath % img_id)

		anno = ET.parse(self._annopath % img_id).getroot()
		ia_boxes = []
		classes = []

		for obj in anno.iter("object"):
			name = obj.find("name").text.lower().strip()
			if name not in self.name2id.keys():
				continue
			classes.append(self.name2id[name])

			_box = obj.find("bndbox")
			box = [
				_box.find("xmin").text,
				_box.find("ymin").text,
				_box.find("xmax").text,
				_box.find("ymax").text,
			]
			box = tuple(
				map(lambda x: x, list(map(float, box)))
			)
			ia_bbox = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
			ia_boxes.append(ia_bbox)

		ia_boxes = BoundingBoxesOnImage(ia_boxes, shape=img.shape)
		return img, ia_boxes, classes

	def __getitem__(self, index):
		if not self.cache:
			img_id = self.img_ids[index]
			img = self._read_img_rgb(self._imgpath % img_id)

			anno = ET.parse(self._annopath % img_id).getroot()
			ia_boxes = []
			classes = []

			for obj in anno.iter("object"):
				name = obj.find("name").text.lower().strip()
				if name not in self.name2id.keys():
					continue
				classes.append(self.name2id[name])

				_box = obj.find("bndbox")
				box = [
					_box.find("xmin").text,
					_box.find("ymin").text,
					_box.find("xmax").text,
					_box.find("ymax").text,
				]
				box = tuple(
					map(lambda x: x, list(map(float, box)))
				)
				ia_bbox = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
				ia_boxes.append(ia_bbox)

			ia_boxes = BoundingBoxesOnImage(ia_boxes, shape=img.shape)
		else:
			img = copy.deepcopy(self.cached_images[index])
			ia_boxes = copy.deepcopy(self.cached_boxes[index])
			classes = copy.deepcopy(self.cached_classes[index])

		img, ia_boxes_pre = self.transform(image=img, bounding_boxes=ia_boxes)
		ia_boxes = ia_boxes_pre.remove_out_of_image().clip_out_of_image()
		# image_after = ia_boxes.draw_on_image(img, size=2, color=[0, 0, 255])
		# cv2.imshow("test", image_after)
		# cv2.waitKey(0)

		boxes = []
		for box in ia_boxes:
			boxes.append([box.x1, box.y1, box.x2, box.y2])

		im_h, im_w, im_c = img.shape
		boxes = self.box_type_convert(boxes, im_h, im_w, mode='yolo')

		targets = []
		for i, box in enumerate(boxes):
			targets.append([index, classes[i], *box])
			# targets.append([0, classes[i], *box])

		img = transforms.ToTensor()(img)
		if not self.preprocessing:
			img = transforms.Normalize(self.mean, self.std)(img)

		# boxes = np.array(boxes, dtype=np.float32)
		# boxes = torch.from_numpy(boxes)
		# classes = torch.LongTensor(classes)

		targets = torch.from_numpy(np.array(targets, dtype=np.float32))
		index = torch.LongTensor([index])

		return img, targets, index

	def get_augmentation(self, split, augmentation):
		if split == 'train' and augmentation:
			seq = iaa.Sequential([
				iaa.PadToSquare(),
				iaa.Crop(percent=(0, 0.2)),
				iaa.Fliplr(0.5),
				iaa.Flipud(0.5),
				iaa.Affine(rotate=(-25, 25)),
				iaa.Resize(int(self.image_size)),
				iaa.AddToHueAndSaturation((-60, 60))
			])
		else:
			seq = iaa.Sequential([
				iaa.PadToSquare(position='center'),
				iaa.Resize(int(self.image_size)),
			])
		return seq

	def box_type_convert(self, boxes, height, width, mode='yolo'):
		if mode == 'xyxy':
			return boxes
		elif mode == 'coco':
			new_boxes = []
			for box in boxes:
				xmin, ymin, xmax, ymax = box
				new_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
			return new_boxes
		elif mode == 'yolo':
			new_boxes = []
			for box in boxes:
				xmin, ymin, xmax, ymax = box
				tx = (xmin + xmax) / (2 * width)
				ty = (ymin + ymax) / (2 * height)
				w = (xmax - xmin) / width
				h = (ymax - ymin) / height
				new_boxes.append([tx, ty, w, h])
			return new_boxes
		else:
			raise ValueError('Invalid box type: {}'.format(mode))


def collate_fn(batch):
	images, targets, indexes = list(zip(*batch))

	images = torch.stack([im for im in images], dim=0)
	indexes = torch.stack([id for id in indexes], dim=0)

	targets = [bboxes for bboxes in targets if bboxes is not None]
	for i, bboxes in enumerate(targets):
		if len(bboxes) == 0:
			continue
		# bboxes[:, 0] = i
	targets = torch.cat(targets, 0)

	return images, targets, indexes


if __name__ == '__main__':
	dataset = URPCDataset('../data/', split='train', use_augmentation=False)
	img, box, cls = dataset[0]
	print(img.shape)
	print(box)
	print(cls)

	img_ = (img.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
	print(img_.shape)
	cv2.imshow('test', img_)
	cv2.waitKey(0)
