import torch

import cv2
import os
import numpy as np
import scipy
from tqdm import tqdm
import argparse

import xml.etree.ElementTree as ET


# based https://github.com/ultralytics/yolov5
def kmean_anchors(data_dir, list_path=None, anchor_num=9, img_size=416, thr=4.0, gen=1000, verbose=True):
	""" Creates kmeans-evolved anchors from training dataset
		Arguments:
			anchor_num: number of anchors
			img_size: image size used for training
			thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
			gen: generations to evolve anchors using genetic algorithm
		Return:
			k: kmeans evolved anchors
		Usage:
			from utils.utils import *; _ = kmean_anchors()
	"""
	thr = 1. / thr
	def metric(k, wh):  # compute metrics
		r = wh[:, None] / torch.tensor(k[None])
		x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
		# x = wh_iou(wh, torch.tensor(k))  # iou metric
		return x, x.max(1)[0]  # x, best_x

	def fitness(k):  # mutation fitness
		_, best = metric(torch.tensor(k, dtype=torch.float32), wh)
		return (best * (best > thr).float()).mean()  # fitness

	def print_results(k):
		k = k[np.argsort(k.prod(1))]  # sort small to large
		x, best = metric(k, wh0)
		bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * anchor_num  # best possible recall, anch > thr
		print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
		print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
			  (anchor_num, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
		for i, x in enumerate(k):
			print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
		return k

	# load label files
	if list_path is not None:
		with open(list_path, "r") as file:
			names = file.readlines()

	# laod labels
	wh0 = []
	boxes = []
	for name in tqdm(names):
		img = cv2.imread(os.path.join(data_dir, 'image', name.strip()+'.jpg'))
		height, width, c = img.shape
		anno = ET.parse(os.path.join(data_dir, 'refined_box', name.strip()+'.xml')).getroot()
		for obj in anno.iter("object"):
			name = obj.find("name").text.lower().strip()
			if name not in ["holothurian", "echinus", "scallop", "starfish"]:
				continue

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

			xmin, ymin, xmax, ymax = box
			tx = (xmin + xmax) / (2 * width)
			ty = (ymin + ymax) / (2 * height)
			w = (xmax - xmin) / width
			h = (ymax - ymin) / height
			boxes.append([tx, ty, w, h])

	boxes = np.array(boxes, dtype=np.float32)

	wh0.append(boxes[:, -2:])
	wh0 = np.concatenate(wh0)
	wh0 = img_size * wh0

	# Filter
	i = (wh0 < 4.0).any(1).sum()
	if i:
		print('WARNING: Extremely small objects found. '
			  '%g of %g labels are < 4 pixels in width or height.' % (i, len(wh0)))
	wh = wh0[(wh0 >= 4.0).any(1)]  # filter > 4 pixels

	# Kmeans calculation
	from scipy.cluster.vq import kmeans
	print('Running kmeans for %g anchors on %g points...' % (anchor_num, len(wh)))
	s = wh.std(0)  # sigmas for whitening
	k, dist = kmeans(wh / s, anchor_num, iter=30)  # points, mean distance
	k *= s
	wh = torch.tensor(wh, dtype=torch.float32)  # filtered
	wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
	k = print_results(k)

	# Plot
	# k, d = [None] * 20, [None] * 20
	# for i in tqdm(range(1, 21)):
	#     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
	# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
	# ax = ax.ravel()
	# ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
	# fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
	# ax[0].hist(wh[wh[:, 0]<100, 0],400)
	# ax[1].hist(wh[wh[:, 1]<100, 1],400)
	# fig.tight_layout()
	# fig.savefig('wh.png', dpi=200)

	# Evolve
	npr = np.random
	f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
	pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
	for _ in pbar:
		v = np.ones(sh)
		while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
			v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
		kg = (k.copy() * v).clip(min=2.0)
		fg = fitness(kg)
		if fg > f:
			f, k = fg, kg.copy()
			pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
			if verbose:
				print_results(k)

	return print_results(k)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--list_path", type=str, default="../data/train.txt", help="path to dataset")
	parser.add_argument("--data_dir", type=str, default="../data", help="path to dataset")
	parser.add_argument("--input_size", type=int, default=512, help="size of input image for network")
	parser.add_argument("--anchor_num", type=int, default=6, help="number of anchors")
	opt = parser.parse_args()
	print(opt)

	kmean_anchors(opt.data_dir, list_path=opt.list_path, anchor_num=opt.anchor_num, img_size=opt.input_size)
