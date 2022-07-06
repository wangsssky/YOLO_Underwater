import os
import json
import torch
import argparse


class Opt():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		self.parser.add_argument("--print_options", default=True, help="print options or not")
		# project root, dataset, checkpoint resume and pretrained model path
		self.parser.add_argument("--project_root", type=str, default=".", help="root directory path of project")
		self.parser.add_argument("--dataset_path", type=str, default="data/",
								 help="directory path of dataset")
		self.parser.add_argument("--cache", action="store_true", help="if true, cache the dataset", default=False)
		self.parser.add_argument("--pretrain", action="store_true", help="if true, do not load optimizer", default=False)

		self.parser.add_argument("--classname_path", type=str, default="data/urpc.names",
								 help="file path of classnames for visualizer")
		self.parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
								 help="directory path of checkpoints")
		self.parser.add_argument("--resume_path", type=str, default="", help="save data (.pth) of previous training")

		# common options that are used in both train and test
		self.parser.add_argument("--manual_seed", type=int, default=22, help="manual_seed of pytorch")
		self.parser.add_argument("--gpu", action="store_true", help="if true, cuda is used", default=False)
		self.parser.add_argument("--num_threads", type=int, default=4,
								 help="# of cpu threads to use for batch generation")
		self.parser.add_argument("--checkpoint_interval", type=int, default=5,
								 help="# interval between saving model weights")
		self.parser.add_argument("--val_interval", type=int, default=1, help="evaluation every # epochs")

		self.parser.add_argument("--model", type=str, default="YOLO-Underwater", help="choose which model to use")
		self.parser.add_argument("--image_size", type=int, default=512, help="size of image")
		self.parser.add_argument("--num_classes", type=int, default=4, help="# of classes of the dataset")
		self.parser.add_argument('--num_epochs', type=int, default=200, help='# of epochs')
		self.parser.add_argument('--begin_epoch', type=int, default=0, help='# of epochs')
		self.parser.add_argument("--batch_size", type=int, default=1, help="batch size")
		self.parser.add_argument("--version", type=int, default=0, help="YOLO Underwater version")
		self.parser.add_argument("--preprocessing", action="store_true", help="if true, use preprocessing module", default=False)

		self.parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer (Adam | SGD)")
		self.parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
		self.parser.add_argument('--momentum', type=float, default=0.9, help="momentum for optimizer")
		self.parser.add_argument('--weight_decay', type=float, default=5e-5, help="weight_decay for optimizer")

		# object detection options
		self.parser.add_argument("--conf_thresh", type=float, default=.25)
		self.parser.add_argument("--nms_thresh", type=float, default=.45)

		self.parser.add_argument("--no_train", action="store_true", default=False, help="training")
		self.parser.add_argument("--no_val", action="store_true", default=False, help="validation")
		self.parser.add_argument("--test", default=False, help="test")

		self.initialized = True

	def print_options(self):
		message = ''
		message += '------------------------ OPTIONS -----------------------------\n'
		for k, v in sorted(vars(self.opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}\n'.format(str(k), str(v), comment)
		message += '------------------------  END   ------------------------------\n'
		print(message)

	def parse(self):
		if not self.initialized:
			self.initialize()

		self.opt = self.parser.parse_args()

		if self.opt.project_root != '':
			self.opt.dataset_path = os.path.join(self.opt.project_root, self.opt.dataset_path)
			self.opt.checkpoint_path = os.path.join(self.opt.project_root, self.opt.checkpoint_path)
			if self.opt.resume_path:
				self.opt.resume_path = os.path.join(self.opt.project_root, self.opt.resume_path)

		os.makedirs(self.opt.checkpoint_path, exist_ok=True)

		with open(os.path.join(self.opt.checkpoint_path, 'opts.json'), 'w') as opt_file:
			json.dump(vars(self.opt), opt_file)

		self.opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		if self.opt.print_options:
			self.print_options()

		return self.opt