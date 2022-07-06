import os
import torch
from terminaltables import AsciiTable

from tqdm import tqdm


def train(model, optimizer, scheduler, dataloader, epoch, opt, logger, best_mAP=0):
	model.train()
	for i, (images, targets, indexes) in enumerate(tqdm(dataloader)):
		optimizer.zero_grad()
		# targets: [index, class_id, x, y, h, w] in yolo format

		rep_targets = []
		for ri in range(torch.cuda.device_count()):
			rep_targets.append(targets.unsqueeze(dim=0))
		rep_targets = torch.cat(rep_targets, dim=0)

		if opt.gpu:
			images = images.cuda()
			indexes = indexes.cuda()
			rep_targets = rep_targets.cuda()
			targets = targets.cuda()

		loss, detections = model.forward(images, rep_targets, indexes)
		# detections = to_cpu(detections)
		if torch.cuda.device_count() > 1:
			loss = loss.sum()

		loss.backward()
		optimizer.step()

		# logging
		if torch.cuda.device_count() > 1:
			metric_keys = model.module.yolo_layers[0].metrics.keys()
			yolo_metrics = [model.module.yolo_layers[i].metrics for i in range(len(model.module.yolo_layers))]
		else:
			metric_keys = model.yolo_layers[0].metrics.keys()
			yolo_metrics = [model.yolo_layers[i].metrics for i in range(len(model.yolo_layers))]

		layer_header = ['YOLO Layer {}'.format(i) for i in range(len(yolo_metrics))]
		metric_table_data = [['Metrics', *layer_header]]
		formats = {m: '%.6f' for m in metric_keys}
		for metric in metric_keys:
			row_metrics = [formats[metric] % ym.get(metric, 0) for ym in yolo_metrics]
			metric_table_data += [[metric, *row_metrics]]
		metric_table_data += [['total loss', '{:.6f}'.format(loss.item()), '', '']]

		# beautify log message
		metric_table = AsciiTable(
			metric_table_data,
			title='[Epoch {:d}/{:d}, Batch {:d}/{:d}, Current best mAP {:4f}]'.format(epoch, opt.num_epochs, i,
																					  len(dataloader), best_mAP))
		metric_table.inner_footing_row_border = True
		logger.print_and_write('{}\n'.format(metric_table.table))

	scheduler.step()

	# print("current best mAP:" + str(best_mAP))

	# save checkpoints
	if torch.cuda.device_count() > 1:
		states = {
			'epoch': epoch + 1,
			'model': opt.model,
			'state_dict': model.module.state_dict(),
			'optimizer': optimizer.state_dict(),
			'scheduler': scheduler.state_dict(),
			'best_mAP': best_mAP,
		}
	else:
		states = {
			'epoch': epoch + 1,
			'model': opt.model,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'scheduler': scheduler.state_dict(),
			'best_mAP': best_mAP,
		}

	save_file_path = os.path.join(opt.checkpoint_path, 'last.pth'.format(epoch))
	torch.save(states, save_file_path)

	if epoch % opt.checkpoint_interval == 0:
		save_file_path = os.path.join(opt.checkpoint_path, 'epoch_{}.pth'.format(epoch))
		torch.save(states, save_file_path)