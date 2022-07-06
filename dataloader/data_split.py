import os
import numpy as np

names = os.listdir('../data/image')

np.random.seed(1234)
np.random.shuffle(names)

train_set = names[:3600]
val_set = names[3600:4000]
test_set = names[4000:]

save_names = ['train', 'val', 'test']
names_list = [train_set, val_set, test_set]

for index, save_name in enumerate(save_names):
	with open('../data/{}.txt'.format(save_name), 'w') as f:
		for name in names_list[index]:
			f.write(name[:-4] + '\n')
