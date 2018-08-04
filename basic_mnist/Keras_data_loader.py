#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mmap
from tqdm import tqdm

def get_num_lines(file_path):
	fp = open(file_path, "r+")
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines

def prepare_data(path, data_limit=0, peak_at_data=False,test=False):
	""" Data format (CSV)

	label,pixel0...,pixel783\n
	{0,9},{0,255}...,{0,255}\n
	{0,9},{0,255}...,{0,255}\n
	"""

	print('Start loading data from "{}".'.format(path))

	targets = []
	input_ = []

	if data_limit:
		total_line = data_limit
	else:
		total_line = get_num_lines(path)-1 #-1 cuz we are always skiping 1st line with lables
	

	with open(path,"r") as file:
		#skip 1st row which is lables 
		next(file)
		
		# counter
		c = 0	
		for line in tqdm(file, total=total_line): # type(line) -> string
			# remove carrige at the end of each line
			pixels = line.strip("\r\n").split(",")
			
			if test:
				input_.append(np.array(pixels,dtype=np.int32).reshape(28,28,1))
			else:
				targets.append(pixels[0])
				input_.append(np.array(pixels[1:],dtype=np.int32).reshape(28,28,1))

			c+=1
			if c == data_limit:
				break

	if test:
		targets = 0

	targets = np.array(targets,dtype=np.int32)
	input_ = np.array(input_,dtype=np.int32)

	# if data_limit:
	# 	assert(np.shape(targets)[0] == data_limit)
	# 	assert(np.shape(input_)[0] == data_limit)

	print('Shape of the targets: {}.'.format(np.shape(targets)))
	print('Shape of the input: {}. \n'.format(np.shape(input_)))
	
	if peak_at_data:
		plt.imshow(input_[0].reshape(28,28),cmap=plt.cm.gray)
		plt.show()

	return targets, input_


def load_data(train_path,test_path,data_limit):

	train_Y, train_X = prepare_data(train_path, data_limit)
	test_Y, test_X = prepare_data(test_path, data_limit,test=True)
	# test_X = 0
	# test_Y = 0


	return train_Y, train_X, test_X