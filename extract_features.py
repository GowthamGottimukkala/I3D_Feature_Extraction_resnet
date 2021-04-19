import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from PIL import Image
import numpy as np
from resnet import i3_res50

def load_frame(frame_file):
	data = Image.open(frame_file)
	data = data.resize((340, 256), Image.ANTIALIAS)
	data = np.array(data)
	data = data.astype(float)
	data = (data * 2 / 255) - 1
	assert(data.max()<=1.0)
	assert(data.min()>=-1.0)
	return data

def load_rgb_batch(frames_dir, rgb_files, frame_indices):
	batch_data = np.zeros(frame_indices.shape + (256,340,3))
	for i in range(frame_indices.shape[0]):
		for j in range(frame_indices.shape[1]):
			batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
	return batch_data

def run(frequency, frames_dir, batch_size=1):
	chunk_size = 16
	# setup the model
	i3d = i3_res50(400)
	i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode

	def forward_batch(b_data):
		b_data = b_data.transpose([0, 4, 1, 2, 3])
		b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224
		b_data = Variable(b_data.cuda(), volatile=True).float()
		inp = {'frames': b_data}
		with torch.no_grad():
			features = i3d(inp)
		return features.cpu().numpy()

	rgb_files = [i for i in os.listdir(frames_dir)]
	rgb_files.sort()
	frame_cnt = len(rgb_files)
	# Cut frames
	assert(frame_cnt > chunk_size)
	clipped_length = frame_cnt - chunk_size
	clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk
	frame_indices = [] # Frames to chunks
	for i in range(clipped_length // frequency + 1):
		frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
	frame_indices = np.array(frame_indices)
	chunk_num = frame_indices.shape[0]
	batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
	frame_indices = np.array_split(frame_indices, batch_num, axis=0)
	
	full_features = [[]]
	for batch_id in range(batch_num): 
		batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])
		batch_data = batch_data[:,:,16:240,58:282,:] # Center Crop  (39, 16, 224, 224, 2)
		assert(batch_data.shape[-2]==224)
		assert(batch_data.shape[-3]==224)
		temp = forward_batch(batch_data)
		temp = temp[0,:,0,0,0]
		full_features[0].append(temp)
	return np.array(full_features)
