from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from extract_features import run
from resnet import i3_res50
import os


def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size):
	Path(outputpath).mkdir(parents=True, exist_ok=True)
	temppath = outputpath+ "/temp/"
	rootdir = Path(datasetpath)
	videos = [str(f) for f in rootdir.glob('**/*.mp4')]
	# setup the model
	i3d = i3_res50(400, pretrainedpath)
	i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode
	for video in videos:
		videoname = video.split("/")[-1].split(".")[0]
		startime = time.time()
		print("Generating for {0}".format(video))
		Path(temppath).mkdir(parents=True, exist_ok=True)
		ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()
		print("Preprocessing done..")
		features = run(i3d, frequency, temppath, 1)
		np.save(outputpath + "/" + videoname, features)
		print("Obtained features of size: ", features.shape)
		shutil.rmtree(temppath)
		print("done in {0}.".format(time.time() - startime))

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str)
	parser.add_argument('--outputpath', type=str)
	parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=1)
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size)    
