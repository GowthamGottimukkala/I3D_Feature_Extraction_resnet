from pathlib import Path
import shutil
import argparse
import subprocess
import sys
import numpy as np
import os
import time
from extract_features import run
import ffmpeg


def generate(datasetpath, outputpath, frequency, batch_size):
	temppath = outputpath+ "/temp/"
	rootdir = Path(datasetpath)
	videos = [str(f) for f in rootdir.glob('**/*.mp4')]
	for video in videos:
		startime = time.time()
		print("Generating for {0}".format(video))
		Path(temppath).mkdir(parents=True, exist_ok=True)
		ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).run()
		print("Preprocessing done..")
		features = run(frequency, temppath, 1)
		savepath = outputpath + "/" + video.split("/")[-1].split(".")[0] + "/"
		Path(savepath).mkdir(parents=True, exist_ok=True)
		np.save(savepath + "feature_rgb", features)
		print("Obtained features of size: ", features.shape)
		shutil.rmtree(temppath)
		print("done in {0}.".format(time.time() - startime))

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str)
	parser.add_argument('--outputpath', type=str)
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=1)
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.frequency, args.batch_size)    
