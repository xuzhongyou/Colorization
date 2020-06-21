import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import PIL
from PIL import Image
from flowlib import read, read_weights_file
from skimage import io, transform
from PIL import Image
import numpy as np
import re
device='cuda'

def toString(num):
	string = str(num)
	while(len(string) < 4):
		string = "0"+string
	return string


class VideoDataset(Dataset):

	def __init__(self, name):
		"""
		looking at the "clean" subfolder for images, might change to "final" later
		root_dir -> path to the location where the "training" folder is kept inside the MPI folder
		"""
		self.root = '../data/'
		self.path = name
		self.image_path = os.path.join(self.root,'video-picture',self.path)
		self.imagelist = os.listdir(self.image_path)
		# ../data/video-picture-flow/file_name/backward_flow
		self.flow_path = os.path.join(self.root,'video-picture-flow',self.path,'backward_flow')
		self.flowlist = os.listdir(self.flow_path)
		self.consistency_path = os.path.join(self.root,'video-picture-flow',self.path,'forward_consistency')
		self.consistencylist = os.listdir(self.consistency_path)
		self.imagelist.sort(key=lambda x: int(re.split('\.|\_',x)[1]))
		self.flowlist.sort(key=lambda x: int(re.split('\.|\_',x)[1]))
		self.consistencylist.sort(key=lambda x: int(re.split('\.|\_',x)[-2]))
		self.lens = len(self.imagelist) -1 
		self.transform = transforms.Compose([transforms.ToTensor()])

	def __len__(self):

		return self.lens

	def __getitem__(self, i):
		"""
		idx must be between 0 to len-1
		assuming flow[0] contains flow in x direction and flow[1] contains flow in y
		"""
		image_path1 = self.imagelist[i]

		image_path2 = self.imagelist[i+1]
		backward_flow_path = self.flowlist[i]
		consistency_path = self.consistencylist[i]
		img1 = Image.open(os.path.join(self.image_path,image_path1))
		img2 = Image.open(os.path.join(self.image_path,image_path2))
		img1 = self.transform(img1)
		img2 = self.transform(img2)
		# mask is numpy and shape is HXWX3
		mask = read_weights_file(os.path.join(self.consistency_path,consistency_path))
		mask = torch.from_numpy(mask).permute(2,0,1).unsqueeze(0)
		flow = read(os.path.join(self.flow_path,backward_flow_path))
		originalflow=torch.from_numpy(flow) 
		# img_flow = Image.fromarray(np.uint8(flow))
		flow = torch.from_numpy(flow).to(device).permute(2,0,1).float()
		# this is for transpose.
		flow[0, :, :] *= float(flow.shape[1])/originalflow.shape[1]
		flow[1, :, :] *= float(flow.shape[2])/originalflow.shape[2]
		print('It now processes image {} !!!'.format(os.path.join(self.image_path,image_path1)))
		return (img1, img2, mask, flow)

if __name__ == "__main__":
	name  = 'windygrassnoaudi'
	video = VideoDataset(name)
	dataloader = torch.utils.data.DataLoader(video,batch_size=1)
	for index,item in enumerate(dataloader):
		print(index)
