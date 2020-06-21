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

	def __init__(self, name='../data/' ):
		"""
		looking at the "clean" subfolder for images, might change to "final" later
		root_dir -> path to the location where the "training" folder is kept inside the MPI folder
		"""
		self.root = name
		self.total_image = []
		self.total_flow = []
		self.total_consistency = []
		for item in os.listdir(os.path.join(self.root,'video-picture')):
			
			print('It is now processing {} !'.format(item))
			tmp_img_dir = os.path.join(self.root,'video-picture',item)
			tmp_flow_dir = os.path.join(self.root,'video-picture-flow',item,'backward_flow')
			tmp_consistency_dir = os.path.join(self.root,'video-picture-flow',item,'forward_consistency')

			tmp_image_list = os.listdir(os.path.join(tmp_img_dir))
			tmp_image_list.sort(key=lambda x: int(re.split('\.|\_',x)[1]))
			tmp_flow_list = os.listdir(os.path.join(tmp_flow_dir))
			tmp_flow_list.sort(key=lambda x: int(re.split('\.|\_',x)[1]))
			tmp_consistency_list = os.listdir(os.path.join(tmp_consistency_dir))
			tmp_consistency_list.sort(key=lambda x: int(re.split('\.|\_',x)[-2]))
			tmp_len = len(tmp_image_list)-1
			for i in range(tmp_len):
				tmp_1 = os.path.join(tmp_img_dir,tmp_image_list[i])
				tmp_2 = os.path.join(tmp_img_dir,tmp_image_list[i+1])
				tmp_image = (tmp_1,tmp_2)
				tmp_flow = os.path.join(tmp_flow_dir,tmp_flow_list[i])
				tmp_consistency = os.path.join(tmp_consistency_dir,tmp_consistency_list[i])
				self.total_image.append(tmp_image)
				self.total_flow.append(tmp_flow)
				self.total_consistency.append(tmp_consistency)
		self.lens = len(self.total_image)
		self.transform = transforms.Compose([transforms.ToTensor()])

	def __len__(self):
		return self.lens

	def __getitem__(self, i):
		"""
		idx must be between 0 to len-1
		assuming flow[0] contains flow in x direction and flow[1] contains flow in y
		"""
		image_path1 = self.total_image[i][0]
		image_path2 = self.total_image[i][1]
		backward_flow_path = self.total_flow[i]
		consistency_path = self.total_consistency[i]
		# print('image_path is ',image_path1)
		# print('backward_flow_path ',backward_flow_path)
		# print('consistency_path ',consistency_path)
		img1 = Image.open(image_path1)
		img2 = Image.open(image_path2)
		img1 = self.transform(img1)
		img2 = self.transform(img2)
		# mask is numpy and shape is HXWX3
		mask = read_weights_file(consistency_path)
		mask = torch.from_numpy(mask).permute(2,0,1)
		flow = read(backward_flow_path)
		originalflow=torch.from_numpy(flow) 
		# img_flow = Image.fromarray(np.uint8(flow))
		flow = torch.from_numpy(flow).permute(2,0,1).float()
		# this is for transpose.
		flow[0, :, :] *= float(flow.shape[1])/originalflow.shape[1]
		flow[1, :, :] *= float(flow.shape[2])/originalflow.shape[2]
		# [3,400,640] [3,400,640] [3,400,640],[ 2, 400, 640]
		return (img1, img2,flow,mask)

if __name__ == "__main__":
	video = VideoDataset()
	print('data lens',len(video))
	dataloader = torch.utils.data.DataLoader(video,batch_size=1)
	for index,item in enumerate(dataloader):
		print(index)
		print(item[0].shape)
		print(item[1].shape)
		print(item[2].shape)
		print(item[3].shape)



