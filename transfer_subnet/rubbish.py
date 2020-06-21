import torch
import os
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision
import torch.nn as nn
import torch

# input_size 维度，隐藏层，层数
rnn = nn.LSTM(10, 20, 2)
# sequence batch,input_size
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, _= rnn(input)
# output, (hn, cn) = rnn(input)
print(len(_))


# writer = SummaryWriter('./logs/')
# path1 = '../data/video-picture/11_ponte_WS_pan_Videv/frame_0001.png'
# path2 = '../data/video-picture/11_ponte_WS_pan_Videv/frame_0002.png'
# image1 = Image.open(path1)
# image2 = Image.open(path2)
# image_transform = transforms.Compose([transforms.ToTensor()])
# tensor1 = image_transform(image1).unsqueeze(0)
# tensor2 = image_transform(image2).unsqueeze(0)
# tensors = torch.cat([tensor1,tensor2],0)
# print(tensor1.shape,tensor2.shape)
# img_grid1 = torchvision.utils.make_grid(tensors)
# img_grid2 = torchvision.utils.make_grid([tensor1.squeeze(0),tensor2.squeeze(0)])
# writer.add_image('four_fashion_mnist_images_0',img_grid2)
# writer.add_image('four_fashion_mnist_images_1', img_grid1)
# writer.add_scalar()
# writer.close()






# 1.set up


# 2.write images with image_grid


# 3 

















# a = np.array([[[1,2],[3,4]]])
# count = 0
# path = '/home/xzy/video/data/video-picture-flow/'
# for item in os.listdir(path):
#     if len(os.listdir(os.path.join(path,item,'forward_consistency')))==0:
#         count +=1
#         print(item)
# print(count)


# a = torch.Tensor([[[1,2],[3,4]]])
# print(a.shape)



# loss = torch.nn.MSELoss()
# a = torch.Tensor([[[1,2],[3,4]]])
# b = torch.Tensor([[[2,3],[3,5]]])
# result = loss(a,b)
# print(result)
# c = torch.sum(a)
# print(c.item())





# a = torch.Tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
# print(a.shape)
# b = a.reshape((2,3,2))
# print(b)
# c = a.permute((1,2,0))
# print(c)



# a = torch.Tensor([[[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],],
#                     [[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2],],
#                     [[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],]]])
# print(a.shape)
# print(a)
# b = a.reshape((1,4,3,4))
# print(b)
# print(b.shape)

# import numpy as np 

# a = np.array([1,2,3])
# b = np.dstack([a]*3)
# print(b.shape)
# print(b)
# c = np.reshape(b,(1,3,3))
# print(c.shape)
# print(c)


# a = np.array([[[[1,1,1,1],[1,1,1,1],[1,1,1,1]],[[2,2,2,2],[2,2,2,2],[2,2,2,2]]]])
# print(a.size,a.shape)
# print(a)
# b = np.array([[1,2],[3,4]])
# print(b.size,b.shape)
# print(b)
# c = np.dstack([b]*3)
# print(c.size,c.shape)
# print(c)

# import os
# in_path = '/home/xzy/video/data/video_picture_flow/'
# count = 0
# for item in os.listdir(in_path):
#     lens = len(os.listdir(os.path.join(in_path,item)))
#     if lens>2:
#         count +=1
# print(count)



# import cv2
# import numpy
# import os
# import torchvision.transforms as transforms
# from PIL import Image
# from torch.utils.data import DataLoader
# from utilities import *
# from dataset import MPIDataset
# from skimage import io, transform
# from torch.utils.data import DataLoader
# import torch


# def test_dataset():
#     path = '/home/xzy/video/WCT2/examples/data/MPI-Sintel-complete/'
#     dataloader = DataLoader(MPIDataset(path), batch_size=1)
#     for itr, (img1, img2, mask, flow) in enumerate(dataloader):
#         print(img1.shape,img2.shape,mask.shape,flow.shape)
#         if(itr==1):
#             break
#         warped=warp(img1,flow)
#         print(img1.squeeze().permute(1,2,0).size())
#         io.imsave("./examples/test_wrap/warped.png", warped.squeeze().permute(1,2,0).cpu().numpy())
#         io.imsave("./examples/test_wrap/img1.png", img1.squeeze().permute(1,2,0).cpu().numpy())
#         io.imsave("./examples/test_wrap/img2.png", img2.squeeze().permute(1,2,0).cpu().numpy())

# test_dataset()



# import datetime
# starttime = datetime.datetime.now()

# endtime = datetime.datetime.now()

# print((endtime - starttime))








# import os
# import sys 
# sys.path.append('./segmentation')
# from segmentation.dataset import round2nearest_multiple

# def make_gray_datasets(path_in,path_out):
#     # path_in = './images/style/'
#     # path_out = './images/gray_reverse/'
#     paths = os.listdir(path_in)
#     for filename in paths:
#         file_path = os.path.join(path_in,filename)
#         print('file path :',file_path)
#         img = cv2.imread(file_path,0)
#         if not os.path.exists(path_out):
#             os.makedirs(path_out)
#         cv2.imwrite(os.path.join(path_out,filename),img)

# make_gray_datasets('./examples/demo_content/','./examples/demo_gray/')

# a =[]
# def func(a):
#     a.append(1)
# func(a)
# print(a)
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--foo', action='store_true')
# parser.add_argument('--bar', action='store_false')
# # Namespace(foo=True, bar=False, baz=True
# config = parser.parse_args()
# print(config.foo)
# print(config.bar)

