import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import cv2

Mse_loss = torch.nn.MSELoss()

def warp(x, flo,mask):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo
    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)

    # mask -=1
    # mask = abs(mask)
    # mask = nn.functional.grid_sample(mask,vgrid)
    # output = output*mask
    return output
    
    # mask = torch.autograd.Variable(torch.ones(x.size()))
    # print(mask)
    # mask = nn.functional.grid_sample(mask, vgrid)
    # print(mask)
    # # if W==128:
    #     # np.save('mask.npy', mask.cpu().data.numpy())
    #     # np.save('warp.npy', output.cpu().data.numpy())
    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1
    # return output*mask

def readFlowFile(name):
    with open(name,'rb') as f:
        tag = np.fromfile(f, np.float32, 1).squeeze()
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    # flow shape like (436,1024,2)
    return flow 

def read_occlusion_file(path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    print(header)
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        vals[i-1] = np.array(list(map(np.float32, line)))
        vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
    # expand to 3 channels
    weights = np.dstack([vals.astype(np.float32)] * 3)
    print(weights.shape)
    return weights


def read_weights_file(path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        # rstrip 删除字符串末尾的字符，默认是空
        line = lines[i].rstrip().split(' ')
        vals[i-1] = np.array(list(map(np.float32, line)))
        vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
    # expand to 3 channels
    weights = np.dstack([vals.astype(np.float32)] * 3)
    return weights

# forward weight
def get_content_weights(forward_path):
    forward_weights = read_weights_file(forward_path)
    return forward_weights

def temporal_loss(x, x_w, c):
  c = torch.from_numpy(c).permute(2,0,1).unsqueeze(0)
  D = torch.sum(c).item()
  x_elements = x.shape[1]*x.shape[2]*x.shape[3]
  loss = (1. / D) * Mse_loss(c * x , c * x_w)
  return loss

def tv_loss(x):
    batch_size = x.shape[0]
    c_x = x.shape[1]
    h_x = x.shape[2]
    w_x = x.shape[3]
    count_h = x[:,:,1:,:].size(1)*x[:,:,1:,:].size(2)*x[:,:,1:,:].size(3)
    count_w = x[:,:,:,1:].size(1)*x[:,:,:,1:].size(2)*x[:,:,:,1:].size(3)
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return (h_tv/count_h+w_tv/count_w)/batch_size 

def test_warp():
    #PIL.Image/numpy.ndarray 数据进转化为torch.FloadTensor,并归一化到[0,1]
    img_transform = transforms.Compose([transforms.ToTensor()])
    img_name = './data/frame_0010.png'
    flow_name = './tmp/frame_0011.flo'
    occlusion_name = './data/occlusion/frame_0011.png'
    flow = readFlowFile(flow_name)
    flow = torch.from_numpy(flow).permute(2, 0, 1).float().unsqueeze(0)
    x = Image.open(img_name)
    x = img_transform(x).unsqueeze(0)
    mask = Image.open(occlusion_name)
    mask = img_transform(mask).unsqueeze(0)
    print("img shape: {}, flow shape: {}, mask shape: {},".format(x.shape,flow.shape,mask.shape))
    warped = warp(x,flow,mask)
    print(warped.shape)
    torchvision.utils.save_image(warped,'warp_result.png')



if __name__ == "__main__":
    # test_warp()

    # out_path = './back_flow_output/frame_01.flo'
    # flow = readFlowFile(out_path)
    # print(flow.shape)
    image_path1 = '../data/video-picture/160825_26_WindTurbines4_1080/frame_0001.png'
    image_path2 = '../data/video-picture/160825_26_WindTurbines4_1080/frame_0002.png'
    weight_path = '../data/video-picture-flow/160825_26_WindTurbines4_1080/forward_consistency/reliable_forward_1.txt'
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    image_transform = transforms.Compose([transforms.ToTensor()])
    image1 = image_transform(image1).unsqueeze(0)
    image2 = image_transform(image2).unsqueeze(0)
    weights = read_weights_file(weight_path)
    loss = temporal_loss(image1,image2,weights)
    print(loss)



