import cv2
import numpy
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch
import re 
# from dataset import MPIDataset

def read_video():
    # read video
    path = './video.mp4'
    cap = cv2.VideoCapture(path)
    print(cap.isOpened())
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cv2.waitKey(30)!=ord('q'):
        retval,image = cap.read()
        print(image.shape)
        # cv2.imshow('video~',image)
    cap.release()


def save_video(input_path,out_path,w,h):

    frames = os.listdir(input_path)
    frames.sort(key = lambda x: int(re.split('\_',x)[1]))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('./output.avi',fourcc, 10.0, (640,480))
    print('output is ',out_path)
    out = cv2.VideoWriter(out_path,fourcc, 10.0, (w,h))

    for item in frames:
        temp = cv2.imread(os.path.join(input_path,item))
        print(temp.shape)
        print('Now, it process frame  ',item)
        out.write(temp)
    out.release()

def get_imgSize():
    path='./stylization_temp_fangao/frame_0001_cat5_decoder_encoder_skip.png'
    img = cv2.imread(path)
    print(img.shape)

def transform_size():
    cw, ch =  640,360                  
    path='/home/xzy/video/WCT2/examples/data/MPI-Sintel-complete/training/clean/temple_2/frame_0001.png'
    img = Image.open(path)
    s_transforms = transforms.Compose([transforms.Resize((360,640), interpolation=Image.NEAREST),transforms.CenterCrop((ch // 16 * 16, cw // 16 * 16)),transforms.ToTensor()])
    c_transforms = transforms.Compose([transforms.Resize((360,640)),transforms.ToTensor()])
    img1 = c_transforms(img)
    img2 = s_transforms(img)
    print(img1.shape)
    print(img2.shape)


def test_dataset():
    path = '/home/xzy/video/WCT2/examples/data/MPI-Sintel-complete'
    dataloader = DataLoader(MPIDataset(path), batch_size=1)
    for itr, (img1, img2, mask, flow) in enumerate(dataloader):
        print(img1.shape,img2.shape,mask.image,flow.shape)
        if(itr==1):
            break


if __name__ == "__main__":
    # 这边w,h一定要正确
    w,h = 640,400 
    input_path = './examples/160825_26_WindTurbines4_1080_adain/'
    out_path = './fangao_160825_26_WindTurbines4_1080_1024_adin.avi'
    save_video(input_path,out_path,w,h)

    # transform_size()
    # test_dataset()
