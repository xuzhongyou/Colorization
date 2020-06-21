import os 
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from util import util
from options.train_options import TrainOptions
from model import network
from model import siggraph
import time


def frozen_layer(layers,net):
    for i,(name, param) in enumerate(net.named_parameters()):
        if i <layers:
            param.requires_grad = False
    return net

def check_module_param(target_name,net):
    for i,(name, param) in enumerate(net.named_parameters()):
        if target_name in name:
            print('name is {} and the param {}'.format(name,param))

def check_module_require_grad(net):
    for i,(name, param) in enumerate(net.named_parameters()):
        print(i,name,param.requires_grad)

if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.sample_p = .125
    record = True
    log_name = './test_logs'
    checkpoint_name = './checkpoints'
    log_dir = ''
    log_dirname = os.path.join(log_name,time.strftime("%Y_%m_%d_%H_%M",time.localtime()))
    if record :
        if not os.path.exists(log_dirname):
            os.makedirs(log_dirname)
        writer = SummaryWriter(log_dirname)
    root = '/home/xzy/project/coloration-xiaoke/src/colorization/dataset/coco'
    root = './'

    state_dict_name = 'checkpoints/2020_03_04_12_27_four_channel_sample125_new/colorization9.pth'

    continue_train = True
    phase = 'FastPhotoStyle'
    gpu_ids= '1'
    device_name = 'cuda:1' #'cpu'
    batch_size = 1
    loss_fre = 1
    img_fre = 1
    device = torch.device(device_name)
    L1oss = torch.nn.L1Loss()
    CEloss = torch.nn.CrossEntropyLoss()
    net = network.Colorization(4,True)
    # net = siggraph.SIGGRAPHGenerator(4,2)
    net.load_state_dict(torch.load(state_dict_name))
    net.eval()
    net.to(device)
    if len(gpu_ids)>1:
        net = nn.DataParallel(net,device_ids=[int(item) for item in gpu_ids.split(',')])  # list of int

    train_datasets = ImageFolder(os.path.join(root,phase),transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    lens = len(train_datasets)
    print('test datasets is [{}] '.format(lens))
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=batch_size,shuffle=False)
    for index , data in enumerate(dataloader):
        data[0] = util.crop_mult(data[0], mult=8)
        data = util.get_colorization_data(data,opt,ab_thresh=0,p=opt.sample_p)
        if(data is None):
            continue
        input = torch.cat((data['A'],data['hint_B'],data['mask_B']),dim=1)
        input = input.to(device)
        outputclass,outputreg = net(input)
        realclass = util.encode_ab_ind(data['B'][:,:,::4,::4],opt).to(device)
        lossreg = L1oss(outputreg,data['B'].to(device))
        lossclass = CEloss(outputclass.type(torch.cuda.FloatTensor),realclass[:,0,:,:].type(torch.cuda.LongTensor))
                        
        image_fake = util.lab2rgb(torch.cat([data['A'].type(torch.cuda.FloatTensor),outputreg.type(torch.cuda.FloatTensor)],dim=1),opt)
        print('  images [{}/{}] loss is [reg: {:.5}/[class: {:.5}]],  '.
                format((index+1)*data['A'].shape[0],lens,lossreg.item()*10,lossclass.item()))
        # torchvision.utils.save_image(image_fake,'hemin.png')
        if index %loss_fre == 0:
            writer.add_scalars('train/loss:',{'reg':lossreg.item()*10,
                                            'class':lossclass.item()},index*batch_size)
        if index % img_fre == 0 :
            image_fake = util.lab2rgb(torch.cat([data['A'].to('cpu'),outputreg.to('cpu')],dim=1),opt)
            image_real = util.lab2rgb(torch.cat([data['A'].to('cpu'),data['B'].to('cpu')],dim=1),opt)
            # image_fake = util.lab2rgb(torch.cat([data['A'].type(torch.cuda.FloatTensor),outputreg.type(torch.cuda.FloatTensor)],dim=1),opt)
            # image_real = util.lab2rgb(torch.cat([data['A'].type(torch.cuda.FloatTensor),data['B'].type(torch.cuda.FloatTensor)],dim=1),opt)
            image_fake = image_fake.clamp_(0, 1)
            image_real = image_real.clamp_(0, 1)
           # torchvision.utils.save_image(image_fake,'./test_result/{}.png'.format(index))
            writer.add_images('Image/test/2020._coco_index_{}'.format(index),torch.cat([image_fake,image_real],dim=0),index)
    writer.close()





























