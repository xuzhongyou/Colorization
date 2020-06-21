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

class Encoder(nn.Module):
    def __init__(self, option_unpool):
        super(Encoder, self).__init__()
        self.option_unpool = option_unpool

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.apool1 = nn.AvgPool2d(2,2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.apool2 = nn.AvgPool2d(2,2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.apool3 = nn.AvgPool2d(2,2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            # print('level x shape',level,x.shape)
        return x,skips

    def encode(self, x, skips, level):

        # cat operation
        if level == 1:
            out = self.conv0(x)
            out = self.relu(self.conv1_1(self.pad(out)))
            return out

        elif level == 2:
            out = self.relu(self.conv1_2(self.pad(x)))
            skips['conv1_2'] = out  
            pool1 = self.apool1(out)    
            skips['pool1'] = pool1   
            out = self.relu(self.conv2_1(self.pad(pool1)))  
            return out

        elif level == 3:
            out = self.relu(self.conv2_2(self.pad(x)))
            skips['conv2_2'] = out
            pool2 = self.apool2(out)
            skips['pool2'] = pool2
            out = self.relu(self.conv3_1(self.pad(pool2)))
            return out

        else:
            out = self.relu(self.conv3_2(self.pad(x)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_4(self.pad(out)))
            skips['conv3_4'] = out
            pool3 = self.apool3(out)
            skips['pool3'] = pool3
            out = self.relu(self.conv4_1(self.pad(pool3)))
            return out



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
    
    record = True
    log_name = './logs'
    checkpoint_name = './checkpoints'
    log_dir = ''
    save_dirname = ''
    if record:
        log_dirname = os.path.join(log_name,time.strftime("%Y_%m_%d_%H_%M",time.localtime()))
        if not os.path.exists(log_dirname):
            os.makedirs(log_dirname)
        save_dirname =  os.path.join(checkpoint_name,time.strftime("%Y_%m_%d_%H_%M",time.localtime()))
        if not os.path.exists(save_dirname):
            os.makedirs(save_dirname)
        writer = SummaryWriter(log_dirname)
        
    root = '/home/xzy/project/coloration-xiaoke/src/colorization/dataset/coco'
    state_dict_name = 'checkpoints/2020_03_03_12_48_four_channel_sample125/colorization14.pth'
    continue_train = True
    phase = 'train'
    gpu_ids= '0,1'
    loss_fre = 100
    img_fre = 2000
    # lr = 0.0005
    beta1 = 0.9
    beta2 = 0.999
    epoches = 10
    lr = 0.000001
    batch_size = 40
    lambdaA = 1
    device_name = 'cuda:0' #'cpu'
    device = torch.device(device_name)
    L1oss = torch.nn.L1Loss()
    CEloss = torch.nn.CrossEntropyLoss()
    net = network.Colorization(4,False)
    if continue_train:
        net.load_state_dict(torch.load(state_dict_name))
    net.to(device)
    frozen_layer(27,net)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()) ,lr=lr,betas=(beta1,beta2))
    # schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    if len(gpu_ids)>1:
        net = nn.DataParallel(net,device_ids=[int(item) for item in gpu_ids.split(',')])  # list of int

    train_datasets = ImageFolder(os.path.join(root,phase),transform=transforms.Compose([
                                                   transforms.RandomChoice([transforms.Resize(opt.loadSize, interpolation=1),
                                                                            transforms.Resize(opt.loadSize, interpolation=2),
                                                                            transforms.Resize(opt.loadSize, interpolation=3),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=1),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=2),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=3)]),
                                                   transforms.RandomChoice([transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=3)]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()]))
    lens = len(train_datasets)
    print('train datasets is [{}] '.format(lens))
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=batch_size,shuffle=True)
    for epoch in range(epoches):
        for index , data in enumerate(dataloader):
            data = util.get_colorization_data(data,opt,p=opt.sample_p)
            if(data is None):
                continue
            input = torch.cat((data['A'],data['hint_B'],data['mask_B']),dim=1)
            input = input.to(device)
            outputclass,outputreg = net(input)
            realclass = util.encode_ab_ind(data['B'][:,:,::4,::4],opt).to(device)
            lossreg = L1oss(outputreg,data['B'].to(device))
            print(outputclass.dtype,realclass.dtype)
            lossclass = CEloss(outputclass.type(torch.cuda.FloatTensor),realclass[:,0,:,:].type(torch.cuda.LongTensor))
            
            if record:
                if index %loss_fre == 0: #100
                    writer.add_scalars('train/loss:',{'reg':lossreg.item()*10,
                                                    'class':lossclass.item()},epoch*lens+index*batch_size)
                if index % img_fre == 0 : # 2000
                    image_fake = util.lab2rgb(torch.cat([data['A'].type(torch.cuda.FloatTensor),outputreg.type(torch.cuda.FloatTensor)],dim=1),opt)
                    image_hint = util.lab2rgb(torch.cat([data['A'].type(torch.cuda.FloatTensor),data['hint_B'].type(torch.cuda.FloatTensor)],dim=1),opt)
                    image_real = util.lab2rgb(torch.cat([data['A'].type(torch.cuda.FloatTensor),data['B'].type(torch.cuda.FloatTensor)],dim=1),opt)
                    image_fake = image_fake.clamp_(0, 1)
                    image_hint = image_hint.clamp_(0, 1)
                    image_real = image_real.clamp_(0, 1)
                    writer.add_images('Image/train/2020._coco_epoch_lr_{}_{}'.format(lr,epoch),torch.cat([image_fake[0,:,:,:].unsqueeze(0),image_hint[0,:,:,:].unsqueeze(0),image_real[0,:,:,:].unsqueeze(0)],dim=0),index)
       
            print('epoch [{}/{}], images [{}/{}] loss is [reg: {:.5}/[class: {:.5}]],  '.
                            format(epoch+1,epoches,(index+1)*data['A'].shape[0],lens,lossreg.item()*10,lossclass.item()))

            loss = lambdaA * lossclass + 10*lossreg
            # check_module_param('conv4.4.weight',net)
            optim.zero_grad() 
            loss.backward()
            optim.step()
            # check_module_param('conv4.4.weight',net)
        if len(gpu_ids)>1:
            torch.save(net.module.state_dict(),os.path.join(save_dirname,'colorization{}.pth'.format(epoch)))
            
        else:
            torch.save(net.state_dict(),os.path.join(save_dirname,'colorization{}.pth'.format(epoch)))
    writer.close()





























