import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision
from torchvision.datasets import ImageFolder
# from torch.utils.tensorboard import SummaryWriter
# from video_dataset import VideoDataset
from torch.autograd import Variable
import os 
import os

class XiaoKeEncoder(nn.Module):
    def __init__(self, option_unpool):
        super(XiaoKeEncoder, self).__init__()
        self.option_unpool = option_unpool

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.apool1 = nn.AvgPool2d(2,2)
        # self.pool1 = WavePool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.apool2 = nn.AvgPool2d(2,2)
        # self.pool2 = WavePool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.apool3 = nn.AvgPool2d(2,2)
        # self.pool3 = WavePool(256)

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


class XiaoKeDecoder(nn.Module):
    def __init__(self, option_unpool):
        super(XiaoKeDecoder, self).__init__()
        self.option_unpool = option_unpool
        
        if option_unpool == 'sum':
            multiply_in = 1
        else:
            multiply_in = 3

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        # self.recon_block3 = WaveUnpool(256, option_unpool)
        self.recon_block3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 直接上采样完成，或者用转卷积来实现
        if option_unpool == 'sum':
            self.conv3_4 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        else:
            self.conv3_4_2 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        # self.recon_block2 = WaveUnpool(128, option_unpool)
        self.recon_block2 = nn.UpsamplingNearest2d(scale_factor=2)
        if option_unpool == 'sum':
            self.conv2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        else:
            self.conv2_2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        # self.recon_block1 = WaveUnpool(64, option_unpool)
        self.recon_block1 = nn.UpsamplingNearest2d(scale_factor=2)
        if option_unpool == 'sum':
            self.conv1_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        else:
            self.conv1_2_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
            # print('level shape is ',level,x.shape)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            pool3 = skips['pool3']
            original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            out = torch.cat([self.recon_block3(out),self.recon_block3(pool3),original],dim=1)

            _conv3_4 = self.conv3_4 if self.option_unpool == 'sum' else self.conv3_4_2
            out = self.relu(_conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            pool2 = skips['pool2']
            original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            out = torch.cat([self.recon_block2(out),self.recon_block2(pool2),original],dim=1)
            _conv2_2 = self.conv2_2 if self.option_unpool == 'sum' else self.conv2_2_2
            return self.relu(_conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            pool1 = skips['pool1']
            original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            out = torch.cat([self.recon_block1(out),self.recon_block1(pool1),original],dim=1)
            _conv1_2 = self.conv1_2 if self.option_unpool == 'sum' else self.conv1_2_2
            return self.relu(_conv1_2(self.pad(out)))
        else:
            return self.conv1_1(self.pad(x))


def warp(x, flo,mask=None):
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

def temporal_loss(x, x_w, c,Mse_loss):
#   c = torch.from_numpy(c).permute(2,0,1).unsqueeze(0)
  D = torch.sum(c).item()
  x_elements = x.shape[1]*x.shape[2]*x.shape[3]
  if D == 0:
      return 0
  loss = (1. / D) * Mse_loss(c * x , c * x_w)
  return loss

def tv_loss(x,batch_size=1):
    batch_size = x.shape[0]
    c_x = x.shape[1]
    h_x = x.shape[2]
    w_x = x.shape[3]
    count_h = x[:,:,1:,:].size(1)*x[:,:,1:,:].size(2)*x[:,:,1:,:].size(3)
    count_w = x[:,:,:,1:].size(1)*x[:,:,:,1:].size(2)*x[:,:,:,1:].size(3)
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    
    return (h_tv/count_h+w_tv/count_w)/batch_size 


if __name__ == "__main__":
    resume_train = True
    log_file = './12_30_log_1'
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    lr = 0.001
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    batch_size = 2

    encoder_model = XiaoKeEncoder('cat')
    decoder_model = XiaoKeDecoder('cat')
    writer = SummaryWriter(log_file)
    root = '../data/'
    en_path = './xiaoke_checkpoints/xiaoke_encoder.pth'
    # de_path = './xiaoke_checkpoints/xiaoke_decoder_89.pth'
    # gpu_ids= '0,1,2,3'
    beta1 = 0.9
    beta2 = 0.999
    recon_weight = 0.5
    feature_weight = 0.1
    tv_weight = 0.1
    consistency_weight = 0.3
    epoches = 40

    device = torch.device('cuda:0')
    pretrained_state_dict = {k:v for k,v in torch.load(en_path).items() if k in encoder_model.state_dict() }
    state_dict = encoder_model.state_dict()
    state_dict.update(pretrained_state_dict)
    encoder_model.load_state_dict(state_dict)
    encoder_model.to(device)
    decoder_model.to(device)
    if resume_train:
        de_path = 'xiaoke_video_checkpoints/xiaoke_latest_0.001_log_lr1.pth'
        decoder_model.load_state_dict(torch.load(de_path))
        decoder_model.to(device)

    for param in encoder_model.parameters():
        param.requires_grad = False
    optim = torch.optim.Adam(decoder_model.parameters(),lr=lr,betas=(beta1,beta2))
    recon_loss = torch.nn.MSELoss()
    feature_loss = torch.nn.MSELoss()
    consistency_loss = torch.nn.MSELoss()
    transforms = transform.Compose([transform.ToTensor()])
    train_datasets = VideoDataset(root)
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=batch_size,shuffle=False)
    lens = len(train_datasets)

    runloss = 0.
    for epoch in range(epoches):
        if epoch ==10:
            lr = 0.0001
        for index , data in enumerate(dataloader):
            image_1 = data[0].to(device)
            image_2 = data[1].to(device)
            warp_image_2 = warp(image_1,data[2].to(device))
            feature1,skips1 = encoder_model(image_1)
            feature2,skips2 = encoder_model(image_2)
            recon_image1 = decoder_model(feature1,skips1)
            recon_image2 = decoder_model(feature2,skips2)
            
            re_loss1 = recon_loss(image_1,recon_image1)
            re_loss2 = recon_loss(image_2,recon_image2) 
 
            feature_recon1,_ = encoder_model(recon_image1)
            feature_recon2,_ = encoder_model(recon_image2)

            fe_loss1 = feature_loss(feature_recon1,feature1.detach())
            fe_loss2 = feature_loss(feature_recon2,feature2.detach())

            tv_loss1 = tv_loss(recon_image1)
            tv_loss2 = tv_loss(recon_image2)

            con_loss = temporal_loss(image_2,warp_image_2,data[3].to(device),consistency_loss)
            re_loss = re_loss1+re_loss1
            fe_loss = fe_loss1+fe_loss2
            tv_losses = tv_loss1+tv_loss2
            loss = recon_weight * re_loss + feature_weight * fe_loss + tv_weight * tv_losses + consistency_weight * con_loss
            # loss = fe_loss *0.5 + tv_losses *0.2 + con_loss*0.3
            runloss += loss.item()
            if index % 300 == 0 :
                writer.add_scalar('Loss/train/2019.12.25_epoch [{}/{}]_lr_{}_index_{}'.format(epoch+1,epoches,lr,index),loss,epoch*lens+index)
                writer.add_images('Image/train/2019.12.25_epoch [{}/{}]_lr_{}_index_{}'.format(epoch+1,epoches,lr,index),
                torch.cat([image_1,recon_image1,image_2,recon_image2],dim=0),index)
                print('epoch [{}/{}], images [{}/{}] loss is [{:.5}/{:.5}/{:.5}/{:.5}], total loss is  {:.5} lr: {}'.
                            format(epoch+1,epoches,(index+1)*data[0].shape[0],lens,re_loss.item(),fe_loss.item(),tv_losses.item(),con_loss,loss.item(),lr))
                runloss = 0.
            if index % 10000 ==0:
                torch.save(decoder_model.state_dict(),'./xiaoke_video_checkpoints/xiaoke_{}_{}_{}.pth'.format(index,lr,log_file[2:]))
            optim.zero_grad() 
            loss.backward()
            optim.step()
            torch.save(decoder_model.state_dict(),'./xiaoke_video_checkpoints/xiaoke_decoder_{}_{}.pth'.format(lr,epoch))
            if epoch % 1==0:
                torch.save(decoder_model.state_dict(),'./xiaoke_video_checkpoints/xiaoke_latest_{}_{}.pth'.format(lr,log_file[2:]))
        if len(gpu_ids)>1:
            torch.save(decoder_model.module.state_dict(),'./xiaoke_checkpoints/xiaoke_decoder_{}mulg.pth'.format(epoch))
        else:
            torch.save(decoder_model.state_dict(),'./xiaoke_checkpoints/xiaoke_decoder_{}.pth'.format(epoch))
    writer.close()



























    # encoder = XiaoKeEncoder(False)
    # x = torch.randn([1,3,224,224])
    # skip = {}
    # for level in [1,2,3,4]:
    #     x = encoder.encode(x,skip,level)
    #     print('level x shape ',level,x.shape)
    # decoder = XiaoKeDecoder(False)
    # x = torch.randn([1,512,28,28])
    # for level in [4,3,2,1]:
    #     x = decoder.decode(x,skip,level)
    #     print('level x shape ',x.shape) 

