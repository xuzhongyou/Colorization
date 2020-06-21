import torch
import torch.nn as nn


class Colorization(nn.Module):
    def __init__(self,in_channels,classification):
        super(Colorization,self).__init__()
# *****************
# ***** conv1 *****
# *****************
        self.classification = classification
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,64,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
            )
        # self.conv1short = nn.Sequential(nn.Conv2d(64,128,3,1,1),nn.ReLU())

# *****************
# ***** conv2 *****
# *****************
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        # self.conv2short = nn.Sequential(nn.Conv2d(128,256,3,1,1),nn.ReLU())

# *****************
# ***** conv3 *****
# *****************
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        # self.conv3short = nn.Sequential(nn.Conv2d(256,512,3,1,1),nn.ReLU())
        
# *****************
# ***** conv4 *****
# *****************
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
            # referseg_conv4norm_rep
            # element wise
        )
# *****************
# ***** conv5 *****
# *****************
        self.conv1short = nn.Sequential(nn.Conv2d(64,128,3,1,1),nn.ReLU())
        self.conv2short = nn.Sequential(nn.Conv2d(128,256,3,1,1),nn.ReLU())
        self.conv3short = nn.Sequential(nn.Conv2d(256,512,3,1,1),nn.ReLU())


        self.conv5_1 = nn.Conv2d(512,512,3,1,2,2)
        self.relu5_1 = nn.ReLU()
        self.rdb5_1 = nn.Conv2d(512,512,3,1,1)
        self.rdb5_1relu = nn.ReLU()
        # concat conv5_1 rdb5_1_relu
        self.rdb5_2 = nn.Conv2d(1024,512,3,1,1)
        self.rdb5_2relu = nn.ReLU()
        # concat conv5_1 rdb5_1_relu rdb5_2_relu
        self.rdb5_3 = nn.Conv2d(1536,512,3,1,1)
        self.rdb5_3relu = nn.ReLU()
        # concat conv5_1 rdb5_1_relu rdb5_2_relu, rdb5_3_relu
        self.rdb5_4 = nn.Conv2d(2048,512,1,1)
        # elementwise 相加
        self.conv5_2 = nn.Conv2d(512,512,3,1,2,2)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512,512,3,1,2,2)
        self.relu5_3 = nn.ReLU()
        self.batch5_3 = nn.BatchNorm2d(512)
        # 在这地方加入reference的信息
        # 进行elementwise sum

# *****************
# ***** conv6 *****
# *****************
        self.conv6_1 = nn.Conv2d(512,512,3,1,2,2)
        self.relu6_1 = nn.ReLU()
        self.rdb6_1 = nn.Conv2d(512,512,3,1,1)
        self.rdb6_1relu = nn.ReLU()
        # concat conv6_1 rdb6_1_relu
        self.rdb6_2 = nn.Conv2d(1024,512,3,1,1)
        self.rdb6_2relu = nn.ReLU()
        # concat conv6_1 rdb6_1_relu rdb6_2_relu
        self.rdb6_3 = nn.Conv2d(1536,512,3,1,1)
        self.rdb6_3relu = nn.ReLU()
        # concat conv6_1 rdb6_1_relu rdb6_2_relu, rdb6_3_relu
        self.rdb6_4 = nn.Conv2d(2048,512,1,1)
        # elementwise 相加 conv6_1 , rdb6_4
        self.conv6_2 = nn.Conv2d(512,512,3,1,2,2)
        self.relu6_2 = nn.ReLU()
        self.conv6_3 = nn.Conv2d(512,512,3,1,2,2)
        self.relu6_3 = nn.ReLU()
        self.batch6_3 = nn.BatchNorm2d(512)
# *****************
# ***** conv7 *****
# *****************
        self.conv7_1 = nn.Conv2d(512,512,3,1,1,1)
        self.relu7_1 = nn.ReLU()
        self.rdb7_1 = nn.Conv2d(512,512,3,1,1)
        self.rdb7_1relu = nn.ReLU()
        # concat conv7_1 rdb7_1_relu
        self.rdb7_2 = nn.Conv2d(1024,512,3,1,1)
        self.rdb7_2relu = nn.ReLU()
        # concat conv7_1 rdb7_1_relu rdb7_2_relu
        self.rdb7_3 = nn.Conv2d(1536,512,3,1,1)
        self.rdb7_3relu = nn.ReLU()
        # concat conv7_1 rdb7_1_relu rdb7_2_relu, rdb7_3_relu
        self.rdb7_4 = nn.Conv2d(2048,512,1,1)
        # elementwise 相加 conv7_1 , rdb7_4
        self.conv7_2 = nn.Conv2d(512,512,3,1,1,1)
        self.relu7_2 = nn.ReLU()
        self.conv7_3 = nn.Conv2d(512,512,3,1,1,1)
        self.relu7_3 = nn.ReLU()
        self.batch7_3 = nn.BatchNorm2d(512)
# *****************
# ***** conv8 *****
# *****************
        # 这边代码给的是4
        self.conv8_1 = nn.ConvTranspose2d(512,256,3,2,1,1)
        self.relu8_1 = nn.ReLU()
        self.conv8_2 = nn.Conv2d(256,256,3,1,1,1)
        self.relu8_2 = nn.ReLU()
        self.conv8_3 = nn.Conv2d(256,256,3,1,1,1)
        self.relu8_3 = nn.ReLU()
# *****************
# ***** class *****
# *****************
        self.classconv = nn.Conv2d(256,256,3,1,1,1)
        self.reluclass = nn.ReLU()
        self.classmodel = nn.Conv2d(256,529,1,1)
        self.softmax = nn.Softmax(dim=1)
# *****************
# ***** conv9 *****
# *****************
        self.conv9_1 = nn.ConvTranspose2d(256,128,3,2,1,1)
        self.relu9_1 = nn.ReLU()
        self.conv9_2 = nn.Conv2d(128,128,3,1,1,1)
        self.relu9_2 = nn.ReLU()
        self.conv9_3 = nn.Conv2d(128,128,3,1,1,1)
        self.relu9_3 = nn.ReLU()
# # *****************
# # ***** conv10 *****
# # *****************
#         self.conv10_1 = nn.ConvTranspose2d(128,64,3,2,1,1)
#         self.relu10_1 = nn.ReLU()
#         self.conv10_2 = nn.Conv2d(64,64,3,1,1,1)
#         self.relu10_2 = nn.ReLU()
#         self.conv10_3 = nn.Conv2d(64,64,3,1,1,1)
#         self.relu10_3 = nn.ReLU()

# *****************
# ***** convfinal *****
# *****************
        self.convfinal_1 = nn.ConvTranspose2d(128,64,3,2,1,1)
        self.relufinal_1 = nn.ReLU()
        self.convfinal_2 = nn.Conv2d(64,64,3,1,1,1)
        self.relufinal_2 = nn.ReLU()
        self.convfinal_3 = nn.Conv2d(64,2,1,1)
        self.tanh = nn.Tanh()

#     def get_current_losses():
#         self.error_cnt += 1
#         errors_ret = OrderedDict()
#         for name in self.loss_names:
#             if isinstance(name, str):
#                 # float(...) works for both scalar tensor and float number avg_loss_alpha=0.986
#                 self.avg_losses[name] = float(getattr(self, 'loss_' + name)) + self.avg_loss_alpha * self.avg_losses[name]
#                 errors_ret[name] = (1 - self.avg_loss_alpha) / (1 - self.avg_loss_alpha**self.error_cnt) * self.avg_losses[name]
#         return errors_ret

    def forward(self,input):
        flag = False
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv1short = self.conv1short(conv1)
        conv2short = self.conv2short(conv2)
        conv3short = self.conv3short(conv3)
        if flag :
                print('conv1 ',conv1.shape) # size 128,feature 64
                print('conv2 ',conv2.shape) # 64
                print('conv3 ',conv3.shape) # 32
                print('conv4 ',conv4.shape)
                print('conv1 short ',conv1short.shape) #size 128, feature 128
                print('conv2 short ',conv2short.shape)
                print('conv3 short ',conv3short.shape)

# conv5_1
        conv5_1 = self.conv5_1(conv4)
        relu5_1 = self.relu5_1(conv5_1)

        rdb5_1 = self.rdb5_1(relu5_1)
        rdb5_1relu = self.rdb5_1relu(rdb5_1)
        rdb5_2 = self.rdb5_2(torch.cat([relu5_1,rdb5_1relu],dim=1))
        rdb5_2relu = self.rdb5_2relu(rdb5_2)
        rdb5_3 = self.rdb5_3(torch.cat([relu5_1,rdb5_1relu,rdb5_2relu],dim=1))
        rdb5_3relu = self.rdb5_3relu(rdb5_3)
        rdb5_4 = self.rdb5_4(torch.cat([relu5_1,rdb5_1relu,rdb5_2relu,rdb5_3relu],dim=1))
        conv5_1_sum = relu5_1 + rdb5_4
        conv5_2 = self.conv5_2(conv5_1_sum)
        relu5_2 = self.relu5_2(conv5_2)
        conv5_3 = self.conv5_3(relu5_2)
        relu5_3 = self.relu5_3(conv5_3)
        batch5_3 = self.batch5_3(relu5_3)
# conv6_1
        conv6_1 = self.conv6_1(batch5_3)
        relu6_1 = self.relu6_1(conv6_1)
        rdb6_1 = self.rdb6_1(relu6_1)
        rdb6_1relu = self.rdb6_1relu(rdb6_1)
        rdb6_2 = self.rdb6_2(torch.cat([relu6_1,rdb6_1relu],dim=1))
        rdb6_2relu = self.rdb6_2relu(rdb6_2)
        rdb6_3 = self.rdb6_3(torch.cat([relu6_1,rdb6_1relu,rdb6_2relu],dim=1))
        rdb6_3relu = self.rdb6_3relu(rdb6_3)
        rdb6_4 = self.rdb6_4(torch.cat([relu6_1,rdb6_1relu,rdb6_2relu,rdb6_3relu],dim=1))
        conv6_1_sum = relu6_1 + rdb6_4
        conv6_2 = self.conv6_2(conv6_1_sum)
        relu6_2 = self.relu6_2(conv6_2)
        conv6_3 = self.conv6_3(relu6_2)
        relu6_3 = self.relu6_3(conv6_3)
        batch6_3 = self.batch6_3(relu6_3)
# conv7_1
        conv7_1 = self.conv7_1(batch6_3)
        relu7_1 = self.relu7_1(conv7_1)
        rdb7_1 = self.rdb7_1(relu7_1)
        rdb7_1relu = self.rdb7_1relu(rdb7_1)
        rdb7_2 = self.rdb7_2(torch.cat([relu7_1,rdb7_1relu],dim=1))
        rdb7_2relu = self.rdb7_2relu(rdb7_2)
        rdb7_3 = self.rdb7_3(torch.cat([relu7_1,rdb7_1relu,rdb7_2relu],dim=1))
        rdb7_3relu = self.rdb7_3relu(rdb7_3)
        rdb7_4 = self.rdb7_4(torch.cat([relu7_1,rdb7_1relu,rdb7_2relu,rdb7_3relu],dim=1))
        conv7_1_sum = relu7_1 + rdb7_4
        conv7_2 = self.conv7_2(conv7_1_sum)
        relu7_2 = self.relu7_2(conv7_2)
        conv7_3 = self.conv7_3(relu7_2)
        relu7_3 = self.relu7_3(conv7_3)
        batch7_3 = self.batch7_3(relu7_3)
# conv8
        batch7_3 = batch7_3 + conv3short
        conv8_1 = self.conv8_1(batch7_3)
        relu8_1 = self.relu8_1(conv8_1)
        conv8_2 = self.conv8_2(relu8_1)
        relu8_2 = self.relu8_2(conv8_2)
        conv8_3 = self.conv8_3(relu8_2)
        relu8_3 = self.relu8_3(conv8_3)
        # print(conv8_1.shape,conv8_2.shape,conv8_3.shape)
        # [1,256,64,64]

# class
        if self.classification:
                classconv = self.classconv(relu8_3)
                reluclass = self.reluclass(classconv)
                classmodel = self.classmodel(reluclass)
                # softmax = self.softmax(classmodel)

        # conv9
                # relu8_3 = relu8_3.detach() + conv2short
                conv9_1 = self.conv9_1(relu8_3.detach() + conv2short.detach())
                relu9_1 = self.relu9_1(conv9_1)
                conv9_2 = self.conv9_2(relu9_1)
                relu9_2 = self.relu9_2(conv9_2)
                conv9_3 = self.conv9_3(relu9_2)
                relu9_3 = self.relu9_3(conv9_3)

        # final 
                # relu9_3 = relu9_3 + conv1short
                convfinal_1 = self.convfinal_1(relu9_3 + conv1short.detach())
                relufinal_1 = self.relufinal_1(convfinal_1)
                convfinal_2 = self.convfinal_2(relufinal_1)
                relufinal_2 = self.relufinal_2(convfinal_2)
                convfinal_3 = self.convfinal_3(relufinal_2)
                tanhfinal_3 = self.tanh(convfinal_3)
        else:
                classconv = self.classconv(relu8_3.detach())
                reluclass = self.reluclass(classconv)
                classmodel = self.classmodel(reluclass)
                # softmax = self.softmax(classmodel)

        # conv9
                # relu8_3 = relu8_3.detach() + conv2short
                conv9_1 = self.conv9_1(relu8_3 + conv2short)
                relu9_1 = self.relu9_1(conv9_1)
                conv9_2 = self.conv9_2(relu9_1)
                relu9_2 = self.relu9_2(conv9_2)
                conv9_3 = self.conv9_3(relu9_2)
                relu9_3 = self.relu9_3(conv9_3)

        # final 
                # relu9_3 = relu9_3 + conv1short
                convfinal_1 = self.convfinal_1(relu9_3 + conv1short)
                relufinal_1 = self.relufinal_1(convfinal_1)
                convfinal_2 = self.convfinal_2(relufinal_1)
                relufinal_2 = self.relufinal_2(convfinal_2)
                convfinal_3 = self.convfinal_3(relufinal_2)
                tanhfinal_3 = self.tanh(convfinal_3)
        return classmodel,tanhfinal_3
     
if __name__ == "__main__":
    net = Colorization(4,False)
    input = torch.randn((1,4,176,176))
    outputclass,outputreg = net(input)
    print(outputclass.shape,outputreg.shape)