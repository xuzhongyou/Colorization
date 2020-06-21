import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    #
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers,classification):
        super(RDN, self).__init__()
        self.G0 = num_features  #64
        self.G = growth_rate    #64
        self.D = num_blocks     #16
        self.C = num_layers     #8
        self.classification = classification
        # shallow feature extraction
        self.sfe1 = nn.Sequential(
            nn.Conv2d(num_channels,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            )
        self.sfe2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.sfe3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.sfe4 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(512))
            # referseg_conv4norm_rep
            # element wise
        # residual dense blocks       in_channel,grow_rate, numlayers
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        # 先上采样一下
        self.upscale_1 = nn.Sequential(
                        nn.ConvTranspose2d(512,256,3,2,1,1),
                        nn.ReLU(),
                        nn.Conv2d(256,256,3,1,1,1),
                        nn.ReLU(),
                        nn.Conv2d(256,256,3,1,1,1),
                        nn.ReLU())

        self.classconv = nn.Conv2d(256,256,3,1,1,1)
        self.reluclass = nn.ReLU()
        self.classmodel = nn.Conv2d(256,529,1,1)
        self.softmax = nn.Softmax(dim=1)

        self.upscale_2 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),
                        nn.ReLU(),
                        nn.Conv2d(128,128,3,1,1,1),
                        nn.ReLU(),
                        nn.Conv2d(128,128,3,1,1,1),
                        nn.ReLU())
        self.convfinal_1 = nn.Sequential(
                        nn.ConvTranspose2d(128,64,3,2,1,1),
                        nn.ReLU(),
                        nn.Conv2d(64,64,3,1,1,1),
                        nn.ReLU(),
                        nn.Conv2d(64,2,1,1),
                        nn.Tanh())


    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        sfe3 = self.sfe3(sfe2)
        sfe4 = self.sfe4(sfe3)
        x = sfe4
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe4  # global residual learning
        
        x = self.upscale_1(x)
        if self.classification:
            classconv = self.classconv(x)
            reluclass = self.reluclass(classconv)
            classmodel = self.classmodel(reluclass)
            x = self.upscale_2(x.detach())
            regression = self.convfinal_1(x)
        else:
            classconv = self.classconv(x.detach())
            reluclass = self.reluclass(classconv)
            classmodel = self.classmodel(reluclass)
            x = self.upscale_2(x)
            regression = self.convfinal_1(x)
        return classmodel,regression

if __name__ == "__main__":
    net = RDN(num_channels=1, num_features=512, growth_rate=512, num_blocks=4, num_layers=3,classification=False)
    input = torch.randn((1,1,176,176))
    print(net)
    classification,regression = net(input)
    print(classification.shape,regression.shape)