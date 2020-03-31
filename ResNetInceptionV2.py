import torch
import torch.nn as nn
import pdb


# OUT = (INPUT - Kernel_size + 2 * padding) / stride + 1


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, with_bn=False):
        super(BasicConv2d, self).__init__()
        self.with_bn = with_bn
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        if with_bn:
            self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        nn.init.kaiming_uniform_(self.conv.weight)


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.stemconv1 = BasicConv2d(in_planes=3, out_planes=32, kernel_size=3, stride=2, padding=0)  # 149*149*32
        self.stemconv2 = BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=0)  # 147*147*32
        self.stemconv3 = BasicConv2d(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1)  # 147*147*64
        self.stemmaxpool41 = nn.MaxPool2d(kernel_size=3, stride=2)  # 73*73*64
        self.stemconv42 = BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=2, padding=0)  # 73*73*96
        # concat stemmaxpool41 and stemconv42
        self.stemconv51 = BasicConv2d(in_planes=160, out_planes=64, kernel_size=1, stride=1, padding=0)  # 73*73*64
        self.stemconv61 = BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=1, padding=0)  # 71*71*96
        self.stemconv52 = BasicConv2d(in_planes=160, out_planes=64, kernel_size=1, stride=1, padding=0)  # 73*73*64
        self.stemconv62 = BasicConv2d(in_planes=64, out_planes=64, kernel_size=(7, 1), stride=1,
                                    padding=(3, 0))  # 73*73*64
        self.stemconv72 = BasicConv2d(in_planes=64, out_planes=64, kernel_size=(1, 7), stride=1,
                                    padding=(0, 3))  # 73*73*64
        self.stemconv82 = BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=1, padding=0)  # 71*71*96
        # concat stemconv61 and stemconv82
        self.stemconv91 = BasicConv2d(in_planes=192, out_planes=192, kernel_size=3, stride=2, padding=0)  # 35*35*192
        self.stemmaxpool92 = nn.MaxPool2d(kernel_size=2, stride=2)  # 35*35*192
        # concat stemconv91 and stemmaxpool92
        # outsize 35*35*384

    def forward(self, x):
        x = self.stemconv1(x)
        x = self.stemconv2(x)
        x = self.stemconv3(x)
        x41 = self.stemmaxpool41(x)
        x42 = self.stemconv42(x)
        x = torch.cat((x41, x42), 1)
        x51 = self.stemconv51(x)
        x61 = self.stemconv61(x51)
        x52 = self.stemconv52(x)
        x62 = self.stemconv62(x52)
        x72 = self.stemconv72(x62)
        x82 = self.stemconv82(x72)
        x = torch.cat((x61, x82), 1)
        x91 = self.stemconv91(x)
        x92 = self.stemmaxpool92(x)
        x = torch.cat((x91, x92), 1)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1, padding=0)  # 35*35*32
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1, padding=0),  # 35*35*32
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)  # 35*35*32
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1, padding=0),  # 35*35*32
            BasicConv2d(in_planes=32, out_planes=48, kernel_size=3, stride=1, padding=1),  # 35*35*48
            BasicConv2d(in_planes=48, out_planes=64, kernel_size=3, stride=1, padding=1)  # 35*35*64
        )
        self.conv2d = nn.Conv2d(in_channels=128, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.maxpool11 = nn.MaxPool2d(kernel_size=3, stride=2)  # 17*17*384
        self.conv12 = BasicConv2d(in_planes=384, out_planes=384, kernel_size=3, stride=2, padding=0)  # 17*17*384
        self.conv13 = BasicConv2d(in_planes=384, out_planes=256, kernel_size=1, stride=1, padding=0)  # 35*35*256
        self.conv23 = BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=1, padding=1)  # 35*35*256
        self.conv33 = BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2, padding=0)  # 17*17*384
    # out 17*17*1152

    def forward(self, x):
        xa = self.maxpool11(x)
        xb = self.conv12(x)
        xc = self.conv13(x)
        xc = self.conv23(xc)
        xc = self.conv33(xc)
        out = torch.cat((xa, xb, xc), 1)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(in_planes=1152, out_planes=192, kernel_size=1, stride=1, padding=0)  # 17*17*192
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=1152, out_planes=128, kernel_size=1, stride=1, padding=0),  # 17*17*128
            BasicConv2d(in_planes=128, out_planes=160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(in_planes=160, out_planes=192, kernel_size=(7, 1), stride=1, padding=(3, 0))  # 17*17*192
        )
        self.conv = nn.Conv2d(in_channels=384, out_channels=1152, kernel_size=1, stride=1, padding=0)  # 17*17*1152
        self.relu = nn.ReLU(inplace=False)
        # out 17*17*1152

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        xx = torch.cat((x0, x1), 1)
        xx = self.conv(xx)
        out = xx * self.scale + x
        out = self.relu(out)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=1152, out_planes=256, kernel_size=1, stride=1, padding=0),  # 17*17*256
            BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2, padding=0)  # 8*8*384
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes=1152, out_planes=256, kernel_size=1, stride=1, padding=0),  # 17*17*256
            BasicConv2d(in_planes=256, out_planes=288, kernel_size=3, stride=2, padding=0)  # 8*8*288
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes=1152, out_planes=256, kernel_size=1, stride=1, padding=0),  # 17*17*256
            BasicConv2d(in_planes=256, out_planes=288, kernel_size=3, stride=1, padding=1),  # 17*17*288
            BasicConv2d(in_planes=288, out_planes=320, kernel_size=3, stride=2, padding=0)  # 8*8*320
        )
        # out 8*8*2144

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(in_planes=2144, out_planes=192, kernel_size=1, stride=1, padding=0)  # 8*8*192
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes=2144, out_planes=192, kernel_size=1, stride=1, padding=0),  # 8*8*192
            BasicConv2d(in_planes=192, out_planes=224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(in_planes=224, out_planes=256, kernel_size=(3, 1), stride=1, padding=(1, 0))  # 8*8*256
        )
        self.conv = nn.Conv2d(in_channels=448, out_channels=2144, kernel_size=1, stride=1, padding=0)  # 17*17*1152
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)
        # out 8*8*2144

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        xx = torch.cat((x0, x1), 1)
        xx = self.conv(xx)
        out = xx * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)


class ResNetInceptionV2(nn.Module):
    def __init__(self, num_class):
        super(ResNetInceptionV2, self).__init__()
        self.stem = Stem()
        self.repeat = nn.Sequential(
            Block35(scale=0.10),
            Block35(scale=0.10),
            Block35(scale=0.10),
            Block35(scale=0.10),
            Block35(scale=0.10)
        )
        self.reductionA = ReductionA()
        self.repeat1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.reductionB = ReductionB()
        self.repeat2 = nn.Sequential(
            Block8(scale=0.10),
            Block8(scale=0.10),
            Block8(scale=0.10),
            Block8(scale=0.10),
            Block8(scale=0.10)
        )
        # self.block8 = Block8(noReLU=True)
        self.AvgPool = nn.AvgPool2d(kernel_size=8, count_include_pad=False)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features=2144, out_features=num_class)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.repeat(x)
        x = self.reductionA(x)
        x = self.repeat1(x)
        x = self.reductionB(x)
        x = self.repeat2(x)
        # x = self.block8(x)
        x = self.AvgPool(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)  # Flat
        x = self.fc(x)
        # x = self.softmax(x)
        # pdb.set_trace()
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)



