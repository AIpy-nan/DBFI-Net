import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, num_group=6):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, groups=num_group, padding=1),
            nn.Dropout(0.1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*out_ch, out_ch, kernel_size=1, stride=stride, padding=0),
            nn.Dropout(0.1)
        )


    def forward(self, x):
        residual = x
        out1_1 = self.conv1(x)
        out1_2 = self.conv2(out1_1)
        out1_3 = torch.cat([out1_1, out1_2], 1)     
        out1_4 = self.conv3(out1_3)
        out1_5 = torch.cat([out1_3, out1_4], 1)
        out = out1_5 + residual

        return out, out1_5

class DMP_CIFAR10(nn.Module):
    def __init__(self):
        super(DMP_CIFAR10, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 54, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(54),
            nn.ReLU(inplace=True)
        )
  
        self.blk = Bottleneck(54, 18)
        self.blk2 = Bottleneck(54, 18)
        self.blk3 = Bottleneck(54, 18)
     
        self.cblk1 = Bottleneck(108, 36)
        self.cblk2 = Bottleneck(108, 36)
        self.cblk3 = Bottleneck(108, 36)
  
        self.mblk1 = Bottleneck(216, 72)
        self.mblk2 = Bottleneck(216, 72)
        self.mblk3 = Bottleneck(216, 72)

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(54, 2*54, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(108),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(3*54, 2*54, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(108),
            nn.ReLU(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(108, 216, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(216),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(3*108, 216, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(216),
            nn.ReLU(inplace=True)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(216, 432, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(432),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(3*216, 432, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(432),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)
        self.convs = nn.Sequential(
            nn.Conv2d(432, 512, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(8192, 500),
            nn.Linear(500, 200),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.blk(x)
        x0_1 = self.blk2(x[0])
        x0_2 = torch.cat([x[1], x0_1[1]], 1)

        x0_3 = self.blk3(x0_1[0])
        x0_4 = torch.cat([x0_3[1], x0_2], 1)

        x0_3 = self.conv1_1(x0_3[0])
        x0_4 = self.conv1_2(x0_4)

        x0_5 = x0_3 + x0_4
        x1_0 = self.avg_pool(x0_5)   
        x1_0 = self.cblk1(x1_0)
        x1_1 = self.cblk2(x1_0[0])
        x1_2 = torch.cat([x1_0[1], x1_1[1]], 1)
        x1_3 = self.cblk3(x1_1[0])
        x1_4 = torch.cat([x1_3[1], x1_2], 1)
        x1_3 = self.conv2_1(x1_3[0])
        x1_4 = self.conv2_2(x1_4)
        x1_5 = x1_3 + x1_4
        x2_0 = self.avg_pool(x1_5)
        x2_0 = self.mblk1(x2_0)
        x2_1 = self.mblk2(x2_0[0])
        x2_2 = torch.cat([x2_0[1], x2_1[1]], 1)
        x2_3 = self.mblk3(x2_1[0])
        x2_4 = torch.cat([x2_2, x2_3[1]], 1)
        x2_3 = self.conv3_1(x2_3[0])
        x2_4 = self.conv3_2(x2_4)
        x2_5 = x2_3 + x2_4
        s = self.convs(x2_5)
        s = self.avg_pool(s)
        s = s.view(s.size()[0], -1)
        s = self.fc(s)
        return s


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    out = DMP_CIFAR10()
    out = out(x)
    print(out.size())

