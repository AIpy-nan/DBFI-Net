#_*_coding:utf8_*_

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*out_ch, out_ch, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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

class DPM(nn.Module):
    def __init__(self):
        super(DPM, self).__init__()
        inchannle = 54
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, 54, kernel_size=3, padding=1, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(54),
            nn.ReLU(inplace=True)
        )
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, 54, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(54),
            nn.ReLU(inplace=True)

        )
        self.pan1_1 = Bottleneck(54, 18)
        self.pan1_2 = Bottleneck(54, 18)
        self.pan1_3 = Bottleneck(54, 18)
        self.ms1_1 = Bottleneck(54, 18)
        self.ms1_2 = Bottleneck(54, 18)
        self.ms1_3 = Bottleneck(54, 18)
        self.bk1_1 = Bottleneck(108, 36)
        self.bk1_2 = Bottleneck(108, 36)
        self.bk1_3 = Bottleneck(108, 36)
        self.bf1_1 = Bottleneck(216, 72)
        self.bf1_2 = Bottleneck(216, 72)
        self.bf1_3 = Bottleneck(216, 72)

        self.convs_pan = nn.Sequential(
            nn.Conv2d(in_channels=54, out_channels=2 * 54, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(2 * 54),
            nn.ReLU(inplace=True)
        )
        self.convx_pan = nn.Sequential(
            nn.Conv2d(3 * 54, 2 * 54, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * 54),
            nn.ReLU(inplace=True)
        )
        self.convs_ms = nn.Sequential(
            nn.Conv2d(in_channels=54, out_channels=2 * 54, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(2 * 54),
            nn.ReLU(inplace=True)
        )
        self.convx_ms = nn.Sequential(
            nn.Conv2d(3 * 54, 2 * 54, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * 54),
            nn.ReLU(inplace=True)
        )

        self.convs1_pan = nn.Sequential(
            nn.Conv2d(in_channels=108, out_channels=2 * 108, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(2 * 108),
            nn.ReLU(inplace=True)
        )
        self.convx1_pan = nn.Sequential(
            nn.Conv2d(3 * 108, 2 * 108, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * 108),
            nn.ReLU(inplace=True)
        )
        self.convs1_ms = nn.Sequential(
            nn.Conv2d(in_channels=108, out_channels=2 * 108, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(2 * 108),
            nn.ReLU(inplace=True)
        )
        self.convx1_ms = nn.Sequential(
            nn.Conv2d(3 * 108, 2 * 108, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * 108),
            nn.ReLU(inplace=True)
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=216, out_channels=2 * 216, kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(2 * 216),
            nn.ReLU(inplace=True)
        )
        self.convx2 = nn.Sequential(
            nn.Conv2d(3 * 216, 2 * 216, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * 216),
            nn.ReLU(inplace=True)
        )


        self.avg_pan = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)
        self.convz = nn.Sequential(
            nn.Conv2d(432, 512, kernel_size=1, padding=0, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(12800, 8192),
            nn.Linear(8192, 500),
            nn.Linear(500, 200),
            nn.Linear(200, 11)
        )

    def forward(self, x, y):
        x = F.relu(self.pre_ms(x))
        y = F.relu(self.pre_pan(y))
        y = self.pan1_1(y)
        y0_1 = y[1]
        y1_1 = self.pan1_2(y[0])
        y0_2 = torch.cat([y0_1, y1_1[1]], 1)
        y1_2 = self.pan1_3(y1_1[0])
        y0_3 = torch.cat([y0_2, y1_2[1]], 1)
        y1_2 = self.convs_pan(y1_2[0])
        y1_3 = self.convx_pan(y0_3)
        y1 = y1_2 + y1_3

        y1_3 = self.avg_pan(y1_3)


        x = self.ms1_1(x)
        x0_1 = x[1]
        x1_1 = self.ms1_2(x[0])
        x0_2 = torch.cat([x0_1, x1_1[1]], 1)
        x1_2 = self.ms1_3(x1_1[0])
        x0_3 = torch.cat([x0_2, x1_2[1]], 1)
        x1_2 = self.convs_ms(x1_2[0])
        x1_3 = self.convx_ms(x0_3)

        x1 = x1_2 + x1_3
        y1 = self.avg_pan(y1)

        xy1_1 = x1_3 + y1
        xy1_2 = self.bk1_1(xy1_1)
        xy1_3 = self.bk1_2(xy1_2[0])
        xy0_1 = torch.cat([xy1_2[1],  xy1_3[1]], 1)
        xy1_4 = self.bk1_3(xy1_3[0])  #上支路完成
        xy0_2 = torch.cat([xy1_4[1], xy0_1], 1)
        xy1_4 = self.convs1_pan(xy1_4[0])
        xy0_3 = self.convx1_pan(xy0_2)
        xy1 = xy1_4 + xy0_3
        xy2_1 = x1 + y1_3
        xy2_2 = self.bk1_1(xy2_1)
        xy2_3 = self.bk1_2(xy2_2[0])
        xy3_1 = torch.cat([xy2_2[1], xy2_3[1]], 1)
        xy2_4 = self.bk1_3(xy2_3[0])
        xy3_2 = torch.cat([xy2_4[1], xy3_1], 1)
        xy2_4 = self.convs1_ms(xy2_4[0])
        xy3_3 = self.convx1_ms(xy3_2)
        xy2 = xy2_4 + xy3_3
        xy3 = xy1 + xy2
        xy3 = self.avg_pan(xy3)

        x_y = self.bf1_1(xy3)
        x_y1 = self.bf1_2(x_y[0])
        x_y2 = torch.cat([x_y[1], x_y1[1]], 1)
        x_y3 = self.bf1_3(x_y1[0])
        x_y4 = torch.cat([x_y2, x_y3[1]], 1)
        x_y3 = self.convs2(x_y3[0])
        x_y4 = self.convx2(x_y4)
        x_y0 = x_y4 +x_y3

        s = self.convz(x_y0)


        s = s.view(s.size()[0], -1)

        s = self.fc(s)

        return s
if __name__ == '__main__':
    x = torch.randn(2, 4, 20, 20)
    y = torch.randn(2, 1, 80, 80)
    out = DPM()
    out = out(x, y)
    print(out.size())