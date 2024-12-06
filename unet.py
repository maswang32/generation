import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dc = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.dc(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_c, out_c)
    def forward(self, x1, x2):
        return self.conv(torch.cat((self.up(x1), x2),dim=1))
    

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.Sequential(
            DoubleConv(in_c, out_c),
            nn.MaxPool2d(2,2)
        )
    
    def forward(self, x):
        return self.down(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inconv = DoubleConv(1, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x_64 = self.inconv(x)

        x_128 = self.down1(x_64)     # 64,  -1
        x_256 = self.down2(x_128)    # 128, -2
        x_512 = self.down3(x_256)    # 256, -3
        x_512_2 = self.down4(x_512)  # 512, -4

        x = self.up1(x_512_2, x_512) # 512, -4 -> 512, -3
        x = self.up2(x, x_256)
        x = self.up3(x, x_128)
        x = self.up4(x, x_64)

        return self.outconv(x)