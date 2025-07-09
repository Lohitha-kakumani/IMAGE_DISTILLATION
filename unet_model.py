import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         def conv_block(in_ch, out_ch):
#             return nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True)
#             )

#         self.enc1 = conv_block(3, 64)
#         self.enc2 = conv_block(64, 128)
#         self.enc3 = conv_block(128, 256)

#         self.pool = nn.MaxPool2d(2)

#         self.middle = conv_block(256, 512)

#         self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec3 = conv_block(512, 256)

#         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec2 = conv_block(256, 128)

#         self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec1 = conv_block(128, 64)

#         self.final = nn.Conv2d(64, 3, kernel_size=1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))

#         m = self.middle(self.pool(e3))

#         d3 = self.up3(m)
#         d3 = self.dec3(torch.cat([d3, e3], dim=1))

#         d2 = self.up2(d3)
#         d2 = self.dec2(torch.cat([d2, e2], dim=1))

#         d1 = self.up1(d2)
#         d1 = self.dec1(torch.cat([d1, e1], dim=1))

#         return self.final(d1)

