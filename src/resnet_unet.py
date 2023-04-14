import torch
import torch.nn as nn
import torchvision.models as models


# First segmentation network
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)  # 256 -> 128

        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)  # 128 -> 64

        self.enc_conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2, 2)  # 64 -> 32

        self.enc_conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                       nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                       nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.pool4 = nn.MaxPool2d(2, 2)  # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
                                             nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU())

        # decoder (upsampling)
        self.upsample1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 16 -> 32
        self.dec_conv1 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                       nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())

        self.upsample2 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 32 -> 64
        self.dec_conv2 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())

        self.upsample3 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 64 -> 128
        self.dec_conv3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())

        self.upsample4 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 128 -> 256
        self.dec_conv4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 1, 1))

    def forward(self, x):
        # encoder
        e1 = self.enc_conv1(x)
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))
        e4 = self.enc_conv4(self.pool3(e3))

        # bottleneck
        b = self.bottleneck_conv(self.pool4(e4))

        # decoder
        d1 = self.dec_conv1(torch.concat([e4, self.upsample1(b)], 1))
        d2 = self.dec_conv2(torch.concat([e3, self.upsample2(d1)], 1))
        d3 = self.dec_conv3(torch.concat([e2, self.upsample3(d2)], 1))
        d4 = self.dec_conv4(torch.concat([e1, self.upsample4(d3)], 1))  # no activation

        return torch.sigmoid(d4)


# Find the original paper on Unet++ at https://arxiv.org/pdf/1807.10165.pdf
class ConvBlock(nn.Module):
    """
    Referred as H in the orginal paper.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 0 (Backbone)
        self.conv_block_0_0 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 480 -> 240
        self.conv_block_1_0 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 240 -> 120
        self.conv_block_2_0 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 120 -> 60
        self.conv_block_3_0 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)  # 60 -> 30
        self.conv_block_4_0 = ConvBlock(512, 1024)

        # Layer 1
        self.upsample_1_0 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 240 -> 480
        self.conv_block_0_1 = ConvBlock(2 * 64, 64)
        self.upsample_2_0 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 120 -> 240
        self.conv_block_1_1 = ConvBlock(2 * 128, 128)
        self.upsample_3_0 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 60 -> 120
        self.conv_block_2_1 = ConvBlock(2 * 256, 256)
        self.upsample_4_0 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 30 -> 60
        self.conv_block_3_1 = ConvBlock(2 * 512, 512)

        # Layer 2
        self.upsample_1_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 240 -> 480
        self.conv_block_0_2 = ConvBlock(3 * 64, 64)
        self.upsample_2_1 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 120 -> 240
        self.conv_block_1_2 = ConvBlock(3 * 128, 128)
        self.upsample_3_1 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 60 -> 120
        self.conv_block_2_2 = ConvBlock(3 * 256, 256)

        # Layer 3
        self.upsample_1_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 240 -> 480
        self.conv_block_0_3 = ConvBlock(4 * 64, 64)
        self.upsample_2_2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 120 -> 240
        self.conv_block_1_3 = ConvBlock(4 * 128, 128)

        # Layer 4 (last layer)
        self.upsample_1_3 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 240 -> 480
        self.conv_block_0_4 = ConvBlock(5 * 64, 64)

        # 1x1 Convolutions + sigmoid for Deep Supervision
        self.ds_0_1 = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.ds_0_2 = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.ds_0_3 = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.ds_0_4 = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Compute layer 0 (Backbone)
        x0_0 = self.conv_block_0_0(x)
        x1_0 = self.conv_block_1_0(self.pool1(x0_0))
        x2_0 = self.conv_block_2_0(self.pool2(x1_0))
        x3_0 = self.conv_block_3_0(self.pool3(x2_0))
        x4_0 = self.conv_block_4_0(self.pool4(x3_0))

        # Compute layer 1
        x0_1 = self.conv_block_0_1(torch.cat([x0_0, self.upsample_1_0(x1_0)], axis=1))
        x1_1 = self.conv_block_1_1(torch.cat([x1_0, self.upsample_2_0(x2_0)], axis=1))
        x2_1 = self.conv_block_2_1(torch.cat([x2_0, self.upsample_3_0(x3_0)], axis=1))
        x3_1 = self.conv_block_3_1(torch.cat([x3_0, self.upsample_4_0(x4_0)], axis=1))

        # Compute layer 2
        x0_2 = self.conv_block_0_2(torch.cat([x0_0, x0_1, self.upsample_1_1(x1_1)], axis=1))
        x1_2 = self.conv_block_1_2(torch.cat([x1_0, x1_1, self.upsample_2_1(x2_1)], axis=1))
        x2_2 = self.conv_block_2_2(torch.cat([x2_0, x2_1, self.upsample_3_1(x3_1)], axis=1))

        # Compute layer 3
        x0_3 = self.conv_block_0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample_1_0(x1_0)], axis=1))
        x1_3 = self.conv_block_1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample_2_0(x2_0)], axis=1))

        # Compute layer 4 (last layer)
        x0_4 = self.conv_block_0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample_1_0(x1_0)], axis=1))

        seg_outputs = torch.cat([self.ds_0_1(x0_1),
                                 self.ds_0_2(x0_2),
                                 self.ds_0_3(x0_3),
                                 self.ds_0_4(x0_4)], axis=1)

        return seg_outputs.mean(axis=1).unsqueeze(1)


# Second stream network -- counting leafs (regression objective)


class LeafCounter(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
        self.counter = models.resnet50()
        for param in self.counter.parameters():
            param.requires_grad = False

        # We will stack the 3 channels of the input image with the mask. Thus, 4 channels in total.
        self.counter.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        for param in self.counter.conv1.parameters():
            param.requires_grad = True

        # We will solve regression problem
        self.counter.fc = torch.nn.Linear(2048, 1)
        for param in self.counter.fc.parameters():
            param.requires_grad = True

        # Segmenter
        self.segmenter = UNetPlusPlus()

