import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext


class Encoder(nn.Module):

    def __init__(self, input_channel, emb_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        # kernel size must be odd
        assert kernel_size % 2==1: "kernle size must be odd."

        self.hid_dim = hid_dim
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)
        self.conv_layer1 = nn.Conv2d(input_channel, 64, kernel=3, stride=1, padding =1)
        self.conv_layer2 = nn.Conv2d(64, 128, kernel=3, stride=1, padding=1)
        self.conv_layer3 = nn.Conv2d(128, 256, kernel=3, stride=1, padding=1)
        self.conv_layer4 = nn.Conv2d(256, 512, kernel=3, stride=1, padding=1)
        self.conv_layer5 = nn.Conv2d(512, 512, kernel=3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # img = [batch, Cin, W, H]
        batch = img.shape[0]
        C_in = img.shape[1]

        
