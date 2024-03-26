import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.ssn.ssn import ssn_iter, sparse_ssn_iter

class HorizontalCircularPadding(nn.Module):
    def __init__(self, padding_width, padding_height):
        super(HorizontalCircularPadding, self).__init__()
        self.padding_width = padding_width
        self.padding_height = padding_height

    def forward(self, x):
        x_padded = F.pad(x, (self.padding_width, self.padding_width, 0, 0), mode='circular')
        x_padded = F.pad(x_padded, (0, 0, self.padding_height, self.padding_height), mode='replicate')
        return x_padded


def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=0, bias=False), 
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class SSNModel(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter

        self.scale1 = nn.Sequential(
            HorizontalCircularPadding(padding_width=1, padding_height=1),
            conv_bn_relu(6, 64),
            HorizontalCircularPadding(padding_width=1, padding_height=1),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            HorizontalCircularPadding(padding_width=1, padding_height=1),
            nn.MaxPool2d(3, 2, padding=0), 
            conv_bn_relu(64, 64),
            HorizontalCircularPadding(padding_width=1, padding_height=1),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            HorizontalCircularPadding(padding_width=1, padding_height=1),
            nn.MaxPool2d(3, 2, padding=0), 
            HorizontalCircularPadding(padding_width=1, padding_height=1),
            conv_bn_relu(64, 64),
            HorizontalCircularPadding(padding_width=1, padding_height=1),
            conv_bn_relu(64, 64)
        )

        self.output_conv = nn.Sequential(
            HorizontalCircularPadding(padding_width=1, padding_height=1),
            nn.Conv2d(64*3+6, feature_dim-6, 3, padding=0), 
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, init_label_map, centers3D):
        pixel_f = self.feature_extract(x)

        if self.training:
            return ssn_iter(pixel_f, self.nspix, self.n_iter, init_label_map, centers3D)
        else:
            return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter, init_label_map, centers3D)


    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return torch.cat([feat, x], 1)
