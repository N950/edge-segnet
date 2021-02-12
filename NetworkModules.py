import torch.nn as nn

class ResidualBottleneckModule(nn.Module):

    def __init__(self):
        
        super(ResidualBottleneckModule, self).__init__()
        self.bn_0 = nn.BatchNorm2d(13)
        self.relu_0 = nn.ReLU()
        
        self.conv_1x1_0 = nn.Conv2d(in_channels=13, out_channels=193, kernel_size=1, stride=1)

        self.bn_1 = nn.BatchNorm2d(193)
        self.relu_1 = nn.ReLU()

        self.conv_3x3_0 = nn.Conv2d(in_channels=193, out_channels=193, kernel_size=3, stride=2, padding=1)

        self.bn_2 = nn.BatchNorm2d(193)
        self.relu_2 = nn.ReLU()

        self.conv_1x1_1 = nn.Conv2d(in_channels=193, out_channels=193, kernel_size=1, stride=1)

        self.conv_3x3_1 = nn.Conv2d(in_channels=13, out_channels=193, kernel_size=3, stride=2, padding=1)

    def forward(self, input_batch):

        x_bn_0 = self.bn_0(input_batch)
        x_relu_0 = self.relu_0(x_bn_0)
        x_conv_3x3_1 = self.conv_3x3_1(x_relu_0)

        x = self.conv_1x1_0(x_relu_0)
        
        x = self.bn_1(x)
        x = self.relu_1(x)
        
        x = self.conv_3x3_0(x)
        
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv_1x1_1(x)

        output = x_conv_3x3_1 + x

        return output

class BottleneckReductionModule(nn.Module):

    def __init__(self):

        super(BottleneckReductionModule, self).__init__()

        self.conv_3x3_0 = nn.Conv2d(in_channels=13, out_channels=193, kernel_size=3, stride=8)
        self.relu_0 = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(in_channels=193, out_channels=193, kernel_size=1, stride=1)
        self.relu_1 = nn.ReLU()
        self.conv_3x3_1 = nn.Conv2d(in_channels=193, out_channels=193, kernel_size=3, stride=1, padding=1)

    def forward(self, input_batch):

        x = self.conv_3x3_0(input_batch)

        x = self.relu_0(x)
        x = self.conv_1x1(x)
        x = self.relu_1(x)
        x = self.conv_3x3_1(x)

        return x

class RefineModule(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(RefineModule, self).__init__()
        
        self.conv_1x1_0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv_1x1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.relu_0 = nn.ReLU()
        self.conv_3x3_0 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU()
        self.conv_3x3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input_batch):

        x_conv_1x1_0 = self.conv_1x1_0(input_batch)

        x = self.conv_1x1_1(x_conv_1x1_0)
        x = self.relu_0(x)
        x = self.conv_3x3_0(x)
        x = self.relu_1(x)
        x = self.conv_3x3_1(x)

        return x + x_conv_1x1_0

class BilinearResizeModule(nn.Module):

    def __init__(self, scale_factor):

        super(BilinearResizeModule, self).__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, input_batch):

        x = self.upsample(input_batch)
        return x
