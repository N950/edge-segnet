import torch
import torch.nn as nn

from NetworkModules import RefineModule, ResidualBottleneckModule, BilinearResizeModule, BottleneckReductionModule


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype_device = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
print("\n*** Device *** :: ",device)



class EdgeSegNet(nn.Module):

    def __init__(self):

        super(EdgeSegNet, self).__init__()

        self.conv_7x7 = nn.Conv2d(in_channels=3, out_channels=13, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Top part of the graph, before the first addition of (32, 32, 193)
        self.bottleneck_module = BottleneckReductionModule()
        self.bilinear_resize_x2_0 = BilinearResizeModule(scale_factor=2)

        # Bottom part of the graph, before the first addition of (32, 32, 193)
        self.residual_module = ResidualBottleneckModule()
        self.refine_module_0 = RefineModule(in_channels=193, out_channels=193)

        # Top part of the graph, after the first addition of (32, 32, 193)
        self.refine_module_1 = RefineModule(in_channels=193, out_channels=217)
        self.bilinear_resize_x2_1 = BilinearResizeModule(scale_factor=2)

        # Bottom part of the graph, maxpool to (64, 64, 217) addition
        self.refine_module_2 = RefineModule(in_channels=13, out_channels=217)

        # After the (64, 64, 217) addition
        self.refine_module_3 = RefineModule(in_channels=217, out_channels=217)
        self.bilinear_resize_x4 = BilinearResizeModule(scale_factor=4)
        self.conv_1x1 = nn.Conv2d(in_channels=217, out_channels=32, kernel_size=1, stride=1)

    def forward(self, input_batch):
        
        x_conv_7x7 = self.conv_7x7(input_batch)
        x_maxpool = self.maxpool(x_conv_7x7)

        x_top = self.bottleneck_module(x_conv_7x7)
        x_top = self.bilinear_resize_x2_0(x_top)

        x_bottom = self.residual_module(x_maxpool)
        x_bottom = self.refine_module_0(x_bottom)

        # (32, 32, 193) addition
        x_top = x_top + x_bottom
        x_top = self.refine_module_1(x_top)
        x_top = self.bilinear_resize_x2_1(x_top)

        x_bottom = self.refine_module_2(x_maxpool)

        # (64, 64, 217) addition
        x_top = x_top + x_bottom

        x_top = self.refine_module_3(x_top)
        x_top = self.bilinear_resize_x4(x_top)
        x_top = self.conv_1x1(x_top)

        return x_top

