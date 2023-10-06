import numpy as np
import torch
from torch.nn import Conv3d, ConvTranspose3d, LeakyReLU
from torch.nn import AvgPool3d, MaxPool3d, Upsample, InstanceNorm3d

class Multiscale_Elab(torch.nn.Module):

    def __init__(self, input_Channels):
        super(Multiscale_Elab, self).__init__()
        self.input_channels = input_Channels
        self.conv_1 = Conv3d(in_channels=self.input_channels, out_channels=12, kernel_size=[1,1,1], 
                             stride=[1,1,1], padding=0, bias=True)
        self.conv_2 = Conv3d(in_channels=self.input_channels, out_channels=12, kernel_size=[3, 3, 3], 
                             stride=[1,1,1], padding=[1,1,1], bias=True)
        self.conv_3 = Conv3d(in_channels=self.input_channels, out_channels=12, kernel_size=[3, 3, 3], 
                             stride=[1,1,1], padding=[1,1,1], bias=True)
        self.conv_3_1 = Conv3d(in_channels=12, out_channels=12, kernel_size=[3, 3, 3], 
                               stride=[1,1,1], padding=[1,1,1], bias=True)
        self.conv_4 = Conv3d(in_channels=self.input_channels, out_channels=12, kernel_size=[3, 3, 3], 
                             stride=[1,1,1], padding=[1,1,1], bias=True)
        self.conv_4_1 = Conv3d(in_channels=12, out_channels=12, kernel_size=[3, 3, 3],
                               stride=[1,1,1], padding=[1,1,1], bias=True)
        self.conv_4_2 = Conv3d(in_channels=12, out_channels=12, kernel_size=[3, 3, 3],
                               stride=[1,1,1], padding=[1,1,1], bias=True)
        self.conv_5 = Conv3d(in_channels=48, out_channels=24, kernel_size=[1,1,1], 
                             stride=[1,1,1], padding=0, bias=True)
        
        self.Relu = LeakyReLU(0.3)
        self.Norm = InstanceNorm3d(num_features=12)

    def forward(self, x_input):

        conv_1 = self.Relu(self.Norm(self.conv_1(x_input)))
        conv_2 = self.Relu(self.Norm(self.conv_2(x_input)))
        conv_3 = self.Relu(self.Norm(self.conv_3(x_input)))
        conv_3_1 = self.Relu(self.Norm(self.conv_3_1(conv_3)))
        conv_4 = self.Relu(self.Norm(self.conv_4(x_input)))
        conv_4_1 = self.Relu(self.Norm(self.conv_4_1(conv_4)))
        conv_4_2 = self.Relu(self.Norm(self.conv_4_2(conv_4_1)))
        concat = torch.cat([conv_1, conv_2, conv_3_1, conv_4_2], dim=1)
        conv_5 = self.Relu(self.Norm(self.conv_5(concat)))

        concat_1 = torch.cat([conv_5, x_input], dim=1)


        return concat_1
    
class Reduction(torch.nn.Module):

    def __init__(self, input_channels):
        super(Reduction, self).__init__()

        self.max_pool = MaxPool3d(kernel_size=[2,2,2], stride=[2,2,2], padding=0)
        self.avg_pool = AvgPool3d(kernel_size=[2,2,2], stride=[2,2,2], padding=0)

        self.conv = Conv3d(in_channels=input_channels, out_channels=8, kernel_size=[2, 2, 2],
                      stride=[2,2,2], padding=0, bias=True)
        self.Norm = InstanceNorm3d(num_features=8)
        self.Relu = LeakyReLU(0.3)
        self.Norm = InstanceNorm3d(num_features=input_channels)

    def forward(self, x_input):

        output_channels = int(x_input.shape[1])
        max_pool = self.max_pool(x_input)
        avg_pool = self.avg_pool(x_input)
        conv = self.Relu(self.Norm(self.conv(x_input)))

        concat = torch.cat([max_pool, avg_pool, conv], dim=1)

        concat_channels = int(concat.shape[1])

        convolution = Conv3d(in_channels=concat_channels, out_channels=output_channels, kernel_size=[1, 1, 1],  
                        stride=[1,1,1], padding=0, bias=True)
        
        torch.nn.init.xavier_uniform_(convolution.weight)
        torch.nn.init.zeros_(convolution.bias)

        conv_1 = self.Relu(self.Norm(convolution(concat)))

        return conv_1

class Expansion(torch.nn.Module):

    def __init__(self, input_channels):
        super(Expansion, self).__init__()

        self.conv_1 = Conv3d(in_channels=input_channels, out_channels= int(input_channels/2), kernel_size=[1, 1, 1], 
                             stride=[1,1,1], padding=0, bias=True)
        self.Norm = InstanceNorm3d(num_features=int(input_channels/2))
        self.Relu = LeakyReLU(0.3)

        self.conv_trans = ConvTranspose3d(in_channels=int(input_channels/2), 
                                          out_channels=int(input_channels/2), 
                                          kernel_size=[3, 3, 3], 
                                          stride=[2,2,2], 
                                          padding=1, output_padding=1, bias=True)
        self.up_sample = Upsample(scale_factor=(2,2,2))
    
    def forward (self, x_input):
        
        output_channels = int(x_input.shape[1]/2)

        conv = self.Relu(self.Norm(self.conv_1(x_input)))
        up_sample = self.up_sample(conv)

        conv_transpose = self.conv_trans(conv)
        concat = torch.cat([up_sample, conv_transpose], dim=1)
        concat_channels = int(concat.shape[1])

        conv_1 = Conv3d(in_channels=concat_channels, out_channels=output_channels, kernel_size=[1, 1, 1], 
                        stride=[1,1,1], padding=0, bias=True)
        torch.nn.init.xavier_uniform_(conv_1.weight)
        torch.nn.init.zeros_(conv_1.bias)

        conv_final = self.Relu(self.Norm(conv_1(concat)))

        return conv_final










