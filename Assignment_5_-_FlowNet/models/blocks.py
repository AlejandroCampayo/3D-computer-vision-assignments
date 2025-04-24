import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

class ConvLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=1):
        super().__init__()

        ######################################################################################################
        # Part3a Q1 Implement ConvLayer
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU(negative_slope=0.1)

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of a simple convolution layer by performing the convolution and the activation.
        :param x: Input to the layer
        :return: Output after the activation function
        """

        ######################################################################################################
        # Part3a Q1 Implement ConvLayer
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        x = self.act(self.conv(x))
        return x

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


class UpConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, output_padding=0):
        super().__init__()

        ######################################################################################################
        # Part3a Q3 Implement UpconvLayer
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1)

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of a simple upconvolution layer by performing the convolution and the activation.
        :param x: Input to the layer
        :return: Output of the upconvolution with subsequent activation
        """
        ######################################################################################################
        # Part3a Q3 Implement UpconvLayer
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        x = self.act(self.conv(x))

        return x

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        ######################################################################################################
        # Part3a Q4 Implement decoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
        self.deconv5 = UpConvLayer(1024, 512)
        self.deconv4 = UpConvLayer(1026, 256)
        self.deconv3 = UpConvLayer(770, 128)
        self.deconv2 = UpConvLayer(386, 64)

        self.flow5 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.flow4 = nn.Conv2d(1026, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.flow3 = nn.Conv2d(770, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.flow2 = nn.Conv2d(386, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.flow1 = nn.Conv2d(194, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.upsample_flow6to5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)#UpConvLayer.forward(2, 2)
        self.upsample_flow5to4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)#UpConvLayer.forward(2, 2)
        self.upsample_flow4to3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)#UpConvLayer.forward(2, 2)
        self.upsample_flow3to2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)#UpConvLayer.forward(2, 2)

        
        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


    def forward(self, x: list):
        """
        Implement the combined decoder for FlowNetS and FlowNetC.
        :param x: A list that contains the output of the bottleneck, as well as the feature maps of the
          required skip connections
        :return: Final flow field
        Keep in mind that the network outputs a flow field at a quarter of the resolution.
        At the end of the decoding, you need to use bilinear upsampling to obtain a flow field at full scale.
        (hint: you can use F.interpolate).
        """

        ######################################################################################################
        # Part3a Q4 Implement decoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
        x_2a, x3_1, x4_1, x5_1, x6_1 = [i for i in x]

        print()
        print("x6_1.shape", x6_1.shape)
        
        up_conv5 = self.deconv5.forward(x6_1)
        print("up_conv5.shape",up_conv5.shape)

        flow_5 = self.flow5(x6_1)
        print("upsampled_flow_5.shape", flow_5.shape)

        upsampled_flow_5 = self.upsample_flow6to5(flow_5)
        print("upsampled_flow_5.shape", upsampled_flow_5.shape)
        
        print()
        print("x5_1.shape", x5_1.shape)

        merged_5 = torch.cat((x5_1, up_conv5, upsampled_flow_5), 1)
        print("merged_5.shape", merged_5.shape)

        up_conv4 = self.deconv4.forward(merged_5) 
        print("up_conv4.shape", up_conv4.shape)
        
        flow_4 = self.flow4(merged_5)
        print("flow_4.shape", flow_4.shape)
        
        upsampled_flow_4 = self.upsample_flow5to4(flow_4)
        print("upsampled_flow_4.shape", upsampled_flow_4.shape)

        print()
        print("x4_1.shape", x4_1.shape)                
        
        merged_4 = torch.cat((x4_1, up_conv4, upsampled_flow_4), 1)
        print("merged_4.shape", merged_4.shape)  
        
        flow_3 = self.flow3(merged_4)
        print("flow_3.shape", flow_3.shape)  
        upsampled_flow_3 = self.upsample_flow4to3(flow_3)
        print("upsampled_flow_3.shape", upsampled_flow_3.shape)  
        up_conv3 = self.deconv3.forward(merged_4)
        print("up_conv3.shape", up_conv3.shape)  

        print()
        print("x3_1.shape", x3_1.shape)  
        
        merged_3 = torch.cat((x3_1, up_conv3, upsampled_flow_3), 1) 
        print("merged_3.shape", merged_3.shape)  
        flow_2 = self.flow2(merged_3)
        print("flow_2.shape", flow_2.shape)  
        upsampled_flow_2 = self.upsample_flow3to2(flow_2)
        print("upsampled_flow_2.shape", upsampled_flow_2.shape)  
        up_conv2 = self.deconv2.forward(merged_3)
        print("up_conv2.shape", up_conv2.shape)  


        print()
        print("x_2a.shape", x_2a.shape) 
        
        merged_2 = torch.cat((x_2a, up_conv2, upsampled_flow_2), 1)    
        print("merged_2.shape", merged_2.shape)               
        flow_1 = self.flow1(merged_2)
        print("flow_1.shape", flow_1.shape)  
         
        return F.interpolate(flow_1, scale_factor=4, mode='bilinear', align_corners=False) 
        up_conv5 = self.deconv5.forward(x6_1)
        print("up_conv5.shape",up_conv5.shape)
        
        concat5 = torch.cat((x5_1, up_conv5), 1)
        print("concat5.shape", concat5.shape)
        
        flow_5 = self.flow5(concat5)
        print("flow_5.shape", flow_5.shape)

        upsampled_flow_5 = self.upsample_flow6to5(flow_5)
        print("upsampled_flow_5.shape", upsampled_flow_5.shape)

        print()
        print("x5_1.shape", x5_1.shape)

        up_conv4 = self.deconv4.forward(concat5)
        print("up_conv4.shape",up_conv4.shape)
        
        concat4 = torch.cat((x4_1, up_conv4, upsampled_flow_5), 1)
        print("concat4.shape", concat4.shape)

        flow_4 = self.flow4(concat4)
        print("flow_4.shape", flow_4.shape)

        upsampled_flow_4 = self.upsample_flow5to4(flow_4)
        print("upsampled_flow_4.shape", upsampled_flow_4.shape)

        print()
        print("x4_1.shape", x4_1.shape)
        
        up_conv3 = self.deconv3.forward(concat4)
        print("up_conv3.shape",up_conv3.shape)
        
        concat3 = torch.cat((x3_1, up_conv3, upsampled_flow_4), 1)
        print("concat3.shape", concat3.shape)

        flow_3 = self.flow3(concat3)
        print("flow_3.shape", flow_3.shape)

        upsampled_flow_3 = self.upsample_flow4to3(flow_3)
        print("upsampled_flow_3.shape", upsampled_flow_3.shape)
        
        print()
        print("x3_1.shape", x3_1.shape)    

        up_conv2 = self.deconv2.forward(concat3)
        print("up_conv2.shape",up_conv2.shape)
        
        concat2 = torch.cat((x_2a, up_conv2, upsampled_flow_3), 1)
        print("concat2.shape", concat2.shape)

        flow_2 = self.flow2(concat2)
        print("flow_2.shape", flow_2.shape)

        upsampled_flow_2 = self.upsample_flow3to2(flow_2)
        print("upsampled_flow_2.shape", upsampled_flow_2.shape)

        return F.interpolate(flow_2, scale_factor=4, mode='bilinear', align_corners=False) 

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
