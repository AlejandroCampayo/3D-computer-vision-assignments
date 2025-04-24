import torch
import torch.nn as nn
from .blocks import ConvLayer, Decoder

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        ######################################################################################################
        # Part3a Q2 Implement encoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv1 = ConvLayer(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvLayer(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = ConvLayer(128, 256, kernel_size=5, stride=2, padding=2) 
        self.conv3_1 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvLayer(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = ConvLayer(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvLayer(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = ConvLayer(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvLayer(512, 1024, kernel_size=3, stride=2, padding=1)

        # Note, that the diagram in the paper does not show this layer.
        # However, in the original Caffe code, the authors an additional layer in the bottleneck.
        # This layer Has the same input and output channels as the output of the previous channel and does not downsaple.
        self.conv6_1 = ConvLayer(1024, 1024, kernel_size=3, stride=1, padding=1)

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor):
        """
        :param x: The two input images concatenated along the channel dimension
        :return: A list of encodings at different stages in the decoder
                As can be seen in the diagram of the FlowNet paper, skip connections branch of at differnt positions.
        """
        ######################################################################################################
        # Part3a Q2 Implement encoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3= self.conv3(x2)
        x3_1 = self.conv3_1(x3)
        x4 = self.conv4(x3_1)
        x4_1 = self.conv4_1(x4)
        x5 = self.conv5(x4_1)
        x5_1 = self.conv5_1(x5)
        x6 = self.conv6(x5_1)
        x6_1 = self.conv6_1(x6)

        return [x2, x3_1, x4_1, x5_1, x6_1]     
        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


class FlowNetS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        """
        Implement the full forward pass of the FlowNetS model.
        For this, you need to implement both the encoder and decoder.
        All parameters are given in the diagram figure of the FlowNet paper.
        :param image1: First image
        :param image2: Second image
        :return: Flow field
        """

        ######################################################################################################
        # Part3a Q5 Put all components together
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        x = torch.cat((image1, image2), 1)
        encoder_output = self.encoder.forward(x)
        decoder_output = self.decoder.forward(encoder_output)        
        return decoder_output
        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

