import torch
import torch.nn as nn
import math
from .blocks import ConvLayer, Decoder

class FeatureExtractor(nn.Module):
    def __init__(self):
        """
        Implement the layers in the Siamese feature extractor of the FlowNetC network.
        The specific parameters can be seen in the diagram of the FlowNet paper.
        """
        super().__init__()

        ######################################################################################################
        # Part3b Q1 Implement feature extractor
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv1 = ConvLayer(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvLayer(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = ConvLayer(128, 256, kernel_size=5, stride=2, padding=2)        

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor):
        """
        :param x: An input image.
        :return: A tuple that returns the output of self.conv2 and self.conv3.
                 In addition to the final output that will be used in the Correlation Layer, we also need the
                 output of self.conv2 as a skip connection to the decoder.
                 Details about this can be found in the diagram of the FlowNet paper.
        """

        ######################################################################################################
        # Part3b Q1 Implement feature extractor
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        x = self.conv1(x)
        conv2_output = self.conv2(x)
        conv3_output = self.conv3(conv2_output)

        return conv2_output, conv3_output

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

class CorrelationLayer(nn.Module):
    def __init__(self, d=20, s1=1, s2=2):
        super().__init__()
        self.s1 = s1
        self.s2 = s2
        self.d = d
        self.padlayer = nn.ConstantPad2d(d, value=0.0)


    def forward(self, features_1: torch.Tensor, features_2: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Correlation Layer.

        :param features_1: the feature map obtained from the first image in the sequence
        :param features_2: the feature map obtained from the second image in the sequence
        :return: The correlation of two patches in the corresponding feature maps.
                Use k=0, d = 20, s1 = 1, s2 = 2
        """

        ######################################################################################################
        # Part3b Q2 Implement correlation
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
        features_2_padded = self.padlayer(features_2)
        Y_coordinates, X_coordinates = torch.meshgrid([torch.arange(0, 2 * self.d + 1, self.s2),
                                            torch.arange(0, 2 * self.d + 1, self.s2)])
        
        batch_size, num_channels, height, width = features_1.shape
        
        output_list = []
        
        for x, y in zip(X_coordinates.reshape(-1), Y_coordinates.reshape(-1)):

            # print("y, x", y, x)
            # print("y+height, x+width", y+height, x+width)
            # Extract the neighbourhood region from features_2_padded using the offsets
            neighbourhood_region = features_2_padded[:, :, y:y+height, x:x+width]
            
            # Compute the cross-correlation by element-wise multiplication and mean pooling
            cross_corr = torch.mean(features_1 * neighbourhood_region, dim=1, keepdim=True)
            
            output_list.append(cross_corr)
            # print("--------------------------------------------")
            # print("features_1.shape", features_1.shape)
            # print("neighbourhood_region.shape", neighbourhood_region.shape)
            # print("cross_corr.shape", cross_corr.shape)
            # print("--------------------------------------------")

        output = torch.cat(output_list, dim=1)
        return output
        
        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


class Encoder(nn.Module):

    def __init__(self):
        """
        Implement the Layers of the FlowNetC encoder
        The specific parameters of each layer can be found in the diagram in the paper
        """
        super().__init__()

        ######################################################################################################
        # Part3b Q3 Implement encoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv3_1 = ConvLayer(473, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvLayer(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = ConvLayer(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvLayer(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = ConvLayer(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvLayer(512, 1024, kernel_size=3, stride=2, padding=1)

        # Note, that the diagram in the paper does not show this layer.
        # See the comment in the FlowNetS architecture for more details.
        self.conv6_1 = ConvLayer(1024, 1024, kernel_size=3, stride=1, padding=1)


        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor):
        """
        The forward pass of the FlowNetC encoder
        :param x: The output of the Correlation Layer
        :return: A List of encodings at different stages in the decoder.
                As can be seen in the diagram of the FlowNet paper, skip connections branch at different positions
                in the encoder into the decoder.
        """

        ######################################################################################################
        # Part3b Q3 Implement encoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
        print(x.shape)
        x3_1 = self.conv3_1(x)
        x4 = self.conv4(x3_1)
        x4_1 = self.conv4_1(x4)
        x5 = self.conv5(x4_1)
        x5_1 = self.conv5_1(x5)
        x6 = self.conv6(x5_1)
        x6_1 = self.conv6_1(x6)

        # Return intermediate encodings
        for i in [x3_1, x4, x4_1, x5, x5_1, x6, x6_1]:
            print(i.shape)
        return [x3_1, x4_1, x5_1, x6_1]

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

class FlowNetC(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.conv_redir = ConvLayer(256, 32, kernel_size=1, stride=1, padding=0)  #nn.Conv2d(256, 32, kernel_size=1, stride=1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.relu = nn.ReLU()
        self.correlationLayer = CorrelationLayer()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the FlowNetC model.
        :param image1: First image
        :param image2: Second image
        :return: Flow field
        """

        ######################################################################################################
        # Part3b Q4 Put components together
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        # Encode both images using the Feature extractor
        conv2_output_1, conv3_output_1 = self.feature_extractor.forward(image1)
        conv2_output_2, conv3_output_2 = self.feature_extractor.forward(image2)

        print("conv3_output_1.shape",conv3_output_1.shape)
        # Establish correlation volume
        correlation_output = self.correlationLayer.forward(conv3_output_1, conv3_output_2)

        # Use the ReLU on the correlation layer
        correlation_output = self.relu(correlation_output)
        conv_redir_output = self.relu(self.conv_redir(conv3_output_1))
        in_conv3_1 = torch.cat((conv_redir_output, correlation_output), 1)
        
        # Feed the output to the encoder
        encoder_output = self.encoder.forward(in_conv3_1)
        encoder_output.insert(0, conv2_output_1)
        

        # Attach the decoder
        decoder_output = self.decoder.forward(encoder_output)        
        return decoder_output
        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def correlate(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        PLEASE SEE EXCERCISE SHEET FOR INFORMATION.
        :param image1: The first image in the image sequence as a torch tensor
        :param image2: The second image in the image sequence as a torch tensor
        :return: The output of the correlation layer after encoding the images
        The goal of is function is to give you an additional opportunity to debug the Correlation layer
        This function will be called in the test_correlation function in the ModelWrapper
        """

        ######################################################################################################
        # Part3b Q4 Put components together
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        # Extract features from both images
        conv2_output_1, conv3_output_1 = self.feature_extractor.forward(image1)
        conv2_output_2, conv3_output_2 = self.feature_extractor.forward(image2)

        # Compute the correlation volume (shape should be [1, 441, 48, 64])
        correlation_output = self.correlationLayer.forward(conv3_output_1, conv3_output_2)

        return correlation_output

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
