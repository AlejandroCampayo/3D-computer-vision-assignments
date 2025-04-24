#!/usr/bin/env python3

import cv2
import numpy as np
import os
import math

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torch.linalg import svd
from tqdm import tqdm


def conv2d(x, w):
    """
    Helper function to convolve 2D tensor x with a 2D weight mask w.
    """
    x, w = x.cuda(), w.cuda()
    sy, sx = w.shape
    padded = F.pad(x.unsqueeze(0), (sx // 2, sx // 2, sy // 2, sy // 2), mode="replicate")
    result = F.conv2d(padded.unsqueeze(0), w.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
    return result

def convolution_kernel(sigma, device):
    """
    Compute convolution kernel: sigma^2 * laplace(gauss(sigma))

    Inputs:
     - sigma: std. deviation
     - device: device

    Returns:
    - mask:tensor(H, W)
    """

    # Kernel size and width
    ks = math.ceil(sigma) * 6 + 3
    kw = ks // 2

    ######################################################################################################
    # TODO Q1: Precompute the kernel for blob filtering                                                  #
    # See lecture 2 Part A slides 41                                                                     #
    # You can use the jupyter notebook to visualize the result                                           #
    ######################################################################################################


    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    kernel = cv2.getGaussianKernel(ks, sigma)
    gaussian_kernel = np.outer(kernel, kernel.T)
    laplacian_of_gaussian = cv2.Laplacian(gaussian_kernel, cv2.CV_64F)
    result = sigma**2 * laplacian_of_gaussian
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Please note: you may have boundary artifacts for which it is fine to crop the final kernel
    result = result[1:-1, 1:-1]
    return result

class SIFTDetector:
    """
    pytorch implementation of SIFT detector detector
    """

    def detect_keypoints(self, I, sigma_min=1, sigma_max=30, window_size=3, threshold=0.1):
        """
        Detect SIFT keypoints.

        Inputs:
         - I: 2D array, input image

        Returns:
        - keypoints:tensor(N, 4) (x, y, scale, orientation)
        """

        assert len(I.shape) == 2, "Image dimensions mismatch, need a grayscale image"
        device = I.device
        h, w = I.shape

        # Compute the number of blob sizes
        n_sigma = sigma_max - sigma_min + 1

        ######################################################################################################
        # TODO Q2: Implement blob detector                                                                   #
        # See lecture 2 Part A slides 41                                                                     #
        # You can use the jupyter notebook to visualize the result                                           #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        R = torch.zeros((h, w, n_sigma), device=device)
        for sigma in range(sigma_min, sigma_max + 1):
            # Compute score map for scale sigma here and insert into R
            R[:, :, sigma - sigma_min] = conv2d(I,torch.Tensor(convolution_kernel(sigma, device)))# ...

        # Threshold score map
        R[torch.abs(R) < torch.abs(R).max()*threshold] = 0

        # Do nms suppression over both, position and scale space, with a 3x3x3 window
        keypoints = []
        # Iterate over the image pixels
        for i in tqdm(range(window_size // 2, h - window_size // 2)):
            for j in range(window_size // 2, w - window_size // 2):
                for k in range(window_size // 2, n_sigma - window_size // 2):
                    # Build a tensor for each pixel containing all the pixels of the window
                    window_tensor = R[i - window_size // 2:i + window_size // 2 + 1, j - window_size // 2:j + window_size // 2 + 1, k - window_size // 2:k + window_size // 2 + 1]
                    # Find the maximum value in each window
                    # Extract keypoints where the maximum value is at the center of the window (and the value is different to zero)
                    if R[i, j, k] == torch.max(window_tensor) and R[i, j, k].item() != 0:
                        keypoints.append([i, j, k])
        keypoints = torch.tensor(keypoints)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return keypoints


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd = SIFTDetector()

    _img1 = cv2.imread("../data/CheckerWarp.png")
    _color1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2RGB)
    _gray1 = cv2.cvtColor(_color1, cv2.COLOR_RGB2GRAY)

    img1 = torch.tensor(_img1, device=device) / 255
    color1 = torch.tensor(_color1, device=device) / 255
    gray1 = torch.tensor(_gray1, device=device) / 255

    I = gray1

    blobs = sd.detect_keypoints(I, threshold=0.7)

    np.savetxt("keypoints.out", blobs.numpy())
    print("Saved result to keypoints.out")
