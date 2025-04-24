#!/usr/bin/env python3

import cv2
import numpy as np
import os

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms import GaussianBlur
from torch.linalg import svd
import matplotlib.pyplot as plt


class Harris:
    """
    pytorch implementation of Harris corner detector
    """
    def show_my_matrix(self, M):
        M_np = M.cpu().numpy()
        plt.imshow(M_np, cmap='gray')


    def compute_score(self, I, sigma=1.0, kernel_size=5, k=0.02):
        """
        Compute the score map of the harris corner detector.

        Inputs:
         - I: 2D array, input image
         - k: The k parameter from the score formula, typically in range [0, 0.2]
         - sigma: Std deviation used for structure tensor

        Returns:
         - R: Score map of size H, W
        """

        assert len(I.shape) == 2, "Image dimensions mismatch, need a grayscale image"
        device = I.device
        w, h = I.shape
        blur_kernel = GaussianBlur(kernel_size, sigma)

        # Apply blur kernel to obtain smooth derivatives
        image_blur_kernel = GaussianBlur(5, 1.0)
        I = image_blur_kernel(I.unsqueeze(0)).squeeze().unsqueeze(0).unsqueeze(0)

        ######################################################################################################
        # TODO Q1: Compute harris corner score of a given image                                              #
        # See lecture 2 Part A slides 18 and 20                                                              #
        # You can use the jupyter notebook to visualize the result                                           #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute derivatives
        sobel_filter_x = torch.tensor([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]],dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_filter_y = torch.tensor([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]],dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        Ix = F.conv2d(I, sobel_filter_x).squeeze()
        Iy = F.conv2d(I, sobel_filter_y).squeeze()

        # Compute structure tensor entries
        Ixx = Ix*Ix
        Ixy = Ix*Iy
        Iyy = Iy*Iy
        #self.show_my_matrix(Iyy)
        # Blur the entries of the structure kernel with blur_kernel

        Sxx = blur_kernel(Ixx.unsqueeze(0))
        Sxy = blur_kernel(Ixy.unsqueeze(0))
        Syy = blur_kernel(Iyy.unsqueeze(0))
        # Make matrices square to be able to compute eigenvals
        Sxx = Sxx.squeeze()
        Sxy = Sxy.squeeze()
        Syy = Syy.squeeze()
        M = torch.stack([Sxx, Sxy, Sxy, Syy], dim=-1)
        M = M.view(Sxx.size(0), Sxx.size(1), 2, 2)
        
        # Compute eigenvalues
        # NOTE: you may use torch.linalg.eigvals(...).real
        eigenvalues = torch.linalg.eigvals(M).real

        lambda1, lambda2 = eigenvalues[:, :, 0], eigenvalues[:, :, 1]
        
        # Compute score R
        R = lambda1*lambda2 - k*(lambda1+lambda2)**2
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return R


    def detect_keypoints(self,  I, threshold=0.2, sigma=1.0, kernel_size=5, k=0.02, window_size=5):
        """
        Perform harris keypoint detection.

        Inputs:
        - I: 2D array, input image
        - threshold: score threshold
        - k: The k parameter for corner_harris, typically in range [0, 0.2]
        - sigma: std. deviation of blur kernel

        Returns:
        - keypoints:tensor(N, 2)
        """

        w, h = I.shape
        R = self.compute_score(I, sigma, kernel_size, k)
        R[R<R.max()*threshold] = 0

        # ######################################################################################################
        # TODO Q2:Non Maximal Suppression for removing adjacent corners.                                       #
        # See lecture 2 Part A slides 22                                                                       #
        # You can use the jupyter notebook to visualize the result                                             #
        # ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Build a tensor that for each pixel contains all the pixels of the window
        keypoints = []

        # Iterate over the image pixels
        for i in range(window_size // 2, w - window_size // 2):
            for j in range(window_size // 2, h - window_size // 2):
                # Build a tensor for each pixel containing all the pixels of the window
                window_tensor = R[i - window_size // 2:i + window_size // 2 + 1, j - window_size // 2:j + window_size // 2 + 1]
                # Find the maximum value in each window
                # Extract keypoints where the maximum value is at the center of the window (and the value is different to zero)
                if R[i, j] == torch.max(window_tensor) and R[i,j].item() != 0:
                    keypoints.append([i, j])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return keypoints


if __name__ == "__main__":
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h = Harris()

    _img1 = cv2.imread("../data/Chess.png")
    _color1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2RGB)
    _gray1 = cv2.cvtColor(_color1, cv2.COLOR_RGB2GRAY)

    img1 = torch.tensor(_img1, device=device) / 255
    color1 = torch.tensor(_color1, device=device) / 255
    gray1 = torch.tensor(_gray1, device=device) / 255

    I = gray1

    keypoints = h.detect_keypoints(I, sigma=1.0, threshold=0.1, k=0.05, window_size=11)

    np.savetxt("harris.out", keypoints.numpy())
    print("Saved result to harris.out")
