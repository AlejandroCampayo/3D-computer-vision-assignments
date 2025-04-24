#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import sys, os
from torchvision.transforms import GaussianBlur


class RBRIEF:
    """
    Brief descriptor.
    """

    def __init__(self, seed):
        """
        Create rotated brief descriptor.

        Inputs:
        - seed: Random seed for pattern
        """
        self._seed = seed

    def pattern(self, device, patch_size=17, num_pairs=256):
        ######################################################################################################
        # TODO Q1: Generate comparison pattern type I                                                        #
        # See lecture 2 part A slide 54                                                                      #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # NOTE: you can use torch.randint
        # Make sure the seed is set up with self._seed
        torch.manual_seed(self._seed)
        point_pairs = torch.randint(size=(num_pairs, 4), low=0, high=patch_size)
        # The returned tensor should be of dim (num_pairs, 4)
        # where the 4 dimensions are x1, y1, x2, y2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return point_pairs

    def compute_descriptors(self, I, keypoints, device="cpu"):
        """
        Extract rBRIEF binary descriptors for given keypoints in image.

        Inputs:
        - img: 2D array, input image
        - keypoint: tensor(N, 6) with fields x, y, angle, octave, response, size
        - device: where a torch.Tensor is or will be allocated

        Returns:
        - descriptor: tensor(num_keypoint,256)
        """

        assert len(I.shape) == 2, "Image dimensions mismatch"

        # Apply blur kernel to obtain smooth derivatives
        image_blur_kernel = GaussianBlur(5, 2.0)
        I = image_blur_kernel(I.unsqueeze(0)).squeeze()

        # Get pattern
        pattern = self.pattern(device)
        
        # Get keypoint values
        points = keypoints[:, 0:2] # x, y
        angle = keypoints[:, 2] # clockwise
        ######################################################################################################
        # TODO Q2: Implement the rotated brief descriptor                                                    #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Make rotation matrices
        cos_angles = torch.cos(angle)
        sin_angles = torch.sin(angle)
        rotation_matrices = torch.zeros((len(angle), 2, 2), device=device)
        rotation_matrices[:, 0, 0] = cos_angles  
        rotation_matrices[:, 0, 1] = -sin_angles
        rotation_matrices[:, 1, 0] = sin_angles
        rotation_matrices[:, 1, 1] = cos_angles

        # Compute rotated patterns
        rotated_patterns = []
        for pat in pattern:
            x1, y1, x2, y2 = pat[0], pat[1], pat[2], pat[3]
            rotated_x1 = rotation_matrices[:, 0, 0] * x1 + rotation_matrices[:, 0, 1] * y1
            rotated_y1 = rotation_matrices[:, 1, 0] * x1 + rotation_matrices[:, 1, 1] * y1
            rotated_x2 = rotation_matrices[:, 0, 0] * x2 + rotation_matrices[:, 0, 1] * y2
            rotated_y2 = rotation_matrices[:, 1, 0] * x2 + rotation_matrices[:, 1, 1] * y2
            rotated_patterns.append(torch.stack([rotated_x1, rotated_y1, rotated_x2, rotated_y2], dim=-1))
        rotated_patterns = torch.stack(rotated_patterns).permute(1, 0, 2)

        # Discard any keypoints that have pixel locations outside the image
        h, w = I.shape

        for i in range(len(points)):
            rotated_patterns[i,:,0] = rotated_patterns[i,:,0] + points[i, 0]
            rotated_patterns[i,:,1] = rotated_patterns[i,:,1] + points[i, 1]
            rotated_patterns[i,:,2] = rotated_patterns[i,:,2] + points[i, 0]
            rotated_patterns[i,:,3] = rotated_patterns[i,:,3] + points[i, 1]

        mask = (rotated_patterns[:, :, 0] >= 0).all(dim=1) & \
                (rotated_patterns[:, :, 0] < w).all(dim=1) & \
                (rotated_patterns[:, :, 1] >= 0).all(dim=1) & \
                (rotated_patterns[:, :, 1] < h).all(dim=1) & \
                (rotated_patterns[:, :, 2] >= 0).all(dim=1) & \
                (rotated_patterns[:, :, 2] < w).all(dim=1) & \
                (rotated_patterns[:, :, 3] >= 0).all(dim=1) & \
                (rotated_patterns[:, :, 3] < h).all(dim=1)
        
        valid_rotated_patterns = rotated_patterns[mask]        
        valid_keypoints = keypoints[mask]
        # Sample image intensities at line segment starts
        x_coords_start = valid_rotated_patterns[:, :, 0]
        y_coords_start = valid_rotated_patterns[:, :, 1]

        int_start = torch.zeros_like(x_coords_start, dtype=I.dtype)  # Initialize with zeros
        for i in range(valid_rotated_patterns.shape[0]):
            for j in range(valid_rotated_patterns.shape[1]):
                x = int(x_coords_start[i, j])
                y = int(y_coords_start[i, j])
                int_start[i, j] = I[y, x]

        # Sample image intensities at line segment ends
        x_coords_end = valid_rotated_patterns[:, :, 2]
        y_coords_end = valid_rotated_patterns[:, :, 3]

        int_end = torch.zeros_like(x_coords_end, dtype=I.dtype)  # Initialize with zeros
        for i in range(valid_rotated_patterns.shape[0]):
            for j in range(valid_rotated_patterns.shape[1]):
                x = int(x_coords_end[i, j])
                y = int(y_coords_end[i, j])
                int_end[i, j] = I[y, x]
       
        # Compare intensities to form a binary descriptor
        descriptors = (int_start > int_end).to(torch.float32)

        # NOTE: output dimension should be (num_desc, dim=256)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return descriptors, valid_keypoints


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np
    import cv2
    import os
    sys.path.append('..')
    from task2a.match import match

    group_id = int(open('../group_id.txt', 'r').read())

    img1 = cv2.imread("../data/NotreDame1.jpg")
    color1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread("../data/NotreDame2.jpg")
    color2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)

    # Fields in keypoints from SIFT detector:
    # x, y, angle, octave, response, size

    keypoints1 = torch.tensor(np.loadtxt('keypoints1.txt'), device=device)
    keypoints2 = torch.tensor(np.loadtxt('keypoints2.txt'), device=device)

    brief = RBRIEF(seed=group_id)
    desc1, _ = brief.compute_descriptors(torch.tensor(gray1, device=device), keypoints1)
    desc2, _ = brief.compute_descriptors(torch.tensor(gray2, device=device), keypoints2)

    matches = match(
        descriptors1=desc1,
        descriptors2=desc2,
        device=device,
        dist="hamming",
        ratio=0.95,
        threshold=160,
    )

    np.savetxt("rbrief.out", matches.numpy())

