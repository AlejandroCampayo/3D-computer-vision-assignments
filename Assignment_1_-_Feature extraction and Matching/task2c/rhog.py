#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import sys, os
from torchvision.transforms import GaussianBlur
import math
import numpy as np

class RHOG:
    """
    Brief descriptor.
    """
    
    def compute_descriptors(self, I, keypoints, device="cpu"):
        """
        Extract rotate hog dsecriptors for the keypoints.

        Inputs:
        - img: 2D array, input image
        - keypoint: tensor(N, 6) with fields x, y, angle, octave, response, size
        - device: where a torch.Tensor is or will be allocated

        Returns:
        - descriptor: tensor(num_keypoint,256)
        """

        assert len(I.shape) == 2, "Image dimensions mismatch"

        # Apply blur kernel to obtain smooth derivatives
        image_blur_kernel = GaussianBlur(5, 1.0)
        I = image_blur_kernel(I.unsqueeze(0)).squeeze()

        # Get keypoint values
        points = keypoints[:, 0:2] # x, y
        angle = keypoints[:, 2] # clockwise

        ######################################################################################################
        # TODO Q1: Implement the rotated hog descriptor                                                      #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Compute image derivatives
        I = I.float().unsqueeze(0).unsqueeze(0)

        sobel_filter_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]],dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_filter_y = torch.tensor([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]],dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        Ix = F.conv2d(I, sobel_filter_x).squeeze()
        Iy = F.conv2d(I, sobel_filter_y).squeeze()
        # Sample patches from image (16x16 squares)
        xs = points[:, 0]
        ys = points[:, 1]
        patch_size = 16
        patches = []
        valid_angles = []
        valid_keypoints = []
        for idx, (x, y) in enumerate(zip(xs, ys)):
            # Convert x and y to integers
            x_item = int(x.item())
            y_item = int(y.item())
            if x_item + patch_size < Ix.shape[0] and y_item + patch_size < Ix.shape[1]:
                # Extract the patch from Ix
                patch_x = Ix[x_item:x_item+patch_size, y_item:y_item+patch_size]
                # Extract the patch from Iy
                patch_y = Iy[x_item:x_item+patch_size, y_item:y_item+patch_size]
                patches.append([patch_x, patch_y])
                # Keep the angle
                valid_angles.append(angle[idx])
                valid_keypoints.append(keypoints[idx])
        valid_keypoints = torch.stack(valid_keypoints)
        # Sample smaller patches (4x4 squares) from 16x16 patch

        small_patch_size = 4
        small_patches = []

        for idx, patch_pair in enumerate(patches):
            patch_x, patch_y = patch_pair

            # Calculate the number of smaller patches in each dimension
            num_patches_x = patch_x.shape[0] // small_patch_size
            num_patches_y = patch_x.shape[1] // small_patch_size

            small_patches_per_patch = []

            for i in range(num_patches_x):
                for j in range(num_patches_y):
                    # Extract smaller patch from larger patch for both Ix and Iy
                    small_patch_x = patch_x[i * small_patch_size:(i + 1) * small_patch_size,
                                            j * small_patch_size:(j + 1) * small_patch_size]
                    small_patch_y = patch_y[i * small_patch_size:(i + 1) * small_patch_size,
                                            j * small_patch_size:(j + 1) * small_patch_size]

                    # Append the small patches to the list for each patch pair
                    small_patches_per_patch.append(torch.stack((torch.tensor(small_patch_x), torch.tensor(small_patch_y))))

            # Append the list of small patches to the main small_patches list
            small_patches_per_patch = torch.stack(small_patches_per_patch)
            small_patches.append(small_patches_per_patch)
        small_patches = torch.stack(small_patches)

        # Rotating the gradient patches
        rotated_patches_x = []
        rotated_patches_y = []
        for pat, ang in zip(small_patches, valid_angles):
            small_rotated_patches_x = []
            small_rotated_patches_y = []
            for small_pat in pat:
                rotation_matrix = torch.tensor([
                [math.cos(ang), -math.sin(ang), 0, 0],
                [math.sin(ang), math.cos(ang), 0, 0],
                [0, 0, math.cos(ang), -math.sin(ang)],
                [0, 0, math.sin(ang), math.cos(ang)]])
                small_rotated_patches_x.append(torch.matmul(rotation_matrix, torch.from_numpy(np.array(small_pat[0]))))
                small_rotated_patches_y.append(torch.matmul(rotation_matrix, torch.from_numpy(np.array(small_pat[1]))))
            small_rotated_patches_x = torch.stack(small_rotated_patches_x)
            small_rotated_patches_y = torch.stack(small_rotated_patches_y)
            rotated_patches_x.append(small_rotated_patches_x)
            rotated_patches_y.append(small_rotated_patches_y)
        rotated_patches_x = torch.stack(rotated_patches_x)
        rotated_patches_y = torch.stack(rotated_patches_y)



        # Assemble gradients from the 4x4 patches into orientation bins
        num_bins = 8
        bin_width = 360.0 / num_bins
        all_bins = []
        for i in range(len(rotated_patches_x)):
            patch_bins = []
            for j in range(16):
                bin = np.zeros(num_bins, dtype=np.float32)
                xs = rotated_patches_x[i,j].view(-1)
                ys = rotated_patches_y[i,j].view(-1)
                for x, y in zip(xs, ys):
                    angle_rad = math.atan2(x, y)
                    norm = math.sqrt(x**2 + y**2)
                    angle_deg = math.degrees(angle_rad)

                    # Calculate the orientation bin index
                    bin_idx = int(angle_deg / bin_width)
                    bin[bin_idx] += norm
                patch_bins.append(bin)
            all_bins.append(patch_bins)
        all_bins = np.array(all_bins)            
        # Slightly smooth along the orientation direction
        smoothed_bins = np.empty_like(all_bins)
        kernel = [0.05, 0.25, 0.4, 0.25, 0.05] 

        for i in range(all_bins.shape[0]):  
            for j in range(all_bins.shape[1]): 
                # Apply cyclic convolution dimension, 360 to 0
                smoothed_bins[i, j] = np.convolve(np.roll(all_bins[i, j], -2), kernel, mode='same')

        # Slightly smooth the orientation histograms spatially
        kernel = [0.05, 0.25, 0.4, 0.25, 0.05] 
        smoothed_bins = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=all_bins)

        # Assemble the histograms into a 128d vector
        descriptors = torch.tensor(smoothed_bins).reshape(-1, 128)  
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return descriptors, valid_keypoints


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np
    import cv2
    import os
    sys.path.append('..')
    from task2a.match import match

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

    hog = RHOG(seed = 0)
    desc1, _ = hog.compute_descriptors(torch.tensor(gray1, device=device), keypoints1)
    desc2, _ = hog.compute_descriptors(torch.tensor(gray2, device=device), keypoints2)

    matches = match(
        descriptors1=desc1,
        descriptors2=desc2,
        device=device,
        dist="euclidean",
        ratio=0.95,
        threshold=0, # Adjust value
    )

    np.savetxt("rhog.out", matches.numpy())

