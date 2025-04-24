#!/usr/bin/env python3
import torch
import numpy as np


def match(descriptors1, descriptors2, device, dist="norm2", threshold=0, ratio=0.5):
    """
    Brute-force descriptor match with Lowe tests and cross consistency check.


    Inputs:
    - descriptors1: tensor(N, feature_size),the descriptors of a keypoint
    - descriptors2: tensor(N, feature_size),the descriptors of a keypoint
    - device: where a torch.Tensor is or will be allocated
    - dist: distance metrics, hamming distance for measuring binary descriptor, and norm-2 distance for others
    - threshold: threshold for first Lowe test
    - ratio: ratio for second Lowe test

    Returns:
    - matches: tensor(M, 2), indices of corresponding matches in first and second set of descriptors,
      where matches[:, 0] denote the indices in the first and
      matches[:, 1] the indices in the second set of descriptors.
    """

    # Exponent for norm
    if dist == "hamming":
        p = 0
    else:
        p = 2.0

    ######################################################################################################
    # TODO Q1: Find the indices of corresponding matches                                                 #
    # See slide 48 of lecture 2 part A                                                                   #
    # Use cross-consistency checking and first and second Lowe test                                      #
    ######################################################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute distances
    # NOTE: you may use torch.cdist with p=p
    distances = torch.cdist(descriptors1, descriptors2, p=p)

    distances_T = distances.t()
    min_distance_T, min_indices_T = distances_T.min(dim=1)
    min_distance, min_indices = distances.min(dim=1)
    # Perform first and second lowe test
    # First lowe test
    matches1 = torch.nonzero(min_distance < threshold)

    # Second lowe tests
    sorted_distances, sorted_indices = distances.sort(dim=1)
    matches2 = torch.nonzero(sorted_distances[:, 0] < ratio * sorted_distances[:, 1])
    
    # Keep the matches that pass both lowe tests
    matches1, matches2 = matches1.numpy(), matches2.numpy()
    valid_matches = np.intersect1d(matches1, matches2)
    valid_matches = torch.tensor(valid_matches)
    # print(valid_matches)
    # Forward backward consistency check
    consistent_matches = []
    for match in valid_matches:
        if min_indices_T[min_indices[match.item()]] == match.item():
            consistent_matches.append([match, min_indices[match.item()]])
    
    # Sort matches using distances from best to worst
    sorted_matches = sorted(consistent_matches, key=lambda x: distances[x[0].item(), sorted_indices[x[0].item(), 0]])
    matches = torch.tensor(sorted_matches)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return matches


if __name__ == "__main__":
    # test your match function under here by using provided image, keypoints, and descriptors
    import numpy as np
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img1 = cv2.imread("../data/Chess.png")
    color1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread("../data/ChessRotated.png")
    color2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

    keypoints1 = np.loadtxt("./keypoints1.in")
    keypoints2 = np.loadtxt("./keypoints2.in")
    keypoints1 = torch.tensor(keypoints1, device=device)
    keypoints2 = torch.tensor(keypoints2, device=device)

    descriptors1 = np.loadtxt("./descriptors1.in")
    descriptors2 = np.loadtxt("./descriptors2.in")
    descriptors1 = torch.tensor(descriptors1, device=device)
    descriptors2 = torch.tensor(descriptors2, device=device)

    matches = match(
        descriptors1=descriptors1,
        descriptors2=descriptors2,
        device=device,
        dist="hamming",
        ratio=0.95,
        threshold=160,
    )

    np.savetxt("./output_matches.out", matches.cpu().numpy())
