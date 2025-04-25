#!/usr/bin/env python3

import os
from util import get_pose_json

# Used to show commands when they are exectued
def run(cmd):
    print("Running <%s>" % cmd)
    os.system(cmd)

# Paths
images_path = "3DCV\Assignment_6_-_COLMAP\Assignment6\images"
workspace_path = "3DCV\Assignment_6_-_COLMAP\Assignment6\workspace"
print(f"Workspace path: {workspace_path}")
print(f"Images path: {images_path}")

######################################################################################################
# Part1 Q1) Run COLMAP to estimate intrinsics and extrinsics from multiview images
######################################################################################################

# ***** START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#run() 

# ***** END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# You should now have the following files:
# workspace/sparse/0/cameras.bin
# workspace/sparse/0/cameras.txt
# workspace/sparse/0/images.bin
# workspace/sparse/0/images.txt
# workspace/sparse/0/points3D.bin
# workspace/sparse/0/points3D.txt

# Convert the poses to JSON to for easy use with the visualization
# This is a common practice in modern NeRF repos
# The code was taken from: https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py
get_pose_json(workspace_path, f"{workspace_path}/poses.json")
