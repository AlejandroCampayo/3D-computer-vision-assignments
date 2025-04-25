#!/usr/bin/env python3

import numpy as np
import json
from PIL import Image
import cv2
import trimesh
from util import visualize_poses

import os
import sys
from natsort import natsorted
import matplotlib.pyplot as plt
from util import get_pose_json, visualize_poses, get_pointcloud

# Paths
# images_path = "images"
# workspace_path = "workspace"
images_path = "3DCV\Assignment_6_-_COLMAP\Assignment6\images"
workspace_path = "3DCV\Assignment_6_-_COLMAP\Assignment6\workspace"
images_list = natsorted(os.listdir(images_path))
pose_json_path = f"{workspace_path}/poses.json"
print(f"Workspace path: {workspace_path}")
print(f"Images path: {images_path}")

# Height and width
h, w = 640, 512

# Create scene for visualization
scene = trimesh.Scene()

# Function to extract transforms from JSON
def get_cam(cam_file, idx):
        cam = cam_file["frames"][idx]
        c2w = np.array(cam["transform_matrix"])
        w2c = np.linalg.inv(c2w)
        return c2w, w2c

# read json file
with open(pose_json_path) as f:
    out = json.load(f)

nframes = len(out["frames"])
poses_original = [get_cam(out, idx) for idx in range(nframes)]
pointcloud, colors = get_pointcloud(out["pointcloud_path"])
poses_vis = visualize_poses(poses_original)
scene = trimesh.Scene()

scene.add_geometry(trimesh.PointCloud(vertices=pointcloud, colors=colors))
scene.add_geometry(poses_vis)

scene.show()




