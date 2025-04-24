# 3D Computer Vision Assignments

This repository contains the code and resources for the assignments completed as part of the 3D Computer Vision course at Saarland University. Each assignment focuses on a specific topic in 3D computer vision, ranging from feature extraction to camera calibration and 3D reconstruction.

## Assignments Overview

1. **Assignment 1 - Feature Extraction and Matching**  
   Focuses on extracting features from images and matching them across multiple views.  
   ![Assignment 1](images/assignment1/matches.png)

2. **Assignment 2 - Transforms, Homogeneous Notation, and Projection**  
   Explores geometric transformations, homogeneous coordinates, and projection techniques.  
   ![Assignment 2](images/assignment2/transformations1.png)

3. **Assignment 3 - Camera Calibration**  
   Covers intrinsic and extrinsic camera calibration using multiple views.  
   ![Assignment 3](images/assignment3/camera_calibration2.png)

4. **Assignment 4 - Semi-Global Matching (SGM)**  
   Implements the SGM algorithm for calculating the depth of each elementin the image based on two images taken from different points of view.
   ![Assignment 4](images/assignment4/reference.png)  
   ![Assignment 4](images/assignment4/disparity.png)

5. **Assignment 5 - FlowNet**  
   Focuses on optical flow estimation using deep learning techniques. This involves training a model to estimate the motion of elements between two images captured at different time points.
   ![Assignment 5](images/assignment5/image_sec1.png)  
   ![Assignment 5](images/assignment5/optical_flow.png)

6. **Assignment 6 - COLMAP**  
   Utilizes COLMAP for 3D reconstruction from multi-view images.

## Installation

To execute the code for any assignment, first install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Each assignment folder contains a Jupyter notebook where the results of the implemented code can be visualized. Navigate to the respective assignment folder and open the notebook to explore the results.
