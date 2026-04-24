#
# This file is part of https://github.com/JingwenWang95/DSP-SLAM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import numpy as np
import os
import cv2
import torch
from reconstruct.loss_utils import get_rays, get_time
from reconstruct.utils import ForceKeyErrorDict
from reconstruct import get_detectors

class RGBDFrame:
    def __init__(self, sequence, frame_id):
        # Load sequence properties
        self.configs = sequence.configs
        self.K = sequence.K_cam
        self.invK = sequence.invK_cam
        self.k1 = sequence.k1
        self.k2 = sequence.k2
        self.depth_factor = sequence.depth_factor
        self.online = sequence.online
        self.detector_2d = sequence.detector_2d
        self.min_mask_area = self.configs.min_mask_area
        self.object_class = self.configs.get("object_class", "chairs")
        self.frame_id = frame_id
        
        rgb_file = os.path.join(sequence.root_dir, sequence.vstrImageFilenamesRGB[frame_id])
        depth_file = os.path.join(sequence.root_dir, sequence.vstrImageFilenamesD[frame_id])
        
        self.img_bgr = cv2.imread(rgb_file)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w, _ = self.img_rgb.shape
        self.depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        self.instances = []

    def get_detections(self):
        # Get 2D Detection
        t1 = get_time()
        det_2d = self.detector_2d.make_prediction(self.img_bgr, object_class=self.object_class)
        t2 = get_time()
        print("2D detector takes %f seconds" % (t2 - t1))

        masks_2d = det_2d["pred_masks"]
        bboxes_2d = det_2d["pred_boxes"]

        # If no 2D detections, return right away
        if masks_2d.shape[0] == 0:
            return

        for i in range(masks_2d.shape[0]):
            mask = masks_2d[i, ...].astype(np.bool8)
            bbox = bboxes_2d[i, ...]
            
            # Extract depth within the mask
            depth_mask = mask & (self.depth_img > 0)
            if np.sum(depth_mask) < self.min_mask_area:
                continue
                
            depth_values = self.depth_img[depth_mask].astype(np.float32) / self.depth_factor
            
            # Get pixel coordinates
            vv, uu = np.where(depth_mask)
            pixels = np.stack([uu, vv], axis=1).astype(np.float32)
            
            # Undistort pixels
            distortion_coef = np.array([self.k1, self.k2, 0.0, 0.0, 0.0])
            pixels_undist = cv2.undistortPoints(pixels.reshape(1, -1, 2), self.K, distortion_coef, P=self.K).squeeze()
            
            # Back-project to 3D
            fx, fy = self.K[0,0], self.K[1,1]
            cx, cy = self.K[0,2], self.K[1,2]
            
            x = (pixels_undist[:, 0] - cx) * depth_values / fx
            y = (pixels_undist[:, 1] - cy) * depth_values / fy
            z = depth_values
            
            surface_points = np.stack([x, y, z], axis=1)
            
            # Compute centroid and PCA for initial pose
            centroid = np.mean(surface_points, axis=0)
            centered_points = surface_points - centroid
            cov = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort eigenvectors by eigenvalues descending
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:,idx]
            
            # Ensure Z axis points roughly towards camera (or away depending on convention)
            if eigenvectors[2, 2] < 0:
                eigenvectors[:, 2] = -eigenvectors[:, 2]
                eigenvectors[:, 1] = -eigenvectors[:, 1]
            
            T_cam_obj = np.eye(4, dtype=np.float32)
            T_cam_obj[:3, :3] = eigenvectors
            T_cam_obj[:3, 3] = centroid
            
            # No rays needed for RGBD, depth is available directly
            rays = None
            depth = None

            instance = ForceKeyErrorDict()
            instance.bbox = bbox
            instance.mask = mask.astype(np.float32) * 255.
            instance.surface_points = surface_points.astype(np.float32)
            instance.T_cam_obj = T_cam_obj
            instance.rays = rays
            instance.depth = depth

            self.instances.append(instance)

class RGBDSequence:
    def __init__(self, data_dir, configs):
        self.root_dir = data_dir

        # Get camera intrinsics
        fs = cv2.FileStorage(configs.slam_config_path, cv2.FILE_STORAGE_READ)
        fx = fs.getNode("Camera.fx").real()
        fy = fs.getNode("Camera.fy").real()
        cx = fs.getNode("Camera.cx").real()
        cy = fs.getNode("Camera.cy").real()
        k1 = fs.getNode("Camera.k1").real()
        k2 = fs.getNode("Camera.k2").real()
        self.depth_factor = fs.getNode("DepthMapFactor").real()
        if self.depth_factor == 0:
            self.depth_factor = 5000.0
            
        self.K_cam = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.invK_cam = np.linalg.inv(self.K_cam)
        self.k1 = k1
        self.k2 = k2

        self.configs = configs
        self.data_type = self.configs.data_type
        self.online = self.configs.detect_online
        
        self.vstrImageFilenamesRGB = []
        self.vstrImageFilenamesD = []
        
        # Parse associations.txt
        assoc_path = os.path.join(self.root_dir, "associations.txt")
        with open(assoc_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        self.vstrImageFilenamesRGB.append(parts[1])
                        self.vstrImageFilenamesD.append(parts[3])

        # Detectors
        self.detector_2d = get_detectors(self.configs)
        self.current_frame = None
        self.detections_in_current_frame = None

    def get_frame_by_id(self, frame_id):
        self.current_frame = RGBDFrame(self, frame_id)
        self.current_frame.get_detections()
        self.detections_in_current_frame = self.current_frame.instances
        return self.detections_in_current_frame
