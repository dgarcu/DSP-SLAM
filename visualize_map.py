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

import os
import argparse
import numpy as np
import open3d as o3d
from reconstruct.utils import color_table, get_configs


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--map_dir', type=str, required=True, help='path to map directory')
    parser.add_argument('-c', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('--display-voxel-size', type=float, default=None,
                        help='Voxel size for display-only downsampling (does not affect saved file)')
    parser.add_argument('--point-size', type=float, default=1.0,
                        help='Rendered point size (default: 1.0)')
    parser.add_argument('--height-axis', type=str, default='-y',
                        choices=['x', 'y', 'z', '-x', '-y', '-z'],
                        help='World axis used as height for coloring (default: -y for KITTI camera convention)')
    return parser


def voxel_downsample(points, voxel_size):
    indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique = np.unique(indices, axis=0, return_index=True)
    return points[unique]


def height_colormap(points, axis='-y'):
    """Color points by height axis: blue (low) → cyan → green → yellow → red (high)."""
    ax = {'x': 0, 'y': 1, 'z': 2, '-x': 0, '-y': 1, '-z': 2}[axis]
    z = points[:, ax] * (-1 if axis.startswith('-') else 1)
    t = (z - z.min()) / (z.max() - z.min() + 1e-9)

    colors = np.zeros((len(t), 3))
    bands = [
        (0.00, 0.25, lambda s: np.c_[np.zeros_like(s),               s,               np.ones_like(s)]),
        (0.25, 0.50, lambda s: np.c_[np.zeros_like(s),  np.ones_like(s),             1 - s]),
        (0.50, 0.75, lambda s: np.c_[s,                 np.ones_like(s),  np.zeros_like(s)]),
        (0.75, 1.00, lambda s: np.c_[np.ones_like(s),             1 - s,  np.zeros_like(s)]),
    ]
    for lo, hi, rgb_fn in bands:
        mask = (t >= lo) & (t < hi if hi < 1.0 else t <= hi)
        s = (t[mask] - lo) / (hi - lo)
        colors[mask] = rgb_fn(s)
    return colors


def set_view(vis, dist=100., theta=np.pi/6.):
    vis_ctr = vis.get_view_control()
    cam = vis_ctr.convert_to_pinhole_camera_parameters()
    # world to eye
    T = np.array([[1., 0., 0., 0.],
                  [0., np.cos(theta), -np.sin(theta), 0.],
                  [0., np.sin(theta), np.cos(theta), dist],
                  [0., 0., 0., 1.]])

    cam.extrinsic = T
    vis_ctr.convert_from_pinhole_camera_parameters(cam)


# 2D and 3D detection and data association
if __name__ == "__main__":
    parser = config_parser()
    parser.parse_args()
    args = parser.parse_args()
    objects_dir = os.path.join(args.map_dir, "objects")
    configs = get_configs(args.config)

    # Visualize results
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = args.point_size

    # Add geometries to map
    filenames = os.listdir(objects_dir)
    for item in filenames:
        if not item.endswith("ply"):
            continue
        obj_id = int(item.split(".")[0])
        mesh = o3d.io.read_triangle_mesh(os.path.join(objects_dir, "%d.ply" % obj_id))
        mesh.compute_vertex_normals()
        pose = np.load(os.path.join(objects_dir, "%d.npy" % obj_id))
        mesh.transform(pose)
        mesh.paint_uniform_color(color_table[obj_id % len(color_table)])
        vis.add_geometry(mesh)
    # Add background points — tries raw float32 binary, then .npy, then .txt
    bin_path = os.path.join(args.map_dir, "LidarPoints.bin")
    npy_path = os.path.join(args.map_dir, "LidarPoints.npy")
    txt_path = os.path.join(args.map_dir, "LidarPoints.txt")
    if os.path.exists(bin_path):
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3).astype(np.float64)
    elif os.path.exists(npy_path):
        pts = np.load(npy_path).astype(np.float64)
    else:
        with open(txt_path, "r") as f_pts:
            pts = np.array([[float(x) for x in line.strip().split()] for line in f_pts])
    # remove some extreme points
    xmin = 1.5 * np.percentile(pts[:, 0], 5)
    xmax = 1.5 * np.percentile(pts[:, 0], 95)
    ymin = 1.5 * np.percentile(pts[:, 1], 5)
    ymax = 1.5 * np.percentile(pts[:, 1], 95)
    zmin = 1.5 * np.percentile(pts[:, 2], 5)
    zmax = 1.5 * np.percentile(pts[:, 2], 95)
    mask = (pts[:, 0] > xmin) & (pts[:, 0] < xmax) & (pts[:, 1] > ymin) & (pts[:, 1] < ymax) & (pts[:, 2] > zmin) & (pts[:, 2] < zmax)
    pts = pts[mask, :]
    if args.display_voxel_size is not None:
        before = len(pts)
        pts = voxel_downsample(pts, args.display_voxel_size)
        print(f"Display downsampling: {before:,} → {len(pts):,} points")
    colors = height_colormap(pts, axis=args.height_axis)
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(pts)
    map_pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(map_pcd)
    # Add coordinate frame for reference
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(configs.viewer.frame_size, np.array([0., 0., 0.]))
    vis.add_geometry(coor_frame)
    # must be put after adding geometries
    set_view(vis, dist=configs.viewer.distance, theta=configs.viewer.tilt * np.pi / 180)
    vis.run()
    vis.destroy_window()