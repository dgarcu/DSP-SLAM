"""
Accumulate KITTI velodyne scans into a globally-registered point cloud
using DSP-SLAM camera poses, with voxel grid downsampling.

Cameras.txt format: each line is a 3x4 [R|t] matrix (camera-to-world), row-major.
calib.txt Tr: 3x4 velodyne-to-camera transform.

Output format: raw float32 binary, 3 values (x, y, z) per point.
Load with: np.fromfile(path, dtype=np.float32).reshape(-1, 3)
"""

import numpy as np
import os
import sys
import argparse
import time


def load_calib_tr(calib_path):
    with open(calib_path) as f:
        for line in f:
            if line.startswith("Tr:"):
                vals = list(map(float, line.strip().split()[1:]))
                Tr = np.eye(4)
                Tr[:3, :] = np.array(vals).reshape(3, 4)
                return Tr
    raise RuntimeError("Tr not found in calib.txt")


def load_poses(cameras_path):
    poses = []
    with open(cameras_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def load_velodyne(bin_path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # drop intensity


def format_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds // 60:.0f}m{seconds % 60:.0f}s"
    return f"{seconds // 3600:.0f}h{(seconds % 3600) // 60:.0f}m"


def format_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def format_pts(n):
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n/1e3:.1f}K"
    return f"{n/1e6:.1f}M"


# float32 x, y, z = 3 * 4 bytes per point.
_BYTES_PER_POINT = 12
_WRITE_EVERY_BYTES = 50 * 1024 * 1024  # flush to disk every 50 MB of new points


def accumulate(velodyne_dir, poses, Tr, voxel_size, output_path,
               max_frames=None, flush_every=50):
    buffer = []
    points = np.empty((0, 3), dtype=np.float32)
    pending = []           # new points not yet written to disk
    pending_bytes = 0
    disk_bytes = 0

    n = len(poses) if max_frames is None else min(max_frames, len(poses))
    print(f"Accumulating {n} frames (voxel_size={voxel_size}m, flush_every={flush_every}) ...")

    t_start = time.time()
    processed = 0
    total_raw_pts = 0

    with open(output_path, "wb") as f_out:
        for i in range(n):
            bin_path = os.path.join(velodyne_dir, f"{i:06d}.bin")
            if not os.path.exists(bin_path):
                continue

            pts = load_velodyne(bin_path)

            ones = np.ones((len(pts), 1))
            pts_h = np.hstack([pts, ones])

            T_world = poses[i] @ Tr
            pts_world = (T_world @ pts_h.T).T[:, :3].astype(np.float32)

            buffer.append(pts_world)
            processed += 1
            total_raw_pts += len(pts_world)

            is_last = (i == n - 1)
            if processed % flush_every == 0 or is_last:
                n_existing = len(points)
                merged = np.vstack([points] + buffer)
                voxel_idx = np.floor(merged / voxel_size).astype(np.int64)
                _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
                points = merged[unique_idx]
                # Only points that came from the buffer and survived voxel filtering
                new_pts = merged[unique_idx[unique_idx >= n_existing]]
                buffer = []

                pending.append(new_pts)
                pending_bytes += len(new_pts) * _BYTES_PER_POINT

                if pending_bytes >= _WRITE_EVERY_BYTES or is_last:
                    for chunk in pending:
                        f_out.write(chunk.tobytes())
                    f_out.flush()
                    disk_bytes = os.path.getsize(output_path)
                    pending = []
                    pending_bytes = 0

            elapsed = time.time() - t_start
            fps = processed / elapsed
            remaining = (n - i - 1) / fps if fps > 0 else 0
            pct = (i + 1) / n * 100
            bar = "#" * (int(pct) // 2) + "-" * (50 - int(pct) // 2)

            avg_pts = total_raw_pts / processed
            est_total_pts = int(avg_pts * n)
            est_size = format_size(est_total_pts * _BYTES_PER_POINT)

            print(
                f"\r[{bar}] {pct:5.1f}%  frame {i+1}/{n}  {fps:.1f} fr/s  ETA {format_eta(remaining)}"
                f"  pts {format_pts(len(points))}/~{format_pts(est_total_pts)}"
                f"  disk {format_size(disk_bytes)}/~{est_size}",
                end="", flush=True,
            )

    print()  # newline after progress bar
    print(f"Done. {len(points):,} points → {format_size(os.path.getsize(output_path))}")
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--velodyne-dir", required=True,
                        help="Directory containing .bin velodyne scans")
    parser.add_argument("--calib", required=True,
                        help="Path to calib.txt")
    parser.add_argument("--cameras", required=True,
                        help="Path to Cameras.txt (DSP-SLAM poses)")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="Voxel grid leaf size in metres (default: 0.1)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit total frames processed")
    parser.add_argument("--output", type=str, required=True,
                        help="Output point cloud file (raw float32 binary)")
    args = parser.parse_args()

    if os.path.exists(args.output):
        ans = input(f"'{args.output}' already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    Tr    = load_calib_tr(args.calib)
    poses = load_poses(args.cameras)
    print(f"Loaded {len(poses)} poses")

    accumulate(args.velodyne_dir, poses, Tr,
               voxel_size=args.voxel_size, max_frames=args.max_frames,
               output_path=args.output)
