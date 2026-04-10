import sys
import numpy as np
try:
    import open3d as o3d
except Exception:
    o3d = None
    print("Warning: open3d not available — 3D visualization will be disabled.", file=sys.stderr)

def visualize_point_cloud(points, colors=None):
    if o3d is None:
        print("3D visualization is disabled because open3d is not available.", file=sys.stderr)
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")

def parse_colmap_file(file_path):
    points = []
    colors = []
    with open(file_path, 'r') as f:
        for line in f:
            ln = line.strip()
            if ln.startswith('#') or not ln:
                continue
            parts = ln.split()
            if len(parts) < 7:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            points.append([x, y, z])
            colors.append([r, g, b])   
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize COLMAP point cloud.")
    parser.add_argument('--file','-f',required=True, type=str, help="Path to COLMAP points3D.txt file")
    args = parser.parse_args()

    pts, cols = parse_colmap_file(args.file)
    visualize_point_cloud(pts, cols)    