#!/usr/bin/env python3
"""
visualize_white_region.py

Visualizes a COLMAP point cloud with a bounding box highlighting the 
predominantly white/light region. The bounding box is rendered in deep orange.

White points are detected as regions where R, G, B channels are all high (>threshold)
and similar to each other (low variance).

Usage:
    python visualize_white_region.py --file points3D.txt
    python visualize_white_region.py --file points3D.txt --white-threshold 200 --save-visualization output.png
"""
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt

try:
    import open3d as o3d
except Exception:
    o3d = None
    print("Warning: open3d not available — 3D visualization will be disabled.", file=sys.stderr)


# -------------------------
# COLMAP file parsing
# -------------------------
def parse_colmap_points3d(file_path):
    """Parse COLMAP points3D.txt file."""
    pids = []
    pts = []
    cols = []
    with open(file_path, 'r') as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) < 7:
                continue
            pid = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            pids.append(pid)
            pts.append([x, y, z])
            cols.append([r, g, b])
    
    if len(pts) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.uint8)
    
    return np.array(pids, dtype=np.int32), np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)


# -------------------------
# White point detection
# -------------------------
def detect_white_points(colors_uint8, threshold=200):
    """
    Detect white/light points in the point cloud.
    White points have R, G, B all high and close to each other.
    
    Args:
        colors_uint8: Nx3 array of RGB colors (uint8)
        threshold: Minimum value for white detection (default: 200/255)
    
    Returns:
        white_mask: Boolean mask for white points
        white_indices: Indices of white points
    """
    r = colors_uint8[:, 0].astype(np.float32)
    g = colors_uint8[:, 1].astype(np.float32)
    b = colors_uint8[:, 2].astype(np.float32)
    
    # All channels must be high
    high_channels = (r > threshold) & (g > threshold) & (b > threshold)
    
    # Channels must be similar (low variance across RGB)
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    channel_variance = max_channel - min_channel
    similar_channels = channel_variance < 30  # Allow small variation
    
    white_mask = high_channels & similar_channels
    white_indices = np.where(white_mask)[0]
    
    return white_mask, white_indices


def get_white_oriented_bounding_box(coords, white_indices):
    """
    Compute oriented bounding box (OBB) for white points.
    
    Args:
        coords: Nx3 array of all point coordinates
        white_indices: Indices of white points
    
    Returns:
        obb: Open3D OrientedBoundingBox object
        white_region_mask: Boolean mask for points within the OBB
    """
    if len(white_indices) == 0:
        return None, None
    
    white_coords = coords[white_indices]
    
    # Create point cloud for white points
    pcd_white = o3d.geometry.PointCloud()
    pcd_white.points = o3d.utility.Vector3dVector(white_coords)
    
    # Get oriented bounding box
    obb = pcd_white.get_oriented_bounding_box()
    
    # Manual point containment check
    # Transform all points to OBB's local coordinate system
    obb_center = np.asarray(obb.center)
    obb_extent = np.asarray(obb.extent)  # Half-extents
    obb_R = np.asarray(obb.R)  # Rotation matrix
    
    # Translate and rotate points to OBB frame
    points_local = np.dot(coords - obb_center, obb_R)
    
    # Check if points are within half-extents
    white_region_mask = np.all(
        np.abs(points_local) <= obb_extent / 2.0,
        axis=1
    )
    
    return obb, white_region_mask


# -------------------------
# Bounding box visualization
# -------------------------
def create_obb_wireframe(obb, color=(1.0, 0.55, 0.0)):
    """
    Create a wireframe representation of an oriented bounding box.
    
    Args:
        obb: Open3D OrientedBoundingBox object
        color: RGB color tuple (default: deep orange)
    
    Returns:
        line_set: Open3D LineSet object
    """
    if obb is None:
        return None
    
    # Get the 8 corners of the OBB
    corners = np.asarray(obb.get_box_points())
    
    # Define the 12 edges connecting the corners
    edges = [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.paint_uniform_color(color)
    
    return line_set


def create_obb_solid_mesh(obb, color=(1.0, 0.55, 0.0)):
    """
    Create a semi-transparent solid OBB mesh for visualization.
    
    Args:
        obb: Open3D OrientedBoundingBox object
        color: RGB color tuple (default: deep orange)
    
    Returns:
        mesh: Open3D TriangleMesh object
    """
    if obb is None:
        return None
    
    # Get the 8 corners
    corners = np.asarray(obb.get_box_points())
    
    # Define 12 triangles (2 per face) to form the box
    triangles = np.array([
        # Bottom face (z=min)
        [0, 1, 2], [0, 2, 3],
        # Top face (z=max)
        [4, 6, 5], [4, 7, 6],
        # Front face (y=min)
        [0, 5, 1], [0, 4, 5],
        # Back face (y=max)
        [2, 7, 3], [2, 6, 7],
        # Left face (x=min)
        [0, 3, 7], [0, 7, 4],
        # Right face (x=max)
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Paint with deep orange
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    
    return mesh


def visualize_white_region_obb(all_coords, all_colors, obb, white_coords=None, 
                               use_solid_bbox=True, show_white_points=True):
    """
    Visualize point cloud with oriented bounding box around white region.
    
    Args:
        all_coords: Nx3 array of all point coordinates
        all_colors: Nx3 array of all RGB colors
        obb: Open3D OrientedBoundingBox object
        white_coords: Optional Nx3 array of white point coordinates to highlight
        use_solid_bbox: If True, use solid semi-transparent box; if False, use wireframe
        show_white_points: If True, display the white points separately
    """
    if o3d is None:
        print("Open3D not available; skipping 3D visualization.", file=sys.stderr)
        return
    
    # Create point cloud for all points
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(all_coords)
    pcd_all.colors = o3d.utility.Vector3dVector(all_colors.astype(float) / 255.0)
    
    geometries = [pcd_all]
    
    # Add white points if available and requested
    if show_white_points and white_coords is not None and len(white_coords) > 0:
        pcd_white = o3d.geometry.PointCloud()
        pcd_white.points = o3d.utility.Vector3dVector(white_coords)
        pcd_white.paint_uniform_color([0.9, 0.9, 0.9])  # Light gray
        geometries.append(pcd_white)
    
    # Create bounding box (deep orange: RGB 1.0, 0.55, 0.0)
    deep_orange = (1.0, 0.55, 0.0)
    
    if use_solid_bbox:
        bbox_mesh = create_obb_solid_mesh(obb, color=deep_orange)
        if bbox_mesh is not None:
            geometries.append(bbox_mesh)
    else:
        bbox_lines = create_obb_wireframe(obb, color=deep_orange)
        if bbox_lines is not None:
            geometries.append(bbox_lines)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(coord_frame)
    
    # Visualize
    window_title = "White Region Oriented Bounding Box (Deep Orange)"
    if show_white_points and white_coords is not None:
        window_title += " | Light gray: White points"
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_title,
        width=1400,
        height=900,
        left=50,
        top=50
    )


def print_white_region_statistics(coords, white_coords, obb, white_region_mask):
    """Print detailed statistics about the white region."""
    print("\n" + "="*60)
    print("WHITE REGION ANALYSIS")
    print("="*60)
    
    print(f"\nTotal points in cloud: {len(coords)}")
    print(f"White points detected: {len(white_coords)} ({100*len(white_coords)/len(coords):.2f}%)")
    
    num_in_region = np.sum(white_region_mask)
    print(f"Points in OBB region: {num_in_region} ({100*num_in_region/len(coords):.2f}%)")
    
    print(f"\nOriented Bounding Box (OBB) Statistics:")
    obb_center = np.asarray(obb.center)
    obb_extent = np.asarray(obb.extent)
    print(f"  Center: [{obb_center[0]:.6f}, {obb_center[1]:.6f}, {obb_center[2]:.6f}]")
    print(f"  Extent (Size): [{obb_extent[0]:.6f}, {obb_extent[1]:.6f}, {obb_extent[2]:.6f}]")
    volume = np.prod(obb_extent)
    print(f"  Volume: {volume:.2f}")
    
    print("="*60 + "\n")


def plot_color_distribution(colors, white_mask, save_path=None):
    """
    Plot color distribution showing detected white points.
    
    Args:
        colors: Nx3 array of RGB colors
        white_mask: Boolean mask for white points
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    white_colors = colors[white_mask]
    non_white_colors = colors[~white_mask]
    
    # Red channel
    axes[0].hist(non_white_colors[:, 0], bins=50, alpha=0.7, label='Non-white', color='red')
    if len(white_colors) > 0:
        axes[0].hist(white_colors[:, 0], bins=50, alpha=0.7, label='White', color='orange')
    axes[0].set_xlabel('Red Channel Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Red Channel Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Green channel
    axes[1].hist(non_white_colors[:, 1], bins=50, alpha=0.7, label='Non-white', color='green')
    if len(white_colors) > 0:
        axes[1].hist(white_colors[:, 1], bins=50, alpha=0.7, label='White', color='orange')
    axes[1].set_xlabel('Green Channel Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Green Channel Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Blue channel
    axes[2].hist(non_white_colors[:, 2], bins=50, alpha=0.7, label='Non-white', color='blue')
    if len(white_colors) > 0:
        axes[2].hist(white_colors[:, 2], bins=50, alpha=0.7, label='White', color='orange')
    axes[2].set_xlabel('Blue Channel Value')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Blue Channel Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved color distribution plot to: {save_path}")
    
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize COLMAP point cloud with a bounding box around the white region"
    )
    parser.add_argument('--file', '-f', required=True, type=str,
                       help="Path to COLMAP points3D.txt file")
    parser.add_argument('--white-threshold', type=int, default=200,
                       help="RGB threshold for white point detection (0-255, default: 200)")
    parser.add_argument('--solid-bbox', action='store_true', default=True,
                       help="Use solid semi-transparent bounding box (default: True)")
    parser.add_argument('--wireframe-bbox', action='store_true',
                       help="Use wireframe bounding box instead of solid")
    parser.add_argument('--hide-white-points', action='store_true',
                       help="Hide the white points in visualization (show only bbox)")
    parser.add_argument('--save-visualization', type=str,
                       help="Save color distribution plot to file")
    parser.add_argument('--output', '-o', type=str,
                       help="Save all points within OBB region to COLMAP points3D.txt format")
    parser.add_argument('--stats-only', action='store_true',
                       help="Print statistics only, skip 3D visualization")
    
    args = parser.parse_args()
    
    # Parse point cloud
    print(f"Loading point cloud from: {args.file}")
    ids, coords, colors = parse_colmap_points3d(args.file)
    
    if len(coords) == 0:
        print("ERROR: No points found in file.")
        sys.exit(1)
    
    print(f"Loaded {len(coords)} points")
    
    # Detect white points
    print(f"\nDetecting white points (threshold: {args.white_threshold})...")
    white_mask, white_indices = detect_white_points(colors, threshold=args.white_threshold)
    
    if len(white_indices) == 0:
        print("ERROR: No white points detected. Try lowering --white-threshold")
        sys.exit(1)
    
    white_coords = coords[white_indices]
    
    # Compute oriented bounding box
    print("Computing white region oriented bounding box (OBB)...")
    obb, white_region_mask = get_white_oriented_bounding_box(coords, white_indices)
    
    # Print statistics
    print_white_region_statistics(coords, white_coords, obb, white_region_mask)
    
    # Save points in OBB region if requested
    if args.output:
        print(f"\nSaving points within OBB to: {args.output}")
        region_indices = np.where(white_region_mask)[0]
        region_ids = ids[region_indices]
        region_coords = coords[region_indices]
        region_colors = colors[region_indices]
        
        with open(args.output, 'w') as f:
            f.write("# COLMAP points3D.txt (extracted from white region OBB)\n")
            f.write(f"# Total points: {len(region_ids)}\n")
            for pid, coord, color in zip(region_ids, region_coords, region_colors):
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                f.write(f"{pid} {coord[0]} {coord[1]} {coord[2]} {r} {g} {b} 0\n")
        print(f"Saved {len(region_ids)} points to {args.output}")
    
    # Plot color distribution if requested
    if args.save_visualization or not args.stats_only:
        if args.save_visualization:
            plot_color_distribution(colors, white_mask, save_path=args.save_visualization)
        else:
            plot_color_distribution(colors, white_mask)
    
    # 3D Visualization
    if not args.stats_only and o3d is not None:
        print("Launching 3D visualization...")
        use_solid = not args.wireframe_bbox
        show_white = not args.hide_white_points
        visualize_white_region_obb(coords, colors, obb, 
                                   white_coords=white_coords if show_white else None,
                                   use_solid_bbox=use_solid, 
                                   show_white_points=show_white)


if __name__ == '__main__':
    main()
