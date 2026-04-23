#!/usr/bin/env python3
"""
visualize_white_region.py

Visualizes a COLMAP point cloud with a bounding box highlighting the 
predominantly white/light region. The bounding box is rendered in deep orange.

White points are detected as regions where R, G, B channels are all high (>threshold).
To ignore outlier white points the OBB is fitted only to the *densest* cluster of white
points (using the same voxel-density approach as visualize_point_cloud_enhanced.py).
The output file contains ALL points — white or not — that fall inside that OBB.

Usage:
    python visualize_white_region.py --file points3D.txt
    python visualize_white_region.py --file points3D.txt --white-threshold 200 --save-visualization output.png
    python visualize_white_region.py --file points3D.txt --output region.txt --voxel-size 0.03 --density-percentile 80
    python visualize_white_region.py --file points3D.txt --recolor-white red --recolor-output recolored.txt
    python visualize_white_region.py --file points3D.txt --recolor-white 255,165,0 --recolor-output recolored.txt
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
    
    white_mask = high_channels #& similar_channels
    white_indices = np.where(white_mask)[0]
    
    return white_mask, white_indices


def detect_dense_white_obb(all_coords, white_indices, voxel_size=0.05, density_percentile=75):
    """
    Find the densest spatial cluster of white points via voxelization, fit an OBB
    to that cluster, then return a mask covering ALL points (white or not) inside
    the OBB.

    The voxel-density approach mirrors visualize_point_cloud_enhanced.py:
      1. Partition white points into voxels of side `voxel_size`.
      2. Keep only voxels whose point count is >= the `density_percentile` threshold.
      3. Build the OBB from those dense-voxel white points — outlier white points
         that land in sparse voxels are excluded, so they cannot inflate the box.
      4. Test ALL cloud points for containment in the OBB and return the boolean mask.

    Args:
        all_coords:        Nx3 float array of every point's XYZ
        white_indices:     Indices (into all_coords) of detected white points
        voxel_size:        Side length of each voxel (same units as the point cloud)
        density_percentile: Percentile of per-voxel counts used as the density cutoff

    Returns:
        obb:               Open3D OrientedBoundingBox fitted to the dense cluster
        obb_mask:          Boolean mask (len == N) — True for every point inside the OBB
        dense_white_coords: Coordinates of white points used to build the OBB
        density_info:      Dict of diagnostic statistics
    """
    if len(white_indices) == 0:
        return None, None, None, {}

    white_coords = all_coords[white_indices]

    # --- Step 1: voxelise the white points ---
    min_coords = white_coords.min(axis=0)
    voxel_idx_array = np.floor((white_coords - min_coords) / voxel_size).astype(int)

    voxel_counts = {}
    voxel_point_map = {}          # voxel key → list of local indices into white_coords
    for local_i, vox_idx in enumerate(voxel_idx_array):
        key = tuple(vox_idx)
        if key not in voxel_counts:
            voxel_counts[key] = 0
            voxel_point_map[key] = []
        voxel_counts[key] += 1
        voxel_point_map[key].append(local_i)

    counts = np.array(list(voxel_counts.values()))
    density_threshold = np.percentile(counts, density_percentile)

    print(f"\nWhite-point voxelization:"
          f"\n  Voxel size:           {voxel_size}"
          f"\n  Total voxels:         {len(voxel_counts)}"
          f"\n  Counts  min/mean/max: {counts.min():.0f} / {counts.mean():.1f} / {counts.max():.0f}"
          f"\n  Density threshold ({density_percentile}th pct): {density_threshold:.1f} pts/voxel")

    # --- Step 2: keep high-density voxels ---
    high_density_keys = [k for k, c in voxel_counts.items() if c >= density_threshold]

    if not high_density_keys:
        print("  ⚠  No high-density voxels found — falling back to all white points.")
        high_density_keys = list(voxel_counts.keys())

    # Gather white-point local indices from those voxels
    dense_local_indices = []
    for key in high_density_keys:
        dense_local_indices.extend(voxel_point_map[key])
    dense_local_indices = np.array(dense_local_indices)

    dense_white_coords = white_coords[dense_local_indices]
    pct_kept = 100.0 * len(dense_local_indices) / len(white_indices)

    print(f"  Dense cluster:        {len(dense_local_indices)} / {len(white_indices)} "
          f"white points ({pct_kept:.1f}%) across {len(high_density_keys)} voxels")

    if pct_kept > 95:
        print(f"  ⚠  WARNING: keeping {pct_kept:.1f}% of white points — the density filter "
              f"has little effect. Try increasing --density-percentile or decreasing --voxel-size.")

    # --- Step 3: fit OBB to the dense white cluster ---
    pcd_dense = o3d.geometry.PointCloud()
    pcd_dense.points = o3d.utility.Vector3dVector(dense_white_coords)
    obb = pcd_dense.get_oriented_bounding_box()

    # --- Step 4: containment test for ALL cloud points ---
    obb_center = np.asarray(obb.center)
    obb_extent = np.asarray(obb.extent)
    obb_R      = np.asarray(obb.R)

    pts_local = np.dot(all_coords - obb_center, obb_R)
    obb_mask  = np.all(np.abs(pts_local) <= obb_extent / 2.0, axis=1)

    density_info = {
        'voxel_size':           voxel_size,
        'density_percentile':   density_percentile,
        'total_white_voxels':   len(voxel_counts),
        'high_density_voxels':  len(high_density_keys),
        'density_threshold':    density_threshold,
        'dense_white_count':    len(dense_local_indices),
        'total_white_count':    len(white_indices),
        'pct_white_in_cluster': pct_kept,
    }

    return obb, obb_mask, dense_white_coords, density_info


# -------------------------
# White point recoloring
# -------------------------

# Named color presets (R, G, B) in uint8
NAMED_COLORS = {
    "red":     (255,   0,   0),
    "green":   (  0, 255,   0),
    "blue":    (  0,   0, 255),
    "orange":  (255, 165,   0),
    "yellow":  (255, 255,   0),
    "magenta": (255,   0, 255),
    "cyan":    (  0, 255, 255),
    "pink":    (255, 105, 180),
    "purple":  (128,   0, 128),
    "lime":    (  0, 255, 128),
}


def parse_recolor(color_str):
    """
    Parse a color string into an (R, G, B) uint8 tuple.
    Accepts:
      - Named colors:  "red", "orange", "blue", etc.
      - Comma-separated RGB values: "255,0,0" or "255, 0, 0"

    Returns:
        (R, G, B) tuple of ints in [0, 255]
    
    Raises:
        ValueError on unrecognised input.
    """
    color_str = color_str.strip().lower()

    if color_str in NAMED_COLORS:
        return NAMED_COLORS[color_str]

    # Try comma-separated
    parts = [p.strip() for p in color_str.split(',')]
    if len(parts) == 3:
        try:
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            if all(0 <= v <= 255 for v in (r, g, b)):
                return (r, g, b)
        except ValueError:
            pass

    available = ', '.join(sorted(NAMED_COLORS.keys()))
    raise ValueError(
        f"Unrecognised color '{color_str}'. "
        f"Use a named color ({available}) or 'R,G,B' notation (e.g. '255,0,0')."
    )


def recolor_white_points(ids, coords, colors, white_mask, new_color):
    """
    Return a copy of the point cloud with detected white points set to new_color.

    Args:
        ids:        Nx1 array of point IDs
        coords:     Nx3 array of XYZ coordinates
        colors:     Nx3 uint8 array of RGB colors
        white_mask: boolean mask of length N — True where a point is "white"
        new_color:  (R, G, B) tuple in [0, 255]

    Returns:
        recolored_colors: Nx3 uint8 array with white points recolored
    """
    recolored = colors.copy()
    recolored[white_mask] = np.array(new_color, dtype=np.uint8)
    return recolored


def save_recolored_points3d(output_path, ids, coords, colors, source_file,
                             white_count, new_color):
    """
    Write a COLMAP points3D.txt file with the recolored point cloud.

    Args:
        output_path:  destination file path
        ids:          Nx1 int array of point IDs
        coords:       Nx3 float array of XYZ coordinates
        colors:       Nx3 uint8 array of (possibly recolored) RGB values
        source_file:  original file path (for the header comment)
        white_count:  number of points that were recolored
        new_color:    (R, G, B) tuple used for recoloring
    """
    r_new, g_new, b_new = new_color
    with open(output_path, 'w') as f:
        f.write("# COLMAP points3D.txt — white points recolored for threshold tuning\n")
        f.write(f"# Source: {source_file}\n")
        f.write(f"# Recolored {white_count} white points to RGB({r_new},{g_new},{b_new})\n")
        f.write(f"# Number of points: {len(ids)}\n")
        for pid, coord, color in zip(ids, coords, colors):
            f.write(
                f"{pid} {coord[0]} {coord[1]} {coord[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )
    print(f"Saved recolored point cloud ({len(ids)} points, "
          f"{white_count} recolored) to: {output_path}")


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


def print_white_region_statistics(all_coords, white_indices, dense_white_coords,
                                   obb, obb_mask, density_info):
    """Print detailed statistics about the white region and density-based OBB."""
    print("\n" + "="*60)
    print("WHITE REGION ANALYSIS")
    print("="*60)

    total = len(all_coords)
    print(f"\nTotal points in cloud:      {total}")
    print(f"White points detected:      {len(white_indices)} "
          f"({100 * len(white_indices) / total:.2f}%)")

    if density_info:
        print(f"\nDensity filtering (voxel_size={density_info['voxel_size']}, "
              f"percentile={density_info['density_percentile']}):")
        print(f"  White voxels total:       {density_info['total_white_voxels']}")
        print(f"  High-density voxels:      {density_info['high_density_voxels']}")
        print(f"  Density threshold:        {density_info['density_threshold']:.1f} pts/voxel")
        print(f"  Dense white cluster size: {density_info['dense_white_count']} "
              f"({density_info['pct_white_in_cluster']:.1f}% of white points)")

    num_in_obb = int(obb_mask.sum())
    print(f"\nAll points inside OBB:      {num_in_obb} "
          f"({100 * num_in_obb / total:.2f}% of cloud)")

    print(f"\nOriented Bounding Box (fitted to dense white cluster):")
    obb_center = np.asarray(obb.center)
    obb_extent = np.asarray(obb.extent)
    print(f"  Center: [{obb_center[0]:.6f}, {obb_center[1]:.6f}, {obb_center[2]:.6f}]")
    print(f"  Extent: [{obb_extent[0]:.6f}, {obb_extent[1]:.6f}, {obb_extent[2]:.6f}]")
    print(f"  Volume: {np.prod(obb_extent):.6f}")

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
    parser.add_argument('--solid-bbox', action='store_true', default=False,
                       help="Use solid semi-transparent bounding box (default: False, use wireframe)")
    parser.add_argument('--wireframe-bbox', action='store_true', default=True,
                       help="Use wireframe bounding box instead of solid")
    parser.add_argument('--hide-white-points', action='store_true',
                       help="Hide the white points in visualization (show only bbox)")
    parser.add_argument('--save-visualization', type=str,
                       help="Save color distribution plot to file")
    parser.add_argument('--output', '-o', type=str,
                       help="Save all points within OBB region to COLMAP points3D.txt format")
    parser.add_argument('--stats-only', action='store_true',
                       help="Print statistics only, skip 3D visualization")

    parser.add_argument('--voxel-size', type=float, default=0.05, metavar='SIZE',
                        help=(
                            "Voxel side length for density-based OBB fitting (default: 0.05). "
                            "Smaller values give finer density resolution but may split the cluster; "
                            "larger values are coarser but more robust to gaps."
                        ))
    parser.add_argument('--density-percentile', type=float, default=75, metavar='PCT',
                        help=(
                            "Percentile of per-voxel white-point counts used as the density cutoff "
                            "(default: 75). Higher values keep only the very densest voxels, "
                            "producing a tighter OBB that excludes more outliers."
                        ))

    # --- Recolor argument ---
    parser.add_argument(
        '--recolor-white', type=str, metavar='COLOR',
        help=(
            "Recolor ALL detected white points to COLOR and save the result. "
            "Accepts named colors (red, orange, green, blue, yellow, magenta, cyan, pink, purple, lime) "
            "or explicit RGB notation (e.g. '255,0,0'). "
            "Use --recolor-output to set the destination file (default: recolored_points3D.txt). "
            "This is useful for visually verifying which points the current threshold is capturing "
            "so you can tune --white-threshold accordingly."
        )
    )
    parser.add_argument(
        '--recolor-output', type=str, default='recolored_points3D.txt', metavar='FILE',
        help=(
            "Output file path for the recolored point cloud "
            "(only used with --recolor-white, default: recolored_points3D.txt)"
        )
    )

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

    print(f"Detected {len(white_indices)} white points "
          f"({100 * len(white_indices) / len(coords):.2f}% of cloud)")

    # --recolor-white: recolor detected white points and save, then exit
    if args.recolor_white:
        try:
            new_color = parse_recolor(args.recolor_white)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"\nRecoloring {len(white_indices)} white points → RGB{new_color} ...")
        recolored_colors = recolor_white_points(ids, coords, colors, white_mask, new_color)
        save_recolored_points3d(
            output_path=args.recolor_output,
            ids=ids,
            coords=coords,
            colors=recolored_colors,
            source_file=args.file,
            white_count=int(white_mask.sum()),
            new_color=new_color,
        )
        print(
            "\nTip: Load the recolored file in your viewer to see which points were "
            "detected as white. Adjust --white-threshold and repeat until only the "
            "intended region is highlighted."
        )
        return  # Skip OBB / visualization steps

    white_coords = coords[white_indices]

    # Compute density-based oriented bounding box
    print("Computing density-based white region OBB...")
    obb, obb_mask, dense_white_coords, density_info = detect_dense_white_obb(
        coords, white_indices,
        voxel_size=args.voxel_size,
        density_percentile=args.density_percentile,
    )

    # Print statistics
    print_white_region_statistics(coords, white_indices, dense_white_coords,
                                   obb, obb_mask, density_info)

    # Save ALL points inside the OBB (white and non-white) if requested
    if args.output:
        region_indices = np.where(obb_mask)[0]
        region_ids    = ids[region_indices]
        region_coords = coords[region_indices]
        region_colors = colors[region_indices]

        with open(args.output, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(region_ids)}\n")
            f.write(f"# Source: density-based OBB of white region "
                    f"(voxel_size={args.voxel_size}, "
                    f"density_percentile={args.density_percentile}, "
                    f"white_threshold={args.white_threshold})\n")
            for pid, coord, color in zip(region_ids, region_coords, region_colors):
                f.write(f"{pid} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} "
                        f"{int(color[0])} {int(color[1])} {int(color[2])} 0.0\n")

        print(f"Saved {len(region_ids)} points (all points inside OBB) to: {args.output}")

    # Plot color distribution if requested
    if args.save_visualization or not args.stats_only:
        if args.save_visualization:
            plot_color_distribution(colors, white_mask, save_path=args.save_visualization)
        else:
            plot_color_distribution(colors, white_mask)

    # 3D Visualization
    if not args.stats_only and o3d is not None:
        print("Launching 3D visualization...")
        use_solid  = not args.wireframe_bbox
        show_white = not args.hide_white_points
        visualize_white_region_obb(coords, colors, obb,
                                   white_coords=white_coords if show_white else None,
                                   use_solid_bbox=use_solid,
                                   show_white_points=show_white)


if __name__ == '__main__':
    main()