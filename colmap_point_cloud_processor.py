import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    import open3d as o3d
except Exception:
    o3d = None
    print("Warning: open3d not available — 3D visualization will be disabled.", file=sys.stderr)

def visualize_point_cloud(points, colors=None, window_name="Point Cloud Visualization"):
    if o3d is None:
        print("3D visualization is disabled because open3d is not available.", file=sys.stderr)
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    o3d.visualization.draw_geometries([pcd], window_name=window_name)

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

def remove_black_points(points, colors, output_file=None):
    """
    Remove all black points (where r=0, g=0, b=0) from the point cloud.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Nx3 array of RGB colors
        output_file (str): Optional path to save the filtered point cloud as COLMAP format
    
    Returns:
        tuple: (filtered_points, filtered_colors)
    """
    # Identify non-black points
    is_not_black = np.any(colors != 0, axis=1)
    
    filtered_points = points[is_not_black]
    filtered_colors = colors[is_not_black]
    
    print(f"Original points: {len(points)}, Filtered points: {len(filtered_points)}")
    print(f"Removed {len(points) - len(filtered_points)} black points")
    
    # Save to file if specified
    if output_file is not None:
        save_colmap_file(filtered_points, filtered_colors, output_file)
    
    return filtered_points, filtered_colors

def save_colmap_file(points, colors, output_file):
    """
    Save point cloud in COLMAP points3D.txt format.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Nx3 array of RGB colors
        output_file (str): Path to save the file
    """
    with open(output_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(len(points)))
        f.write("#\n")
        
        for idx, (point, color) in enumerate(zip(points, colors)):
            x, y, z = point
            r, g, b = color
            # Format: POINT3D_ID X Y Z R G B ERROR
            f.write(f"{idx + 1} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} 0.0\n")
    
    print(f"Saved filtered point cloud to: {output_file}")

def visualize_bounding_box(points, colors=None):
    """
    Visualize the point cloud with its bounding box.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Optional Nx3 array of RGB colors
    """
    if o3d is None:
        print("Bounding box visualization is disabled because open3d is not available.", file=sys.stderr)
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # Create bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)  # Red color for bounding box
    
    # Also compute oriented bounding box
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)  # Green color for oriented bounding box
    
    # Print bounding box info
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    extent = max_bound - min_bound
    
    print("\n=== Bounding Box Information ===")
    print(f"Min bounds: {min_bound}")
    print(f"Max bounds: {max_bound}")
    print(f"Extent (Size): {extent}")
    print(f"Volume: {np.prod(extent):.2f}")
    print(f"Center: {bbox.get_center()}")
    print("================================\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd, bbox, obb],
        window_name="Point Cloud with Bounding Box (Red: Axis-aligned, Green: Oriented)"
    )

def compute_point_curvature(points, radius=0.1, max_neighbors=100):
    """
    Compute curvature for each point in the point cloud.
    Uses local surface fitting to estimate principal curvatures.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        radius (float): Search radius for neighbors
        max_neighbors (int): Maximum number of neighbors to use
    
    Returns:
        np.ndarray: Curvature values for each point (mean of principal curvatures)
    """
    if o3d is None:
        print("Curvature computation requires open3d.", file=sys.stderr)
        return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbors))
    
    # Compute curvature using local covariance analysis
    curvatures = []
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    for i in range(len(points)):
        # Find neighbors
        k, idx, dist = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        
        if k < 4:  # Need at least 4 points for covariance
            curvatures.append(0.0)
            continue
        
        # Get neighbor points
        neighbor_points = np.asarray(pcd.points)[idx[:k]]
        
        # Compute covariance
        mean = neighbor_points.mean(axis=0)
        centered = neighbor_points - mean
        cov = np.cov(centered.T)
        
        # Eigenvalues represent principal curvatures
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)
        
        # Curvature is related to the ratio of eigenvalues
        # For surface points, the smallest should be ~0, and two others are principal curvatures
        if eigenvalues[0] > 1e-6:
            curvature = eigenvalues[0]  # Smallest eigenvalue indicates surface deviation
        else:
            curvature = eigenvalues[1]  # Use second smallest
        
        curvatures.append(curvature)
    
    return np.array(curvatures)

def detect_hemispheric_region(points, colors=None, curvature_threshold=None, min_region_size=100):
    """
    Detect hemispherical region based on curvature analysis.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Optional colors
        curvature_threshold (float): Threshold for high curvature (auto-computed if None)
        min_region_size (int): Minimum number of points for valid region
    
    Returns:
        tuple: (hemisphere_points, hemisphere_colors, hemisphere_indices)
    """
    print("\nDetecting hemispherical region...")
    
    # Compute curvature
    curvatures = compute_point_curvature(points, radius=0.1)
    
    if curvatures is None:
        return None, None, None
    
    # Auto-threshold if not provided
    if curvature_threshold is None:
        # Use points above 75th percentile as potentially hemispherical
        curvature_threshold = np.percentile(curvatures, 75)
    
    print(f"Curvature range: [{np.min(curvatures):.6f}, {np.max(curvatures):.6f}]")
    print(f"Curvature threshold: {curvature_threshold:.6f}")
    
    # Find high-curvature points
    high_curv_mask = curvatures > curvature_threshold
    high_curv_indices = np.where(high_curv_mask)[0]
    
    print(f"High-curvature points: {len(high_curv_indices)}")
    
    if len(high_curv_indices) < min_region_size:
        print(f"Warning: Only {len(high_curv_indices)} high-curvature points (need {min_region_size})")
        return None, None, None
    
    hemisphere_points = points[high_curv_indices]
    hemisphere_colors = colors[high_curv_indices] if colors is not None else None
    
    return hemisphere_points, hemisphere_colors, high_curv_indices

def fit_sphere_to_points(points, max_iterations=1000, distance_threshold=0.1):
    """
    Fit a sphere to a point cloud using least squares fitting.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        max_iterations (int): Max iterations for refinement
        distance_threshold (float): Inlier threshold for RANSAC
    
    Returns:
        tuple: (center, radius) or (None, None) if fitting fails
    """
    if len(points) < 4:
        print("Not enough points to fit sphere")
        return None, None
    
    # Initial estimate: center as mean
    center = np.mean(points, axis=0)
    
    # Estimate radius
    distances = np.linalg.norm(points - center, axis=1)
    radius = np.median(distances)
    
    print(f"Initial sphere fit - Center: {center}, Radius: {radius:.4f}")
    
    # Refine using least squares
    for iteration in range(max_iterations):
        # Calculate distances from current center
        distances = np.linalg.norm(points - center, axis=1)
        
        # Use inliers (points within distance_threshold of expected radius)
        inlier_mask = np.abs(distances - radius) < distance_threshold
        inliers = points[inlier_mask]
        
        if len(inliers) < 4:
            break
        
        # Update center and radius
        new_center = np.mean(inliers, axis=0)
        new_radius = np.mean(np.linalg.norm(inliers - new_center, axis=1))
        
        # Check convergence
        center_shift = np.linalg.norm(new_center - center)
        radius_shift = abs(new_radius - radius)
        
        center = new_center
        radius = new_radius
        
        if center_shift < 1e-6 and radius_shift < 1e-6:
            print(f"Sphere fit converged at iteration {iteration}")
            break
    
    print(f"Final sphere fit - Center: {center}, Radius: {radius:.4f}")
    print(f"Inlier points: {np.sum(np.abs(np.linalg.norm(points - center, axis=1) - radius) < distance_threshold)} / {len(points)}")
    
    return center, radius

def visualize_with_hemisphere(points, colors=None, hemisphere_center=None, hemisphere_radius=None):
    """
    Visualize point cloud with fitted hemisphere.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Optional colors
        hemisphere_center (np.ndarray): Center of sphere
        hemisphere_radius (float): Radius of sphere
    """
    if o3d is None:
        print("Visualization disabled because open3d is not available.", file=sys.stderr)
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    geometries = [pcd]
    
    # Add sphere mesh if parameters provided
    if hemisphere_center is not None and hemisphere_radius is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=hemisphere_radius)
        sphere.translate(hemisphere_center)
        sphere.paint_uniform_color([1, 0, 0])  # Red sphere
        geometries.append(sphere)
        
        # Add center point as sphere
        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector([hemisphere_center])
        center_pcd.paint_uniform_color([0, 1, 0])  # Green for center
        geometries.append(center_pcd)
        
        print(f"Fitted sphere: Center={hemisphere_center}, Radius={hemisphere_radius:.4f}")
    
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with Hemispherical Fit")

def detect_densest_region(points, colors=None, voxel_size=0.05, density_percentile=75):
    """
    Detect the densest region in the point cloud using voxelization.
    Returns filtered points in the densest region and a bounding box.
    """
    print(f"\nDetecting densest region (voxel_size={voxel_size}, percentile={density_percentile})...")
    
    if o3d is None:
        print("Density detection requires open3d.", file=sys.stderr)
        return None, None, None, None, None
    
    # Manually partition points into voxels using coordinate-based binning
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # Calculate voxel indices for each point
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(int)
    
    # Count points per voxel
    voxel_counts = {}
    voxel_point_indices = {}
    
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        if voxel_key not in voxel_counts:
            voxel_counts[voxel_key] = 0
            voxel_point_indices[voxel_key] = []
        voxel_counts[voxel_key] += 1
        voxel_point_indices[voxel_key].append(i)
    
    counts = np.array(list(voxel_counts.values()))
    density_threshold = np.percentile(counts, density_percentile)
    
    print(f"Total voxels with points: {len(voxel_counts)}")
    print(f"Voxel point counts - Min: {np.min(counts)}, Max: {np.max(counts)}, Mean: {np.mean(counts):.1f}")
    print(f"Density threshold ({density_percentile}th percentile): {density_threshold:.1f} points/voxel")
    
    # Find high-density voxels
    high_density_voxels = [voxel_key for voxel_key, count in voxel_counts.items() if count >= density_threshold]
    print(f"High-density voxels: {len(high_density_voxels)} / {len(voxel_counts)}")
    
    if len(high_density_voxels) == 0:
        print("No high-density voxels found")
        return None, None, None, None, None
    
    # Convert voxel indices back to spatial coordinates for bounding box
    high_density_voxels = np.array(high_density_voxels)
    min_voxel_idx = high_density_voxels.min(axis=0)
    max_voxel_idx = high_density_voxels.max(axis=0)
    
    # Convert to spatial coordinates
    bbox_min = min_coords + min_voxel_idx * voxel_size
    bbox_max = min_coords + (max_voxel_idx + 1) * voxel_size
    
    print(f"Bounding box min: {bbox_min}")
    print(f"Bounding box max: {bbox_max}")
    
    # Collect all points in high-density voxels
    point_indices_in_dense_region = []
    for voxel_idx in high_density_voxels:
        # Convert numpy array voxel index back to tuple for dictionary key
        voxel_key = tuple(voxel_idx)
        point_indices_in_dense_region.extend(voxel_point_indices[voxel_key])
    
    point_indices_in_dense_region = np.array(point_indices_in_dense_region)
    filtered_points = points[point_indices_in_dense_region]
    filtered_colors = colors[point_indices_in_dense_region] if colors is not None else None
    
    percent_kept = 100 * len(point_indices_in_dense_region) / len(points)
    print(f"Points in high-density region: {len(point_indices_in_dense_region)} / {len(points)} ({percent_kept:.1f}%)")
    
    # If keeping more than 95% of points, warn user
    if percent_kept > 95:
        print(f"⚠️  WARNING: Keeping {percent_kept:.1f}% of points - filtering may be too lenient")
        print(f"   Try increasing --density-percentile (current: {density_percentile}) or decreasing --voxel-size (current: {voxel_size})")
    
    density_info = {
        'voxel_size': voxel_size,
        'total_voxels': len(voxel_counts),
        'high_density_voxels': len(high_density_voxels),
        'density_threshold': density_threshold,
        'filtered_point_count': len(point_indices_in_dense_region),
        'percent_kept': percent_kept
    }
    
    return filtered_points, filtered_colors, bbox_min, bbox_max, density_info

def visualize_density_bbox(points, colors=None, bbox_min=None, bbox_max=None):
    """
    Visualize point cloud with bounding box around densest region.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Optional colors
        bbox_min (np.ndarray): Minimum corner of bounding box
        bbox_max (np.ndarray): Maximum corner of bounding box
    """
    if o3d is None:
        print("Visualization disabled because open3d is not available.", file=sys.stderr)
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    geometries = [pcd]
    
    # Add bounding box if provided
    if bbox_min is not None and bbox_max is not None:
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
        bbox.color = (1, 0, 0)  # Red bounding box
        geometries.append(bbox)
        
        print(f"Density-based bounding box: Min={bbox_min}, Max={bbox_max}")
        extent = bbox_max - bbox_min
        print(f"Bounding box extent: {extent}")
        print(f"Bounding box volume: {np.prod(extent):.2f}")
    
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with Density Bounding Box")

def visualize_combined_bboxes(points, colors=None, density_bbox=None, density_points=None, show_axis_aligned=False):
    """
    Visualize point cloud with multiple bounding boxes in a single window.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Optional colors
        density_bbox (tuple): (bbox_min, bbox_max) for density-based bbox (deprecated, use density_points)
        density_points (np.ndarray): Points in the density region to compute OBB for
        show_axis_aligned (bool): Whether to compute and show axis-aligned and oriented bboxes
    """
    if o3d is None:
        print("Visualization disabled because open3d is not available.", file=sys.stderr)
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    geometries = [pcd]
    
    # Add axis-aligned and oriented bboxes
    if show_axis_aligned:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = (0, 1, 0)  # Green for axis-aligned
        geometries.append(bbox)
        
        obb = pcd.get_oriented_bounding_box()
        obb.color = (0, 0, 1)  # Blue for oriented
        geometries.append(obb)
        
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        extent = max_bound - min_bound
        
        print("\n=== Bounding Box Information ===")
        print(f"Min bounds: {min_bound}")
        print(f"Max bounds: {max_bound}")
        print(f"Extent (Size): {extent}")
        print(f"Volume: {np.prod(extent):.2f}")
        print(f"Center: {bbox.get_center()}")
    
    # Add density-based bbox (oriented)
    if density_points is not None and len(density_points) > 0:
        density_pcd = o3d.geometry.PointCloud()
        density_pcd.points = o3d.utility.Vector3dVector(density_points)
        
        # Compute oriented bounding box for density region
        density_obb = density_pcd.get_oriented_bounding_box()
        density_obb.color = (1, 0, 0)  # Red for density-based
        geometries.append(density_obb)
        
        extent = density_obb.extent
        center = density_obb.center
        volume = np.prod(extent)
        
        print(f"\n=== Density-based Oriented Bounding Box ===")
        print(f"Center: {center}")
        print(f"Extent: {extent}")
        print(f"Volume: {volume:.2f}")
    
    if show_axis_aligned and density_points is not None:
        print("\n=== Legend ===")
        print("Green (axis-aligned) | Blue (oriented OBB) | Red (density-based OBB)")
    
    window_name = "Point Cloud Visualization"
    if show_axis_aligned and density_points is not None:
        window_name = "Point Cloud with Bounding Boxes (Green: AA, Blue: OBB, Red: Density OBB)"
    elif show_axis_aligned:
        window_name = "Point Cloud with Bounding Boxes (Green: AA, Blue: OBB)"
    elif density_points is not None:
        window_name = "Point Cloud with Density Bounding Box (Oriented)"
    
    o3d.visualization.draw_geometries(geometries, window_name=window_name)

def plot_distribution_histogram(points, num_bins=50, save_path=None):
    """
    Plot histogram of point distribution along the longest axis of the oriented bounding box.
    
    Args:
        points (np.ndarray): Nx3 array of point coordinates
        num_bins (int): Number of bins for the histogram (default: 50)
        save_path (str): Optional path to save the histogram as an image
    """
    if o3d is None:
        print("Histogram visualization is disabled because open3d is not available.", file=sys.stderr)
        return
    
    # Create point cloud and compute oriented bounding box
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_oriented_bounding_box()
    
    # Get OBB properties
    extents = obb.extent  # Dimensions along each principal axis
    rotation = obb.R  # Rotation matrix (3x3)
    center = obb.center  # Center point
    
    # Find longest axis
    longest_axis = np.argmax(extents)
    axis_names = ['1st', '2nd', '3rd']
    axis_name = axis_names[longest_axis]
    
    # Get the direction vector for the longest axis from rotation matrix
    longest_direction = rotation[:, longest_axis]
    
    # Project all points onto this direction
    # First translate points relative to center, then project
    points_centered = points - center
    axis_points = np.dot(points_centered, longest_direction)
    
    # Calculate statistics
    mean_val = np.mean(axis_points)
    median_val = np.median(axis_points)
    std_val = np.std(axis_points)
    
    print(f"\n=== Distribution Statistics (OBB {axis_name} principal axis - Longest edge) ===")
    print(f"Longest axis extent: {extents[longest_axis]:.2f}")
    print(f"Projection range: [{np.min(axis_points):.2f}, {np.max(axis_points):.2f}]")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Std Dev: {std_val:.2f}")
    print(f"Direction vector: {longest_direction}")
    print("=========================================\n")
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    counts, bins, patches = ax.hist(axis_points, bins=num_bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel(f'Oriented Bounding Box {axis_name} Principal Axis', fontsize=12, fontweight='bold')
    ax.set_ylabel('Point Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Point Distribution along OBB Longest Axis (Extent: {extents[longest_axis]:.2f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Enhanced COLMAP point cloud visualization.")
    parser.add_argument('--file', '-f', required=True, type=str, help="Path to COLMAP points3D.txt file")
    parser.add_argument('--remove-black', action='store_true', help="Remove black points and create a new file")
    parser.add_argument('--output', '-o', type=str, help="Output file path for filtered point cloud (used with --remove-black or --filter-by-density)")
    parser.add_argument('--show-bbox', action='store_true', help="Visualize bounding box of the point cloud")
    parser.add_argument('--histogram', action='store_true', help="Plot distribution histogram along longest bounding box axis")
    parser.add_argument('--bins', type=int, default=50, help="Number of bins for histogram (default: 50)")
    parser.add_argument('--save-histogram', type=str, help="Save histogram to file (requires --histogram)")
    parser.add_argument('--detect-hemisphere', action='store_true', help="Detect and visualize hemispherical end (residual limb)")
    parser.add_argument('--curvature-threshold', type=float, help="Curvature threshold for hemisphere detection (auto if not specified)")
    parser.add_argument('--filter-by-density', action='store_true', help="Filter point cloud by density to remove sparse noise regions")
    parser.add_argument('--voxel-size', type=float, default=0.05, help="Voxel size for density computation (default: 0.05)")
    parser.add_argument('--density-percentile', type=float, default=75, help="Percentile threshold for high-density voxels (default: 75)")
    
    args = parser.parse_args()

    # Parse the original point cloud
    pts, cols = parse_colmap_file(args.file)
    print(f"Loaded point cloud: {len(pts)} points")
    
    # Handle remove-black operation
    if args.remove_black:
        # Determine output file path
        if args.output:
            output_path = args.output
        else:
            # Generate default output filename
            base_path = os.path.splitext(args.file)[0]
            output_path = f"{base_path}_no_black.txt"
        
        pts, cols = remove_black_points(pts, cols, output_file=output_path)
    
    # Plot histogram if requested
    if args.histogram:
        plot_distribution_histogram(pts, num_bins=args.bins, save_path=args.save_histogram)
    
    # Store density filtered points if needed
    density_filtered_pts = None
    density_filtered_cols = None
    
    # Filter by density if requested
    if args.filter_by_density:
        filtered_pts, filtered_cols, bbox_min, bbox_max, density_info = detect_densest_region(
            pts, cols,
            voxel_size=args.voxel_size,
            density_percentile=args.density_percentile
        )
        if filtered_pts is not None:
            density_filtered_pts = filtered_pts
            density_filtered_cols = filtered_cols
            
            # Save all points within bbox to output file if specified
            if args.output is not None:
                # Get all points that fall within the density bounding box
                mask = (pts >= bbox_min) & (pts <= bbox_max)
                point_indices_in_bbox = np.where(np.all(mask, axis=1))[0]
                bbox_pts = pts[point_indices_in_bbox]
                bbox_cols = cols[point_indices_in_bbox]
                
                save_colmap_file(bbox_pts, bbox_cols, args.output)
                print(f"Saved {len(bbox_pts)} filtered points (all points within density bounding box, black points removed) to '{args.output}'")
            
            print("\nNote: Use filtered point cloud data for further processing if needed")
        else:
            print("Failed to detect densest region")
    
    # Detect and visualize hemisphere if requested
    if args.detect_hemisphere:
        hemisphere_pts, hemisphere_cols, _ = detect_hemispheric_region(
            pts, cols, 
            curvature_threshold=args.curvature_threshold
        )
        if hemisphere_pts is not None:
            center, radius = fit_sphere_to_points(hemisphere_pts)
            if center is not None and radius is not None:
                visualize_with_hemisphere(pts, cols, center, radius)
            else:
                print("Failed to fit sphere to hemisphere points")
        else:
            print("Failed to detect hemispherical region")
        sys.exit(0)  # Exit after hemisphere visualization
    
    # Visualize with combined bboxes or individual visualizations
    if args.show_bbox and args.filter_by_density and density_filtered_pts is not None:
        # Combined visualization with both density and standard bboxes
        visualize_combined_bboxes(pts, cols, density_points=density_filtered_pts, show_axis_aligned=True)
    elif args.filter_by_density and density_filtered_pts is not None:
        # Only density bbox
        visualize_combined_bboxes(pts, cols, density_points=density_filtered_pts, show_axis_aligned=False)
    elif args.show_bbox:
        # Only standard bboxes
        visualize_bounding_box(pts, cols)
    else:
        # Just the point cloud
        visualize_point_cloud(pts, cols)
