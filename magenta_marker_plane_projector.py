#!/usr/bin/env python3
"""
magenta_marker_plane_projector.py

Extracts magenta points from COLMAP point cloud, fits RANSAC plane,
projects points onto plane, and generates comprehensive outputs.

"""
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Optional Open3D
try:
    import open3d as o3d
except Exception:
    o3d = None
    print("Warning: open3d not available — 3D visualization will be disabled.", file=sys.stderr)


# -------------------------
# Parse COLMAP points3D.txt
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
# Magenta mask helper
# -------------------------
def magenta_mask_from_colors_uint8(colors_uint8, saturation_min=0.4, value_min=0.4):
    """
    Create boolean mask for magenta points using HSV color space.
    Magenta has hue in range 270-330 degrees (or 0.75-0.92 in normalized [0,1]).
    
    Args:
        colors_uint8: Nx3 array of RGB colors (uint8)
        saturation_min: Minimum saturation threshold (0-1)
        value_min: Minimum value/brightness threshold (0-1)
    
    Returns:
        Boolean mask for magenta points
    """
    from colorsys import rgb_to_hsv
    
    # Normalize RGB to [0, 1]
    rgb_normalized = colors_uint8.astype(np.float32) / 255.0
    
    # Convert to HSV
    hsv = np.zeros_like(rgb_normalized)
    for i in range(len(rgb_normalized)):
        h, s, v = rgb_to_hsv(rgb_normalized[i, 0], rgb_normalized[i, 1], rgb_normalized[i, 2])
        hsv[i] = [h, s, v]
    
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]
    
    # Magenta hue range: 270-330 degrees = 0.75-0.92 in normalized [0,1]
    # Also include wrap-around from red (~0) to magenta (~0.9)
    #magenta_hue = ((h >= 0.75) & (h <= 0.92)) | ((h >= 0.9) & (h <= 1.0))
    magenta_hue = ((h >= 0.75) & (h <= 0.92)) 

    # High saturation (vivid color, not washed out)
    high_saturation = s >= saturation_min
    
    # Reasonable brightness (not too dark)
    reasonable_brightness = v >= value_min
    
    return magenta_hue & high_saturation & reasonable_brightness



# -------------------------
# Whitespace detection
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
# Plane RANSAC (basic version for grid search)
# -------------------------
def fit_plane_ransac(points, threshold=0.01, max_iterations=1000):
    """Fit plane using RANSAC algorithm (basic version for grid search)."""
    if points is None or len(points) < 3:
        return None, None, np.array([], dtype=int)
    
    best_inliers = []
    best_n = 0
    best_normal = None
    best_point = None
    n = len(points)
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(max_iterations):
        if n < 3:
            break
        
        # Randomly select 3 points
        idx = np.random.choice(n, 3, replace=False)
        a, b, c = points[idx]
        
        # Calculate plane normal
        v1 = b - a
        v2 = c - a
        nrm = np.cross(v1, v2)
        norm_len = np.linalg.norm(nrm)
        
        if norm_len < 1e-9:
            continue
        
        nrm = nrm / norm_len
        
        # Calculate distances to plane
        dists = np.abs(np.dot(points - a, nrm))
        inliers = np.where(dists < threshold)[0]
        
        # Update best model
        if len(inliers) > best_n:
            best_n = len(inliers)
            best_inliers = inliers
            best_normal = nrm
            best_point = a
    
    if best_normal is None:
        return None, None, np.array([], dtype=int)
    
    return best_normal, best_point, np.array(best_inliers, dtype=int)


def evaluate_color_threshold(
    ids,
    coords,
    colors,
    color_threshold,
    plane_threshold,
    plane_iterations
):
    """
    Evaluate a single color threshold.

    Returns
    -------
    ok : bool
        Whether this threshold produced a valid plane
    result : dict or None
        Evaluation results
    """

    # --- Extract magenta points ---
    mask = magenta_mask_from_colors_uint8(colors, color_threshold)
    mag_idx = np.nonzero(mask)[0]

    if len(mag_idx) < 3:
        return False, None

    mag_coords = coords[mag_idx]

    # --- Plane RANSAC ---
    plane_normal, plane_point, plane_inliers = fit_plane_ransac(
        mag_coords,
        threshold=plane_threshold,
        max_iterations=plane_iterations
    )

    if plane_normal is None or len(plane_inliers) < 3:
        return False, None

    return True, {
        "color_threshold": color_threshold,
        "num_magenta": len(mag_coords),
        "num_inliers": len(plane_inliers),
        "inlier_ratio": len(plane_inliers) / len(mag_coords),
        "plane_normal": plane_normal,
        "plane_point": plane_point,
        "plane_inliers": plane_inliers,
        "mag_idx": mag_idx
    }


def grid_search_color_threshold(
    ids,
    coords,
    colors,
    color_thresholds,
    plane_threshold,
    plane_iterations
):
    """
    Perform 2D Cartesian product grid search over saturation and brightness.
    Tests all combinations: for each S in color_thresholds, for each V in color_thresholds.
    This creates a full 2D exploration without overlapping parameter regions.
    
    Returns best (S, V) pair based on maximum inlier count.
    """
    best_result = None
    all_results = []

    n_thresholds = len(color_thresholds)
    total_evaluations = n_thresholds * n_thresholds

    print("\nRunning 2D Cartesian product grid search...")
    print(f"  Saturation (S) values:  {n_thresholds}")
    print(f"  Brightness (V) values:  {n_thresholds}")
    print(f"  Total evaluations:      {total_evaluations}")
    print("-" * 60)

    # Cartesian product: test all (S, V) combinations
    for s_thresh in color_thresholds:
        for v_thresh in color_thresholds:
            # Apply magenta mask with current (S, V) pair
            mask = magenta_mask_from_colors_uint8(colors, s_thresh, v_thresh)
            mag_idx = np.nonzero(mask)[0]

            if len(mag_idx) < 3:
                continue

            mag_coords = coords[mag_idx]

            # Plane RANSAC
            plane_normal, plane_point, plane_inliers = fit_plane_ransac(
                mag_coords,
                threshold=plane_threshold,
                max_iterations=plane_iterations
            )

            if plane_normal is None or len(plane_inliers) < 3:
                continue

            res = {
                "saturation_min": s_thresh,
                "brightness_min": v_thresh,
                "num_magenta": len(mag_coords),
                "num_inliers": len(plane_inliers),
                "inlier_ratio": len(plane_inliers) / len(mag_coords),
                "plane_normal": plane_normal,
                "plane_point": plane_point,
                "plane_inliers": plane_inliers,
                "mag_idx": mag_idx
            }

            print(
                f"  S={s_thresh:.2f}, V={v_thresh:.2f} | "
                f"magenta={res['num_magenta']:4d} | "
                f"inliers={res['num_inliers']:4d} | "
                f"ratio={res['inlier_ratio']:.3f}"
            )

            all_results.append(res)

    if not all_results:
        return None

    # Find best result (maximum inliers)
    best_result = max(all_results, key=lambda r: (r['num_inliers'], r['inlier_ratio']))
    
    peak_s = best_result['saturation_min']
    peak_v = best_result['brightness_min']
    peak_inliers = best_result['num_inliers']
    peak_ratio = best_result['inlier_ratio']
    
    print(
        f"\n  Peak found at S={peak_s:.2f}, V={peak_v:.2f} with "
        f"{peak_inliers} inliers (ratio={peak_ratio:.3f})"
    )

    return best_result


def _find_first_local_maximum_filtered(all_results, min_inliers=30, min_ratio=0.85):
    """
    Helper function to find first significant local maximum in the results.
    """
    # Filter for significant results
    significant_results = [
        r for r in all_results 
        if r['num_inliers'] >= min_inliers and r['inlier_ratio'] >= min_ratio
    ]
    
    if not significant_results:
        # Relax criteria if no results meet threshold
        significant_results = [r for r in all_results if r['num_inliers'] >= min_inliers]
    
    if not significant_results:
        # Last resort: use all results
        significant_results = all_results
    
    inlier_counts = [r['num_inliers'] for r in significant_results]
    
    # Find first local maximum
    for i in range(len(significant_results)):
        is_local_max = True
        
        # Check if greater than or equal to previous
        if i > 0 and inlier_counts[i] < inlier_counts[i-1]:
            is_local_max = False
        
        # Check if greater than or equal to next
        if i < len(significant_results) - 1 and inlier_counts[i] < inlier_counts[i+1]:
            is_local_max = False
        
        if is_local_max:
            # Found first local maximum
            # Check for ties (same inlier count nearby)
            peak_inlier_count = inlier_counts[i]
            candidates = [
                r for j, r in enumerate(significant_results)
                if abs(j - i) <= 1 and r['num_inliers'] == peak_inlier_count
            ]
            
            if len(candidates) > 1:
                return max(candidates, key=lambda r: r['inlier_ratio'])
            else:
                return significant_results[i]
    
    # No local maximum found, return max inliers among significant results
    return max(significant_results, key=lambda r: (r['num_inliers'], r['inlier_ratio']))

# -------------------------
# Plane RANSAC (deterministic multi-threshold)
# -------------------------
def fit_plane_ransac_deterministic(
    pts_all,
    pts_magenta,
    plane_thresholds,
    num_iterations=1000,
    seed=42
):
    """
    Deterministic multi-threshold RANSAC plane fitting
    with geometry-first candidate evaluation.

    Returns
    -------
    plane_normal : (3,)
    plane_point  : (3,)
    inliers      : ndarray
    diagnostics  : dict
    """

    best_score = -np.inf
    best_result = None

    N_all = len(pts_all)
    N_mag = len(pts_magenta)

    # Relative inlier requirement
    min_inliers = max(20, int(0.15 * N_mag))

    for i, thresh in enumerate(plane_thresholds):

        # ---- decorrelated RNG per threshold ----
        rng = np.random.default_rng(seed + i * 10_000)

        candidates = []

        # ---- RANSAC proposal generation ----
        for _ in range(num_iterations):

            idx = rng.choice(N_mag, 3, replace=False)
            a, b, c = pts_magenta[idx]

            v1 = b - a
            v2 = c - a
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-9:
                continue

            normal /= norm

            dists = np.abs((pts_magenta - a) @ normal)
            inliers = np.where(dists < thresh)[0]

            if len(inliers) >= min_inliers:
                candidates.append((normal, a, inliers))

        if not candidates:
            continue

        # ---- Candidate evaluation ----
        for normal, point, inliers in candidates:

            # -----------------------------
            # Global balance (dead-zone)
            # -----------------------------
            d_all = (pts_all - point) @ normal
            eps = 0.5 * thresh

            N_pos = np.sum(d_all >= 0)
            N_neg = np.sum(d_all < 0)

            if N_pos == 0 or N_neg == 0:
                continue

            global_balance = abs(N_pos - N_neg) / (N_pos + N_neg)

            # -----------------------------
            # Magenta balance (dead-zone)
            # -----------------------------
            pts_in = pts_magenta[inliers]
            d_mag = (pts_in - point) @ normal

            M_pos = np.sum(d_mag >= 0)
            M_neg = np.sum(d_mag < 0)

            mag_balance = abs(M_pos - M_neg) / max(M_pos + M_neg, 1)

            # -----------------------------
            # Robust distance metric
            # -----------------------------
            mean_dist = np.median(np.abs(d_mag)) / thresh

            # -----------------------------
            # Transverse spread (stable)
            # -----------------------------
            proj = pts_in - np.outer(d_mag, normal)
            proj -= proj.mean(axis=0)

            _, S, _ = np.linalg.svd(proj, full_matrices=False)
            eps_svd = 1e-12
            transverse_spread = S[-1] / max(S[0], eps_svd)
            transverse_spread = np.clip(transverse_spread, 0.0, 1.0)

            # -----------------------------
            # Inlier ratio (secondary)
            # -----------------------------
            inlier_ratio = len(inliers) / N_mag

            # -----------------------------
            # Final score (geometry-first)
            # -----------------------------
            score = (
                + 0.40 * inlier_ratio
                + 0.20 * (1-global_balance)     
                + 0.25 * (1-mag_balance)        
                + 0.05 * (1-transverse_spread)
                + 0.1 * (1-mean_dist)
            )

            # Deterministic tie-break
            if (
                score > best_score or
                (
                    np.isclose(score, best_score) and
                    thresh < best_result["plane_threshold"]
                )
            ):
                best_score = score
                best_result = {
                    "normal": normal,
                    "point": point,
                    "inliers": inliers,
                    "plane_threshold": thresh,
                    "score": score,
                    "inlier_ratio": inlier_ratio,
                    "global_balance": global_balance,
                    "magenta_balance": mag_balance,
                    "mean_distance": mean_dist,
                    "transverse_spread": transverse_spread,
                    "num_inliers": len(inliers),
                }

    if best_result is None:
        raise RuntimeError("No valid plane found")

    return (
        best_result["normal"],
        best_result["point"],
        best_result["inliers"],
        best_result,
    )

# -------------------------
# Plane projection
# -------------------------
def project_points_to_plane(points, plane_normal, plane_point):
    """Project 3D points onto a plane."""
    d = np.dot(points - plane_point, plane_normal)
    return points - np.outer(d, plane_normal)


def compute_plane_basis(plane_normal):
    """Compute orthonormal basis (u, v) for the plane."""
    temp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(plane_normal, temp)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    u = np.cross(plane_normal, temp)
    u = u / np.linalg.norm(u)
    
    v = np.cross(plane_normal, u)
    v = v / np.linalg.norm(v)
    
    return u, v


def get_plane_equation(plane_normal, plane_point):
    """Get plane equation in form: ax + by + cz + d = 0"""
    a, b, c = plane_normal
    d = -np.dot(plane_normal, plane_point)
    return a, b, c, d


# -------------------------
# Visualization functions
# -------------------------
def create_plane_mesh(plane_normal, plane_point, magenta_coords, size_multiplier=1.5):
    """
    Create a double-sided mesh representation of the fitted plane.
    Visible from both sides.
    """
    # Compute plane basis
    u, v = compute_plane_basis(plane_normal)
    
    # Project magenta points onto plane to get bounds
    projected = project_points_to_plane(magenta_coords, plane_normal, plane_point)
    center = np.mean(projected, axis=0)
    rel = projected - center
    
    # Get 2D coordinates
    coords_u = np.dot(rel, u)
    coords_v = np.dot(rel, v)
    
    # Determine plane mesh size based on magenta point spread
    u_min, u_max = np.min(coords_u), np.max(coords_u)
    v_min, v_max = np.min(coords_v), np.max(coords_v)
    
    # Expand bounds by size_multiplier
    u_range = u_max - u_min
    v_range = v_max - v_min
    u_min -= u_range * (size_multiplier - 1) / 2
    u_max += u_range * (size_multiplier - 1) / 2
    v_min -= v_range * (size_multiplier - 1) / 2
    v_max += v_range * (size_multiplier - 1) / 2
    
    # Create mesh vertices (4 corners of rectangle)
    corners = [
        center + u_min * u + v_min * v,
        center + u_max * u + v_min * v,
        center + u_max * u + v_max * v,
        center + u_min * u + v_max * v
    ]
    
    vertices = np.array(corners, dtype=np.float64)
    
    # Create double-sided triangles (4 triangles: 2 for front, 2 for back)
    triangles = np.array(
        [[0, 1, 2], [0, 2, 3],  # Front face
         [0, 2, 1], [0, 3, 2]],  # Back face (reversed winding)
        dtype=np.int32
    )
    
    # Create mesh
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Set semi-transparent cyan color
    plane_mesh.paint_uniform_color([0.0, 0.8, 0.8])
    plane_mesh.compute_vertex_normals()
    
    return plane_mesh


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


def visualize_3d_magenta_points(all_coords, all_colors, magenta_coords, plane_normal, plane_point, white_coords=None, obb=None):
    """Visualize point cloud with highlighted magenta points, fitted plane, and white region OBB using Open3D."""
    if o3d is None:
        print("Open3D not available; skipping 3D visualization.")
        return
    
    # Create point cloud for all points
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(all_coords)
    pcd_all.colors = o3d.utility.Vector3dVector(all_colors.astype(float) / 255.0)
    
    # Create point cloud for magenta points
    pcd_magenta = o3d.geometry.PointCloud()
    pcd_magenta.points = o3d.utility.Vector3dVector(magenta_coords)
    pcd_magenta.paint_uniform_color([1.0, 0.0, 1.0])  # Bright magenta
    
    # Create plane mesh
    plane_mesh = create_plane_mesh(plane_normal, plane_point, magenta_coords)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
    # Build geometry list
    geometries = [pcd_all, pcd_magenta, plane_mesh, coord_frame]
    
    # Add white region if available
    if white_coords is not None and len(white_coords) > 0:
        pcd_white = o3d.geometry.PointCloud()
        pcd_white.points = o3d.utility.Vector3dVector(white_coords)
        pcd_white.paint_uniform_color([0.9, 0.9, 0.9])  # Light gray
        geometries.append(pcd_white)
    
    if obb is not None:
        obb_wireframe = create_obb_wireframe(obb, color=(1.0, 0.55, 0.0))
        if obb_wireframe is not None:
            geometries.append(obb_wireframe)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Magenta Points with Fitted Plane and White Region",
        width=1400,
        height=900,
        left=50,
        top=50
    )


def save_plane_visualization_with_histogram(coords_2d, output_file="plane_visualization.png", n_bins=20):
    """Save 2D plane projection with histogram showing point distribution."""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    
    # Main scatter plot
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(coords_2d[:, 0], coords_2d[:, 1], s=3, c='magenta', alpha=0.6)
    ax1.set_xlabel('u (plane coordinate)', fontsize=12)
    ax1.set_ylabel('v (plane coordinate)', fontsize=12)
    ax1.set_title('Magenta Points Projected onto Plane', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Histogram on the right
    ax2 = fig.add_subplot(gs[1])
    
    # Create histogram bins
    y_vals = coords_2d[:, 1]
    counts, bin_edges = np.histogram(y_vals, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot horizontal bars
    ax2.barh(bin_centers, counts, height=(bin_edges[1] - bin_edges[0]) * 0.9, 
             color='magenta', alpha=0.7, edgecolor='darkmagenta')
    
    # Add count labels
    for i, (center, count) in enumerate(zip(bin_centers, counts)):
        if count > 0:
            ax2.text(count + max(counts) * 0.02, center, str(count), 
                    va='center', ha='left', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('v (plane coordinate)', fontsize=12)
    ax2.set_xlabel('Point Count', fontsize=12)
    ax2.set_title('Distribution Histogram', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_ylim(ax1.get_ylim())
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved plane visualization with histogram to {output_file}")


def save_point_data_txt(point_ids, coords_3d, coords_2d, output_file="pointData.txt"):
    """Save filtered magenta point data to text file."""
    with open(output_file, 'w') as f:
        f.write("# Magenta Point Data\n")
        f.write("# Format: POINT_ID X Y Z X_2D Y_2D\n")
        f.write(f"# Total points: {len(point_ids)}\n")
        f.write("#\n")
        
        for pid, coord3d, coord2d in zip(point_ids, coords_3d, coords_2d):
            f.write(f"{pid} {coord3d[0]:.6f} {coord3d[1]:.6f} {coord3d[2]:.6f} "
                   f"{coord2d[0]:.6f} {coord2d[1]:.6f}\n")
    
    print(f"Saved point data to {output_file}")


def save_npz_bundle(point_ids, coords_3d, coords_2d, plane_normal, plane_point, 
                    all_coords, magenta_coords, output_file="magenta_plane_data.npz"):
    """Save all data to NPZ file for future visualizations."""
    np.savez_compressed(
        output_file,
        point_ids=point_ids,
        coords_3d=coords_3d,
        coords_2d=coords_2d,
        plane_normal=plane_normal,
        plane_point=plane_point,
        all_coords=all_coords,
        magenta_coords=magenta_coords
    )
    print(f"Saved NPZ bundle to {output_file}")


# -------------------------
# Main processing function
# -------------------------
def process_magenta_points(file_path, color_threshold=0.3, plane_threshold=0.03, 
                          plane_iterations=1000, visualize_3d=True, n_histogram_bins=20):
    """Main processing pipeline."""
    
    print("="*60)
    print("MAGENTA POINT EXTRACTION AND PLANE PROJECTION")
    print("="*60)
    
    # Load point cloud
    print(f"\nLoading point cloud from: {file_path}")
    ids, coords, colors = parse_colmap_points3d(file_path)
    
    if len(coords) == 0:
        print("ERROR: No points found in file.")
        return None
    
    print(f"Loaded {len(coords)} points")
    
    # Detect white points and compute oriented bounding box
    print("\nDetecting white/light points for region constraint...")
    white_mask, white_indices = detect_white_points(colors, threshold=230)
    white_coords = coords[white_indices] if len(white_indices) > 0 else None
    
    obb = None
    obb_indices = None
    ids_in_region = ids
    coords_in_region = coords
    colors_in_region = colors
    
    if white_coords is not None and len(white_coords) > 0:
        obb, white_region_mask = get_white_oriented_bounding_box(coords, white_indices)
        obb_indices = np.where(white_region_mask)[0]
        
        # Filter to only points in the OBB region
        ids_in_region = ids[obb_indices]
        coords_in_region = coords[obb_indices]
        colors_in_region = colors[obb_indices]
        
        print(f"Found {len(white_coords)} white points ({100*len(white_coords)/len(coords):.2f}% of total)")
        print(f"OBB region contains {len(obb_indices)} points ({100*len(obb_indices)/len(coords):.2f}% of total)")
    else:
        print("No white points detected - using all points")
    
    # Extract magenta points via grid search over color thresholds (within OBB region)
    print(f"\nExtracting magenta points from OBB region...")
    color_grid = np.linspace(0.15, 0.5, 8)
    #color_grid = np.linspace(0.10, 0.25, 16)
    #color_grid = [0.15]

    best = grid_search_color_threshold(
        ids_in_region,
        coords_in_region,
        colors_in_region,
        color_thresholds=color_grid,
        plane_threshold=plane_threshold,
        plane_iterations=plane_iterations
    )

    if best is None:
        print("ERROR: No valid color threshold found.")
        return None

    # Use best parameters from grid search
    mag_idx_in_region = best["mag_idx"]
    
    # Map indices back to original array
    mag_idx = obb_indices[mag_idx_in_region] if obb_indices is not None else mag_idx_in_region

    print("\nSelected best color thresholds (2D Cartesian product):")
    print(f"  saturation_min  = {best['saturation_min']:.2f}")
    print(f"  brightness_min  = {best['brightness_min']:.2f}")
    print(f"  magenta points  = {best['num_magenta']}")
    print(f"  plane inliers   = {best['num_inliers']}")

    if len(mag_idx) == 0:
        print("ERROR: No magenta points found with current threshold.")
        return None
    
    mag_coords = coords[mag_idx]
    mag_ids = ids[mag_idx]
    
    print(f"Found {len(mag_coords)} magenta points ({100*len(mag_coords)/len(coords):.2f}% of total)")
    
    # Fit plane using RANSAC
    print(f"\nFitting RANSAC plane (iterations: {plane_iterations})...")
    plane_thresholds = np.linspace(0.01, 0.12, 12)

    plane_normal, plane_point, plane_inliers, plane_stats = (
        fit_plane_ransac_deterministic(
            coords,
            mag_coords,
            plane_thresholds=plane_thresholds,
            num_iterations=plane_iterations
        )
    )

    print("\nSelected plane via multi-threshold RANSAC:")
    for k, v in plane_stats.items():
        print(f"  {k}: {v}")

    if plane_normal is None:
        print("ERROR: Failed to fit plane.")
        return None
    
    print(f"Plane fitted with {len(plane_inliers)} inliers ({100*len(plane_inliers)/len(mag_coords):.2f}% of magenta points)")
    
    # Get inlier points
    plane_mag_coords = mag_coords[plane_inliers]
    plane_mag_ids = mag_ids[plane_inliers]
    
    # Project points onto plane
    print("\nProjecting points onto plane...")
    projected = project_points_to_plane(plane_mag_coords, plane_normal, plane_point)
    
    # Compute plane basis and 2D coordinates
    u, v = compute_plane_basis(plane_normal)
    center = np.mean(projected, axis=0)
    rel = projected - center
    coords_2d = np.column_stack([np.dot(rel, u), np.dot(rel, v)])
    
    # Get plane equation
    a, b, c, d = get_plane_equation(plane_normal, plane_point)
    
    # Calculate distance statistics of all magenta points to the plane
    dists_all_magenta = np.abs(np.dot(mag_coords - plane_point, plane_normal))
    print(f"\nDistance stats of all magenta points to plane:")
    print(f"  Min distance: {np.min(dists_all_magenta):.6f}")
    print(f"  Max distance: {np.max(dists_all_magenta):.6f}")
    print(f"  Mean distance: {np.mean(dists_all_magenta):.6f}")
    print(f"  Median distance: {np.median(dists_all_magenta):.6f}")

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nTotal magenta points found: {len(mag_coords)}")
    print(f"Points on plane (inliers): {len(plane_mag_coords)}")
    print(f"Inlier percentage: {100*len(plane_mag_coords)/len(mag_coords):.2f}%")
    
    print(f"\nPlane Normal: [{plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f}]")
    print(f"Plane Point: [{plane_point[0]:.6f}, {plane_point[1]:.6f}, {plane_point[2]:.6f}]")
    print(f"\nPlane Equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    
    print("\n2D Projection Statistics:")
    print(f"  u range: [{np.min(coords_2d[:, 0]):.6f}, {np.max(coords_2d[:, 0]):.6f}]")
    print(f"  v range: [{np.min(coords_2d[:, 1]):.6f}, {np.max(coords_2d[:, 1]):.6f}]")
    
    # Save outputs
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    save_point_data_txt(plane_mag_ids, plane_mag_coords, coords_2d)
    save_npz_bundle(plane_mag_ids, plane_mag_coords, coords_2d, plane_normal, plane_point, 
                    coords, mag_coords)
    save_plane_visualization_with_histogram(coords_2d, n_bins=n_histogram_bins)
    
    # 3D Visualization
    if visualize_3d and o3d is not None:
        print("\nLaunching 3D visualization...")
        visualize_3d_magenta_points(coords, colors, mag_coords, plane_normal, plane_point, 
                                   obb=obb)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    return {
        'point_ids': plane_mag_ids,
        'coords_3d': plane_mag_coords,
        'coords_2d': coords_2d,
        'plane_normal': plane_normal,
        'plane_point': plane_point,
        'plane_equation': (a, b, c, d),
        'all_coords': coords,
        'magenta_coords': mag_coords
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract magenta points and fit RANSAC plane from COLMAP point cloud"
    )
    parser.add_argument('--file', '-f', required=True, 
                       help="Path to COLMAP points3D.txt file")
    parser.add_argument('--color-threshold', '-c', type=float, default=0.3,
                       help="Color threshold for magenta detection (0-1, default: 0.3)")
    parser.add_argument('--plane-threshold', '-p', type=float, default=0.03,
                       help="RANSAC plane distance threshold (default: 0.03)")
    parser.add_argument('--plane-iterations', '-i', type=int, default=1000,
                       help="RANSAC maximum iterations (default: 1000)")
    parser.add_argument('--no-viz', action='store_true',
                       help="Skip 3D visualization")
    parser.add_argument('--histogram-bins', type=int, default=20,
                       help="Number of bins for histogram (default: 20)")
    
    args = parser.parse_args()
    
    results = process_magenta_points(
        file_path=args.file,
        color_threshold=args.color_threshold,
        plane_threshold=args.plane_threshold,
        plane_iterations=args.plane_iterations,
        visualize_3d=not args.no_viz,
        n_histogram_bins=args.histogram_bins
    )
    
    if results is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
