import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree   
from scipy.integrate import quad

try:
    import open3d as o3d
except Exception:
    o3d = None
    print("Warning: open3d not available — 3D visualization will be disabled.", file=sys.stderr)

def parse_pointData(file_path):
    '''parse 2d points info from pointData.txt'''
    pids = []
    pts_3d = []
    pts_2d = []
    with open(file_path, 'r') as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) < 6:
                continue
            pid = int(parts[0])
            x3d, y3d, z3d = float(parts[1]), float(parts[2]), float(parts[3])
            x2d, y2d = float(parts[4]), float(parts[5])
            pids.append(pid)
            pts_3d.append([x3d, y3d, z3d])
            pts_2d.append([x2d, y2d])

    if len(pids) == 0:
        raise ValueError("No valid point data found in the file.")
    return np.array(pids, dtype=np.int32), np.array(pts_3d, dtype=np.float32), np.array(pts_2d, dtype=np.float32)

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


def prune_curve_endpoints_3d(
    pts_3d_sorted,
    radius,
    min_neighbors
):
    """
    Iteratively prune 3D curve endpoints based on neighbor density.
    """

    tree = cKDTree(pts_3d_sorted)
    start = 0
    end = len(pts_3d_sorted) - 1

    # --- Prune from start ---
    while start < end:
        neighbors = tree.query_ball_point(
            pts_3d_sorted[start], r=radius
        )
        if len(neighbors) - 1 >= min_neighbors:
            break
        start += 1

    # --- Prune from end ---
    while end > start:
        neighbors = tree.query_ball_point(
            pts_3d_sorted[end], r=radius
        )
        if len(neighbors) - 1 >= min_neighbors:
            break
        end -= 1

    return pts_3d_sorted[start:end+1], (start, end)

def compute_arc_length_parametric(fx, fy, fz, t0, t1):
    """
    Compute arc length of a parametric 3D curve using UnivariateSpline.

    Parameters
    ----------
    fx, fy, fz : callable
        Parametric coordinate functions (UnivariateSpline objects)
    t0, t1 : float
        Parameter interval

    Returns
    -------
    length : float
        Arc length
    """

    dfx = fx.derivative()
    dfy = fy.derivative()
    dfz = fz.derivative()

    def integrand(t):
        dx = dfx(t)
        dy = dfy(t)
        dz = dfz(t)
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    length, _ = quad(integrand, t0, t1, limit=200)
    return length


def compute_arc_length_poly(fx, fy, fz, t0, t1):
    """
    Compute arc length of a parametric 3D curve using polynomial functions.

    Parameters
    ----------
    fx, fy, fz : np.poly1d
        Parametric coordinate polynomials
    t0, t1 : float
        Parameter interval

    Returns
    -------
    length : float
        Arc length
    """

    dfx = fx.deriv()
    dfy = fy.deriv()
    dfz = fz.deriv()

    def integrand(t):
        dx = dfx(t)
        dy = dfy(t)
        dz = dfz(t)
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    length, _ = quad(integrand, t0, t1, limit=200)
    return length


def fit_curve_pca_parametric_3d(
    pts_3d,
    poly_order=None,
    smoothing_factor=None,
    prune_radius=None,
    min_neighbors=None
):
    """
    Fit a 3D curve using PCA-based parameterization with endpoint pruning.
    Adaptively chooses parameters based on dataset size if not explicitly provided.
    """

    n_points = len(pts_3d)
    
    # --- Adaptive parameter selection based on dataset size ---
    # Small datasets (< 100 points): use polynomial fitting without smoothing
    # Large datasets (>= 100 points): use smoothing spline
    if n_points < 100:
        if poly_order is None:
            poly_order = 3
        if prune_radius is None:
            prune_radius = 0.3
        if min_neighbors is None:
            min_neighbors = 1
        # Ensure smoothing_factor is not used for small datasets
        smoothing_factor = None
    else:
        if smoothing_factor is None:
            smoothing_factor = 10.0
        if prune_radius is None:
            prune_radius = 0.01
        if min_neighbors is None:
            min_neighbors = 4
        if poly_order is None:
            poly_order = 3  # Fallback if neither specified

    # --- PCA ---
    centroid = pts_3d.mean(axis=0)
    X = pts_3d - centroid

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    e1 = Vt[0]    # dominant direction (curve axis)
    e2 = Vt[1]
    e3 = Vt[2]

    # --- Intrinsic parameter ---
    t = X @ e1
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    pts_sorted = pts_3d[sort_idx]

    # --- Optional endpoint pruning ---
    if prune_radius is not None and min_neighbors is not None:
        pts_pruned, (i0, i1) = prune_curve_endpoints_3d(
            pts_sorted,
            radius=prune_radius,
            min_neighbors=min_neighbors
        )
        t = t[i0:i1+1]
        pts_sorted = pts_pruned

    # --- Fit parametric model ---
    if smoothing_factor is not None:
        fx = UnivariateSpline(t, pts_sorted[:, 0], s=smoothing_factor)
        fy = UnivariateSpline(t, pts_sorted[:, 1], s=smoothing_factor)
        fz = UnivariateSpline(t, pts_sorted[:, 2], s=smoothing_factor)
        curve_length = compute_arc_length_parametric(fx, fy, fz, t.min(), t.max())
    elif poly_order is not None:
        fx = np.poly1d(np.polyfit(t, pts_sorted[:, 0], poly_order))
        fy = np.poly1d(np.polyfit(t, pts_sorted[:, 1], poly_order))
        fz = np.poly1d(np.polyfit(t, pts_sorted[:, 2], poly_order))
        curve_length = compute_arc_length_poly(fx, fy, fz, t.min(), t.max())
    else:
        raise ValueError("Specify poly_order or smoothing_factor.")

    return {
        "centroid": centroid,
        "principal_axis": e1,
        "secondary_axis": e2,
        "tertiary_axis": e3,
        "t": t,
        "fx": fx,
        "fy": fy,
        "fz": fz,
        "length": curve_length
    }

def visualize_3d(pts_3d, color=None, model_3d=None, n_samples=500):
    """
    Visualize original 3D points and fitted PCA-based 3D curve.

    Parameters
    ----------
    pts_3d : (N,3) ndarray
        Original 3D points
    model_3d : dict or None
        Output of fit_curve_pca_parametric_3d
    n_samples : int
        Number of samples used to render the curve
    """

    if o3d is None:
        print("Open3D is not available. Skipping 3D visualization.", file=sys.stderr)
        return

    # --- Original point cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color.astype(float) / 255.0)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    geometries = [pcd]

    # --- Fitted curve ---
    if model_3d is not None:
        t_min, t_max = model_3d["t"].min(), model_3d["t"].max()
        t_model = np.linspace(t_min, t_max, n_samples)

        curve_3d = np.column_stack((
            model_3d["fx"](t_model),
            model_3d["fy"](t_model),
            model_3d["fz"](t_model),
        ))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(curve_3d)
        line_set.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(curve_3d) - 1)]
        )
        line_set.paint_uniform_color([0.0, 1.0, 0.0])

        geometries.append(line_set)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Curve Fit Visualization"
    )


def process_3d_curve_fitting(file_path,original_path=None):
    '''Process 3D curve fitting from point data file'''
    print("="*60)
    print("Processing 3D curve fitting from:", file_path)
    print("="*60)
    pids, pts_3d, pts_2d = parse_pointData(file_path)
    print(f"Parsed {len(pids)} points from the file.")
    # Adaptive parameter selection automatically happens inside fit_curve_pca_parametric_3d
    fitted_3d = fit_curve_pca_parametric_3d(pts_3d)
    if original_path is not None:
        print("Also visualizing original data from:", original_path)
        pids_orig, pts_3d_orig, pts_3d_color = parse_colmap_points3d(original_path)
        visualize_3d(pts_3d_orig,color=pts_3d_color, model_3d=fitted_3d)
        print(f"Original model has {len(pids_orig)} points.")
    else:
        visualize_3d(pts_3d, model_3d=fitted_3d)
    print("Fitted 3D curve length:", fitted_3d["length"])
    

def main():
    parser = argparse.ArgumentParser(description="Curve Fitting from pointData.txt")
    parser.add_argument('--data','-d', required=True, help='Path to pointData.txt file')
    parser.add_argument('--original','-o', required=False, help='Path to original data file')
    args = parser.parse_args()
    process_3d_curve_fitting(args.data, original_path=args.original)

if __name__ == "__main__":
    main()
