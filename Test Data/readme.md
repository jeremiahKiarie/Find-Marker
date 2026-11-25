This folder contains test data output from the marker detection and curve-fitting pipeline. The data is organized into subfolders for each test case, with the following structure:

```Marker Detection and Curve Fitting Pipeline Test Data Structure
├── Marker Design
│   ├── ground_truth.ply
│   ├── run_1(18.75_0_1.0).ply
│   ├── run_2(18.75_45_5.0).ply
│   ├── run_3(18.75_90_7.5).ply
│   ├── run_4(37.5_0_7.5).ply
│   ├── run_5(37.5_45_1.0).ply
│   ├── run_6(37.5_90_5.0).ply
│   ├── run_7(75.0_0_5.0).ply
│   ├── run_8(75.0_45_7.5).ply
│   ├── run_9(75.0_90_1.0).ply
│   ├── run_10(75.0_0_1.0).ply
├── Clinical Investigation
│   ├── 3D scan (Ground Truth)
|   |   ├── p1_scan.ply
|   |   ├── p2_scan.ply
|   |   ├── p3_scan.ply
|   |   ├── p4_scan.ply
|   |   ├── p5_scan.ply
|   |   ├── p6_scan.ply
|   |── DA3
|   |   ├── p1_da3.ply
|   |   ├── p2_da3.ply
|   |   ├── p3_da3.ply
|   |   ├── p4_da3.ply
|   |   ├── p5_da3.ply
|   |   ├── p6_da3.ply
|   |── DA3 + Colmap
|   |   ├── p1_colmap.ply
|   |   ├── p2_colmap.ply
|   |   ├── p3_colmap.ply
|   |   ├── p4_colmap.ply
|   |   ├── p5_colmap.ply
|   |   ├── p6_colmap.ply
|   |── Ours (PCA-RANSAC)
|   |   ├── p1_our.ply
|   |   ├── p2_our.ply
|   |   ├── p3_our.ply
|   |   ├── p4_our.ply
|   |   ├── p5_our.ply
|   |   ├── p6_our.ply