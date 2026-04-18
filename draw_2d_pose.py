import os
import cv2
import json
import argparse
import numpy as np

def project_points(points_3d, fx, fy, cx, cy):
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    
    z = np.where(z == 0, 1e-6, z)
    
    u = (fx * x / z + cx).astype(int)
    v = (fy * y / z + cy).astype(int)
    return np.column_stack((u, v))

def detect_camera_params(path, fx, fy, cx, cy):
    """Auto-detect camera intrinsics based on depth data format."""
    depth_path = os.path.join(path, "depth.npz")
    if os.path.exists(depth_path):
        depth = np.load(depth_path)['depth']
        # Synthetic data: float32, values in cm range, square image
        if depth.dtype == np.float32 and depth.max() < 200:
            if depth.shape[0] == depth.shape[1]:
                cx = depth.shape[1] / 2.0
                cy = depth.shape[0] / 2.0
                print(f"[Auto] Detected synthetic data -> cx={cx}, cy={cy}")
    return fx, fy, cx, cy

def draw_2d_pose(path, fx=912.481, fy=910.785, cx=644.943, cy=353.497):
    # Auto-detect camera params for synthetic data
    fx, fy, cx, cy = detect_camera_params(path, fx, fy, cx, cy)

    json_path = os.path.join(path, "grasp_pose.json")
    img_path = os.path.join(path, "image.png")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("Grasp pose data is empty.")
        return

    R = np.array(data['rotation'])
    t = np.array(data['translation'])
    w = data['width']
    depth = 0.04 # Standard gripper finger depth
    
    # Local coordinates of the gripper (U shape + stem)
    local_pts = np.array([
        [-w/2, 0, 0],       # 0: Left tip
        [w/2, 0, 0],        # 1: Right tip
        [-w/2, 0, -depth],  # 2: Left base
        [w/2, 0, -depth],   # 3: Right base
        [0, 0, -depth],     # 4: Center base
        [0, 0, -depth*2]    # 5: Arm stem
    ])
    
    # Transform to camera coordinates
    world_pts = (R @ local_pts.T).T + t
    
    # Project to 2D
    pts_2d = project_points(world_pts, fx, fy, cx, cy)
    
    # Draw on image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read {img_path}")
        return
        
    color = (255, 255, 0) # BGR for Cyan/Light Blue
    thickness = 3
    
    # Lines
    cv2.line(img, tuple(pts_2d[0]), tuple(pts_2d[2]), color, thickness) # Left finger
    cv2.line(img, tuple(pts_2d[1]), tuple(pts_2d[3]), color, thickness) # Right finger
    cv2.line(img, tuple(pts_2d[2]), tuple(pts_2d[3]), color, thickness) # Base
    cv2.line(img, tuple(pts_2d[4]), tuple(pts_2d[5]), color, thickness) # Stem
    
    out_path = os.path.join(path, "pose_2d.png")
    cv2.imwrite(out_path, img)
    print(f"✅ Saved 2D pose image to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Draw 3D grasp pose bounding box on 2D image")
    parser.add_argument("path", help="Directory containing image.png and grasp_pose.json")
    parser.add_argument("--fx", type=float, default=912.481)
    parser.add_argument("--fy", type=float, default=910.785)
    parser.add_argument("--cx", type=float, default=644.943)
    parser.add_argument("--cy", type=float, default=353.497)
    args = parser.parse_args()
    draw_2d_pose(args.path, args.fx, args.fy, args.cx, args.cy)

