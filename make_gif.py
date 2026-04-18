import os
import argparse
import numpy as np
import open3d as o3d
from PIL import Image

def export_pose_gif(path, out_filename="pose_output.gif"):
    cloud_path = os.path.join(path, "cloud.ply")
    grasp_path = os.path.join(path, "grasp.obj")
    
    if not os.path.exists(cloud_path) or not os.path.exists(grasp_path):
        print(f"Error: Missing cloud.ply or grasp.obj in {path}")
        return

    pcd = o3d.io.read_point_cloud(cloud_path)
    gg = o3d.io.read_triangle_mesh(grasp_path)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(gg)
    
    # Customize viewpoint if needed
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.2, 0.2, 0.2]) # Dark gray background

    images = []
    ctr = vis.get_view_control()
    # We rotate the view 36 times (dx in pixels per frame)
    for i in range(36):
        ctr.rotate(20.0, 0.0) 
        vis.poll_events()
        vis.update_renderer()
        
        img = vis.capture_screen_float_buffer(False)
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))
        
    vis.destroy_window()
    
    if images:
        out_path = os.path.join(path, out_filename)
        images[0].save(out_path, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
        print(f"✅ Saved animated pose GIF to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate animated GIF for grasp pose")
    parser.add_argument("path", help="Directory containing cloud.ply and grasp.obj")
    args = parser.parse_args()
    
    export_pose_gif(args.path)
