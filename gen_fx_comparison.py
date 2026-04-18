import numpy as np, json, cv2

with open('data/sample_demo_1/grasp_pose.json') as f:
    data = json.load(f)
R = np.array(data['rotation'])
t = np.array(data['translation'])
gw = data['width']
gdepth = 0.04
local_pts = np.array([[-gw/2,0,0],[gw/2,0,0],[-gw/2,0,-gdepth],[gw/2,0,-gdepth],[0,0,-gdepth],[0,0,-gdepth*2]])
world_pts = (R @ local_pts.T).T + t

cx, cy = 600.0, 600.0
thickness = 4
color = (0, 255, 255)

images = []
for fx in [600, 912, 1039, 1200, 1500, 1666]:
    fy = fx
    img = cv2.imread('data/sample_demo_1/image.png')
    pts2d = []
    for p in world_pts:
        u = int(fx * p[0] / p[2] + cx)
        v = int(fy * p[1] / p[2] + cy)
        pts2d.append((u, v))
    cv2.line(img, pts2d[0], pts2d[2], color, thickness)
    cv2.line(img, pts2d[1], pts2d[3], color, thickness)
    cv2.line(img, pts2d[2], pts2d[3], color, thickness)
    cv2.line(img, pts2d[4], pts2d[5], color, thickness)
    cv2.putText(img, f'fx={fx}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 5)
    img_small = cv2.resize(img, (400, 400))
    images.append(img_small)

row1 = np.hstack(images[:3])
row2 = np.hstack(images[3:])
grid = np.vstack([row1, row2])
cv2.imwrite('data/sample_demo_1/fx_comparison.png', grid)
print('Saved fx_comparison.png')
