"""
Extract one sample from FreeGraspData (parquet) and save it in the
format expected by demo.py / run.py:
  data/sample_demo/
    image.png
    depth.npz
    task.txt

Usage:
    python extract_sample.py [scene_id]
    # scene_id is optional; first scene is used if not specified.
"""

import os
import io
import sys
import numpy as np
import pandas as pd
from PIL import Image

PARQUET_FILES = [
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
]
NPZ_DIR = "data/npz_file"
OUTPUT_DIR = "data/sample_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print("Loading parquet files (this may take a moment)...")
df = pd.concat([pd.read_parquet(p) for p in PARQUET_FILES])
print(f"Total rows: {len(df)}, columns: {list(df.columns)}")

# Pick scene
if len(sys.argv) > 1:
    scene_id = int(sys.argv[1])
else:
    scene_id = int(df["sceneId"].iloc[0])

scene_rows = df[df["sceneId"] == scene_id]
if scene_rows.empty:
    print(f"Scene {scene_id} not found. Available first 10 scene IDs: {df['sceneId'].unique()[:10].tolist()}")
    sys.exit(1)

row = scene_rows.iloc[0]
print(f"\nExtracting scene {scene_id}...")
print(f"  Columns available: {list(row.index)}")

# --- Save image ---
image_bytes = row["image"]["bytes"]
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
image_path = os.path.join(OUTPUT_DIR, "image.png")
image.save(image_path)
print(f"  Saved image → {image_path}  ({image.size})")

# --- Save depth (from npz_file) ---
npz_path = os.path.join(NPZ_DIR, f"{scene_id}.npz")
if os.path.exists(npz_path):
    npz_data = np.load(npz_path, allow_pickle=True)
    print(f"  NPZ keys: {list(npz_data.keys())}")

    # Try to find a depth-like array
    depth_key = None
    for k in npz_data.keys():
        arr = npz_data[k]
        if hasattr(arr, "shape") and arr.ndim == 2:
            depth_key = k
            break

    if depth_key:
        depth = npz_data[depth_key]
        depth_out = os.path.join(OUTPUT_DIR, "depth.npz")
        np.savez_compressed(depth_out, depth=depth)
        print(f"  Saved depth (key='{depth_key}') → {depth_out}  shape={depth.shape}")
    else:
        print("  ⚠️  No 2-D array found in npz; creating dummy depth.")
        h, w = image.size[1], image.size[0]
        dummy = np.ones((h, w), dtype=np.float32) * 0.5
        np.savez_compressed(os.path.join(OUTPUT_DIR, "depth.npz"), depth=dummy)
else:
    print(f"  ⚠️  NPZ not found for scene {scene_id}; creating dummy depth.")
    h, w = image.size[1], image.size[0]
    dummy = np.ones((h, w), dtype=np.float32) * 0.5
    np.savez_compressed(os.path.join(OUTPUT_DIR, "depth.npz"), depth=dummy)

# --- Save task.txt ---
task_col = None
for col in ["instruction", "task", "text", "query", "prompt"]:
    if col in row.index:
        task_col = col
        break

if task_col:
    task_text = str(row[task_col])
else:
    # Pick first non-image string column as fallback
    task_text = "Pick the target object."
    for col in row.index:
        if col != "image" and isinstance(row[col], str) and len(row[col]) > 3:
            task_text = row[col]
            task_col = col
            break

task_path = os.path.join(OUTPUT_DIR, "task.txt")
with open(task_path, "w") as f:
    f.write(task_text)
print(f"  Saved task (col='{task_col}') → {task_path}")
print(f"  Task text: {task_text[:120]}")

print(f"\n✅ Sample ready in: {OUTPUT_DIR}")
print("   Run demo:    python demo.py")
print("   Or run.py:   edit run.py → images = ['data/sample_demo']")
