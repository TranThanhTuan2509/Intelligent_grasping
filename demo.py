import os
import cv2
import trimesh
import numpy as np
import gradio as gr
from pathlib import Path

from run import compute_grasp_pose
from models.FGC_graspnet.utils.data_utils import CameraInfo

TMP_DIR = os.path.join(os.getcwd(), 'data', 'demo')
EXAMPLES_DIR = os.path.join(os.getcwd(), 'data', 'real_examples', 'hard')
os.makedirs(TMP_DIR, exist_ok=True)


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content


def exists(label, ext):
    file = os.path.join(TMP_DIR, f"{label}.{ext}")
    if os.path.exists(file):
        return file
    return None


def _build_feedback_strings(feedback_log: list) -> tuple[str, str]:
    """Convert feedback_log list into (status_str, detail_str) for Gradio display."""
    if not feedback_log:
        return "⚠️ No feedback data", ""

    total = len(feedback_log)
    last = feedback_log[-1]

    if last["is_correct"]:
        status = f"✅ Correct  ({total} attempt{'s' if total > 1 else ''})"
    else:
        status = f"❌ Could not verify after {total} attempt{'s' if total > 1 else ''}"

    lines = []
    for entry in feedback_log:
        icon = "✅" if entry["is_correct"] else "❌"
        lines.append(
            f"{icon} Attempt {entry['attempt']}:\n"
            f"   Output  : {entry['output']}\n"
            f"   Verdict : {entry['reason']}"
        )
        if not entry["is_correct"] and entry.get("corrected_id") is not None:
            hint = f"id={entry['corrected_id']}"
            if entry.get("corrected_class"):
                hint += f", class='{entry['corrected_class']}'"
            lines.append(f"   Hint    : {hint}")
        lines.append("")

    return status, "\n".join(lines)


def get_grasp_pose(text_prompt, rgb_image, depth_file, _fx, _fy, _cx, _cy, _scale):
    tmp = Path(TMP_DIR)
    for file in tmp.iterdir():
        if file.is_file():
            file.unlink()

    if depth_file is None:
        raise ValueError("Depth file is required")

    with np.load(depth_file.name) as data:
        depth = data['depth']

    np.savez_compressed(f'{TMP_DIR}/depth.npz', depth=depth)

    rgb_image = np.array(rgb_image)
    _height, _width, _ = rgb_image.shape
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{TMP_DIR}/image.png', rgb_image)

    prompt_path = f"{TMP_DIR}/task.txt"
    with open(prompt_path, "w") as file:
        file.write(text_prompt)

    camera = CameraInfo(width=_width, height=_height, fx=_fx,
                        fy=_fy, cx=_cx, cy=_cy, scale=_scale)

    grasp_dict, feedback_log = compute_grasp_pose(TMP_DIR, camera)
    molmo = exists("molmo_label", "png")

    gpt_path = exists("log", "txt")
    gpt = read_file(gpt_path) if gpt_path is not None else ""

    mask = exists("image_mask_1", "png")
    pcd = create_pcd()

    feedback_status, feedback_detail = _build_feedback_strings(feedback_log)

    return (
        gr.update(value=molmo),
        gpt,
        gr.update(value=mask),
        gr.update(value=pcd),
        grasp_dict,
        feedback_status,
        feedback_detail,
    )


def create_pcd():
    glbscene = trimesh.Scene()

    cloud_file_path = exists("cloud", "ply")
    grasp_file_path = exists("grasp", "obj")

    if grasp_file_path != None:
        point_cloud = trimesh.load(cloud_file_path)
        glbscene.add_geometry(point_cloud)

    if grasp_file_path != None:
        grasp_pose = trimesh.load(grasp_file_path)
        glbscene.add_geometry(grasp_pose)

    if len(glbscene.geometry) == 0:
        return None

    glb_path = os.path.join(TMP_DIR, f'visualization.glb')
    glbscene.export(glb_path)

    return glb_path


def interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# 🦾 FreeGrasp: Free-form language-based robotic reasoning and grasping")

        with gr.Row():
            # ── Left column: Inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## Input")
                input_text = gr.Textbox(label="Textual Prompt")
                rgb_input = gr.Image(label="RGB", type="pil")
                depth_input = gr.File(
                    label="Depth (.npz)", file_types=[".npz"])

                gr.Markdown("## Camera intrinsics")
                with gr.Group():
                    with gr.Row(equal_height=True):
                        fx = gr.Number(label="fx", value=912.481)
                        cx = gr.Number(label="cx", value=644.943)

                    with gr.Row(equal_height=True):
                        fy = gr.Number(label="fy", value=910.785)
                        cy = gr.Number(label="cy", value=353.497)

                    with gr.Row():
                        scale = gr.Number(label="scale", value=1000.0)

                submit_btn = gr.Button("Submit")

            # ── Middle column: Pipeline ──────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("## Pipeline")
                molmo_output = gr.Image(label="Molmo")
                gpt_output = gr.Textbox(
                    label=(
                        "You are a robotic system for bin picking, using a parallel gripper. "
                        "I labeled all objects id in the image.\n\nTask:\nGiven a target object "
                        "description as input, determine the first object that needs to be grasped "
                        "to enable picking the target object. If the target object is free of "
                        "obstacles, return the target object ID itself. Otherwise, identify an object "
                        "that is occluding the target and is itself free of obstacles. If multiple "
                        "objects could be removed, return any one valid option.\n"
                    ),
                    lines=1,
                )
                mask_output = gr.Image(label="LangSAM")

                # ── VLM Feedback section ─────────────────────────────────────
                with gr.Accordion("🔍 VLM Feedback", open=True):
                    feedback_status_output = gr.Textbox(
                        label="Verification Status",
                        interactive=False,
                        lines=1,
                    )
                    feedback_detail_output = gr.Textbox(
                        label="Attempt Details",
                        interactive=False,
                        lines=6,
                    )

            # ── Right column: Output ─────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("## Output")
                grasp_output = gr.Model3D(label="GraspNet")
                grasp_json_output = gr.JSON()

        submit_btn.click(
            fn=get_grasp_pose,
            inputs=[input_text, rgb_input, depth_input, fx, fy, cx, cy, scale],
            outputs=[
                molmo_output,
                gpt_output,
                mask_output,
                grasp_output,
                grasp_json_output,
                feedback_status_output,
                feedback_detail_output,
            ],
        )

        examples = []

        for i in range(1, 4):
            example = []
            tmp_path = os.path.join(EXAMPLES_DIR, str(i))

            task = read_file(os.path.join(tmp_path, "task.txt"))
            example.append(task)

            example.append(os.path.join(tmp_path, "image.png"))
            example.append(os.path.join(tmp_path, "depth.npz"))
            examples.append(example)

        gr.Examples(
            examples=examples,
            inputs=[input_text, rgb_input, depth_input, fx, fy, cx, cy, scale],
            cache_examples=False,
        )

    demo.launch()


if __name__ == "__main__":
    interface()
