import os
from dotenv import load_dotenv
load_dotenv()  # phải load trước khi đọc os.environ
import cv2
import json
import logging

MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")

from grasp_model import grasp_model
from models.langsam import langsamutils

from models.FGC_graspnet.utils.data_utils import CameraInfo

from molmo_eval import process_and_send_to_gpt

from utils.utils import *
from utils.config import *
from utils.config import call_llm_with_retry
from utils.graspnet_utils import get_correct_pose
from utils.vlm_feedback import verify_vlm_output

# Maximum number of smart-retry attempts after a failed verification
MAX_VERIFY_RETRIES = 2



def compute_grasp_pose(path, camera_info):
    parser = argparse.ArgumentParser('RUN an experiment with real data', parents=[get_args_parser()])
    args = parser.parse_args()
    
    try:
        path = str(path)
        if not path.startswith('/'):
            path = '/' + path
        image_path = os.path.join(path, "image.png")
        depth_path = os.path.join(path, "depth.npz")
        text_path = os.path.join(path, "task.txt")
        
        prompt = "Point out all objects on the table. Do not point out the robotic arm or gripper."

        # Use molmo pre-process (label number) the image
        base64_labeled_image, labeled_text = process_and_send_to_gpt(image_path, prompt, path)
        
        img_ori = cv2.imread(image_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        with open(text_path, 'r') as file:
            text = file.read()

        depth_ori = np.load(depth_path)
        depth_ori = depth_ori['depth']
        
        # Dynamically adapt camera dimensions to depth map to avoid AssertionError
        camera_info.height = depth_ori.shape[0]
        camera_info.width = depth_ori.shape[1]

        # Auto-detect synthetic data (float32 depth in cm) vs real data (uint16 depth in mm)
        if depth_ori.dtype == np.float32 and depth_ori.max() < 200:
            logging.info(f"[DepthAuto] Detected SYNTHETIC depth: dtype={depth_ori.dtype}, "
                         f"range=[{depth_ori.min():.2f}, {depth_ori.max():.2f}] -> using scale=100 (cm)")
            camera_info.scale = 100.0
            # Relax collision threshold for denser synthetic point clouds
            args.collision_thresh = 0.02
            # Adjust principal point to image center for square synthetic renders
            if depth_ori.shape[0] == depth_ori.shape[1]:
                camera_info.cx = depth_ori.shape[1] / 2.0
                camera_info.cy = depth_ori.shape[0] / 2.0
                logging.info(f"[DepthAuto] Adjusted cx={camera_info.cx}, cy={camera_info.cy}, "
                             f"collision_thresh={args.collision_thresh}")
        else:
            logging.info(f"[DepthAuto] Detected REAL depth: dtype={depth_ori.dtype}, "
                         f"range=[{depth_ori.min()}, {depth_ori.max()}] -> keeping scale={camera_info.scale}")

        image_pil = langsamutils.load_image(image_path)

        input_text = text

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robotic system for bin picking, using a parallel gripper. "
                    "I labeled all objects id in the image.\n\n"

                    "1. remove obstacle, object_id: Move the specified object out of the way. Can only be performed if the specified object is itself free of obstacles.\n"
                    "2. pick object, object_id: Pick up the specified object. Can only be performed if the object is free of obstacles.\n\n"
                    
                    "IMPORTANT: The robotic arm/gripper is NOT an object to be picked or removed. Ignore the robot arm and gripper entirely.\n\n"

                    "## Output Format\n"
                    "Respond with ONLY the object ID in this exact format, nothing else:\n"
                    "[object_id, color class_name]\n"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_labeled_image}"}}
                ]
            }
        ]

        result = {
            "selected_object": None,
            "cropping_box": None,
            "objects": []
        }

        # ── Step 1: First LLM call ───────────────────────────────────────────
        output = call_llm_with_retry(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            top_p=1,
        )

        # ── Step 2: Verify + Smart Retry loop ────────────────────────────────
        feedback_log = []   # keeps track of every attempt for UI display
        current_messages = list(messages)  # mutable copy

        for attempt in range(MAX_VERIFY_RETRIES + 1):
            feedback = verify_vlm_output(
                labeled_image_b64=base64_labeled_image,
                input_text=input_text,
                vlm_output_text=output,
            )
            attempt_entry = {
                "attempt": attempt + 1,
                "output": output,
                "is_correct": feedback["is_correct"],
                "reason": feedback["reason"],
                "corrected_id": feedback.get("corrected_id"),
                "corrected_class": feedback.get("corrected_class"),
            }
            feedback_log.append(attempt_entry)

            logging.info(
                f"[VLMFeedback] attempt={attempt+1}, is_correct={feedback['is_correct']}, "
                f"reason={feedback['reason']!r}"
            )

            if feedback["is_correct"]:
                break  # good answer — stop retrying

            if attempt < MAX_VERIFY_RETRIES:
                # Smart Retry: inject the wrong answer + verifier feedback into history
                # so the LLM has context to self-correct (pure retry with same prompt
                # would produce identical output at temperature=0).
                correction_hint = (
                    f"Your previous answer was INCORRECT.\n"
                    f"Reason: {feedback['reason']}\n"
                )
                if feedback.get("corrected_id") is not None:
                    correction_hint += (
                        f"Hint: the correct object to choose might be "
                        f"id={feedback['corrected_id']}"
                    )
                    if feedback.get("corrected_class"):
                        correction_hint += f", class='{feedback['corrected_class']}'"
                    correction_hint += ".\n"
                correction_hint += "Please reconsider and provide the correct object ID."

                current_messages = current_messages + [
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": correction_hint},
                ]
                output = call_llm_with_retry(
                    model=MODEL_NAME,
                    messages=current_messages,
                    temperature=0,
                    top_p=1,
                )

        # ── Step 3: Parse final output ───────────────────────────────────────
        result = process_grasping_result(output, text)
        
        # Goal object predicted by GPT-4o (reasoning part)
        goal = result['class_name']
        if 'selected_object_id' not in result:
            raise ValueError(f"Model output not parseable into [id, class]: {output!r}")
        goal_id = result['selected_object_id']
        goal_coor = get_coordinates(labeled_text, goal_id)

        if goal_coor is None:
            # Parse available IDs from labeled_text for debugging
            if isinstance(labeled_text, str):
                avail_lines = labeled_text.strip().split("\n")[1:]
            else:
                avail_lines = labeled_text[1:]
            avail_ids = [line.split()[0] for line in avail_lines if line.strip()]
            raise ValueError(
                f"Goal ID {goal_id} not found in Molmo labels. "
                f"Available IDs: {avail_ids}. Raw LLM output: {output!r}"
            )

        # ── Step 4: Write log ────────────────────────────────────────────────
        with open(f"{path}/log.txt", "a") as file:
            file.write(f"I have to remove the object with id = {str(goal_id)}, named {goal}\n")
            file.write(f"\n--- VLM Feedback Log ({len(feedback_log)} attempt(s)) ---\n")
            for entry in feedback_log:
                status = "CORRECT" if entry["is_correct"] else "INCORRECT"
                file.write(
                    f"  Attempt {entry['attempt']}: [{status}] output={entry['output']!r}  "
                    f"reason={entry['reason']!r}\n"
                )

        masks, boxes, phrases, logits = langsam_actor.predict(image_pil, goal)

        # Selected the mask of the object we want based on the coordinates generate from molmo (if there are multi same class objects)
        goal_mask, mask_index = get_goal_mask_with_index(masks, goal_coor)
        goal_bbox = boxes[mask_index]
        goal_bbox = goal_bbox.cpu().numpy()

        cropping_box = create_cropping_box_from_boxes(
            goal_bbox, (img_ori.shape[1], img_ori.shape[0]))
        
        goal_mask = goal_mask.unsqueeze(0)

        if args.viz:
            visualize_cropping_box(img_ori, cropping_box)

        langsam_actor.save(masks, boxes, phrases, logits, image_pil, path, viz=args.viz)
        
        endpoint, pcd = get_and_process_data(
            cropping_box, img_ori, depth_ori, camera_info, viz=args.viz)
        
        grasp_net = grasp_model(args=args, device="cuda",
                                image=img_ori, mask=goal_mask, camera_info=camera_info)
            
        gg, _ = grasp_net.forward(endpoint, pcd, path)
        
        if len(gg) == 0:
            data = {}
        else:
            R, t, w = get_correct_pose(gg[0], path, args.viz)
            
            data = {
                'translation': t.tolist(),
                'rotation': R.tolist(),
                'width': w
            }
            
            save_path = os.path.join(path, "grasp_pose.json")
            with open(save_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

        return data, feedback_log
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        with open(f"{path}/log.txt", "w") as file:
            file.write(f"Pipeline error:\n{error_msg}")
        return {}, []



if __name__ == "__main__":
    # NOTE: to change if you want to try with your own images
    camera =  CameraInfo(width=1280, height=720, fx=912.481, fy=910.785, cx=644.943, cy=353.497, scale=1000.0)

    images = [
        "/home/chien/data/FreeGrasp_code/data/sample_demo_3",

        ]
    
    for i in images:
        print(compute_grasp_pose(i, camera))
    
        
