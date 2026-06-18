#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PFHP multi-image prompting with Hugging Face `AutoProcessor` + `AutoModelForImageTextToText`.

This script mirrors an OpenAI-style flow where you interleave TEXT and IMAGE turns in a single
chat message, but runs fully on HF Transformers by:
  1) Building chat `messages` (text + image slots in order),
  2) Rendering them via `processor.apply_chat_template(...)` to a prompt string,
  3) Passing the actual pixel inputs via `processor(text=..., images=[...])` with the same
     image count and order as in your messages,
  4) Generating with `model.generate(...)`, and
  5) Writing results to JSONL.

Notes
-----
- The number of {"type": "image", ...} entries in `messages` MUST EQUAL len(images list).
- The ORDER of images MUST MATCH the order of image entries in `messages`.
- We use local files (PIL images). We also include "url": "file://..." in message image entries
  just to mirror your OpenAI payload; the HF processor does not fetch them when images= is given.
- For large models (e.g., Gemma-3-12B), ensure you have enough GPU memory; consider device_map="auto".
"""

import argparse
import json
import os
import sys
import logging
from typing import List, Tuple, Union

import torch
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

from vllm import LLM, SamplingParams

# -----------------------------
# Your prompt builders (provided by you)
# -----------------------------
from open_source_evals.prompts_builder import (
    build_prediction_prompt,
    build_prediction_2Dimage_prompt,
    build_planning_2Dimage_prompt,
)


# -----------------------------
# Argparse
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PFHP predictions with vLLM over text step folders."
    )

    # Model / vLLM
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Hub path or local path of the model.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="vLLM tensor parallel size.",
    )

    parser.add_argument(
        "--mamba",
        action="store_true",
    )

    # Task / data
    parser.add_argument(
        "--task",
        default="Planning_2D",
        help="Task name used to locate data and organize output.",
    )
    parser.add_argument(
        "--json-path",
        nargs="+",
        default=["./data/PFHP_Planning_Data.json"],
        help="Path to the main JSON file with records.",
    )
    parser.add_argument(
        "--img-text-root",
        default="./img_text_outputs",
        help="Root folder containing PFHP_* folders with textual steps.",
    )
    parser.add_argument(
        "--location_img",
        default="./paper_structure.png"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many PFHP folders (useful for smoke tests).",
    )
   
    # Output
    parser.add_argument(
        "--save-dir",
        default="./pfhp_open_source_evals",
        help="Directory under which results are written if --save-file is not provided.",
    )
    parser.add_argument(
        "--logs",
        default="./logs",
        help="Directory under which results are written if --save-file is not provided.",
    )
    parser.add_argument(
        "--save-file",
        default=None,
        help=(
            "Full path to save JSONL. If not set, will be constructed as "
            "<save-dir>/<TASK>/<model_basename>.jsonl"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=12288)

    return parser.parse_args()


# -----------------------------
# Logging
# -----------------------------
def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Redirect prints
    sys.stdout = open(log_file, "w", buffering=1)  # line-buffered
    sys.stderr = sys.stdout

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging initialized.")


# -----------------------------
# Utilities
# -----------------------------
def torch_dtype_from_str(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16


def load_json_records(paths: List[str]) -> List[dict]:
    records: List[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"Expected list in JSON file: {p}, got {type(obj).__name__}")
        records.extend(obj)
    return records


def list_image_paths(img_dir: str, *, exclude_substr: List[str] = None) -> List[str]:
    exclude_substr = exclude_substr or []
    if not os.path.isdir(img_dir):
        return []
    out = []
    for f in os.listdir(img_dir):
        if f.startswith("."):
            continue
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if any(s in f for s in exclude_substr):
            continue
        out.append(os.path.join(img_dir, f))
    return sorted(out)


def pil_list(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def build_messages_for_two_images(p1: str, p2: str, p3: str, img1_path: str, img2_path: str) -> Tuple[list, list]:
    """
    Build chat messages with TEXT + IMAGE + TEXT + IMAGE + TEXT (single user turn),
    and the corresponding PIL image list in the same order.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": p1},
                {"type": "image", "url": f"{img1_path}"},
                {"type": "text", "text": p2},
                {"type": "image", "url": f"{img2_path}"},
                {"type": "text", "text": p3},
            ],
        }
    ]
    imgs = pil_list([img1_path, img2_path])
    return messages, imgs


def build_messages_for_many_images(prompt: str, img_paths: List[str]) -> Tuple[list, list]:
    """
    Build chat messages with TEXT followed by N IMAGE entries (single user turn),
    and the corresponding PIL image list in the same order.
    """
    content = [{"type": "text", "text": prompt}]
    for p in img_paths:
        content.append({"type": "image", "url": f"file://{p}"})
    messages = [{"role": "user", "content": content}]
    imgs = pil_list(img_paths)
    return messages, imgs


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    model_basename = args.model_name.split("/")[-1]
    save_file = args.save_file or os.path.join(args.save_dir, args.task, f"{model_basename}_{args.task}.jsonl")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    log_file = os.path.join(args.logs, f"run_{model_basename}_{args.task}.log")
    setup_logging(log_file)

    if os.path.exists(save_file) and not args.overwrite:
        raise FileExistsError(f"Output file exists: {save_file} (use --overwrite to replace)")

    # ----- Device & dtype -----
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dtype = torch_dtype_from_str(args.dtype)
    # logging.info(f"Using device={device}, dtype={dtype}")

    # # ----- Model / Processor -----
    # # trust_remote_code: user flag takes precedence; fallback: enable if model name hints custom code (NVILA)
    # trust_remote = args.trust_remote_code or ("NVILA" in args.model_name)

    logging.info(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)

    # logging.info(f"Loading model: {args.model_name} (trust_remote_code={trust_remote})")
    # if args.device_map.strip().lower() == "auto":
    #     # Let HF auto-shard across devices
    #     model = AutoModelForImageTextToText.from_pretrained(
    #         args.model_name,
    #         torch_dtype=dtype,
    #         device_map="auto",
    #         trust_remote_code=trust_remote,
    #     )
    # else:
    #     model = AutoModelForImageTextToText.from_pretrained(
    #         args.model_name,
    #         torch_dtype=dtype,
    #         trust_remote_code=trust_remote,
    #     )
    #     model.to(device)

    # model.eval()
    # logging.info("Model loaded.")

    # ----- Load records -----
    records = load_json_records(args.json_path)
    data_by_id = {item["id"]: item for item in records}
    logging.info(f"Loaded {len(records)} records from {len(args.json_path)} file(s).")

    # ----- Enumerate PFHP_* folders -----
    folder_names = [
        d for d in sorted(os.listdir(args.img_text_root))
        if d.startswith("PFHP_") and os.path.isdir(os.path.join(args.img_text_root, d))
    ]
    if args.limit is not None:
        folder_names = folder_names[: args.limit]

    tasks_out = []
    logging.info(f"Scanning {len(folder_names)} folders under {args.img_text_root} ...")

    for folder_name in tqdm(folder_names, desc="Processing folders"):
        id_ = folder_name
        folder_path = os.path.join(args.img_text_root, folder_name)
        img_dir = os.path.join(folder_path, "img")

        if not os.path.isdir(img_dir):
            print(f" No 'img/' directory in {folder_name}")
            continue

        # Collect images
        if args.task == "Planning_2D_images":
            # The earlier code excluded 'initial' images
            image_paths = list_image_paths(img_dir, exclude_substr=["initial"])
        else:
            image_paths = list_image_paths(img_dir)

        if not image_paths:
            print(f" No valid images in {img_dir}")
            continue

        item = data_by_id.get(id_)
        if not item:
            print(f" No JSON record for {id_}")
            continue

        # Pull fields commonly used by your prompt builders
        initial_holes = item.get("initialHoles", [])
        result_holes = item.get("resultHoles", [])
        num_folds = item.get("numberofFoldingSteps", None)

        # ----- Build messages + image list based on task -----
        messages: List[dict]
        imgs: List[Image.Image]

        if args.task == "Prediction":
            # build_prediction_prompt(id, text_data, initial_holes) in your earlier script — here we assume it returns a single text prompt
            # If your implementation differs, adjust accordingly.
            # Try reading the textual steps if needed
            txt_path = os.path.join(folder_path, "text", f"{id_}_textual_steps.txt")
            if not os.path.isfile(txt_path):
                print(f" Missing text file for {id_}")
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                text_data = f.read().strip()
            if not initial_holes:
                print(f" No initial holes for {id_}")
                continue

            single_prompt: str = build_prediction_prompt(id_, text_data, initial_holes)

            # As in your older flow, include the location image at the end
            img_paths_for_chat = image_paths + [args.location_img]
            messages, imgs = build_messages_for_many_images(single_prompt, img_paths_for_chat)

        elif args.task == "Prediction_2D_images":
            # Assume your builder returns a single text prompt that refers to many images
            if not initial_holes:
                print(f" No initial holes for {id_}")
                continue
            single_prompt: str = build_prediction_2Dimage_prompt(id_, initial_holes)
            # Use all images (or customize selection)
            messages, imgs = build_messages_for_many_images(single_prompt, image_paths)

        elif args.task == "Planning_2D_images":
            # Your previous code expected (p1, p2, p3) and used two images: location + a 'result' image
            if not result_holes:
                print(f"No result holes for {id_}")
                continue
            if num_folds is None:
                print(f"No folding step count for {id_}")
                continue
            p1, p2, p3 = build_planning_2Dimage_prompt(id_, num_folds, result_holes)

            loc_img_path = args.location_img
            # Choose a "result" image; by default, take the last in the folder (customize as needed)
            result_img_path = image_paths[0] if image_paths else None
            messages, imgs = build_messages_for_two_images(p1, p2, p3, loc_img_path, result_img_path)

        else:
            raise ValueError(f"Unsupported task: {args.task}")

        # ----- Render messages to a prompt string (DO NOT tokenize yet) -----
        # This inserts any necessary special tokens per model template.
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,      # we want a rendered string
            return_dict=False,
        )
        images = imgs 
        
        llm_input = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": images}
        }
        tasks_out.append(llm_input)

    vlm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.9,
        # limit_mm_per_prompt={"video":1}
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    outputs = vlm.generate(tasks_out, sampling_params=sampling_params)
    # print(outputs)
    # Save JSONL
    print(f"Saving outputs to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        for idx, row in tqdm(enumerate(outputs), desc="Writing JSONL"):
            prompt = row.prompt
            generated_text = row.outputs[0].text
            f.write(json.dumps({"id": folder_names[idx], "prompt": prompt, "response": generated_text}) + "\n")

    # ----- Save -----
    print(f"Saving outputs to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        for row in tqdm(tasks_out, desc="Writing JSONL"):
            f.write(json.dumps(row) + "\n")

    print("Process Finished!!")


if __name__ == "__main__":
    main()
