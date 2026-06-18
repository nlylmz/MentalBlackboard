#!/usr/bin/env python3
import argparse
import json
import os
import textwrap
from typing import Tuple, List
import sys
import logging

from tqdm import tqdm
from transformers import pipeline
import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from open_source_evals.prompts_builder import (
    build_prediction_2Dimage_prompt,
    build_prediction_prompt,
)

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
        default="Prediction_2D_images",
        help="Task name used to locate data and organize output.",
    )
    parser.add_argument(
        "--json-path",
        nargs="+",
        default=["./data/PFHP_Prediction_Data.json"],
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

def setup_logging(log_file: str):
    # Redirect all prints
    sys.stdout = open(log_file, "w", buffering=1)   # line-buffered
    sys.stderr = sys.stdout  # redirect errors too

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Also set vllm logger to INFO and propagate
    logging.getLogger("vllm").setLevel(logging.INFO)
    logging.getLogger("vllm").propagate = True


def main() -> None:
    args = parse_args()
    model_name = args.model_name
    task = args.task

    # Resolve save path
    model_basename = model_name.split("/")[-1]
    if args.save_file:
        save_file = args.save_file
    else:
        print('Creating directory')
        save_dir = os.path.join(args.save_dir, task)
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"{model_basename}_{task}.jsonl")

    setup_logging(f"{args.logs}/run_{model_basename}_{task}.log")

    if os.path.exists(save_file) and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {save_file} (use --overwrite to replace)"
        )

    if task == "Prediction_2D_images":
        build_prompt = build_prediction_2Dimage_prompt
    else:
        raise ValueError(f"Task {task} not supported")
    
    # Load JSON records
    data = []
    for jp in args.json_path:
        with open(jp, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data.extend(loaded)
            else:
                raise ValueError(f"Expected a list in JSON file: {jp}, got {type(loaded).__name__}")

    data_by_id = {item["id"]: item for item in data}
    # Build tasks
    tasks: List[str] = []

    # Deterministic order & optional limit
    folder_names = [
        d
        for d in sorted(os.listdir(args.img_text_root))
        if d.startswith("PFHP_")
        and os.path.isdir(os.path.join(args.img_text_root, d))
    ]
    if args.limit is not None:
        folder_names = folder_names[: args.limit]

    trust_remote = True
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=trust_remote)
    

    for folder_name in tqdm(folder_names, desc="Preparing prompts"):
        id_ = folder_name
        folder_path = os.path.join(args.img_text_root, folder_name)
        txt_path = os.path.join(folder_path, "text", f"{id_}_textual_steps.txt")

        img_dir = os.path.join(folder_path, "img")

        image_files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "result_holes" not in f
        ])

        if not image_files:
            print(f" No valid images in {img_dir}")
            continue
        
        image_files.append(args.location_img)

        # Validate image directory
        if not os.path.isdir(img_dir):
            print(f" No 'img/' directory in {folder_name}")
            continue

        if not os.path.isfile(txt_path):
            print(f" Missing text file for {id_}")
            continue
        if id_ not in data_by_id:
            print(f" No JSON record found for {id_}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text_data = f.read().strip()

        item = data_by_id[id_]
        initial_holes = item.get("initialHoles", [])
        if not initial_holes:
            print(f" No initial holes for {id_}")
            continue
        
        
        content = [{"type": "image"} for url in image_files]
        content.append({"type": "text", "text": build_prompt(id_, text_data, initial_holes)})
        messages = [{
            "role": "user",
            "content": content
        }]
        prompt = processor.apply_chat_template(
            messages,
            tokenizer=False,
            add_generation_prompt=True,
        )
        images = [Image.open(url) for url in image_files]
        task = {
            "prompt": prompt,
            "multi_modal_data": {"image": images}
        }

        # outputs = vlm(text=task, max_new_tokens=args.max_tokens)
        tasks.append(task)
        # break
        # break
    print(f" {len(tasks)} tasks created.")

    if not tasks:
        print("No tasks to run. Exiting.")
        return

    vlm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.9,
        # limit_mm_per_prompt={"video":1}
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    outputs = vlm.generate(tasks, sampling_params=sampling_params)
    # print(outputs)
    # Save JSONL
    print(f"Saving outputs to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        for idx, row in tqdm(enumerate(outputs), desc="Writing JSONL"):
            prompt = row.prompt
            generated_text = row.outputs[0].text
            f.write(json.dumps({"id": folder_names[idx], "prompt": prompt, "response": generated_text}) + "\n")

    print("Process Finished!!")

if __name__ == "__main__":
    main()
