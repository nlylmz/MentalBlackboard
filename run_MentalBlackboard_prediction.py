import argparse
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from planning_converter import extract_json_from_response



REPO_TYPE = "dataset"

RESPONSE_FORMAT = """
Think step-by-step, then provide your answer in the required JSON format.

```json
{ 
  "totalNumberofHoles": <number>, 
  "unfoldingTypes": ["H1-F", "H2-F", "V1-F", "V2-F", "D1-F", "D2-F", "D3-F", "D4-F"],
  "resultHoles": [
    {
      "shape": "<circle | ellipse | triangle | star | letter | trapezoid>",
      "size": "<small | large>",
      "direction": <0 | 90 | 180 | 270>,
      "location": <1-32>
    }
  ]
}
```"""


# --------------------------------------------------
# Utils
# --------------------------------------------------

def normalize_direction(direction_str):
    if not direction_str or str(direction_str).strip() == "":
        return 0
    first = str(direction_str).split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return 0


def parse_initial_holes(initial_holes):
    if initial_holes is None:
        return []
    if isinstance(initial_holes, str):
        try:
            return json.loads(initial_holes)
        except Exception:
            return []
    if isinstance(initial_holes, list):
        return initial_holes
    return []


def safe_open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


# --------------------------------------------------
# Model
# --------------------------------------------------

def load_model(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if torch.cuda.is_available():
        dtype = torch.bfloat16  # safer for these VLMs
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    last_err = None
    model = None
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception as e:
        last_err = e

    if model is None:
        raise RuntimeError(f"Failed to load model. Last error: {repr(last_err)}")

    model.eval()
    return model, processor


# --------------------------------------------------
# Prompt (text)
# --------------------------------------------------

def build_prompt(example: Dict[str, Any]) -> str:
    holes = parse_initial_holes(example.get("initialHoles"))

    hole_list = "\n".join([
        f"- Shape: {h.get('shape')}, "
        f"Size: {h.get('size')}, "
        f"Location: {h.get('location')}, "
        f"Direction: {normalize_direction(h.get('direction'))}"
        for h in holes
    ])

    # IMPORTANT: We describe the ordering explicitly:
    # Image 1 = paper structure, Images 2..N = folding frames
    prompt = textwrap.dedent(f"""
        You are an AI system with the ability to mentally visualize and manipulate folded paper. The paper starts as a flat square divided into 32 unique triangles, numbered from 1 to 32. These numbers increase row by row from left to right, starting at the top-left corner. The first image shows the number locations at the paper structure.

        The remaining images are ordered frames from a 3D animation illustrating a paper folding and hole-punching task. The paper is folded one or more times according to a defined sequence of actions. The paper may also undergo a rotation. Each image frame in the sequence—excluding last action which is hole-punching step—represents either a folding or a rotation action.  

        The holes are depicted as black shapes or marks. In the given example, the following initial holes are punched:

        {hole_list}

        Your task is mentally unfold the paper step by step. Then, provide:
        - the sequence of unfolding steps,
        - the total number of resulting holes,
        - and the final position, size, direction, and shape of each resulting hole on the original paper.

        When determining the unfolding steps, do not reverse the rotation. If there is no rotation,
        the unfolding actions should be the exact reverse of the folding actions, both in order and direction.
        If there is a rotation, identify the physically accurate unfolding action by accounting for the rotated orientation.

        Unfolding step choices:
        H1-F — Unfold horizontally from top to bottom
        H2-F — Unfold horizontally from bottom to top
        V1-F — Unfold vertically from left to right
        V2-F — Unfold vertically from right to left
        D1-F — Unfold diagonally from top-left to bottom-right
        D2-F — Unfold diagonally from top-right to bottom-left
        D3-F — Unfold diagonally from bottom-left to top-right
        D4-F — Unfold diagonally from bottom-right to top-left

        {RESPONSE_FORMAT}
    """)
    return prompt.strip()


# --------------------------------------------------
# Image loading
# --------------------------------------------------

def load_frames_from_hf(
    repo_id: str,
    frame_paths: List[str],
    cache_dir: Optional[str] = None,
) -> List[Image.Image]:

    images: List[Image.Image] = []

    for p in frame_paths:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=p,
                repo_type=REPO_TYPE,
                cache_dir=cache_dir,
            )
            images.append(safe_open_rgb(local_path))

        except Exception:
            # skip missing or failed downloads silently
            continue

    return images



def load_paper_image_local(paper_image_path: str) -> Image.Image:
    if not os.path.exists(paper_image_path):
        raise FileNotFoundError(f"paper_image not found: {paper_image_path}")
    return safe_open_rgb(paper_image_path)


def _ensure_list_of_paths(frame_paths) -> List[str]:
    """
    Accept a list[str] or a JSON / Python-list string and return a List[str].
    Raises ValueError if it cannot coerce.
    """
    if isinstance(frame_paths, list):
        return frame_paths
    if isinstance(frame_paths, str):
        s = frame_paths.strip()
        # try JSON
        try:
            import json as _json
            parsed = _json.loads(s)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass
        # try ast.literal_eval
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass
        raise ValueError("frame_paths is a string and not a JSON/list-string convertible to List[str]. "
                         "Please pass a List[str].")
    raise TypeError("frame_paths must be List[str] (or a JSON/string-convertible list).")
    
# --------------------------------------------------
# Inference
# --------------------------------------------------

@torch.no_grad()
def run_inference(images, prompt_text, model, processor, max_new_tokens=2500):
    name = (getattr(model.config, "_name_or_path", "") or "").lower()
    
    # 1. Identify Model Family
    is_qwen = "qwen" in name
    is_llava = "llava" in name
    is_gemma = "gemma-3" in name or "gemma3" in name

    # 2. Try Chat Template (Standard way)
    try:
        if hasattr(processor, "apply_chat_template"):
            # Most InternVL 2.5/3.5 HF-compatible processors support this
            messages = [{
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt_text}],
            }]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            raise ValueError
            
    except Exception:
        # 3. Manual Fallback for LLaVA / Qwen
        if is_llava:
            image_tokens = "<image>\n" * len(images)
            chat_text = f"USER: {image_tokens}{prompt_text}\nASSISTANT:"
        elif is_qwen:
            image_tokens = "".join([f"Picture {i+1}: <img></img>" for i in range(len(images))])
            chat_text = f"<|im_start|>user\n{image_tokens}{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            chat_text = prompt_text

    # 4. Build Tensors
    inputs = processor(
        text=[chat_text],
        images=images,
        padding=True,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Critical for LLaVA: Match image tensor precision to model precision
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # 5. Generate with specific InternVL settings
    out = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=False,
        use_cache=True,
    )

    # 6. Clean decoding
    input_len = inputs["input_ids"].shape[-1]
    gen_tokens = out[0][input_len:]
    
    return processor.decode(gen_tokens, skip_special_tokens=True).strip()
# --------------------------------------------------

# --------------------------------------------------
# Main
# --------------------------------------------------

def main(
    dataset_name: str,
    config_name: str,
    split: str,
    model_id: str,
    output_path: str,
    paper_image_path: str,
    streaming: bool,
    cache_dir: Optional[str],
):
    os.makedirs(output_path, exist_ok=True)

    out_file = os.path.join(
        output_path,
        f"{model_id.replace('/', '_')}_{config_name}_{split}.jsonl"
    )

    model, processor = load_model(model_id)
    dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming)

    # Load once (same for all examples)
    paper_img = load_paper_image_local(paper_image_path)

    # safe loop
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in tqdm(dataset, desc="Running inference", unit="ex"):
            ex_id = ex.get("id")
            raw_frame_paths = ex.get("foldingFrames", [])

            # Print a short repr for debugging
            print("frame_paths sample:", (repr(raw_frame_paths)[:300] + '...') if isinstance(raw_frame_paths, str) else raw_frame_paths[:10])

            # Ensure frame_paths is actually a list of strings
            try:
                frame_paths = _ensure_list_of_paths(raw_frame_paths)
            except Exception as e:
                # If we cannot coerce, skip this example but log the problem
                row = {
                    "id": ex_id,
                    "error": f"invalid_frame_paths: {repr(e)}",
                    "raw_response": None,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[SKIP] Example {ex_id}: invalid frame_paths, skipping.")
                continue

            # initialize response so except block can safely reference it
            response = None
            try:
                if isinstance(frame_paths, str):
                    frame_paths = json.loads(frame_paths)
                
                frames = load_frames_from_hf(dataset_name, frame_paths, cache_dir=cache_dir)
                
                print("Loaded frames:", len(frames))

                # paper image MUST be first (caller expects this)
                images = [paper_img] + frames

                prompt = build_prompt(ex)

                # run inference (may raise)
                response = run_inference(images, prompt, model, processor)

                # parsed may be None or not dict; guard it
                parsed = extract_json_from_response(response)
                if not isinstance(parsed, dict):
                    parsed = {}

                row = {
                    "id": ex_id,
                    "totalNumberofHoles": parsed.get("totalNumberofHoles") if isinstance(parsed, dict) else None,
                    "unfoldingTypes": ",".join(parsed.get("unfoldingTypes", [])) if isinstance(parsed, dict) else "",
                    "resultHoles": parsed.get("resultHoles", []) if isinstance(parsed, dict) else []
                }

            except Exception as e:
                # Never reference variables that may be undefined; use locals().get as extra safety
                raw_resp = locals().get("response", None)
                row = {
                    "id": ex_id,
                    "error": repr(e),
                    "raw_response": raw_resp,
                }
                # log exception locally so you can debug later
                print(f"[ERROR] ex_id={ex_id} -> {e!r}")

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved predictions to: {out_file}")

    score_pre_infer("PFHP_Prediction_Data.json", out_file)


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nlylmz/MentalBlackboard")
    parser.add_argument("--config", default="prediction")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_path", default="predictions")
    parser.add_argument("--paper_image", default="paper_structure.png")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        model_id=args.model,
        output_path=args.output_path,
        paper_image_path=args.paper_image,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
    )
