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

from planning_converter import extract_json_from_response, convert_to_target_format

REPO_TYPE = "dataset"

RESPONSE_FORMAT = """
Think step-by-step, then provide your answer in the required JSON format.

```json
{ 
  "folds": ["H1-F", "H2-F", "V1-F", "V2-F", "D1-F", "D2-F", "D3-F", "D4-F"],
  "holes": [
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


def parse_result_holes(result_holes):
    if result_holes is None:
        return []
    if isinstance(result_holes, str):
        try:
            return json.loads(result_holes)
        except Exception:
            return []
    if isinstance(result_holes, list):
        return result_holes
    return []


def safe_open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")

def hf_download_image(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> Image.Image:
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=REPO_TYPE, cache_dir=cache_dir)
    return safe_open_rgb(local_path)


# --------------------------------------------------
# Model
# --------------------------------------------------

def load_model(model_id: str):
    # use_fast=False avoids the Qwen2VL warning; trust_remote_code handles custom architectures
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    if torch.cuda.is_available():
        dtype = torch.bfloat16  
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    last_err = None
    model = None

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,              # Changed from torch_dtype to dtype to fix warning
            device_map=device_map,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        print(f"Successfully loaded model")
    except Exception as e:
        last_err = e

    if model is None:
        raise RuntimeError(f"Failed to load model {model_id}. Last error: {repr(last_err)}")

    model.eval()
    return model, processor

# --------------------------------------------------
# Prompt (text)
# --------------------------------------------------

def build_prompt(example: Dict[str, Any]) -> str:
    holes = parse_result_holes(example.get("resultHoles"))

    hole_list = "\n".join([
        f"- Shape: {h.get('shape')}, "
        f"Size: {h.get('size')}, "
        f"Location: {h.get('location')}, "
        f"Direction: {normalize_direction(h.get('direction'))}"
        for h in holes
    ])

    num_folds = example.get("numberofFoldingSteps")


    prompt = textwrap.dedent(f"""
        You are an AI system with the ability to mentally visualize and manipulate folded paper. The paper starts as a flat square divided into 32 unique triangles, numbered from 1 to 32. These numbers increase row by row from left to right, starting at the top-left corner. The first image shows the number locations at the paper structure.

        The second image shows a paper that has been fully unfolded after undergoing a series of folds and hole punches. In this image, white triangles represent visible (unfolded) areas, and green shapes indicate the final positions of punched holes after unfolding:
   
        In the given example, the following hole(s) appear in the final unfolded result:

        {hole_list}

        Your task is determine the sequence of folding actions in the correct order that would produce the given pattern of holes. Then identify the original location, size, direction, and shape of initial hole(s) on the folded paper where the punch(es) were made before unfolding.
    
        You are allowed to punch up to two initial holes on the folded paper. To complete the task, you must use exactly {num_folds} folding steps—no more, no less. Rotation folds are not allowed; only horizontal, vertical, or diagonal folds can be used.
    
        Here are the folding options you may choose from:
        H1-F — fold horizontally from top to bottom
        H2-F — fold horizontally from bottom to top
        V1-F — fold vertically from left to right
        V2-F — fold vertically from right to left
        D1-F — fold diagonally from top-left to bottom-right
        D2-F — fold diagonally from top-right to bottom-left
        D3-F — fold diagonally from bottom-left to top-right
        D4-F — fold diagonally from bottom-right to top-left

        {RESPONSE_FORMAT}
    """)
    return prompt.strip()


# --------------------------------------------------
# Image loading
# --------------------------------------------------

def load_paper_image_local(paper_image_path: str) -> Image.Image:
    if not os.path.exists(paper_image_path):
        raise FileNotFoundError(f"paper_image not found: {paper_image_path}")
    return safe_open_rgb(paper_image_path)


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
    is_internvl = "internvl" in name

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
        # 3. Manual Fallback for InternVL / LLaVA / Qwen
        if is_internvl:
            # InternVL 2.5+ usually expects ChatML with <image> placeholders
            image_tokens = "".join([f"Image-{i+1}: <image>\n" for i in range(len(images))])
            chat_text = f"<|im_start|>user\n{image_tokens}{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        elif is_llava:
            image_tokens = "<image>\n" * len(images)
            chat_text = f"USER: {image_tokens}{prompt_text}\nASSISTANT:"
        elif is_qwen:
            image_tokens = "".join([f"Picture {i+1}: <img></img>" for i in range(len(images))])
            chat_text = f"<|im_start|>user\n{image_tokens}{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            chat_text = prompt_text

    # 4. Build Tensors (InternVL needs pixel_values to be model's dtype)
    inputs = processor(
        text=[chat_text],
        images=images,
        padding=True,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Critical for InternVL/LLaVA: Match image tensor precision to model precision
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # 5. Generate with specific InternVL settings
    out = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=False,
        use_cache=True,
        # InternVL often performs better with this:
        # temperature=0.0 
    )

    # 6. Clean decoding
    input_len = inputs["input_ids"].shape[-1]
    gen_tokens = out[0][input_len:]
    
    return processor.decode(gen_tokens, skip_special_tokens=True).strip()
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

    with open(out_file, "w", encoding="utf-8") as f:
        for ex in tqdm(dataset, desc="Running inference", unit="ex"):
            ex_id = ex.get("id")

            try:
                # Directly get the single image path
                result_path = ex.get("resultImg", None),

                if result_path is None:
                    raise ValueError(f"Example id={ex_id} has no 'result_img' field")

                    # Unpack single-element tuples/lists that sometimes appear due to dataset formatting
                if isinstance(result_path, (list, tuple)):
                    if len(result_path) == 0:
                        raise ValueError(f"Empty result_img list for id={ex_id}")
                    # handle ('path',) or ['path']
                    result_path = result_path[0]

                if not isinstance(result_path, str):
                    raise ValueError(f"Unsupported type for result_img (id={ex_id}): {type(result_path)}")

                repo_filename = result_path.replace("\\", "/")

                print(f"Downloading from HF repo '{dataset_name}': {repo_filename}")
                local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=repo_filename,
                    repo_type=REPO_TYPE,
                    cache_dir=cache_dir,
                )
                result_img = safe_open_rgb(local_path)

                images = []
                if paper_img is not None:
                    images.append(paper_img)
                images.append(result_img)

                prompt = build_prompt(ex)
                response = run_inference(images, prompt, model, processor)

                print(f"Converting response format...")
                parsed = extract_json_from_response(response)
                formatted_row = convert_to_target_format(ex_id, parsed)

                row = formatted_row

            except Exception as e:
                row = {
                    "id": ex_id,
                    "error": str(e),
                    "raw_response": response,
                }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

        print(f"Saved model responses - plannings to: {out_file}")


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nlylmz/MentalBlackboard")
    parser.add_argument("--config", default="planning")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_path", default="plannings")
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
