import argparse
import json
import os
import sys
import logging
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import av  # PyAV

# ---------- Your video-aware prompt builder ----------
from open_source_evals.prompts_builder import build_back_prediction_video_batch


# -----------------------------
# Your requested PyAV function
# -----------------------------
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def read_all_frames_with_pyav(video_path: str) -> np.ndarray:
    """
    Use `read_video_pyav` to decode **all** frames from `video_path` into
    a numpy array of shape (T, H, W, 3), dtype=uint8.
    """
    container = av.open(video_path)

    # Try to get frame count from stream metadata (may be 0/None for some codecs).
    try:
        stream = container.streams.video[0]
        total = int(stream.frames) if stream.frames is not None else 0
    except Exception:
        total = 0

    if total and total > 0:
        indices = list(range(total))
        clip = read_video_pyav(container, indices)
        container.close()
        return clip

    # Fallback: first pass to count frames, second pass to decode via your function
    idxs = []
    container.seek(0)
    for i, _ in enumerate(container.decode(video=0)):
        idxs.append(i)
    container.seek(0)
    if not idxs:
        container.close()
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    clip = read_video_pyav(container, idxs)
    container.close()
    return clip


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PFHP predictions with Llava-OneVision from (video + image + text) using ALL frames via PyAV."
    )

    # Model
    p.add_argument("--model-name", default="llava-hf/llava-onevision-qwen2-7b-si-hf",
                   help="HF model id or local path for Llava-OneVision variant.")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="", help='Set "auto" to shard across devices; else we .to(device).')
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Enable if the repo needs custom code.")

    # Data / Inputs
    p.add_argument("--json-path",
                   default="./data/PFHP_Back_Prediction_Data.json",
                   help="Path to PFHP records (list[dict] with 'id', etc.).")
    p.add_argument("--video_root",
                   default="./folding_videos",
                   help="Folder with videos named like 'PF_123_animation.mp4'.")
    p.add_argument("--location-img",
                   default="./paper_structure.png",
                   help="Optional extra image (e.g., location).")
    p.add_argument("--limit", type=int, default=None, help="Process at most this many videos.")

    # Generation
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # Output / Logging
    p.add_argument("--save-dir", default="./pfhp_open_source_evals")
    p.add_argument("--save-file", default=None,
                   help="If not set: <save-dir>/Prediction_Video/<model>_Prediction.jsonl")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--logs", default="./logs")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# -----------------------------
# Logging
# -----------------------------
def setup_logging(logs_dir: str, model: str) -> str:
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"run_{model.split('/')[-1]}_video_pyav_allframes.log")
    sys.stdout = open(log_file, "w", buffering=1)
    sys.stderr = sys.stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging initialized.")
    return log_file


# -----------------------------
# Helpers
# -----------------------------
def torch_dtype_from_str(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_json_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a list in {path}, got {type(obj).__name__}")
    return obj


def map_video_filename_to_id(video_filename: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(video_filename))[0]
    if base.startswith("PF_") and base.endswith("_animation"):
        id_number = base.replace("PF_", "").replace("_animation", "")
        return f"PFHP_{id_number}"
    return None


def build_messages_onevision(base_prompt: str, include_image: bool) -> list:
    """
    Build a single user message with ONE video + optional ONE image + text.
    The template will insert <video>/<image> tokens for these.
    """
    content = [{"type": "video"}]
    if include_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": base_prompt})
    return [{"role": "user", "content": content}]


def extract_from_prompt_length(processor, outputs, inputs) -> str:
    """
    Extract the assistant response by slicing off the prompt token length.

    Args:
        processor: HF processor with .decode/.batch_decode or tokenizer
        outputs:   torch.LongTensor from model.generate(...) (shape [1, L] or [L])
        inputs:    dict returned by processor(...), must contain 'input_ids'

    Returns:
        str: decoded assistant text (may be empty if model produced nothing new)
    """
    # Normalize outputs to 1D tensor of token ids
    if hasattr(outputs, "shape"):  # torch tensor
        gen_ids = outputs[0] if outputs.dim() == 2 else outputs
    else:
        # fall back if outputs is a list/np-like
        gen_ids = outputs[0]

    # Determine prompt length from input_ids
    prompt_len = None
    try:
        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            if hasattr(input_ids, "shape"):  # torch tensor
                prompt_len = input_ids.shape[-1]
            else:
                # list-like
                prompt_len = len(input_ids[0]) if isinstance(input_ids[0], (list, tuple)) else len(input_ids)
    except Exception:
        prompt_len = None

    # Prefer decoding ONLY newly generated tokens
    def _decode(ids) -> str:
        # Try processor.decode, then tokenizer.decode, then batch_decode
        txt = ""
        try:
            if hasattr(processor, "decode"):
                txt = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            elif hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                txt = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            elif hasattr(processor, "batch_decode"):
                txt = processor.batch_decode(ids.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        except Exception:
            # last resort: try tokenizer without clean_up param
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                txt = tok.decode(ids, skip_special_tokens=True)
        return (txt or "").strip()

    # 1) If we have a valid prompt length and generation is longer, slice
    if prompt_len is not None and prompt_len < gen_ids.shape[0]:
        sliced = gen_ids[prompt_len:]
        text = _decode(sliced)
        if text:
            return text

    # 2) Fallback: decode full sequence
    text_full = _decode(gen_ids)
    if text_full:
        return text_full

    # 3) Absolute fallback: decode without skipping specials (debuggy but non-empty)
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            return tok.decode(gen_ids, skip_special_tokens=False).strip()
    except Exception:
        pass

    return ""

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    # Output path
    model_basename = args.model_name.split("/")[-1]
    save_file = args.save_file or os.path.join(
        args.save_dir, "Prediction_Video", f"{model_basename}_Prediction.jsonl"
    )
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Logging
    setup_logging(args.logs, args.model_name)

    # Overwrite guard
    if os.path.exists(save_file) and not args.overwrite:
        raise FileExistsError(f"Output exists: {save_file} (use --overwrite to replace)")

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device / dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype_from_str(args.dtype)
    logging.info(f"Using device={device}, dtype={dtype}")

    # Model / Processor
    trust_remote = args.trust_remote_code or ("NVILA" in args.model_name)
    logging.info(f"Loading processor: {args.model_name} (trust_remote_code={trust_remote})")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=trust_remote)

    logging.info(f"Loading model: {args.model_name} (trust_remote_code={trust_remote})")
    if args.device_map.strip().lower() == "auto":
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=trust_remote
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name, torch_dtype=dtype, trust_remote_code=trust_remote
        )
        model.to(device)
    model.eval()
    logging.info("Model loaded.")

    # Token IDs for safer generation
    tokenizer = getattr(processor, "tokenizer", None)
    eos_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None) or (
        tokenizer.eos_token_id if tokenizer is not None else None
    )
    pad_id = getattr(getattr(model, "generation_config", None), "pad_token_id", None) or (
        getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    )

    # Load PFHP JSON
    records = load_json_records(args.json_path)
    data_by_id = {item["id"]: item for item in records}
    logging.info(f"Loaded {len(records)} records.")

    # Enumerate videos
    if not os.path.isdir(args.video_root):
        raise FileNotFoundError(f"Video root not found: {args.video_root}")

    video_files = [f for f in sorted(os.listdir(args.video_root))
                   if f.lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".avi"))]
    if args.limit is not None:
        video_files = video_files[: args.limit]

    # Optional location image
    location_img_pil: Optional[Image.Image] = None
    if args.location_img and os.path.isfile(args.location_img):
        try:
            location_img_pil = Image.open(args.location_img).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load location image {args.location_img}: {e}")
            location_img_pil = None

    rows_out = []
    logging.info(f"Found {len(video_files)} candidate videos under {args.video_root}.")

    for vf in sorted(os.listdir(args.video_root)):
        video_path = os.path.join(args.video_root, vf)

        if not os.path.isfile(video_path) or not vf.lower().endswith(('.mp4', '.mov', '.webm')):
            continue

        base_name = os.path.splitext(vf)[0]  # e.g., "PF_1_animation"

        # Convert PF_1_animation → PFHP_1
        if base_name.startswith("PF_") and base_name.endswith("_animation"):
            id_number = base_name.replace("PF_", "").replace("_animation", "")
            id_ = f"PFHP_{id_number}"
        else:
            print(f"Skipping invalid filename: {vf}")
            continue

        # Get initial hole info
        item = data_by_id[id_]
        initial_holes = item.get("initialHoles", [])
    
        if not initial_holes:
            print(f" No initial holes for {id_}")
            continue

        # Build task prompt
        try:
            base_prompt = build_back_prediction_video_batch(id_, initial_holes)
        except TypeError:
            base_prompt = build_back_prediction_video_batch(id_, item)

        # Read ALL frames via your PyAV function
        try:
            video_clip = read_all_frames_with_pyav(video_path)  # np.ndarray (T, H, W, 3)
        except Exception as e:
            logging.error(f"Failed to read frames for {vf}: {e}")
            continue

        if video_clip.size == 0:
            if args.verbose:
                print(f"[skip] No frames extracted from {vf}")
            continue

        messages = build_messages_onevision(base_prompt, include_image=(location_img_pil is not None))

        try:
            prompt_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                return_dict=False,
            )
        except Exception as e:
            logging.error(f"apply_chat_template failed for {id_}: {e}")
            
            parts = []
            parts.append("<video>")
            if location_img_pil is not None:
                parts.append("<image>")
            parts.append(base_prompt.strip())
            parts.append("ASSISTANT:")
            prompt_text = "\n".join(parts)

        # Build processor inputs (text + ALL video frames + optional image)
        proc_kwargs = dict(
            text=prompt_text,
            videos=video_clip,  # shape (T, H, W, 3), dtype uint8
            padding=True,
            return_tensors="pt",
        )
        if location_img_pil is not None:
            proc_kwargs["images"] = location_img_pil

        inputs = processor(**proc_kwargs)

        # Move to device (if not using device_map="auto")
        if args.device_map.strip().lower() != "auto":
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        # Generate
        gen_kwargs = dict(
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})

        # Decode and extract assistant part
        try:
            decoded_full = processor.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0].strip()
            # print('output: ', decoded_full)
        except Exception:
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                decoded_full = tok.batch_decode(
                    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0].strip()
            else:
                decoded_full = ""

        response_text = extract_from_prompt_length(processor, outputs, inputs)

        # Fallback: slice off prompt tokens if available and empty after extraction
        if not response_text:
            try:
                start = inputs["input_ids"].shape[-1]
                response_text = processor.decode(outputs[0][start:], skip_special_tokens=True).strip()
            except Exception:
                pass

        rows_out.append({
            "id": id_,
            "prompt": prompt_text,
            "response": response_text,
        })

    # Save JSONL
    print(f"Saving outputs to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        for row in tqdm(rows_out, desc="Writing JSONL"):
            f.write(json.dumps(row) + "\n")

    print("Process Finished!!")


if __name__ == "__main__":
    main()
