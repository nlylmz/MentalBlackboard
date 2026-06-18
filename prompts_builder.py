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

from vllm import LLM, SamplingParams
# ---------- your builder ----------
from open_source_evals.prompts_builder import build_prediction_video_prompt


# -----------------------------
# PyAV decoding
# -----------------------------
def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder using the given indices.
    Returns: np.ndarray (num_frames, H, W, 3) uint8
    """
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
    Decode ALL frames to a uint8 numpy array (T, H, W, 3) via read_video_pyav().
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        total = int(stream.frames) if stream.frames is not None else 0
    except Exception:
        total = 0

    if total > 0:
        indices = list(range(total))
        clip = read_video_pyav(container, indices)
        container.close()
        return clip

    # Fallback: count frames then decode
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
# Processor/model helpers
# -----------------------------
def is_internvl(processor, model_name: str) -> bool:
    cls = processor.__class__.__name__
    t = str(type(processor)).lower()
    mn = model_name.lower()
    return ("internvl" in mn) or ("internvl" in t) or ("InternVL" in cls)


def _get_size_dict(proc_obj):
    s = getattr(proc_obj, "size", None)
    if isinstance(s, dict) and "height" in s and "width" in s:
        return {"height": int(s["height"]), "width": int(s["width"])}
    return None


def _round_to_multiple(x: int, mul: int = 14) -> int:
    # round to nearest multiple of mul (bias upward if tie)
    return int(round(x / mul)) * mul or mul


def choose_internvl_common_size(processor) -> dict:
    """
    Decide a patch-aligned common size for InternVL. Prefer 448x448 if present,
    else use the larger of image/video sizes (rounded to multiples of 14).
    """
    ip = getattr(processor, "image_processor", None)
    vp = getattr(processor, "video_processor", None)
    img_sz = _get_size_dict(ip) if ip is not None else None
    vid_sz = _get_size_dict(vp) if vp is not None else None

    # If either is exactly 448x448, pick that.
    for sz in (img_sz, vid_sz):
        if sz and sz.get("height") == 448 and sz.get("width") == 448:
            return {"height": 448, "width": 448}

    # Else pick the larger dims we see, then round to multiples of 14.
    h = max((img_sz or {"height": 0})["height"], (vid_sz or {"height": 0})["height"], 448)
    w = max((img_sz or {"width": 0})["width"], (vid_sz or {"width": 0})["width"], 448)
    h = _round_to_multiple(h, 14)
    w = _round_to_multiple(w, 14)
    return {"height": h, "width": w}


def set_internvl_common_size(processor, common: dict):
    """
    Force both image_processor.size and video_processor.size to `common`.
    """
    ip = getattr(processor, "image_processor", None)
    vp = getattr(processor, "video_processor", None)
    if ip is not None:
        ip.size = {"height": common["height"], "width": common["width"]}
        setattr(ip, "do_resize", True)
    if vp is not None:
        vp.size = {"height": common["height"], "width": common["width"]}
        setattr(vp, "do_resize", True)


def np_clip_to_pil_list(clip: np.ndarray) -> List[Image.Image]:
    return [Image.fromarray(frame) for frame in clip]


# -----------------------------
# Prompt/messages
# -----------------------------
def build_messages_onevision(base_prompt: str, include_image: bool) -> list:
    """
    Single user turn: [VIDEO] + (optional [IMAGE]) + [TEXT].
    Template will place <video>/<image> tokens.
    """
    content = [{"type": "video"}]
    if include_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": base_prompt})
    return [{"role": "user", "content": content}]


# -----------------------------
# Decode: slice by prompt length
# -----------------------------
def extract_from_prompt_length(processor, outputs, inputs) -> str:
    """
    Extract assistant response by slicing off prompt token length.
    """
    # normalize outputs to 1D tensor
    if hasattr(outputs, "shape"):
        gen_ids = outputs[0] if outputs.dim() == 2 else outputs
    else:
        gen_ids = outputs[0]

    prompt_len = None
    try:
        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            if hasattr(input_ids, "shape"):
                prompt_len = input_ids.shape[-1]
            else:
                prompt_len = len(input_ids[0]) if isinstance(input_ids[0], (list, tuple)) else len(input_ids)
    except Exception:
        prompt_len = None

    def _decode(ids) -> str:
        txt = ""
        try:
            if hasattr(processor, "decode"):
                txt = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            elif getattr(processor, "tokenizer", None) is not None:
                txt = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            elif hasattr(processor, "batch_decode"):
                txt = processor.batch_decode(ids.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        except Exception:
            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                txt = tok.decode(ids, skip_special_tokens=True)
        return (txt or "").strip()

    if prompt_len is not None and prompt_len < gen_ids.shape[0]:
        sliced = gen_ids[prompt_len:]
        text = _decode(sliced)
        if text:
            return text

    text_full = _decode(gen_ids)
    if text_full:
        return text_full

    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            return tok.decode(gen_ids, skip_special_tokens=False).strip()
    except Exception:
        pass
    return ""


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PFHP predictions from (video + image + text) using ALL frames via PyAV (Llava-OneVision & InternVL)."
    )

    # Model
    p.add_argument("--model-name", default="llava-hf/llava-onevision-qwen2-7b-si-hf")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default="", help='Set "auto" to shard; else manual .to(device).')
    p.add_argument("--trust-remote-code", action="store_true")

    # Data / Inputs
    p.add_argument("--json-path",
                   default="./data/PFHP_Prediction_Data.json")
    p.add_argument("--video_root",
                   default="./folding_videos")
    p.add_argument("--location-img",
                   default="./paper_structure.png")
    p.add_argument("--limit", type=int, default=None)

    # Generation
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # Output / Logging
    p.add_argument("--save-dir", default="./pfhp_open_source_evals")
    p.add_argument("--save-file", default=None,
                   help="If unset: <save-dir>/Prediction_Video/<model>_Prediction.jsonl")
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

    # Seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device / dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    logging.info(f"Using device={device}, dtype={dtype}")

    # Processor / Model
    trust_remote = args.trust_remote_code or ("NVILA" in args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=trust_remote)
    # logging.info(f"Loaded processor: {processor.__class__.__name__}")

    # if args.device_map.strip().lower() == "auto":
    #     model = AutoModelForImageTextToText.from_pretrained(
    #         args.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=trust_remote
    #     )
    # else:
    #     model = AutoModelForImageTextToText.from_pretrained(
    #         args.model_name, torch_dtype=dtype, trust_remote_code=trust_remote
    #     )
    #     model.to(device)
    # model.eval()
    logging.info("Model loaded.")

    # tokenizer = getattr(processor, "tokenizer", None)
    # eos_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None) or (
    #     tokenizer.eos_token_id if tokenizer is not None else None
    # )
    # pad_id = getattr(getattr(model, "generation_config", None), "pad_token_id", None) or (
    #     getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    # )

    # InternVL-only: force common, patch-aligned size and let processor resize
    # use_internvl = is_internvl(processor, args.model_name)
    # common_size = None
    # if use_internvl:
    #     common_size = choose_internvl_common_size(processor)
    #     set_internvl_common_size(processor, common_size)
    #     logging.info(f"[InternVL] Forced common processor size: {common_size} (patch-aligned)")
    use_internvl = False

    # Load PFHP JSON
    with open(args.json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {args.json_path}, got {type(records).__name__}")
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

    llm_inputs = []
    ids = []
    for vf in tqdm(video_files, desc="Videos"):
        video_path = os.path.join(args.video_root, vf)
        base = os.path.splitext(os.path.basename(vf))[0]
        if base.startswith("PF_") and base.endswith("_animation"):
            id_num = base.replace("PF_", "").replace("_animation", "")
            id_ = f"PFHP_{id_num}"
            ids.append(id_)
        else:
            if args.verbose:
                print(f"[skip] Unrecognized video name: {vf}")
            continue

        item = data_by_id.get(id_)
        if not item:
            if args.verbose:
                print(f"[skip] No JSON record for {id_}")
            continue

        initial_holes = item.get("initialHoles", [])
        if not initial_holes:
            if args.verbose:
                print(f"[skip] No initial holes for {id_}")
            continue

        # Build base prompt
        try:
            base_prompt = build_prediction_video_prompt(id_, initial_holes)
        except TypeError:
            base_prompt = build_prediction_video_prompt(id_, item)

        # Read ALL frames via PyAV
        try:
            video_clip_np = read_all_frames_with_pyav(video_path)  # (T, H, W, 3) uint8
        except Exception as e:
            logging.error(f"Failed to read frames for {vf}: {e}")
            continue

        if video_clip_np.size == 0:
            if args.verbose:
                print(f"[skip] No frames extracted from {vf}")
            continue

        # For InternVL: pass frames as list[PIL] so processor applies its own resizing precisely
        if use_internvl:
            video_frames = np_clip_to_pil_list(video_clip_np)
        else:
            # Non-InternVL processors are fine with numpy or PIL; numpy is efficient
            video_frames = video_clip_np

        # Build messages: one video + optional image + text
        messages = build_messages_onevision(base_prompt, include_image=(location_img_pil is not None))

        # Render chat template
        # try:
        #     prompt_text = processor.apply_chat_template(
        #         messages,
        #         add_generation_prompt=True,
        #         tokenize=False,
        #         return_dict=False,
        #     )
        # except Exception as e:
        #     logging.error(f"apply_chat_template failed for {id_}: {e}")
        #     # Fallback raw prompt with tokens
        #     parts = ["<video>"]
        #     if location_img_pil is not None:
        #         parts.append("<image>")
        #     parts.append(base_prompt.strip())
        #     parts.append("ASSISTANT:")
        #     prompt_text = "\n".join(parts)

        # Build processor inputs
        # proc_kwargs = dict(
        #     # text=prompt_text,
        #     videos=video_frames,      # list[PIL] for InternVL, numpy(T,H,W,3) otherwise
        #     padding=True,
        #     return_tensors="pt",
        # )
        # if location_img_pil is not None:
        #     proc_kwargs["images"] = location_img_pil  # single PIL image
        prompt = processor.apply_chat_template(
            messages,
            tokenizer=False,
            add_generation_prompt=True,
        )
        image_inputs = [location_img_pil]
        llm_input = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs, "video": video_frames}
        }
        llm_inputs.append(llm_input)
        # break
        
        # try:
        #     inputs = processor(**proc_kwargs)
        #     print('input_keys: ', list(inputs.keys()))
        # except Exception as e_first:
        #     # InternVL can still be picky—retry without auxiliary image if a concat/reshape mismatch occurs
        #     if use_internvl and ("Sizes of tensors must match" in str(e_first) or "invalid for input of size" in str(e_first)):
        #         logging.warning("[InternVL] Preprocess mismatch; retrying without auxiliary image.")
        #         proc_kwargs.pop("images", None)
        #         inputs = processor(**proc_kwargs)
        #     else:
        #         logging.error(f"Processor call failed for {id_}: {e_first}")
        #         continue

        # Move to device if needed
        # if args.device_map.strip().lower() != "auto":
        #     inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        # Generate
        # gen_kwargs = dict(
        #     max_new_tokens=int(args.max_new_tokens),
        #     do_sample=bool(args.do_sample),
        #     temperature=float(args.temperature),
        #     top_p=float(args.top_p),
        #     eos_token_id=eos_id,
        #     pad_token_id=pad_id,
        # )
        # with torch.no_grad():
        #     outputs = model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})

        # # Decode via prompt-length slicing
        # response_text = extract_from_prompt_length(processor, outputs, inputs)

        # rows_out.append({
        #     "id": id_,
        #     "prompt": prompt_text,
        #     "response": response_text,
        # })

    vlm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.9,
        # limit_mm_per_prompt={"video":1}
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    outputs = vlm.generate(llm_inputs, sampling_params=sampling_params)
    # print(outputs)
    # Save JSONL
    print(f"Saving outputs to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        for idx, row in tqdm(enumerate(outputs), desc="Writing JSONL"):
            prompt = row.prompt
            generated_text = row.outputs[0].text
            f.write(json.dumps({"id": ids[idx], "prompt": prompt, "response": generated_text}) + "\n")

    print("Process Finished!!")


if __name__ == "__main__":
    main()
