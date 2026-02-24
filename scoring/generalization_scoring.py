import json
import os
from collections import Counter
from copy import deepcopy
import re

import argparse


def load_json_or_jsonl(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    with open(file_path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        elif ext == ".json":
            return json.load(f)
        else:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]

def holes_equal(gt_hole: dict, pred_hole: dict) -> bool:
    """
    True if the two holes are considered the same for *exact-match* purposes.

    • shape, size and location must be identical (case-insensitive for strings).
    • direction follows the flexible rule used in field-level scoring:
        - overlaps ⇒ match       (e.g. "90,270" vs "270")
        - blank/None in GT ⇒ wildcard (always match)
    """
    # --- quick checks on the easy fields ------------------------------------
    for field in ("shape", "size"):
        if str(gt_hole.get(field, "")).strip().lower() != \
           str(pred_hole.get(field, "")).strip().lower():
            return False

    # location is numeric (GT loader already normalises triples → int)
    if int(gt_hole.get("location", -1)) != int(pred_hole.get("location", -1)):
        return False

    # --- flexible direction comparison --------------------------------------
    return direction_match(gt_hole.get("direction"), pred_hole.get("direction"))


def parse_direction_value(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    nums = re.findall(r'-?\d+', s)
    if not nums:
        return None
    return {str(int(n) % 360) for n in nums}

def direction_match(gt_dir, pred_dir):
    gt_set = parse_direction_value(gt_dir)
    pred_set = parse_direction_value(pred_dir)
    if gt_set is None:
        return True  # wildcard
    if pred_set is None:
        return False
    return not gt_set.isdisjoint(pred_set)

def pos_to_index(row: int, col: int, tri: int) -> int:
    return row * 8 + col * 2 + tri + 1

def extract_result_holes(record):
    holes = record.get("resultHoles", [])
    extracted = []
    for h in holes:
        try:
            location = h.get("location")
            if isinstance(location, list) and len(location) == 3:
                location = pos_to_index(*location)
            elif not isinstance(location, int):
                continue
            extracted.append({
                "shape": str(h.get("shape", "")).lower().strip(),
                "size": str(h.get("size", "")).lower().strip(),
                "direction": str(h.get("direction", "")).strip(),
                "location": location
            })
        except Exception as e:
            print(f"  Skipping invalid hole in record {record.get('id', 'UNKNOWN')}: {e}")
    return extracted

def extract_prediction(record):
    try:
        content_str = ""
        if "response" in record and isinstance(record["response"], str):
            content_str = record["response"].strip()
        elif "result" in record and "message" in record["result"]:
            content_blocks = record["result"]["message"].get("content", [])
            text_blocks = [c.get("text", "") for c in content_blocks if c.get("type") == "text"]
            content_str = "\n".join(text_blocks).strip()
        elif "response" in record and isinstance(record["response"], dict):
            content_str = record["response"]["body"]["choices"][0]["message"]["content"].strip()
        else:
            raise ValueError("Unknown record format")

        if not content_str:
            raise ValueError("Empty response content")

        fenced = re.findall(r"```json([\s\S]*?)```", content_str, flags=re.IGNORECASE)
        if fenced:
            json_str = fenced[-1].strip()
        else:
            matches = re.findall(r"\{[\s\S]*\}", content_str)
            if not matches:
                preview = content_str[:200].replace("\n", " ")
                raise ValueError(f"No JSON found. Content preview: {preview}")
            json_str = matches[-1].strip()

        return json.loads(json_str)

    except Exception as e:
        print(f" Failed to parse prediction for {record.get('custom_id', record.get('id', 'UNKNOWN'))}: {e}")
        return {"resultHoles": [], "totalNumberofHoles": 0}

def normalize_field_value(val, field):
    return str(val).strip().lower()

def compare_result_holes(record_a, record_b):
    gt_holes = extract_result_holes(record_a)
    pred_holes = extract_result_holes(record_b)

    total_gt = record_a.get("totalNumberofHoles", len(gt_holes))
    total_pred = record_b.get("totalNumberofHoles", len(pred_holes))
    extra_holes = max(0, total_pred - total_gt)

    matched_pairs = []
    used_pred = set()
    for i, gt in enumerate(gt_holes):
        for j, pred in enumerate(pred_holes):
            if j in used_pred:
                continue
            if holes_equal(gt, pred):
                matched_pairs.append((gt, pred))
                used_pred.add(j)
                break

    matched_count = len(matched_pairs)
    penalty = max(0, total_pred - total_gt)
    denominator = total_gt + penalty
    custom_score = matched_count / denominator if denominator > 0 else 0.0

    precision = matched_count / total_pred if total_pred > 0 else 1.0
    recall = matched_count / total_gt if total_gt > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match_accuracy = 1.0 if (total_pred == total_gt and matched_count == total_gt) else 0.0

    field_counts = {}
    for field in ["shape", "size", "direction", "location"]:
        gt_vals = [normalize_field_value(h.get(field, ""), field) for h in gt_holes]
        pred_vals = [normalize_field_value(h.get(field, ""), field) for h in pred_holes]

        if field == "direction":
            matched = 0
            used = [False] * len(pred_vals)
            for gt_val in gt_vals:
                gt_set = parse_direction_value(gt_val)
                if gt_set is None:
                    matched += 1  # wildcard
                    continue
                for i, pred_val in enumerate(pred_vals):
                    if used[i]:
                        continue
                    pred_set = parse_direction_value(pred_val)
                    if pred_set and not gt_set.isdisjoint(pred_set):
                        matched += 1
                        used[i] = True
                        break
            gt_total = len(gt_vals)
            pred_total = len(pred_vals)
        else:
            gt_counter = Counter(gt_vals)
            pred_counter = Counter(pred_vals)
            matched = sum(min(gt_counter[val], pred_counter[val]) for val in gt_counter)
            gt_total = sum(gt_counter.values())
            pred_total = sum(pred_counter.values())

        extra = max(0, pred_total - gt_total)
        denom = gt_total + extra
        field_counts[field] = {
            "matched": matched,
            "gt_total": gt_total,
            "pred_total": pred_total,
            "extra": extra,
            "hole_custom_score": matched / denom if denom > 0 else 0.0,
        }

    return {
        "exact_match_accuracy": exact_match_accuracy,
        "custom_score": custom_score,
        "recall": recall,
        "f1_score": f1,
        "correct": matched_count,
        "incorrect": total_pred - matched_count,
        "missing": total_gt - matched_count,
        "extra_holes": extra_holes,
        "field_accuracy": field_counts,
    }


def get_group_from_id(rec_id: str):
    rec_id_lower = rec_id.lower()
    if "direction" in rec_id_lower:
        return "direction"
    elif "size" in rec_id_lower:
        return "size"
    elif "shape" in rec_id_lower:
        return "shape"
    elif "location" in rec_id_lower:
        return "location"
    return "other"


def compare_json_files(file_a, file_b):
    data_a = load_json_or_jsonl(file_a)
    data_b = load_json_or_jsonl(file_b)

    dict_a = {
        record.get("custom_id", record.get("id")): record
        for record in data_a
    }

    group_stats = {g: {"total": 0, "exact_match": 0} for g in ["direction", "size", "shape", "location", "other"]}
    all_records_stats = {}
    overall_stats = {"correct": 0, "incorrect": 0, "missing": 0}
    extra_hole_records = 0
    missing_hole_records = 0

    global_field_totals = {f: {"matched": 0, "gt_total": 0, "pred_total": 0, "extra": 0}
                           for f in ["shape", "size", "direction", "location"]}
    gt_total_shape = 0
    for record_b_raw in data_b:
        custom_id = record_b_raw.get("custom_id", record_b_raw.get("id"))
        mod_id = custom_id + "_mod"

        if mod_id not in dict_a:
            continue

        record_a = dict_a[mod_id]
        gt_total_shape += len(record_a.get("resultHoles", []))

        record_b = extract_prediction(record_b_raw)
        stats = compare_result_holes(record_a, record_b)

        if stats["exact_match_accuracy"] == 1.0:
            print(f" Exact Hole Match: {mod_id}")

        group = get_group_from_id(mod_id)
        group_stats[group]["total"] += 1
        if stats["exact_match_accuracy"] == 1.0:
            group_stats[group]["exact_match"] += 1

        total_gt = record_a.get("totalNumberofHoles", len(record_a.get("resultHoles", [])))
        total_pred = record_b.get("totalNumberofHoles", len(record_b.get("resultHoles", [])))
        if total_pred > total_gt:
            extra_hole_records += 1
        elif total_pred < total_gt:
            missing_hole_records += 1

        for field, fstats in stats['field_accuracy'].items():
            global_field_totals[field]["matched"] += fstats["matched"]
            global_field_totals[field]["gt_total"] += fstats["gt_total"]
            global_field_totals[field]["pred_total"] += fstats["pred_total"]
            global_field_totals[field]["extra"] += fstats["extra"]

        all_records_stats[custom_id] = stats
        overall_stats["correct"] += stats["correct"]
        overall_stats["incorrect"] += stats["incorrect"]
        overall_stats["missing"] += stats["missing"]
    print("GT total shape field holes processed:", gt_total_shape)
    total_records = 240
    total_exact_match = sum(1 for s in all_records_stats.values() if s["exact_match_accuracy"] == 1.0)
    exact_match_accuracy = total_exact_match / total_records if total_records > 0 else 0.0

    print("\n==============================")
    print("OVERALL METRICS")
    print(f"  - Exact Match Accuracy (per record): {exact_match_accuracy:.2%} ({total_exact_match}/{total_records})")
    print(f"  - Records with Extra Holes:          {extra_hole_records} / {total_records}")
    print(f"  - Records with Missing Holes:        {missing_hole_records} / {total_records}")

    print("\nHOLE INFORMATION ACCURACY:")
    for field, fstats in global_field_totals.items():
        matched = fstats["matched"]
        gt_total = 708 #fstats["gt_total"]
        extra = fstats["extra"]
        denom = gt_total + extra
        custom_score = matched / denom if denom > 0 else 0.0
        print(f"  • {field}: {matched} / ({gt_total} + {extra}) = {custom_score:.2%}")

    print("\nGROUP-WISE PERFORMANCE:")
    for g, stats in group_stats.items():
        total = 60
        exact = stats["exact_match"]
        if total > 0:
            print(f"  - {g.capitalize()}: Exact {exact}/{total} = {exact/total:.2%}")
        else:
            print(f"  - {g.capitalize()}: no records")

    return all_records_stats, {
        "exact_match_accuracy": exact_match_accuracy,
        "records_with_extra_holes": extra_hole_records,
        "records_with_missing_holes": missing_hole_records,
        **overall_stats
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare JSON files.")
    parser.add_argument("ground_truth", help="Path to ground truth JSON file")
    parser.add_argument("model_results", help="Path to model results JSONL file")

    args = parser.parse_args()

    overall_stats, field_accuracy = compare_json_files(
        args.ground_truth,
        args.model_results
    )
