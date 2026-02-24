import json
from collections import Counter
from copy import deepcopy
import re
import os

# ------------------------------
# Utility Functions
# ------------------------------

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


def pos_to_index(row: int, col: int, tri: int) -> int:
    """Convert (row, col, tri) to a single 1-based index."""
    return row * 8 + col * 2 + tri + 1


# ------------------------------
# Direction Matching (flexible)
# ------------------------------

def parse_direction_value(val):
    """Parse direction like '0,180' → {'0','180'}, handle wildcards."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None  # wildcard
    nums = re.findall(r'-?\d+', s)
    if not nums:
        return None
    return {str(int(n) % 360) for n in nums}

def direction_match(gt_dir, pred_dir):
    """Flexible direction comparison allowing multiple valid values."""
    gt_set = parse_direction_value(gt_dir)
    pred_set = parse_direction_value(pred_dir)
    if gt_set is None:
        return True  # wildcard
    if pred_set is None:
        return False
    return not gt_set.isdisjoint(pred_set)


# ------------------------------
# Hole Normalization
# ------------------------------

def normalize_hole(hole):
    location = hole.get("location")
    if isinstance(location, list) and len(location) == 3:
        location = pos_to_index(*location)
    else:
        try:
            location = int(location)
        except (TypeError, ValueError):
            location = -1

    return {
        "shape": str(hole.get("shape", "")),
        "size": str(hole.get("size", "")),
        "direction": str(hole.get("direction", "")),
        "location": location,
    }


# ------------------------------
# ID and Group Helpers
# ------------------------------

def get_group(pfhp_number: int) -> int:
    if 1   <= pfhp_number <= 20:   return 9
    if 21  <= pfhp_number <= 40:   return 8
    if 41  <= pfhp_number <= 60:   return 7
    if 61  <= pfhp_number <= 80:   return 6
    if 81  <= pfhp_number <= 100:  return 5
    if 101 <= pfhp_number <= 120:  return 4
    if 121 <= pfhp_number <= 140:  return 3
    if 141 <= pfhp_number <= 160:  return 2
    if 161 <= pfhp_number <= 180:  return 1
    return 0


def extract_pfhp_number(rec_id: str) -> int:
    match = re.search(r'PFHP_(\d+)', rec_id)
    return int(match.group(1)) if match else -1

'''
def extract_prediction(record):
    """Extracts the JSON content from prediction record text."""
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

    except json.JSONDecodeError:
        json_str = re.sub(r'\b([DHV]\d-[BF])\b', r'"\1"', json_str)
        return json.loads(json_str)
    except Exception as e:
        print(f" Failed to parse prediction for {record.get('custom_id', record.get('id', 'UNKNOWN'))}: {e}")
        return {"resultHoles": [], "totalNumberofHoles": 0, "unfoldingTypes": []}
'''

import json
import re

def extract_prediction(record):
    """
    Extracts and parses JSON content from a prediction record.
    Handles multiple response formats and common JSON formatting errors.
    """
    try:
        # STEP 1: Extract content string
        content_str = ""

        # Case 1: Simple string response
        if "response" in record and isinstance(record["response"], str):
            content_str = record["response"].strip()

        # Case 2: Structured OpenAI-like format
        elif "result" in record and "message" in record["result"]:
            content_blocks = record["result"]["message"].get("content", [])
            text_blocks = [c.get("text", "") for c in content_blocks if c.get("type") == "text"]
            content_str = "\n".join(text_blocks).strip()

        # Case 3: Another variant with dictionary response
        elif "response" in record and isinstance(record["response"], dict):
            content_str = record["response"]["body"]["choices"][0]["message"]["content"].strip()

        else:
            raise ValueError("Unknown record format")

        if not content_str:
            raise ValueError("Empty response content")

        # STEP 2: Extract JSON from code block or inline
        fenced = re.findall(r"```json([\s\S]*?)```", content_str, flags=re.IGNORECASE)
        if fenced:
            json_str = fenced[-1].strip()
        else:
            matches = re.findall(r"\{[\s\S]*\}", content_str)
            if not matches:
                preview = content_str[:200].replace("\n", " ")
                raise ValueError(f"No JSON found. Content preview: {preview}")
            json_str = matches[-1].strip()

        # STEP 3: Try to parse raw JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[Warning] Initial JSON decode failed: {e}")
            print(f"Attempting to fix common issues...")

            # STEP 4: Clean common issues
            json_str = re.sub(r'\b([DHV]\d-[BF])\b', r'"\1"', json_str)  # unquoted IDs
            json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)           # trailing commas
            json_str = json_str.replace("'", '"')                        # single to double quotes

            # STEP 5: Retry parsing
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e2:
                print(f"[Error] Failed to parse JSON after cleanup: {e2}")
                print(f"Partial JSON string:\n{json_str[:500]}")
                raise

    except Exception as e:
        record_id = record.get('custom_id', record.get('id', 'UNKNOWN'))
        print(f"[Error] Failed to extract prediction for record {record_id}: {e}")
        return {
            "resultHoles": [],
            "totalNumberofHoles": 0,
            "unfoldingTypes": []
        }


# ------------------------------
# Hole Comparison
# ------------------------------

def compare_result_holes(record_a, record_b):
    gt_holes = [normalize_hole(h) for h in record_a.get("resultHoles", [])]
    pred_holes = [normalize_hole(h) for h in record_b.get("resultHoles", [])]

    total_gt = record_a.get("totalNumberofHoles", len(gt_holes))
    total_pred = record_b.get("totalNumberofHoles", len(pred_holes))
    extra_holes = max(0, total_pred - total_gt)

    # --- Exact match (flexible direction)
    matched_pairs = []
    used_pred = set()
    for gt in gt_holes:
        for j, pred in enumerate(pred_holes):
            if j in used_pred:
                continue
            same_non_dir = (
                gt["shape"] == pred["shape"] and
                gt["size"] == pred["size"] and
                gt["location"] == pred["location"]
            )
            if same_non_dir and direction_match(gt.get("direction", ""), pred.get("direction", "")):
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

    # --- Field accuracies
    fields_to_check = ["shape", "size", "direction", "location"]
    field_counts = {}

    for field in fields_to_check:
        if field == "direction":
            gt_vals = [h.get("direction", "") for h in gt_holes]
            pred_vals = [h.get("direction", "") for h in pred_holes]
            matched = 0
            used_pred = set()
            for gt_val in gt_vals:
                for j, pred_val in enumerate(pred_vals):
                    if j in used_pred:
                        continue
                    if direction_match(gt_val, pred_val):
                        matched += 1
                        used_pred.add(j)
                        break
            gt_total = len(gt_vals)
            pred_total = len(pred_vals)
            extra = max(0, pred_total - gt_total)
            denom = gt_total + extra
            field_counts[field] = {
                "matched": matched,
                "gt_total": gt_total,
                "pred_total": pred_total,
                "extra": extra,
                "hole_custom_score": matched / denom if denom > 0 else 0.0,
            }
        else:
            gt_vals = [h.get(field, "") for h in gt_holes]
            pred_vals = [h.get(field, "") for h in pred_holes]
            gt_counter = Counter(gt_vals)
            pred_counter = Counter(pred_vals)
            matched = sum(min(gt_counter[v], pred_counter[v]) for v in gt_counter)
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


# ------------------------------
# File Comparison
# ------------------------------

def compare_json_files(file_a, file_b):
    data_a = load_json_or_jsonl(file_a)
    data_b = load_json_or_jsonl(file_b)

    dict_b = {r.get("custom_id", r.get("id")): r for r in data_b}

    group_stats = {i: {"total": 0, "exact_match": 0, "unfolding_exact": 0} for i in range(1, 10)}
    all_records_stats = {}

    # Aggregates
    extra_hole_records = 0
    missing_hole_records = 0
    unfolding_exact_match = 0
    unfolding_partial_match_total = 0
    unfolding_total_steps = 0

    global_field_totals = {}

    for record_a in data_a:
        rec_id = record_a["id"]
        if rec_id not in dict_b:
            continue
        record_b = extract_prediction(dict_b[rec_id])
        stats = compare_result_holes(record_a, record_b)

        # Count exact unfolding matches
        gt_unf = record_a.get("unfoldingTypes", [])
        pr_unf = record_b.get("unfoldingTypes", [])
        if gt_unf == pr_unf:
            unfolding_exact_match += 1
        unfolding_partial_match_total += sum(1 for i in range(min(len(gt_unf), len(pr_unf))) if gt_unf[i] == pr_unf[i])
        unfolding_total_steps += len(gt_unf)

        total_gt = record_a.get("totalNumberofHoles", len(record_a.get("resultHoles", [])))
        total_pred = record_b.get("totalNumberofHoles", len(record_b.get("resultHoles", [])))
        if total_pred > total_gt:
            extra_hole_records += 1
        elif total_pred < total_gt:
            missing_hole_records += 1

        pfhp_number = extract_pfhp_number(rec_id)
        group = get_group(pfhp_number)
        if group in group_stats:
            group_stats[group]["total"] += 1
            if stats["exact_match_accuracy"] == 1.0:
                group_stats[group]["exact_match"] += 1
            if gt_unf == pr_unf:
                group_stats[group]["unfolding_exact"] += 1

        for field, fstats in stats["field_accuracy"].items():
            if field not in global_field_totals:
                global_field_totals[field] = {"matched": 0, "gt_total": 0, "pred_total": 0, "extra": 0}
            global_field_totals[field]["matched"] += fstats["matched"]
            global_field_totals[field]["gt_total"] += fstats["gt_total"]
            global_field_totals[field]["pred_total"] += fstats["pred_total"]
            global_field_totals[field]["extra"] += fstats["extra"]

        all_records_stats[rec_id] = stats

    total_records = len(all_records_stats)
    total_exact_match = sum(1 for s in all_records_stats.values() if s["exact_match_accuracy"] == 1.0)
    exact_match_accuracy = total_exact_match / total_records if total_records > 0 else 0.0

    overall_matched = sum(s["correct"] for s in all_records_stats.values())
    overall_gt_total = sum(s["correct"] + s["missing"] for s in all_records_stats.values())
    overall_penalty = sum(s["extra_holes"] for s in all_records_stats.values())
    overall_custom_score = (
        overall_matched / (overall_gt_total + overall_penalty)
        if (overall_gt_total + overall_penalty) > 0 else 0.0
    )

    print("\n==============================")
    print("OVERALL METRICS")
    print(f"  - Exact Match Accuracy:       {exact_match_accuracy:.2%} ({total_exact_match}/{total_records})")
    print(f"  - Overall Custom Score:       {overall_custom_score:.2%}")
    print(f"  - Records with Extra Holes:   {extra_hole_records}/{total_records} = {(extra_hole_records/total_records):.2%}")
    print(f"  - Records with Missing Holes: {missing_hole_records}/{total_records} = {(missing_hole_records/total_records):.2%}")
    print(f"  - Unfolding Exact Matches:    {unfolding_exact_match}/{total_records} = {(unfolding_exact_match/total_records):.2%}")
    print(f"  - Step-wise Unfolding Match:  {unfolding_partial_match_total}/{unfolding_total_steps} = {(unfolding_partial_match_total/unfolding_total_steps):.2%}")

    print("\nHOLE INFORMATION ACCURACY:")
    for field, fstats in global_field_totals.items():
        matched, gt_total, extra = fstats["matched"], fstats["gt_total"], fstats["extra"]
        denom = gt_total + extra
        custom_score = matched / denom if denom > 0 else 0.0
        fstats["custom_score"] = custom_score
        print(f"  • {field}: {matched}/({gt_total}+{extra}) = {custom_score:.2%}")

    print("\nGROUP-WISE PERFORMANCE:")
    for g in sorted(group_stats.keys(), reverse=True):
        total = group_stats[g]["total"]
        exact = group_stats[g]["exact_match"]
        unfold = group_stats[g]["unfolding_exact"]
        if total > 0:
            print(f"  - Group {g}: Holes Exact {exact}/{total} = {exact/total:.2%},  Unfolding Exact {unfold}/{total} = {unfold/total:.2%}")
        else:
            print(f"  - Group {g}: no records")

    return all_records_stats, {
        "exact_match_accuracy": exact_match_accuracy,
        "overall_custom_score": overall_custom_score,
        "field_accuracy": global_field_totals,
        "exact_unfolding_match_count": unfolding_exact_match,
        "partial_unfolding_match_counts": unfolding_partial_match_total,
        "total_unfolding_steps": unfolding_total_steps,
        "records_with_extra_holes": extra_hole_records,
        "records_with_missing_holes": missing_hole_records,
    }


# ------------------------------
# Run
# ------------------------------

if __name__ == "__main__":
    overall_stats, summary = compare_json_files(
        "../data_MBB/BackwardPrediction/PFHP_Back_Prediction_Data.json",
        "../model_results/BackwardPrediction/gpt-5.1_backward_pediction_video_frame_result.json"
    )
