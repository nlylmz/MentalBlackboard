import json
from collections import Counter
from copy import deepcopy
import re
import os

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
    """Convert (row, col, tri) back to # index (1-based)."""
    return row * 8 + col * 2 + tri + 1

def parse_direction_value(val):
    """
    Parses a direction string like '90,270' or '(90, 270)' into a set of normalized angle strings.
    Returns:
        - None → wildcard
        - set of str → normalized angles like {'90', '270'}
    """
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
    """
    Returns True if the predicted direction matches any acceptable ground truth direction.
    Wildcard GT ('') or None matches anything.
    """
    gt_set = parse_direction_value(gt_dir)
    pred_set = parse_direction_value(pred_dir)
    if gt_set is None:
        return True  # Wildcard
    if pred_set is None:
        return False
    return not gt_set.isdisjoint(pred_set)

def normalize_hole(hole):
    # Accept None
    if not isinstance(hole, dict):
        return {"shape": "", "size": "", "direction": "", "location": -1}

    # Get location
    location = hole.get("location")

    # Convert from [row, col, tri] list to flat index
    if isinstance(location, list) and len(location) == 3:
        try:
            location = pos_to_index(*location)
        except Exception:
            location = -1
    elif location is None:
        location = -1
    else:
        try:
            location = int(location)
        except (TypeError, ValueError):
            location = -1

    return {
        "shape": str(hole.get("shape", "") or ""),
        "size": str(hole.get("size", "") or ""),
        "direction": str(hole.get("direction", "") or ""),
        "location": location,
    }

def get_group(pfhp_number: int) -> int:
    """Map PFHP_number ranges to groups"""
    if 1 <= pfhp_number <= 100: return 9
    if 101 <= pfhp_number <= 200: return 8
    if 201 <= pfhp_number <= 300: return 7
    if 301 <= pfhp_number <= 400: return 6
    if 401 <= pfhp_number <= 500: return 5
    if 501 <= pfhp_number <= 600: return 4
    if 601 <= pfhp_number <= 700: return 3
    if 701 <= pfhp_number <= 800: return 2
    if 801 <= pfhp_number <= 900: return 1
    return 0  # fallback if out of range

def extract_pfhp_number(rec_id: str) -> int:
    """Extract the numeric part from an id like 'PFHP_123'."""
    match = re.search(r'PFHP_(\d+)', rec_id)
    return int(match.group(1)) if match else -1

def _repair_json_like(s: str) -> str:
    # Quote bare unfolding tokens like D1-F, H2-F, V1-F, etc.
    s = re.sub(r'(?<!")\b([DHV]\d-F)\b(?!")', r'"\1"', s)
    # Remove trailing commas before closing ] or }
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    # If someone used single quotes everywhere, we can try to replace them (careful)
    # s = s.replace("'", '"')
    return s

def normalize_unfolding(unfolding):
    """
    Normalize unfoldingTypes to a list of tokens.
    Accepts:
      - None -> []
      - list -> strip tokens
      - string 'A,B,C' -> split by comma
      - single token string -> [token]
    """
    if unfolding is None:
        return []
    if isinstance(unfolding, list):
        return [str(x).strip() for x in unfolding if str(x).strip()]
    if isinstance(unfolding, str):
        # if empty string -> []
        s = unfolding.strip()
        if s == "":
            return []
        # split by comma or whitespace+comma
        parts = [p.strip() for p in re.split(r'[,\s]+', s) if p.strip()]
        return parts
    # fallback: try to coerce to string and split
    return normalize_unfolding(str(unfolding))

def extract_prediction(record):
    """
    From a prediction record (which may have different shapes), extract a dict:
      {
        "resultHoles": [...],  # list of hole dicts
        "totalNumberofHoles": int,
        "unfoldingTypes": [...],  # normalized list
      }
    If record has an "error" key, return an empty prediction structure so it is scored as missing.
    """
    try:
        # Handle explicit error records
        if "error" in record:
            return {
                "resultHoles": [],
                "totalNumberofHoles": 0,
                "unfoldingTypes": []
            }

        # If record already appears to be the parsed JSON with fields we want, use them
        if any(k in record for k in ("resultHoles", "totalNumberofHoles", "unfoldingTypes")):
            res_holes = record.get("resultHoles", []) or []
            total = record.get("totalNumberofHoles", len(res_holes))
            unfolding = normalize_unfolding(record.get("unfoldingTypes", []))
            return {
                "resultHoles": res_holes,
                "totalNumberofHoles": total if isinstance(total, int) else int(total) if str(total).isdigit() else len(res_holes),
                "unfoldingTypes": unfolding
            }

        # Otherwise, many model responses are embedded in a "response" text blob or nested structure.
        content_str = ""
        if isinstance(record.get("response"), str):
            content_str = record.get("response").strip()
        elif isinstance(record.get("raw_response"), str):
            # sometimes raw_response contains text
            content_str = record.get("raw_response").strip()
        elif "result" in record and isinstance(record["result"], dict) and "message" in record["result"]:
            blocks = record["result"]["message"].get("content", [])
            text_blocks = [c.get("text", "") for c in blocks if c.get("type") == "text"]
            content_str = "\n".join(text_blocks).strip()
        elif isinstance(record.get("response"), dict):
            # OpenAI-like response object
            try:
                content_str = record["response"]["body"]["choices"][0]["message"]["content"].strip()
            except Exception:
                content_str = ""
        else:
            # fallback: try to stringify record
            content_str = json.dumps(record)

        if not content_str:
            raise ValueError("Empty response content")

        # 2) Collect candidate JSON blobs
        candidates = []

        # Prefer fenced ```json ... ``` blocks
        candidates += re.findall(r"```json\s*([\s\S]*?)```", content_str, flags=re.IGNORECASE)
        # Also accept generic fenced blocks (``` ... ```) in case the tag isn't "json"
        candidates += re.findall(r"```\s*([\s\S]*?)```", content_str)
        # If nothing fenced, fall back to any {...} blocks
        if not candidates:
            brace_blocks = re.findall(r"\{[\s\S]*\}", content_str)
            preferred = [b for b in brace_blocks if ("resultHoles" in b or "totalNumberofHoles" in b or "unfoldingTypes" in b)]
            candidates = preferred if preferred else brace_blocks

        if not candidates:
            raise ValueError("No JSON-looking block found in response content.")

        # 3) Try candidates from the end (the final block is usually the answer)
        for raw in reversed(candidates):
            raw_strip = raw.strip()
            # First try as-is
            try:
                parsed = json.loads(raw_strip)
                # Normalize fields for safety
                parsed_out = {
                    "resultHoles": parsed.get("resultHoles", []) or [],
                    "totalNumberofHoles": parsed.get("totalNumberofHoles", len(parsed.get("resultHoles", []) or [])),
                    "unfoldingTypes": normalize_unfolding(parsed.get("unfoldingTypes", []))
                }
                return parsed_out
            except json.JSONDecodeError:
                pass
            # Then try repairing typical issues (unquoted D/H/V tokens, trailing commas)
            repaired = _repair_json_like(raw_strip)
            try:
                parsed = json.loads(repaired)
                parsed_out = {
                    "resultHoles": parsed.get("resultHoles", []) or [],
                    "totalNumberofHoles": parsed.get("totalNumberofHoles", len(parsed.get("resultHoles", []) or [])),
                    "unfoldingTypes": normalize_unfolding(parsed.get("unfoldingTypes", []))
                }
                return parsed_out
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON could be parsed from candidates.")

    except Exception as e:
        # If parsing fails, return an empty/safe prediction (counts as missing)
        # Optionally log: print(f" Failed to parse prediction for {record.get('custom_id', record.get('id', 'UNKNOWN'))}: {e}")
        return {
            "resultHoles": [],
            "totalNumberofHoles": 0,
            "unfoldingTypes": []
        }

def compare_result_holes(record_a, record_b):
    gt_holes = [normalize_hole(h) for h in record_a.get("resultHoles", [])]
    pred_holes = [normalize_hole(h) for h in record_b.get("resultHoles", [])]

    total_gt = record_a.get("totalNumberofHoles", len(gt_holes))
    total_pred = record_b.get("totalNumberofHoles", len(pred_holes))
    extra_holes = max(0, total_pred - total_gt)

    # --- Exact hole match (respects multi-direction + wildcard)
    unmatched_gt = deepcopy(gt_holes)
    unmatched_pred = deepcopy(pred_holes)
    matched_pairs = []
    used_pred = set()

    for i, gt in enumerate(unmatched_gt):
        for j, pred in enumerate(unmatched_pred):
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

    fields_to_check = ["shape", "size", "direction", "location"]
    field_counts = {}

    for field in fields_to_check:
        if field == "direction":
            gt_vals = [h.get("direction", "") for h in gt_holes]
            pred_vals = [h.get("direction", "") for h in pred_holes]

            matched = 0
            used_pred = set()
            for i, gt_val in enumerate(gt_vals):
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

    # --- Compound field matching
    def match_compound(gt_list, pred_list, fields):
        used_pred = set()
        matched = 0
        for i, gt in enumerate(gt_list):
            for j, pred in enumerate(pred_list):
                if j in used_pred:
                    continue
                ok = True
                for f in fields:
                    if f == "direction":
                        if not direction_match(gt.get("direction", ""), pred.get("direction", "")):
                            ok = False
                            break
                    else:
                        if gt.get(f) != pred.get(f):
                            ok = False
                            break
                if ok:
                    matched += 1
                    used_pred.add(j)
                    break
        gt_total = len(gt_list)
        pred_total = len(pred_list)
        extra = max(0, pred_total - gt_total)
        denom = gt_total + extra
        return {
            "matched": matched,
            "gt_total": gt_total,
            "pred_total": pred_total,
            "extra": extra,
            "hole_custom_score": matched / denom if denom > 0 else 0.0,
        }

    compound_fields = {
        "location_shape": ("location", "shape"),
        "location_direction": ("location", "direction"),
        "shape_direction": ("shape", "direction"),
        "location_shape_direction": ("location", "shape", "direction")
    }

    for name, fields in compound_fields.items():
        field_counts[name] = match_compound(gt_holes, pred_holes, fields)

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

def compare_json_files(file_a, file_b):
    data_a = load_json_or_jsonl(file_a)
    data_b = load_json_or_jsonl(file_b)

    dict_b = {
        record.get("custom_id", record.get("id")): record
        for record in data_b
    }

    group_stats = {i: {"total": 0, "exact_match": 0, "unfolding_exact": 0} for i in range(1, 10)}

    all_records_stats = {}
    overall_stats = {
        "correct": 0,
        "incorrect": 0,
        "missing": 0,
    }

    extra_hole_records = 0
    missing_hole_records = 0

    unfolding_exact_match = 0
    unfolding_partial_match_total = 0
    unfolding_total_steps = 0

    # Detect if any prediction contains direction field (after extraction)
    pred_has_direction = False
    for record in dict_b.values():
        pred = extract_prediction(record)
        if any(h.get("direction") not in (None, "", "None") for h in pred.get("resultHoles", [])):
            pred_has_direction = True
            break

    # Always include basic fields
    fields_to_track = ["shape", "size", "location", "location_shape"]

    # Only include direction-related fields if direction is present in predictions
    if pred_has_direction:
        fields_to_track += [
            "direction",
            "location_direction",
            "shape_direction",
            "location_shape_direction"
        ]

    global_field_totals = {
        field: {"matched": 0, "gt_total": 0, "pred_total": 0, "extra": 0}
        for field in fields_to_track
    }

    for record_a in data_a:
        rec_id = record_a["id"]

        if rec_id not in dict_b:
            # skip missing record in predictions
            continue

        record_b_raw = dict_b[rec_id]
        record_b = extract_prediction(record_b_raw)

        # Ensure unfoldingTypes normalized on both sides
        record_a_norm = dict(record_a)  # shallow copy
        record_a_norm["unfoldingTypes"] = normalize_unfolding(record_a.get("unfoldingTypes", []))
        record_b["unfoldingTypes"] = normalize_unfolding(record_b.get("unfoldingTypes", []))

        # Ensure totalNumberofHoles exists
        record_a_norm["totalNumberofHoles"] = record_a.get("totalNumberofHoles", len(record_a.get("resultHoles", []) or []))
        record_b["totalNumberofHoles"] = record_b.get("totalNumberofHoles", len(record_b.get("resultHoles", []) or []))

        stats = compare_result_holes(record_a_norm, record_b)

        if stats["exact_match_accuracy"] == 1.0:
            print(f" Exact Hole Match: {rec_id}")

        if record_a_norm.get("unfoldingTypes", []) == record_b.get("unfoldingTypes", []):
            print(f" Unfolding Exact Match: {rec_id}")

        pfhp_number = extract_pfhp_number(rec_id)
        group = get_group(pfhp_number)

        if group in group_stats:
            group_stats[group]["total"] += 1

            # record-level exact hole matches
            if stats["exact_match_accuracy"] == 1.0:
                group_stats[group]["exact_match"] += 1

            # record-level exact unfolding matches
            if record_a_norm.get("unfoldingTypes", []) == record_b.get("unfoldingTypes", []):
                group_stats[group]["unfolding_exact"] += 1

        total_gt = record_a_norm.get("totalNumberofHoles", len(record_a_norm.get("resultHoles", [])))
        total_pred = record_b.get("totalNumberofHoles", len(record_b.get("resultHoles", [])))

        if total_pred > total_gt:
            extra_hole_records += 1
            print(f" Extra holes in: {rec_id}  (GT={total_gt}, Pred={total_pred})")
        elif total_pred < total_gt:
            missing_hole_records += 1
            print(f" Missing holes in: {rec_id}  (GT={total_gt}, Pred={total_pred})")

        gt_unfolding = record_a_norm.get("unfoldingTypes", [])
        pred_unfolding = record_b.get("unfoldingTypes", [])

        if gt_unfolding == pred_unfolding:
            unfolding_exact_match += 1

        min_len = min(len(gt_unfolding), len(pred_unfolding))
        correct_in_position = sum(
            1 for i in range(min_len) if gt_unfolding[i] == pred_unfolding[i]
        )
        unfolding_partial_match_total += correct_in_position
        unfolding_total_steps += len(gt_unfolding)

        for field, fstats in stats['field_accuracy'].items():
            if field in global_field_totals:
                global_field_totals[field]["matched"] += fstats["matched"]
                global_field_totals[field]["gt_total"] += fstats["gt_total"]
                global_field_totals[field]["pred_total"] += fstats["pred_total"]
                global_field_totals[field]["extra"] += fstats["extra"]

        all_records_stats[rec_id] = stats
        overall_stats["correct"] += stats["correct"]
        overall_stats["incorrect"] += stats["incorrect"]
        overall_stats["missing"] += stats["missing"]

    total_records = 900  # keep your original baseline if desired
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
    print(f"  - Exact Match Accuracy (per record): {exact_match_accuracy:.2%} ({total_exact_match}/{total_records})")
    print(f"  - Overall Custom Score:              {overall_custom_score:.2%}")
    print(f"  - Records with Extra Holes:          {extra_hole_records} / {total_records} = {(extra_hole_records / total_records):.2%}")
    print(f"  - Records with Missing Holes:        {missing_hole_records} / {total_records} = {(missing_hole_records / total_records):.2%}")
    print(f"  - Exact Unfolding Matches:           {unfolding_exact_match} / {total_records} = {(unfolding_exact_match / total_records):.2%}")
    print(f"  - Unfolding Steps Correctly Matched: {unfolding_partial_match_total} / {unfolding_total_steps} = {(unfolding_partial_match_total / unfolding_total_steps) if unfolding_total_steps>0 else 0:.2%}")

    print("HOLE INFORMATION ACCURACY:")
    for field, fstats in global_field_totals.items():
        matched = fstats["matched"]
        gt_total = fstats["gt_total"]
        extra = fstats["extra"]
        denom = gt_total + extra
        custom_score = matched / denom if denom > 0 else 0.0
        print(f"  • {field}: {matched} / ({gt_total} + {extra}) = {custom_score:.2%}")
        fstats["custom_score"] = custom_score

    print("\nGROUP-WISE PERFORMANCE:")
    for g in sorted(group_stats.keys(), reverse=True):  # groups 9 → 1
        total = group_stats[g]["total"]
        exact = group_stats[g]["exact_match"]
        unfold = group_stats[g]["unfolding_exact"]
        if total > 0:
            exact_pct = exact / total
            unfold_pct = unfold / total
            print(f"  - Group {g}: Holes Exact {exact} / {total} = {exact_pct:.2%},  Unfolding Exact {unfold} / {total} = {unfold_pct:.2%}")
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
        **overall_stats
    }
