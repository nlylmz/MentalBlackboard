import argparse
import json
from collections import Counter
from copy import deepcopy
import re

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def normalize_hole(hole):
    return {
        "shape": str(hole["shape"]),
        "size": str(hole["size"]),
        "direction": str(hole["direction"]),
        "location": int(hole["location"]),
    }

def compare_number_of_folding_steps(gt_steps, pred_steps):
    return gt_steps == pred_steps

def safe_get_steps(record):
    return int(record.get("numberofFoldingSteps", -1))

def parse_direction_value(val):
    """Return a set of ints (mod 360) or None for blank."""
    if val is None: return None
    s = str(val).strip()
    if s == "": return None
    nums = re.findall(r'-?\d+', s)
    return {int(n) % 360 for n in nums} if nums else None

def direction_match(gt_dir, pred_dir):
    gt_set = parse_direction_value(gt_dir)
    pred_set = parse_direction_value(pred_dir)
    if gt_set is None:            # blank GT ⇒ wildcard
        return True
    if pred_set is None:
        return False
    return not gt_set.isdisjoint(pred_set)

def holes_equal(gt_hole, pred_hole):
    """Flexible equality: shape/size/location exact, direction via overlap."""
    for field in ("shape", "size"):
        if str(gt_hole.get(field, "")).strip().lower() != \
           str(pred_hole.get(field, "")).strip().lower():
            return False
    if int(gt_hole.get("location", -1)) != int(pred_hole.get("location", -1)):
        return False
    return direction_match(gt_hole.get("direction"), pred_hole.get("direction"))

def count_initial_holes(record):
    """Return the number of initial holes from the record (prediction)."""
    ih = record.get("initialHoles", [])
    if isinstance(ih, list):
        return len(ih)
    for k in ("totalNumberofInitialHoles", "numberOfInitialHoles", "initialHoleCount"):
        if k in record:
            try:
                return int(record[k])
            except Exception:
                pass
    return 0

def extract_pfhp_number(rec_id: str) -> int:
    match = re.search(r'PFHP_(\d+)', rec_id)
    return int(match.group(1)) if match else -1

def get_group(pfhp_number: int) -> int:
    if 1 <= pfhp_number <= 100: return 4
    if 101 <= pfhp_number <= 200: return 3
    if 201 <= pfhp_number <= 300: return 2
    if 301 <= pfhp_number <= 400: return 1
    return 0

def compare_result_holes(record_a, record_b):
    gt_holes = [normalize_hole(h) for h in record_a.get("resultHoles", [])]
    pred_holes = [normalize_hole(h) for h in record_b.get("resultHoles", [])]

    total_gt = record_a.get("totalNumberofHoles", len(gt_holes))
    total_pred = record_b.get("totalNumberofHoles", len(pred_holes))
    extra_holes = max(0, total_pred - total_gt)

    unmatched_gt = deepcopy(gt_holes)
    unmatched_pred = deepcopy(pred_holes)
    matched_pairs = []
    used_pred = set()

    for gt in unmatched_gt:
        for j, pred in enumerate(unmatched_pred):
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

    # Field-wise accuracy
    field_counts = {}
    for field in ["shape", "size", "direction", "location"]:
        gt_vals = [hole[field] for hole in gt_holes]
        pred_vals = [hole[field] for hole in pred_holes]
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

    # Compound fields
    compound_fields = {
        "location_shape": ("location", "shape"),
        "location_direction": ("location", "direction"),
        "shape_direction": ("shape", "direction"),
        "location_shape_direction": ("location", "shape", "direction")
    }
    for combo_name, fields in compound_fields.items():
        gt_keys = [tuple(hole[f] for f in fields) for hole in gt_holes]
        pred_keys = [tuple(hole[f] for f in fields) for hole in pred_holes]
        gt_counter = Counter(gt_keys)
        pred_counter = Counter(pred_keys)
        matched = sum(min(gt_counter[k], pred_counter[k]) for k in gt_counter)
        gt_total = sum(gt_counter.values())
        pred_total = sum(pred_counter.values())
        extra = max(0, pred_total - gt_total)
        denom = gt_total + extra
        field_counts[combo_name] = {
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

def holes_exact_match_content_only(record_a, record_b):
    """Compare resultHoles purely by content, ignoring totals and overrides."""
    def to_keys(rec):
        holes = [normalize_hole(h) for h in rec.get("resultHoles", [])]
        return Counter((h["shape"], h["size"], h["direction"], h["location"]) for h in holes)
    return to_keys(record_a) == to_keys(record_b)

def holes_match(list_a, list_b):
    """Check if two hole lists match by content only."""
    return Counter(
        (h["shape"], h["size"], h["direction"], int(h["location"]))
        for h in [normalize_hole(h) for h in list_a]
    ) == Counter(
        (h["shape"], h["size"], h["direction"], int(h["location"]))
        for h in [normalize_hole(h) for h in list_b]
    )

def compare_json_files(file_a, file_b):
    data_a = load_json(file_a)
    data_b = load_json(file_b)

    dict_b = {record["id"]: record for record in data_b}
    all_records_stats = {}

    wrongStepcorrectAnswer = 0

    overall_stats = {"correct": 0, "incorrect": 0, "missing": 0}
    extra_hole_records = 0
    missing_hole_records = 0

    records_marked_wrong = 0
    records_with_step_mismatch = 0
    records_with_invalid_initialHoles = 0

    fields_to_track = [
        "shape", "size", "direction", "location",
        "location_shape", "location_direction", "shape_direction", "location_shape_direction"
    ]
    global_field_totals = {
        field: {"matched": 0, "gt_total": 0, "pred_total": 0, "extra": 0}
        for field in fields_to_track
    }

    group_stats = {i: {"total": 0, "exact_match": 0} for i in range(1, 5)}

    for record_a in data_a:
        rec_id = record_a["id"]
        record_b = dict_b.get(rec_id, {
            "id": rec_id,
            "resultHoles": [],
            "totalNumberofHoles": 0,
            "numberofFoldingSteps": -1,
            "initialHoles": []
        })

        stats = compare_result_holes(record_a, record_b)
        if stats["exact_match_accuracy"] == 1.0:
            print(f" Exact Hole Match: {rec_id}")

            # (optional) also check unfolding match
        if record_a.get("unfoldingTypes", []) == record_b.get("unfoldingTypes", []):
            print(f" Unfolding Exact Match: {rec_id}")

        # Group assignment
        pfhp_number = extract_pfhp_number(rec_id)
        group = get_group(pfhp_number)
        if group in group_stats:
            group_stats[group]["total"] += 1

        total_gt = record_a.get("totalNumberofHoles", len(record_a.get("resultHoles", [])))
        total_pred = record_b.get("totalNumberofHoles", len(record_b.get("resultHoles", [])))
        if total_pred > total_gt:
            extra_hole_records += 1
            #print(f" Extra holes in: {rec_id}  (GT={total_gt}, Pred={total_pred})")
        elif total_pred < total_gt:
            missing_hole_records += 1
            #print(f" Missing holes in: {rec_id}  (GT={total_gt}, Pred={total_pred})")

        # Rules
        pred_initial_holes = count_initial_holes(record_b)
        invalid_initial_holes = pred_initial_holes > 2
        if invalid_initial_holes:
            records_with_invalid_initialHoles += 1

        steps_match = compare_number_of_folding_steps(
            safe_get_steps(record_a), safe_get_steps(record_b)
        )
        if not steps_match:
            records_with_step_mismatch += 1
            #print(f" Step mismatch: {rec_id} "
                  #f"(GT={safe_get_steps(record_a)}, Pred={safe_get_steps(record_b)})")

        # wrongStepcorrectAnswer: holes match content-only,
        # rule violation occurs, but exclude trivial initial=final cases
        holes_match_content_only = holes_exact_match_content_only(record_a, record_b)
        initial_equals_result = holes_match(record_b.get("initialHoles", []),
                                            record_b.get("resultHoles", []))

        if holes_match_content_only and ((not steps_match) or (pred_initial_holes > 2)):
            if not initial_equals_result:
                wrongStepcorrectAnswer += 1
                print(f" wrongStepcorrectAnswer: {rec_id}  "
                      f"(steps_match={steps_match}, initialHoles={pred_initial_holes})")

        # Override: if rule violation, mark wrong and reset exact_match_accuracy
        if invalid_initial_holes or not steps_match:
            records_marked_wrong += 1
            stats['exact_match_accuracy'] = 0.0

        if stats["exact_match_accuracy"] == 1.0:
            group_stats[group]["exact_match"] += 1

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
    print(f"  - Exact Match Accuracy (per record): {exact_match_accuracy:.2%} ({total_exact_match}/{total_records})")
    print(f"  - Overall Custom Score:              {overall_custom_score:.2%}")
    print(f"  - Records with Extra Holes:          {extra_hole_records} / {total_records}")
    print(f"  - Records with Missing Holes:        {missing_hole_records} / {total_records}")

    print("\nRECORD-LEVEL RULE VIOLATIONS:")
    print(f"  - Records marked WRONG (steps mismatch OR initialHoles > 2): {records_marked_wrong} / {total_records}")
    print(f"     • Step mismatches:                {records_with_step_mismatch}")
    print(f"     • initialHoles > 2:               {records_with_invalid_initialHoles}")
    print(f"     • wrongStepcorrectAnswer:         {wrongStepcorrectAnswer}")

    print("\nHOLE INFORMATION ACCURACY:")
    for field, fstats in global_field_totals.items():
        matched = fstats["matched"]
        gt_total = fstats["gt_total"]
        extra = fstats["extra"]
        denom = gt_total + extra
        custom_score = matched / denom if denom > 0 else 0.0
        print(f"  • {field}: {matched} / ({gt_total} + {extra}) = {custom_score:.2%}")
        fstats["custom_score"] = custom_score

    print("\nGROUP-WISE PERFORMANCE:")
    for g in sorted(group_stats.keys(), reverse=True):
        total = group_stats[g]["total"]
        correct = group_stats[g]["exact_match"]
        if total > 0:
            pct = correct / total
            print(f"  - Group {g}: {correct} / {total} = {pct:.2%}")
        else:
            print(f"  - Group {g}: no records")

    return all_records_stats, {
        "exact_match_accuracy": exact_match_accuracy,
        "overall_custom_score": overall_custom_score,
        "field_accuracy": global_field_totals,
        "wrongStepcorrectAnswer": wrongStepcorrectAnswer,
        "records_marked_wrong": records_marked_wrong,
        "records_with_step_mismatch": records_with_step_mismatch,
        "records_with_invalid_initialHoles": records_with_invalid_initialHoles,
        "records_with_extra_holes": extra_hole_records,
        "records_with_missing_holes": missing_hole_records,
        **overall_stats
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare JSON files.")
    parser.add_argument("ground_truth", help="Path to ground truth JSON file")
    parser.add_argument("model_results", help="Path to model results JSON file")

    args = parser.parse_args()

    overall_stats, field_accuracy = compare_json_files(
        args.ground_truth,
        args.model_results
    )