#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate PFHP predictions *without* using the `direction` attribute.
"""

import json
import os
import re
from collections import Counter
from copy import deepcopy

import argparse as argparse


# ──────────────────────────────── I/O helpers ──────────────────────────────── #
def load_json_or_jsonl(file_path: str):
    """Return a list (JSONL) or dict/list (JSON) depending on file content."""
    ext = os.path.splitext(file_path)[-1].lower()
    with open(file_path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        elif ext == ".json":
            return json.load(f)
        else:  # best-effort fallback
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]


# ──────────────────────────────── utilities ───────────────────────────────── #
def pos_to_index(row: int, col: int, tri: int) -> int:
    """Convert (row, col, tri) to the 1-based flattened location index."""
    return row * 8 + col * 2 + tri + 1


def extract_pfhp_number(rec_id: str) -> int:
    """Pull the numeric part out of an ID like  'PFHP_123' → 123 ."""
    m = re.search(r"PFHP_(\d+)", rec_id)
    return int(m.group(1)) if m else -1


def get_group(pfhp_number: int) -> int:
    """Map PFHP numbers to 9 difficulty buckets; bucket 0 = out of range."""
    if 1 <= pfhp_number <= 100:   return 9
    if 101 <= pfhp_number <= 200: return 8
    if 201 <= pfhp_number <= 300: return 7
    if 301 <= pfhp_number <= 400: return 6
    if 401 <= pfhp_number <= 500: return 5
    if 501 <= pfhp_number <= 600: return 4
    if 601 <= pfhp_number <= 700: return 3
    if 701 <= pfhp_number <= 800: return 2
    if 801 <= pfhp_number <= 900: return 1
    return 0


# ──────────────────────────────── extraction ──────────────────────────────── #
def _repair_json_like(s: str) -> str:
    """Lightweight fixer for common JSON-formatting errors."""
    s = re.sub(r'(?<!")\b([DHV]\d-F)\b(?!")', r'"\1"', s)  # quote bare tokens
    s = re.sub(r',(\s*[}\]])', r'\1', s)                   # strip trailing ,
    return s


def extract_prediction(record):
    """
    Robustly locate & parse the JSON answer embedded in a record.
    Returns a dict with at least keys:
       • resultHoles          (list)
       • totalNumberofHoles   (int)
       • unfoldingTypes       (list)
    On any error, returns an empty-answer stub.
    """
    try:
        content = ""
        # multiple dataset formats supported:
        if isinstance(record.get("response"), str):
            content = record["response"].strip()
        elif "result" in record and "message" in record["result"]:
            blocks = record["result"]["message"].get("content", [])
            text_blocks = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            content = "\n".join(text_blocks).strip()
        elif isinstance(record.get("response"), dict):
            content = record["response"]["body"]["choices"][0]["message"]["content"].strip()
        else:
            raise ValueError("unknown record format")

        if not content:
            raise ValueError("empty response")

        # candidate JSON snippets
        cand = []
        cand += re.findall(r"```json\s*([\s\S]*?)```", content, flags=re.I)
        cand += re.findall(r"```\s*([\s\S]*?)```",        content, flags=re.I)
        if not cand:
            braces = re.findall(r"\{[\s\S]*\}", content)
            pref   = [b for b in braces if ("resultHoles" in b or "totalNumberofHoles" in b)]
            cand   = pref or braces

        if not cand:
            raise ValueError("no JSON-looking block found")

        for raw in reversed(cand):
            try:
                return json.loads(raw.strip())
            except json.JSONDecodeError:
                try:
                    return json.loads(_repair_json_like(raw))
                except json.JSONDecodeError:
                    continue

        raise ValueError("all candidate blocks failed to parse")

    except Exception as e:
        rid = record.get("custom_id", record.get("id", "UNKNOWN"))
        print(f"  ✖ prediction parse failed for {rid}: {e}")
        return {"resultHoles": [], "totalNumberofHoles": 0, "unfoldingTypes": []}


# ──────────────────────────────── normalization ───────────────────────────── #
import re

def _parse_location(raw, default_tri=0, sentinel=-1):
    """
    Convert location → 1-based integer index.
    Returns `sentinel` (default −1) when location is missing/blank.
    """
    # ─── 0. Missing or explicitly None ────────────────────────────
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return sentinel          # <-- guarantees downstream code still runs

    # ─── 1. List/tuple cases ──────────────────────────────────────
    if isinstance(raw, (list, tuple)):
        if len(raw) == 3:
            return pos_to_index(*map(int, raw))
        if len(raw) == 2:
            return pos_to_index(int(raw[0]), int(raw[1]), default_tri)

    # ─── 2. Scalar numeric or numeric‐string ──────────────────────
    s = str(raw).strip()
    if s.lstrip("-").isdigit():
        return int(s)

    # ─── 3. Comma / parenthesized string ─────────────────────────
    nums = re.findall(r'-?\d+', s)
    if len(nums) == 3:
        return pos_to_index(*map(int, nums))
    if len(nums) == 2:
        r, c = map(int, nums)
        return pos_to_index(r, c, default_tri)

    # ─── 4. Anything else ─────────────────────────────────────────
    return sentinel              # or raise if you *want* to stop



import json
import re

def _coerce_hole(h):
    """
    Accepts either a dict (normal case) or a string.
    • If it's already a dict, return as-is.
    • If it's a string, try to parse JSON; if that fails, do a best-effort
      regex pull for shape/size/location.
    """
    if isinstance(h, dict):
        return h

    # 1. Try JSON first
    if isinstance(h, str):
        s = h.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # 2. Regex fallback:  "ellipse large (2,1,1)"  etc.
        m = re.search(r'(?P<shape>[A-Za-z]+)\s+'
                      r'(?P<size>small|large)\s+'
                      r'\(?\s*(?P<row>\d+)[,\s]+(?P<col>\d+)[,\s]+(?P<tri>\d+)\s*\)?',
                      s, flags=re.I)
        if m:
            return {
                "shape": m.group("shape").lower(),
                "size":  m.group("size").lower(),
                "location": [int(m.group("row")),
                             int(m.group("col")),
                             int(m.group("tri"))]
            }

    # 3. Give up—return an empty stub so the hole simply won’t match
    return {"shape": "", "size": "", "location": None}


def normalize_hole(raw_hole, default_size=""):
    """
    Canonicalize one hole object.  Safe for dicts *or* strings.
    Direction is deliberately ignored.
    """
    hole = _coerce_hole(raw_hole)      # <── new line

    return {
        "shape": str(hole.get("shape", "")),
        "size":  str(hole.get("size",  default_size)),
        "location": _parse_location(hole.get("location")),
    }




# ──────────────────────────────── core comparison ─────────────────────────── #
def compare_result_holes(gt_rec, pred_rec):
    """Compute per-record metrics, ignoring direction completely."""
    gt_holes   = [normalize_hole(h) for h in gt_rec.get("resultHoles", [])]
    pred_holes = [normalize_hole(h) for h in pred_rec.get("resultHoles", [])]

    total_gt   = gt_rec.get("totalNumberofHoles",   len(gt_holes))
    total_pred = pred_rec.get("totalNumberofHoles", len(pred_holes))
    extra_holes = max(0, total_pred - total_gt)

    # --- match holes exactly on (shape, size, location) -------------------- #
    unmatched_pred = deepcopy(pred_holes)
    matched_pairs  = []
    used_pred      = set()

    for gt in gt_holes:
        for j, pred in enumerate(unmatched_pred):
            if j in used_pred:
                continue
            if (gt["shape"] == pred["shape"]
                    and gt["size"] == pred["size"]
                    and gt["location"] == pred["location"]):
                matched_pairs.append((gt, pred))
                used_pred.add(j)
                break

    matched_count = len(matched_pairs)
    penalty       = extra_holes
    denom         = total_gt + penalty
    custom_score  = matched_count / denom if denom else 0.0

    precision = matched_count / total_pred if total_pred else 1.0
    recall    = matched_count / total_gt   if total_gt   else 1.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    exact_acc = 1.0 if (total_pred == total_gt == matched_count) else 0.0

    # --- field-level accuracies ------------------------------------------- #
    fields = ["shape", "size", "location"]
    field_acc = {}

    for field in fields:
        gt_vals   = [h[field] for h in gt_holes]
        pred_vals = [h[field] for h in pred_holes]
        gt_cnt    = Counter(gt_vals)
        pred_cnt  = Counter(pred_vals)
        matched   = sum(min(gt_cnt[v], pred_cnt[v]) for v in gt_cnt)
        extra     = max(0, len(pred_vals) - len(gt_vals))
        denom_f   = len(gt_vals) + extra
        field_acc[field] = {
            "matched": matched,
            "gt_total": len(gt_vals),
            "pred_total": len(pred_vals),
            "extra": extra,
            "hole_custom_score": matched / denom_f if denom_f else 0.0
        }

    # compound: (location, shape)
    def match_compound(gts, preds, keys):
        used = set(); m = 0
        for gt in gts:
            for j, pr in enumerate(preds):
                if j in used:
                    continue
                if all(gt[k] == pr[k] for k in keys):
                    m += 1
                    used.add(j)
                    break
        extra = max(0, len(preds) - len(gts))
        denom_c = len(gts) + extra
        return {
            "matched": m,
            "gt_total": len(gts),
            "pred_total": len(preds),
            "extra": extra,
            "hole_custom_score": m / denom_c if denom_c else 0.0
        }

    field_acc["location_shape"] = match_compound(gt_holes, pred_holes, ("location", "shape"))

    # ---------------------------------------------------------------------- #
    return {
        "exact_match_accuracy": exact_acc,
        "custom_score": custom_score,
        "recall": recall,
        "f1_score": f1,
        "correct": matched_count,
        "incorrect": total_pred - matched_count,
        "missing": total_gt - matched_count,
        "extra_holes": extra_holes,
        "field_accuracy": field_acc
    }


# ──────────────────────────────── file-level driver ───────────────────────── #
def compare_json_files(gt_file: str, pred_file: str):
    gt_data   = load_json_or_jsonl(gt_file)
    pred_data = load_json_or_jsonl(pred_file)

    # id-indexed lookup for predictions
    pred_by_id = {rec.get("custom_id", rec.get("id")): rec for rec in pred_data}

    groups = {i: {"total": 0, "exact_match": 0, "unfolding_exact": 0} for i in range(1, 10)}

    record_stats     = {}
    overall_counts   = {"correct": 0, "incorrect": 0, "missing": 0}
    extra_hole_recs  = 0
    missing_hole_recs = 0

    unfold_exact = 0
    unfold_pos_correct = 0
    unfold_total_steps = 0

    # field totals for shape/size/location/location_shape
    track_fields = ["shape", "size", "location", "location_shape"]
    field_totals = {f: {"matched": 0, "gt_total": 0, "pred_total": 0, "extra": 0} for f in track_fields}

    for gt in gt_data:
        rid = gt["id"]
        if rid not in pred_by_id:
            continue  # skip missing predictions

        pred_raw = pred_by_id[rid]
        pred     = extract_prediction(pred_raw)

        stats = compare_result_holes(gt, pred)
        record_stats[rid] = stats

        # difficulty group bookkeeping
        g = get_group(extract_pfhp_number(rid))
        if g:
            groups[g]["total"] += 1
            if stats["exact_match_accuracy"] == 1.0:
                groups[g]["exact_match"] += 1
            if gt.get("unfoldingTypes", []) == pred.get("unfoldingTypes", []):
                groups[g]["unfolding_exact"] += 1

        # extra / missing holes at record level
        gt_n   = gt.get("totalNumberofHoles",   len(gt.get("resultHoles", [])))
        pred_n = pred.get("totalNumberofHoles", len(pred.get("resultHoles", [])))
        if pred_n > gt_n:
            extra_hole_recs += 1
        elif pred_n < gt_n:
            missing_hole_recs += 1

        # unfolding
        gt_un = gt.get("unfoldingTypes", [])
        pr_un = pred.get("unfoldingTypes", [])
        if gt_un == pr_un:
            unfold_exact += 1
        common_len = min(len(gt_un), len(pr_un))
        unfold_pos_correct += sum(1 for i in range(common_len) if gt_un[i] == pr_un[i])
        unfold_total_steps += len(gt_un)

        # global field totals
        for f, fstats in stats["field_accuracy"].items():
            field_totals[f]["matched"]    += fstats["matched"]
            field_totals[f]["gt_total"]   += fstats["gt_total"]
            field_totals[f]["pred_total"] += fstats["pred_total"]
            field_totals[f]["extra"]      += fstats["extra"]

        # overall hole counts
        overall_counts["correct"]   += stats["correct"]
        overall_counts["incorrect"] += stats["incorrect"]
        overall_counts["missing"]   += stats["missing"]

    # ------------------------- overall summary ------------------------------ #
    TOTAL_RECS = 315  # static dataset size
    exact_rec  = sum(1 for s in record_stats.values() if s["exact_match_accuracy"] == 1.0)
    exact_rec_acc = exact_rec / TOTAL_RECS

    total_matched   = sum(s["correct"] for s in record_stats.values())
    total_gt_holes  = sum(s["correct"] + s["missing"] for s in record_stats.values())
    total_penalty   = sum(s["extra_holes"] for s in record_stats.values())
    overall_custom  = total_matched / (total_gt_holes + total_penalty) if (total_gt_holes + total_penalty) else 0.0

    # print block
    print("\n==============================")
    print("OVERALL METRICS (direction ignored)")
    print(f"  - Exact Match Accuracy (per record): {exact_rec_acc:.2%} ({exact_rec}/{TOTAL_RECS})")
    print(f"  - Overall Custom Score:              {overall_custom:.2%}")
    print(f"  - Records with Extra Holes:          {extra_hole_recs}/{TOTAL_RECS} = {(extra_hole_recs/TOTAL_RECS):.2%}")
    print(f"  - Records with Missing Holes:        {missing_hole_recs}/{TOTAL_RECS} = {(missing_hole_recs/TOTAL_RECS):.2%}")
    print(f"  - Exact Unfolding Matches:           {unfold_exact}/{TOTAL_RECS} = {(unfold_exact/TOTAL_RECS):.2%}")
    if unfold_total_steps:
        print(f"  - Unfolding Steps Correctly Matched: {unfold_pos_correct}/{unfold_total_steps} = {(unfold_pos_correct/unfold_total_steps):.2%}")

    print("\nHOLE INFORMATION ACCURACY:")
    for f, fs in field_totals.items():
        matched = fs["matched"]; gt_t = fs["gt_total"]; extra = fs["extra"]
        denom   = gt_t + extra
        sc      = matched/denom if denom else 0.0
        print(f"  • {f}: {matched} / ({gt_t} + {extra}) = {sc:.2%}")

    print("\nGROUP-WISE PERFORMANCE:")
    for g in sorted(groups, reverse=True):
        t = groups[g]["total"]
        if not t:
            print(f"  - Group {g}: no records")
            continue
        em = groups[g]["exact_match"]/t
        um = groups[g]["unfolding_exact"]/t
        print(f"  - Group {g}: Holes Exact {groups[g]['exact_match']}/{t} = {em:.2%},  "
              f"Unfolding Exact {groups[g]['unfolding_exact']}/{t} = {um:.2%}")

    # return a structured object in case the caller wants it
    return record_stats, {
        "exact_match_accuracy": exact_rec_acc,
        "overall_custom_score": overall_custom,
        "field_accuracy": field_totals,
        "exact_unfolding_match_count": unfold_exact,
        "partial_unfolding_match_counts": unfold_pos_correct,
        "total_unfolding_steps": unfold_total_steps,
        "records_with_extra_holes": extra_hole_recs,
        "records_with_missing_holes": missing_hole_recs,
        **overall_counts
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
