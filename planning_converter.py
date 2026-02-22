import json
import re

def extract_json_from_response(response_text):
    """
    Extract JSON block from model response.
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    else:
        raise ValueError("No JSON block found in response")


def convert_to_target_format(ex_id, parsed_json):
    """
    Convert model JSON to required dataset format.
    """
    return {
        "id": ex_id,
        "folds": ",".join(parsed_json.get("folds", [])),
        "holes": parsed_json.get("holes", [])
    }
