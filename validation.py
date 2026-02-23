import itertools

#added backward folding
folding_types_all = [
    "H1-F", "H1-B", "H2-F", "H2-B",
    "V1-F", "V1-B", "V2-F", "V2-B",
    "D1-F", "D1-B", "D2-F", "D2-B", "D3-F", "D3-B", "D4-F", "D4-B"
]

folding_types_backward = [
    "H1-B", "H2-B",
    "V1-B", "V2-B",
    "D1-B", "D2-B",
    "D3-B", "D4-B"
]


folding_types = [
    "H1-F", "H2-F",
    "V1-F", "V2-F",
    "D1-F", "D2-F", "D3-F", "D4-F"
]

rotation_types = [
    "R-90", "R-180", "R-270"
]

all_actions = folding_types + rotation_types

def is_valid_folding_sequence(sequence):
    """
    Checks a given paper folding sequence (a list of fold strings) against these rules:
      1. No more than 2 horizontal folds total.
      2. No more than 2 vertical folds total.
      3. A diagonal fold is allowed as the first step.
      4. A diagonal fold (when not following a diagonal) must come after a combination of one horizontal and one vertical fold,
         and if there are more than one H or V fold before it, then the diagonal fold is not allowed.
      5. If there are already 2 horizontal and 2 vertical folds, then only a restricted set of diagonal types (here: D1 and D2) are allowed.
      6. When a diagonal fold comes immediately after a diagonal fold, only a subset of 2 out of 4 (front/behind options give 4 possibilities) is allowed.
      7. The same two diagonal folding directions never come consecutively.
      8. No more than 2 diagonal folds total.
      9. After two diagonal folding, there can be one vertical and one horizontal move and vice versa (more than three steps).
    """

    horizontal_count = 0
    vertical_count = 0
    diagonal_count = 0  # Track the number of diagonal folds

    def get_fold_info(fold):
        parts = fold.split('-')
        if len(parts) != 2:
            return None, None
        return parts[0], parts[1]

    allowed_after_diagonal = {
        "D1": {"D2", "D3"},
        "D2": {"D1", "D4"},
        "D3": {"D1", "D4"},
        "D4": {"D2", "D3"}
    }
    restricted_diagonals = {"D1", "D2"}

    for i, fold in enumerate(sequence):
        fold_type, _ = get_fold_info(fold)
        if fold_type is None:
            return False, f"Invalid fold representation at position {i}: {fold}"

        if fold_type.startswith("H"):
            horizontal_count += 1
            if horizontal_count > 2:
                return False, f"More than 2 horizontal folds at position {i}."
        elif fold_type.startswith("V"):
            vertical_count += 1
            if vertical_count > 2:
                return False, f"More than 2 vertical folds at position {i}."
        elif fold_type.startswith("D"):
            diagonal_count += 1
            if diagonal_count > 2:  # Enforce maximum of 2 diagonal folds
                return False, "More than 2 diagonal folds in sequence."

            diag_family = fold_type
            if i == 0:
                # First step diagonal fold is allowed
                pass
            else:
                prev_fold = sequence[i - 1]
                prev_type, _ = get_fold_info(prev_fold)
                if prev_type.startswith("D"):
                    allowed_set = allowed_after_diagonal.get(prev_type, set())
                    if horizontal_count == 2 and vertical_count == 2:
                        allowed_set = allowed_set.intersection(restricted_diagonals)
                    if diag_family not in allowed_set:
                        return False, f"Diagonal fold {fold} at position {i} is not allowed after {prev_fold}."
                else:
                    # NEW RULE 4:
                    # When not following a diagonal, a diagonal fold must come immediately after exactly one horizontal and one vertical fold.
                    if horizontal_count != 1 or vertical_count != 1:
                        return False, f"Diagonal fold {fold} at position {i} must follow exactly one horizontal and one vertical fold."

            if horizontal_count == 2 and vertical_count == 2 and diag_family not in restricted_diagonals:
                return False, f"Diagonal fold {fold} at position {i} is not allowed after max H and V folds."
        else:
            return False, f"Unknown fold type {fold_type} at position {i}."

        # RULE 9: If the sequence is longer than 3 steps and has exactly 2 diagonal folds,
        # the remaining moves must be exactly one horizontal and one vertical fold (in any order).
    if len(sequence) > 3 and diagonal_count == 2:
        remaining_moves = len(sequence) - 2  # After 2 diagonal folds, remaining moves count
        if horizontal_count != 1 or vertical_count != 1 or remaining_moves != 2:
            return False, "After two diagonal folds, exactly one horizontal and one vertical move must follow."

    return True, "Valid sequence"


def is_valid_rotation_sequence(sequence):
    def get_action_type(action):
        parts = action.split('-')
        if len(parts) != 2:
            return None, None
        return parts[0], parts[1]

    if not sequence:
        return False, "Sequence is empty."

    first_type, _ = get_action_type(sequence[0])
    if first_type is None:
        return False, "Invalid first action."
    if first_type.startswith("R"):
        return False, "Sequence cannot start with a rotation."

    i = 0
    rotation_count = 0
    fold_count = 0
    fold_types_used = set()

    while i < len(sequence):
        action_type, _ = get_action_type(sequence[i])
        if action_type is None:
            return False, f"Invalid action at position {i}: {sequence[i]}"

        if action_type.startswith("R"):
            rotation_count += 1
            if i > 0:
                prev_type, _ = get_action_type(sequence[i - 1])
                if prev_type.startswith("R"):
                    return False, f"Consecutive rotations not allowed at position {i}."

        elif action_type.startswith("D"):
            # Diagonal only allowed at first position
            if i != 0:
                return False, f"Diagonal fold only allowed at first position, found at {i}."
            fold_count += 1
            fold_types_used.add("D")

        elif action_type.startswith("H") or action_type.startswith("V"):
            fold_count += 1
            fold_types_used.add(action_type[0])

        else:
            return False, f"Unknown action type {action_type} at position {i}."

        i += 1

    # Check total number of folds
    if fold_count > 3:
        return False, f"More than 3 fold steps used: {fold_count}."

    if fold_count == 3:
        if sequence[0][0] != "D":
            return False, "If 3 folds used, first must be diagonal."
        if not fold_types_used.issubset({"D", "H", "V"}):
            return False, f"Invalid fold types used: {fold_types_used}"

    # Rotation count constraint
    if rotation_count < 1 or rotation_count > 3:
        return False, f"Must have 1 to 3 rotations. Found: {rotation_count}."

    return True, "Valid rotation sequence"



def generate_valid_structures():

    # 1-Step Folding: single fold H, D , V
    one_step = folding_types
    # Generate valid sequences of different lengths using itertools.product (allows repeats)
    # HV, DH,
    two_step = [combo for combo in itertools.product(folding_types, repeat=2) if is_valid_folding_sequence(combo)[0]]
    # HVD, DHH
    three_step = [combo for combo in itertools.product(folding_types, repeat=3) if is_valid_folding_sequence(combo)[0]]
    # DHVD
    four_step = [combo for combo in itertools.product(folding_types, repeat=4) if is_valid_folding_sequence(combo)[0]]

    # e.g., H → R, D → R, etc.
    two_step_with_rotation = [
        combo for combo in itertools.product(all_actions, repeat=2)
        if is_valid_rotation_sequence(combo)[0]
    ]

    # 3-Step
    three_step_with_rotation  = [
        combo for combo in itertools.product(all_actions, repeat=3)
        if is_valid_rotation_sequence(combo)[0]
    ]

    # 4-Step
    four_step_with_rotation  = [
        combo for combo in itertools.product(all_actions, repeat=4)
        if is_valid_rotation_sequence(combo)[0]
    ]


    # 5-Step
    five_step_with_rotation  = [
        combo for combo in itertools.product(all_actions, repeat=5)
        if is_valid_rotation_sequence(combo)[0]
    ]

    # 6-Step
    six_step_with_rotation = [
        combo for combo in itertools.product(all_actions, repeat=6)
        if is_valid_rotation_sequence(combo)[0]
    ]
    return one_step, two_step, three_step, four_step, two_step_with_rotation, three_step_with_rotation, four_step_with_rotation, five_step_with_rotation, six_step_with_rotation

# List of possible folding types:
# - 4 horizontal moves: 2 types ("H1" and "H2"), each with front ("F") and behind ("B")
# - 4 vertical moves: 2 types ("V1" and "V2"), each with front and behind
# - 8 diagonal moves: 4 types ("D1", "D2", "D3", "D4"), each with front and behind

#D1: TopRight to BottomLeft   D2: BottomRight to TopLeft   D3: TopLeft to BottomRight   D4: BottomLeft to TopRight

