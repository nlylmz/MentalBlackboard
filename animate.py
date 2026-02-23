
import sys
import random
import subprocess
import validation
import json
import os

from run import create_videos_for_all

if __name__ == "__main__":
    python_exe = sys.executable

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py prediction [structure_group] [count]")
        print("  python main.py planning")
        sys.exit(1)

    run_mode = sys.argv[1].lower()

    if run_mode == "prediction":
        if len(sys.argv) < 4:
            print("Usage for prediction: python main.py prediction [structure_group] [count]")
            print("Example: python main.py prediction one_step 8")
            sys.exit(1)

        group_name = sys.argv[2]
        try:
            num_folds = int(sys.argv[3])
        except ValueError:
            print("The count must be an integer.")
            sys.exit(1)

        # Generate all structure groups
        one_step, two_step, three_step, four_step, \
        two_step_with_rotation, three_step_with_rotation, \
        four_step_with_rotation, five_step_with_rotation, \
        six_step_with_rotation = validation.generate_valid_structures()

        structure_options = {
            "one_step": one_step,
            "two_step": two_step,
            "three_step": three_step,
            "four_step": four_step,
            "two_step_with_rotation": two_step_with_rotation,
            "three_step_with_rotation": three_step_with_rotation,
            "four_step_with_rotation": four_step_with_rotation,
            "five_step_with_rotation": five_step_with_rotation,
            "six_step_with_rotation": six_step_with_rotation
        }

        if group_name not in structure_options:
            print(f"Invalid structure group '{group_name}'. Valid options are:")
            print(", ".join(structure_options.keys()))
            sys.exit(1)

        selected_group = structure_options[group_name]
        if not selected_group:
            print(f"No folds available in group '{group_name}'.")
            sys.exit(1)

        if num_folds < 1 :
            print(f"Invalid number of folds. Must be bigger than 0.")
            sys.exit(1)

        group_len = len(selected_group)
        if num_folds <= group_len:
            selected_structure = random.sample(selected_group, num_folds)
        else:
            # take all unique folds once, then sample remaining with replacement
            sel = selected_group[:]  # copy
            remaining = num_folds - group_len
            extra = random.choices(selected_group, k=remaining)
            selected_structure = sel + extra
            # shuffle so duplicates don't always appear at the end
            random.shuffle(selected_structure)

        print("Selected folds:", selected_structure)

        for i, fold in enumerate(selected_structure):
            print(f"\n=== Starting prediction run #{i+1} with fold: {fold} ===")
            fold_arg = fold if isinstance(fold, str) else ",".join(fold)
            args = [python_exe, "run.py", "prediction", fold_arg]
            result = subprocess.run(args, capture_output=True, text=True)

            if result.returncode != 0:
                print("Subprocess failed with error:")
                print(result.stderr)
                print("Subprocess output:")
                print(result.stdout)
                raise RuntimeError("Subprocess failed.")
            else:
                print("Subprocess succeeded.")
                print(result.stdout)


    elif run_mode == "planning":

        if len(sys.argv) < 3:
            print("Error: Missing required input_file.")
            print("Usage: python main.py planning [input_file]")
            sys.exit(1)

        input_file = sys.argv[2]

        if not os.path.exists(input_file):
            print(f"Planning input file '{input_file}' not found.")
            sys.exit(1)

        print(f"Using planning input file: {input_file}")

        with open(input_file, "r") as f:
            planning_data = json.load(f)

        if not isinstance(planning_data, list):
            print("Expected a list of planning runs in JSON.")
            sys.exit(1)

        for i, run in enumerate(planning_data):
            print(f"\n=== Starting planning run #{i+1} ===")
            fold_arg = run.get("folds")
            hole_specs = run.get("holes")
            id = run.get("id")

            if not fold_arg or not hole_specs:
                print("Missing 'folds' or 'holes' in planning run data.")
                continue

            args = [python_exe, "run.py", "planning", fold_arg, str(hole_specs), id]
            result = subprocess.run(args, capture_output=True, text=True)

            if result.returncode != 0:
                print("Subprocess failed with error:")
                print(result.stderr)
                print("Subprocess output:")
                print(result.stdout)
                raise RuntimeError("Subprocess failed.")
            else:
                print("Subprocess succeeded.")
                print(result.stdout)

    elif run_mode == "create_video":

        print("Creating folding video...")
        create_videos_for_all("folding_frames", fps=2, quality=9)

        print("Creating unfolding video...")
        create_videos_for_all("unfolding_frames", fps=3, quality=9)

        print(" Video creation complete.")

    else:
        print("Invalid mode. Use 'prediction', 'planning', or 'create_video'.")
        sys.exit(1)
