import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, Ellipse
import numpy as np
import os
import math
import json
import random
# === Triangle ID Mapping ===
# 4x4 grid = 16 squares, 2 triangles each = 32 triangles
# triangle IDs: 1 to 32
# Layout:
# Triangle ID = (row * 4 + col) * 2 + (0 or 1)

# It does not generate two or more DIAGONAL folds. There is a limitation on that.

GRID_ROWS = 4
GRID_COLS = 4
ALL_TRIANGLE_IDS = set(range(1, 33))  # 1 to 32

# ------------------------------------------------------------------
# helper:  integer grid extent of the currently visible sheet
# ------------------------------------------------------------------


def _visible_sheet_box(centroids):
    xs = [p[0] for p in centroids.values()]
    ys = [p[1] for p in centroids.values()]

    x_min = math.floor(min(xs))
    x_max = math.ceil (max(xs))
    y_min = math.floor(min(ys))
    y_max = math.ceil (max(ys))
    return x_min, x_max, y_min, y_max


def get_folded_triangles(action, unfolded_ids):
    """
    Return the triangles to fold (convert to black) based on action.
    Operates only on the remaining unfolded triangles.
    """

    # SWAP incorrect diagonal names to match actual behavior
    if action == 'diagonal_topRight_to_bottomLeft':
        action = 'diagonal_bottomRight_to_topLeft'
    elif action == 'diagonal_bottomRight_to_topLeft':
        action = 'diagonal_topRight_to_bottomLeft'
    elif action == 'diagonal_topLeft_to_bottomRight':
        action = 'diagonal_bottomLeft_to_topRight'
    elif action == 'diagonal_bottomLeft_to_topRight':
        action = 'diagonal_topLeft_to_bottomRight'

    centroids = {tid: triangle_id_to_centroid(tid) for tid in unfolded_ids}
    folded = set()

    if not centroids:
        return folded

    x_min, x_max, y_min, y_max = _visible_sheet_box(centroids)

    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2

    for tid, (x, y) in centroids.items():

        # === Horizontal Folds ===
        if action == 'horizontal_top_to_bottom':
            if y < mid_y:  # fold top half
                folded.add(tid)

        elif action == 'horizontal_bottom_to_top':
            if y >= mid_y:  # fold bottom half
                folded.add(tid)

        # === Vertical Folds ===
        elif action == 'vertical_right_to_left':
            if x >= mid_x:  # fold right half
                folded.add(tid)

        elif action == 'vertical_left_to_right':
            if x < mid_x:  # fold left half
                folded.add(tid)

        elif action == 'diagonal_bottomLeft_to_topRight':
            if x + y < (x_min + y_min + x_max + y_max) / 2:
                folded.add(tid)

        elif action == 'diagonal_topRight_to_bottomLeft':
            if x + y > (x_min + y_min + x_max + y_max) / 2:
                folded.add(tid)

        elif action == 'diagonal_topLeft_to_bottomRight':
            if x - y < (x_min - y_max + x_max - y_min) / 2:
                folded.add(tid)

        elif action == 'diagonal_bottomRight_to_topLeft':
            if x - y > (x_min - y_max + x_max - y_min) / 2:
                folded.add(tid)


    return folded


def triangle_id_to_position(triangle_id):
    """Return (row, col, triangle_in_cell) for given triangle_id"""
    triangle_id -= 1
    row = triangle_id // 8
    col = (triangle_id % 8) // 2
    tri = triangle_id % 2
    return (row, col, tri)

def triangle_id_to_centroid(triangle_id):
    """Calculate the centroid of a triangle by its ID."""
    row, col, tri = triangle_id_to_position(triangle_id)
    x, y = col, row  # match drawing orientation
    flip = (row + col) % 2 == 1

    if not flip:
        if tri == 0:
            pts = np.array([(x, y + 1), (x, y), (x + 1, y + 1)])
        else:
            pts = np.array([(x + 1, y), (x, y), (x + 1, y + 1)])
    else:
        if tri == 0:
            pts = np.array([(x, y), (x + 1, y), (x, y + 1)])
        else:
            pts = np.array([(x + 1, y + 1), (x + 1, y), (x, y + 1)])

    return pts.mean(axis=0)  # centroid (x, y)

def _reflect_across_line(pt, p1, p2):
    """Reflect point `pt` across the infinite line through p1‑p2."""
    # ensure floating‑point arithmetic
    p = np.asarray(pt, dtype=float)
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)

    d  = b - a
    d /= np.linalg.norm(d)          # now d is safely float64
    proj = a + d * np.dot(p - a, d)
    return tuple(2 * proj - p)

def mirror_triangles(
        fold_set,
        unfolded_ids,
        action,
        *,                 # make the 3rd argument keyword‑only
        search_ids=None):  # triangles that may lie underneath
    """
    Parameters
    ----------
    fold_set      : triangles that are moving (will turn black).
    unfolded_ids  : *visible* triangles – used to work out the current
                    size of the sheet and therefore the exact fold‑line.
    search_ids    : triangles that are allowed to appear **under** the
                    fold.  By default only the visible ones are searched,
                    but for rule #3 we usually pass ALL_TRIANGLE_IDS.
    """
    if not fold_set:
        return set()

    if search_ids is None:
        search_ids = unfolded_ids        # back‑wards compatible

    C = {tid: triangle_id_to_centroid(tid) for tid in unfolded_ids}
    x_min, x_max, y_min, y_max = _visible_sheet_box(C)
    mid_x, mid_y = (x_min + x_max) / 2, (y_min + y_max) / 2

    if action.startswith('horizontal'):
        p1, p2 = (x_min, mid_y), (x_max, mid_y)
    elif action.startswith('vertical'):
        p1, p2 = (mid_x, y_min), (mid_x, y_max)
    elif any(k in action for k in ('bottomLeft_to_topRight',
                                   'topRight_to_bottomLeft')):
        p1, p2 = (x_min, y_max), (x_max, y_min)
    elif any(k in action for k in ('topLeft_to_bottomRight',
                                   'bottomRight_to_topLeft')):
        p1, p2 = (x_min, y_min), (x_max, y_max)
    else:
        raise ValueError(f'Unknown fold action {action!r}')

    # --------- centroids that *could* be covered after the fold ----------
    C_search = {tid: triangle_id_to_centroid(tid) for tid in search_ids}

    mirror = set()
    for tid in fold_set:
        src_pt  = triangle_id_to_centroid(tid)
        tgt_pt  = _reflect_across_line(src_pt, p1, p2)

        # find which triangle owns that reflected point
        for cand, c_pt in C_search.items():
            if np.allclose(c_pt, tgt_pt, atol=1e-6):
                mirror.add(cand)
                break
    return mirror

def get_textual_representation_compact_string(black_triangles, punched_holes=None, force_all_white=False):
    """
    Return compact string format for grid:
    Example:
    11, a0, 00, 1C,
    01, 10, 10, 11,
    ...
    """
    SHAPE_TO_LETTER = {
        'circle': 'C',
        'square': 'Q',
        'ellipse': 'E',
        'star': 'S',
        'triangle': 'A',
        'rectangle': 'R',
        'trapezoid': 'Z',
        'letter': 'T',
        'text': 'X'
    }

    # Build hole map: triangle_id -> symbol
    hole_map = {}
    if punched_holes:
        for hole in punched_holes:
            tid = hole["location"]
            shape = hole.get("shape", "circle")
            size = hole.get("size", "small")
            base = SHAPE_TO_LETTER.get(shape, '?')
            symbol = base.lower() if size == "small" else base
            hole_map[tid] = symbol

    result = []
    tid = 1
    for _ in range(4):
        row = []
        for _ in range(4):
            square = []
            for _ in range(2):
                if tid in hole_map:
                    square.append(hole_map[tid])
                elif force_all_white:
                    square.append("1")
                elif tid in black_triangles:
                    square.append("0")
                else:
                    square.append("1")
                tid += 1
            row.append("".join(square))
        result.append(", ".join(row) + ",")
    return "\n".join(result)



# === Drawing ===

def draw_checkerboard_triangle_grid(ax, black_mark_ids):
    triangle_id = 1
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x, y = col, row
            if (row + col) % 2 == 0:
                tri1 = [(x, y + 1), (x, y), (x + 1, y + 1)]
                tri2 = [(x + 1, y), (x, y), (x + 1, y + 1)]
                label1_pos = (x + 0.2, y + 0.7)
                label2_pos = (x + 0.6, y + 0.2)
            else:
                tri1 = [(x, y), (x + 1, y), (x, y + 1)]
                tri2 = [(x + 1, y + 1), (x + 1, y), (x, y + 1)]

                label1_pos = (x + 0.2, y + 0.2)
                label2_pos = (x + 0.6, y + 0.6)

            color1 = 'black' if triangle_id in black_mark_ids else 'white'
            ax.add_patch(plt.Polygon(tri1, facecolor=color1, edgecolor='gray'))
            #ax.text(*label1_pos, str(triangle_id), fontsize=14, color='red')
            triangle_id += 1

            color2 = 'black' if triangle_id in black_mark_ids else 'white'
            ax.add_patch(plt.Polygon(tri2, facecolor=color2, edgecolor='gray'))
            #ax.text(*label2_pos, str(triangle_id), fontsize=14, color='red')
            triangle_id += 1

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()


def add_shape_legend(ax, origin_x=4.5, origin_y=0.5, size=0.2, initial_holes=None):
    spacing = 0.5
    items = [
        {'shape': 'triangle', 'color': 'black', 'label': 'Folded region'},
        {'shape': 'triangle', 'color': 'white', 'label': 'Remaining paper'},
    ]

    # Dynamically add unique shapes from initial_holes
    if initial_holes:
        shape_counts = {}
        for hole in initial_holes:
            shape = hole.get("shape", "circle")
            shape_counts[shape] = shape_counts.get(shape, 0) + 1

        for shape, count in shape_counts.items():
            label = f'Punched hole'
            items.append({'shape': shape, 'color': 'green', 'label': label})

    for i, item in enumerate(items):
        y = origin_y + i * spacing
        x = origin_x
        shape = item['shape']
        color = item['color']
        label = item['label']

        if shape == 'triangle':
            tri = Polygon([
                (x, y), (x + size, y), (x, y + size)
            ], closed=True, facecolor=color, edgecolor='gray')
            ax.add_patch(tri)
        elif shape == 'circle':
            circ = Circle((x + size / 2, y + size / 2), radius=size / 2, color=color)
            ax.add_patch(circ)
        elif shape == 'square':
            sq = Rectangle((x, y), size, size, color=color)
            ax.add_patch(sq)
        elif shape == 'ellipse':
            ell = Ellipse((x + size / 2, y + size / 2), width=size, height=size / 2, color=color)
            ax.add_patch(ell)
        elif shape == 'rectangle':
            rect = Rectangle((x, y), size * 1.2, size * 0.6, color=color)
            ax.add_patch(rect)
        elif shape == 'trapezoid':
            trap = Polygon([
                (x, y + size), (x + size, y + size), (x + size * 0.8, y), (x + size * 0.2, y)
            ], closed=True, color=color)
            ax.add_patch(trap)
        elif shape == 'star':
            r_outer, r_inner = size / 2, size / 4
            angles = np.linspace(0, 2 * np.pi, 11)
            points = [((x + size / 2) + np.cos(a) * (r_outer if i % 2 == 0 else r_inner),
                       (y + size / 2) + np.sin(a) * (r_outer if i % 2 == 0 else r_inner))
                      for i, a in enumerate(angles)]
            star = Polygon(points, closed=True, color=color)
            ax.add_patch(star)
        elif shape in ['letter', 'text']:
            display_text = "T" if shape == 'letter' else "TA"
            ax.text(x + size / 2, y + size / 2, display_text, fontsize=12,
                    ha='center', va='center', color=color)

        ax.text(x + size + 0.1, y + size / 4, label,
                fontsize=10, color='black', va='center', ha='left')


def place_mark_on_triangle(ax, triangle_id, shape='circle', size='small', direction=0, text='TA', color='blue'):
    """
    Places a mark in a triangle cell on a 4x4 checkerboard grid.

    Parameters:
    - ax: The matplotlib axes
    - triangle_id: ID from 1 to 32
    - shape: 'circle', 'ellipse', 'triangle', 'trapezoid', 'star', 'letter'
    - size: 'small' or 'large'
    - direction: rotation in degrees (counterclockwise)
    - color: fill or text color
    """
    # Get row and col from triangle ID
    center = triangle_id_to_centroid(triangle_id)

    # === Flip direction for specific shapes ===
    if shape in ['triangle', 'trapezoid']:
        if direction == 0:
            direction = 180
        elif direction == 180:
            direction = 0
    if shape in ["star"]:
        if direction == 270:
            direction =0
        elif direction == 90:
            direction = 180
        elif direction == 0:
            direction = 270
        elif direction == 180:
            direction = 90



    triangle_id -= 1
    cell_row = triangle_id // 8
    cell_col = (triangle_id % 8) // 2
    tri_in_cell = triangle_id % 2  # 0: first triangle, 1: second triangle
    row = cell_row
    col = cell_col
    x, y = col, row  # flip y for top-down display

    # Size scaling
    scale = 0.1 if size == 'small' else 0.175

    # Shape creation with rotation if applicable
    def rotated_shape(points, angle_deg):
        theta = np.radians(angle_deg)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        return (points - center) @ rot_matrix.T + center

    if shape == 'circle':
        mark = Circle(center, radius=scale, color=color)
    elif shape in ['square', 'rectangle']:
        width = height = scale * 1 if shape == 'square' else (scale * 1.8, scale)
        rect = Rectangle(
            center - np.array([width[0] / 6, width[1] / 2]) if isinstance(width, tuple) else center - scale,
            width=width[0] if isinstance(width, tuple) else width,
            height=width[1] if isinstance(width, tuple) else width,
            angle=direction,
            color=color)
        mark = rect
    elif shape == 'ellipse':
        mark = Ellipse(center, width=scale * 2, height=scale, angle=direction, color=color)
    elif shape == 'triangle':
        tri = np.array([[0, scale], [-scale, -scale], [scale, -scale]]) + center
        mark = Polygon(rotated_shape(tri, direction), color=color)
    elif shape == 'trapezoid':
        trap = np.array([[-scale, -scale], [scale, -scale], [scale * 0.275, scale], [-scale * 0.25, scale]]) + center
        mark = Polygon(rotated_shape(trap, direction), color=color)
    elif shape == 'star':
        r_outer, r_inner = scale, scale * 0.3
        angles = np.linspace(0, 2 * np.pi, 11)
        points = [center + [np.cos(a) * (r_outer if i % 2 == 0 else r_inner),
                            np.sin(a) * (r_outer if i % 2 == 0 else r_inner)] for i, a in enumerate(angles)]
        mark = Polygon(rotated_shape(np.array(points), direction), color=color)
    elif shape in ['letter', 'text']:
        display_text = "T" if shape == 'letter' else "TA"
        ax.text(center[0], center[1], display_text, color=color,
                fontsize=12 if size == 'small' else 18,
                ha='center', va='center', rotation=direction)
        return
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    ax.add_patch(mark)


def simulate_folding(actions, task_id="default_task", base_folder='img_text_outputs',
                     initial_holes=None, result_holes=None):
    """
    Simulate a folding task and save:

      • images  ─► <base>/<task>/img/…
      • text    ─► <base>/<task>/text/…

    Includes Step 0 (“initial sheet” all-white).
    """
    # ───────────────────────── directories ────────────────────────────
    task_root = os.path.join(base_folder, task_id)
    img_dir   = os.path.join(task_root, "img")
    txt_dir   = os.path.join(task_root, "text")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(txt_dir,  exist_ok=True)

    # ───────────────────────── step 0 (blank) ─────────────────────────
    black_triangles      = set()
    unfolded_triangles   = set(ALL_TRIANGLE_IDS)
    step_representations = []

    fig, ax = plt.subplots(figsize=(6, 5))
    draw_checkerboard_triangle_grid(ax, black_mark_ids=black_triangles)
    add_shape_legend(ax)
    ax.axis('off'); fig.tight_layout()
    plt.savefig(os.path.join(img_dir, f'{task_id}_fold_0_initial.png'), dpi=150)
    plt.close(fig)

    grid0 = get_textual_representation_compact_string(
        black_triangles=set(), punched_holes=None, force_all_white=True)
    step_representations.append(f"Step 0: initial sheet\n{grid0}\n")

    # ───────────────────────── folds ──────────────────────────────────
    for i, action in enumerate(actions, start=1):

        if action.startswith("rotation-"):
            angle = int(action.split('-')[1])
            black_triangles   = {rotate_triangle_id(t, angle) for t in black_triangles}
            unfolded_triangles = {rotate_triangle_id(t, angle) for t in unfolded_triangles}
        else:
            fold_set   = get_folded_triangles(action, unfolded_triangles)
            mirror_set = mirror_triangles(fold_set, unfolded_ids=unfolded_triangles,
                                          search_ids=ALL_TRIANGLE_IDS, action=action)

            was_black = black_triangles.copy()
            black_triangles.update(fold_set)
            black_triangles.difference_update(mirror_set & was_black)
            unfolded_triangles.difference_update(fold_set)
            unfolded_triangles.update(mirror_set - black_triangles)

        # --- image
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_checkerboard_triangle_grid(ax, black_mark_ids=black_triangles)
        add_shape_legend(ax)
        ax.axis('off'); fig.tight_layout()
        plt.savefig(os.path.join(img_dir, f'{task_id}_fold_{i}_{action}.png'), dpi=150)
        plt.close(fig)

        # --- text
        grid = get_textual_representation_compact_string(black_triangles, punched_holes=None)
        step_representations.append(f"Step {i}: \n{grid}\n")

    # ───────────────────────── punching holes ─────────────────────────
    if initial_holes:
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_checkerboard_triangle_grid(ax, black_mark_ids=black_triangles)
        add_shape_legend(ax, initial_holes=initial_holes)

        for hole in initial_holes:
            place_mark_on_triangle(
                ax,
                triangle_id=hole["location"],
                shape=hole.get("shape", "circle"),
                size=hole.get("size", "small"),
                direction=parse_direction(hole.get("direction", 0)),
                text=hole.get("text", "T"),
                color='green'
            )
        ax.axis('off'); fig.tight_layout()
        plt.savefig(os.path.join(img_dir, f'{task_id}_punching_holes.png'), dpi=150)
        plt.close(fig)

        grid = get_textual_representation_compact_string(
            black_triangles, punched_holes=initial_holes)
        step_representations.append(f"Hole Punching:\n{grid}\n")

    # ───────────────────────── write all step text ────────────────────
    steps_file = os.path.join(txt_dir, f'{task_id}_textual_steps.txt')
    with open(steps_file, "w") as f:
        f.write("\n".join(step_representations))
    print(f"Saved step-by-step text in '{steps_file}'")

    # ───────────────────────── final result (unfolded) ────────────────
    if result_holes:
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_checkerboard_triangle_grid(ax, black_mark_ids=set())
        add_shape_legend(ax, initial_holes=result_holes)
        for hole in result_holes:
            place_mark_on_triangle(
                ax,
                triangle_id=hole["location"],
                shape=hole.get("shape", "circle"),
                size=hole.get("size", "small"),
                direction=parse_direction(hole.get("direction", 0)),
                text=hole.get("text", "T"),
                color='green'
            )
        ax.axis('off'); fig.tight_layout()
        plt.savefig(os.path.join(img_dir, f'{task_id}_result_holes.png'), dpi=150)
        plt.close(fig)

        result_str = get_textual_representation_compact_string(
            black_triangles=set(), punched_holes=result_holes, force_all_white=True)
        final_txt = os.path.join(txt_dir, f'{task_id}_textual_final_result.txt')
        with open(final_txt, "w") as f:
            f.write("Final Result:\n")
            f.write(result_str)
        print(f"Saved final result text in '{final_txt}'")

    print(f"Saved images in '{img_dir}' and text files in '{txt_dir}'")


def rotate_triangle_id(tid: int, angle: int) -> int:
    """
    Map a triangle ID after rotating the (possibly already-folded)
    4×4 sheet *counter-clockwise* by 90 °, 180 ° or 270 °.

    The mapping is found geometrically:
        1. take the triangle’s centroid,
        2. rotate that point CCW around the sheet centre,
        3. return the ID whose centroid coincides with the new point.
    """
    if angle not in (90, 180, 270):
        raise ValueError("angle must be 90, 180 or 270")

    # current centroid
    c_old = np.array(triangle_id_to_centroid(tid))

    # rotate CCW around the sheet’s centre
    sheet_centre = np.array([GRID_COLS / 2, GRID_ROWS / 2])

    # The drawing uses y-down coordinates (invert_yaxis), so a mathematically *negative* angle gives a visible CCW turn.
    theta = -np.radians(angle)
    R     = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    c_new = sheet_centre + R @ (c_old - sheet_centre)

    # 3. find which triangle owns that centroid --------------------------
    for cand in range(1, GRID_ROWS * GRID_COLS * 2 + 1):
        if np.allclose(triangle_id_to_centroid(cand), c_new, atol=1e-6):
            return cand

    raise RuntimeError("Could not map triangle after rotation.")

def parse_direction(value):
    if value is None or value == "":
        return 0

    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        value_lower = value.lower()
        if "180" in value_lower:
            return 180
        elif "90" in value_lower:
            return 90

    return 0


def load_all_tasks(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    tasks = []
    for task in data:
        task_id = task.get("id", "unknown_task")
        actions = [step["foldType"] for step in task.get("foldingTypes", [])]
        initial_holes = task.get("initialHoles", [])
        result_holes = task.get("resultHoles",[])
        tasks.append({
            "id": task_id,
            "actions": actions,
            "initial_holes": initial_holes,
            "result_holes": result_holes
        })

    return tasks


def has_invalid_shapes(initial_holes):
    """Return True if any hole is a square or rectangle."""
    return any(h.get("shape") in {"square", "rectangle", "text"} for h in initial_holes)

def has_sequential_diagonal_folds(actions):
    """Return True if two or more sequential diagonal folds exist."""

    diagonal_keywords = [
        "diagonal_topLeft_to_bottomRight",
        "diagonal_bottomRight_to_topLeft",
        "diagonal_topRight_to_bottomLeft",
        "diagonal_bottomLeft_to_topRight"
    ]
    for i in range(len(actions) - 1):
        if actions[i] in diagonal_keywords and actions[i + 1] in diagonal_keywords:
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load and filter tasks.")
    parser.add_argument("json_file_path", help="Path to the input JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.json_file_path):
        raise FileNotFoundError(f"File not found: {args.json_file_path}")
    all_tasks = load_all_tasks(args.json_file_path)

    # Filter tasks
    filtered_tasks = []

    for task in all_tasks:
        actions = task["actions"]
        holes   = task["initial_holes"]

        if has_invalid_shapes(holes):
            continue  # Skip task with disallowed shape

        if has_sequential_diagonal_folds(actions):
            continue  # Skip task with sequential diagonals which cause corrupted shape

        filtered_tasks.append(task)

    print(f" Filtered down to {len(filtered_tasks)} tasks after applying shape and folding constraints.")

    for task in filtered_tasks:
        task_id = task["id"]
        actions = task["actions"]
        initial_holes = task["initial_holes"]
        result_holes  = task["result_holes"]

        print(f" Running task: {task_id} with {len(actions)} fold(s)")

        simulate_folding(
            actions=actions,
            task_id=task_id,
            initial_holes=initial_holes,
            result_holes=result_holes
        )
