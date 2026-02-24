import math
import random
import itertools
import json
import sys
import ast
import time
import imageio.v2 as imageio
from PIL import ImageGrab
import os
import re

from vpython import vertex, triangle, vector, rate, scene, color, curve, box, cylinder, ellipsoid, extrusion, shapes, \
    text, label

# ------------------------------
# Global Constants and Variables
# ------------------------------
HOLE_Z = 0.1  # All holes are drawn at z=0.1 so they appear on top.
vertex_cache = {}
fold_history = []  # Each fold record holds fold type, crease info, moving triangles, etc.
punched_holes = []  # Each hole record holds the stable copy, a moving copy, its base position, etc.
triangle_id_counter = 1  # For assigning unique numbers (1..32) to triangles
fold_crease_dashes = None
initial_label_pos = None
current_rotation_deg = 0        # 0, 90, 180, 270  (mod 360)

# ------------------------------
# Colors
# ------------------------------
base_color = color.gray(0.95)
dark_color = color.gray(0.3)

# ------------------------------
# Outer Border Drawing
# ------------------------------
outer_border_curve = None


# ------------------------------
# Helper Functions: Paper Geometry
# ------------------------------
def get_vertex(p, col):
    key = (round(p.x, 5), round(p.y, 5), round(p.z, 5))
    if key in vertex_cache:
        return vertex_cache[key]
    v = vertex(pos=p, color=col)
    vertex_cache[key] = v
    return v


def make_quadrant_triangles_32(A, B, C, D, col):
    global triangle_id_counter
    E = (A + B) / 2
    F = (B + C) / 2
    G = (C + D) / 2
    H = (D + A) / 2
    M = (A + B + C + D) / 4

    def make_tri(p0, p1, p2):
        global triangle_id_counter
        t = triangle(v0=get_vertex(p0, col),
                     v1=get_vertex(p1, col),
                     v2=get_vertex(p2, col))
        # Tag with unique ID.
        t.tri_id = triangle_id_counter
        triangle_id_counter += 1
        return t

    return [
        make_tri(A, E, M),
        make_tri(E, B, M),
        make_tri(B, F, M),
        make_tri(F, C, M),
        make_tri(C, G, M),
        make_tri(G, D, M),
        make_tri(D, H, M),
        make_tri(H, A, M)
    ]


def create_paper_triangles(paper_size=16):
    global vertex_cache, triangle_id_counter
    vertex_cache = {}  # Reset cache
    triangle_id_counter = 1  # Reset numbering
    half = paper_size / 2
    col = base_color
    # Four quadrants (8 triangles each → 32 total)
    A_tl = vector(-half, half, 0)
    B_tl = vector(0, half, 0)
    C_tl = vector(0, 0, 0)
    D_tl = vector(-half, 0, 0)
    tris_tl = make_quadrant_triangles_32(A_tl, B_tl, C_tl, D_tl, col)

    A_tr = vector(0, half, 0)
    B_tr = vector(half, half, 0)
    C_tr = vector(half, 0, 0)
    D_tr = vector(0, 0, 0)
    tris_tr = make_quadrant_triangles_32(A_tr, B_tr, C_tr, D_tr, col)

    A_bl = vector(-half, 0, 0)
    B_bl = vector(0, 0, 0)
    C_bl = vector(0, -half, 0)
    D_bl = vector(-half, -half, 0)
    tris_bl = make_quadrant_triangles_32(A_bl, B_bl, C_bl, D_bl, col)

    A_br = vector(0, 0, 0)
    B_br = vector(half, 0, 0)
    C_br = vector(half, -half, 0)
    D_br = vector(0, -half, 0)
    tris_br = make_quadrant_triangles_32(A_br, B_br, C_br, D_br, col)

    # Combine all triangles
    all_tris = tris_tl + tris_tr + tris_bl + tris_br


    # --- Assign custom IDs based on centroid position ---
    custom_ids = [
        26, 27, 30, 31, 25, 28, 29, 32,
        17, 20, 21, 24, 18, 19, 22, 23,
        10, 11, 14, 15, 9, 12, 13, 16,
        1, 4, 5, 8, 2, 3, 6, 7
    ]

    def centroid(tri):
        c = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        return (round(c.y, 5), round(c.x, 5))  # Sort Y then X

    sorted_tris = sorted(all_tris, key=centroid)

    id_map = {}
    for new_id, tri in zip(custom_ids, sorted_tris):
        old_id = tri.tri_id
        id_map[old_id] = new_id
        tri.tri_id = new_id  # Overwrite with custom ID

    return all_tris


def compute_bounding_box(all_triangles):
    verts = []
    for tri in all_triangles:
        verts.extend([tri.v0.pos, tri.v1.pos, tri.v2.pos])
    min_x = min(v.x for v in verts)
    max_x = max(v.x for v in verts)
    min_y = min(v.y for v in verts)
    max_y = max(v.y for v in verts)
    return min_x, max_x, min_y, max_y


def get_current_color(t, base, dark):
    if t <= 0.5:
        f = t / 0.5
        return base * (1 - f) + dark * f
    else:
        f = (t - 0.5) / 0.5
        return dark * (1 - f) + base * f


def update_outer_border(all_triangles):
    global outer_border_curve
    edge_dict = {}
    for tri in all_triangles:
        for (v1, v2) in [(tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)]:
            key = tuple(sorted((id(v1), id(v2))))
            edge_dict.setdefault(key, []).append((v1, v2, tri))
    outer_edges = [(v1, v2) for lst in edge_dict.values() if len(lst) == 1 for (v1, v2, _) in lst]
    connectivity = {}
    for v1, v2 in outer_edges:
        connectivity.setdefault(v1, []).append(v2)
        connectivity.setdefault(v2, []).append(v1)
    if connectivity:
        start = list(connectivity.keys())[0]
        ordered = [start]
        current = start
        prev = None
        while True:
            neighbors = connectivity[current]
            next_v = None
            for nb in neighbors:
                if nb is not prev:
                    next_v = nb
                    break
            if next_v is None or next_v == start:
                ordered.append(start)
                break
            ordered.append(next_v)
            prev, current = current, next_v
        if outer_border_curve is None:
            outer_border_curve = curve(color=color.black, radius=0.04)
        else:
            outer_border_curve.clear()
        for v in ordered:
            outer_border_curve.append(pos=v.pos)

def get_ordered_border_vertices(all_triangles):
    edge_dict = {}
    for tri in all_triangles:
        for (v1, v2) in [(tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)]:
            key = tuple(sorted((id(v1), id(v2))))
            edge_dict.setdefault(key, []).append((v1, v2, tri))
    outer_edges = [(v1, v2) for lst in edge_dict.values() if len(lst) == 1 for (v1, v2, _) in lst]
    connectivity = {}
    for v1, v2 in outer_edges:
        connectivity.setdefault(v1, []).append(v2)
        connectivity.setdefault(v2, []).append(v1)
    if connectivity:
        start = list(connectivity.keys())[0]
        ordered = [start]
        current = start
        prev = None
        while True:
            neighbors = connectivity[current]
            next_v = None
            for nb in neighbors:
                if nb is not prev:
                    next_v = nb
                    break
            if next_v is None or next_v == start:
                ordered.append(start)
                break
            ordered.append(next_v)
            prev, current = current, next_v
        return [v.pos for v in ordered]
    else:
        return []


def draw_dashed_line(pts, dash_length=0.5, gap_length=0.3, dash_color=color.white, dash_radius=0.04):
    dashes = []
    for i in range(len(pts) - 1):
        p_start = pts[i]
        p_end = pts[i + 1]
        seg_vector = p_end - p_start
        seg_length = seg_vector.mag
        direction = seg_vector.norm()
        d = 0
        while d < seg_length:
            dash_start = p_start + direction * d
            d_end = d + dash_length
            dash_end = p_end if d_end > seg_length else p_start + direction * d_end
            dash = curve(color=dash_color, radius=dash_radius)
            dash.append(pos=dash_start)
            dash.append(pos=dash_end)
            dashes.append(dash)
            d += dash_length + gap_length
    return dashes

# ------------------------------
# Crease Drawing Functions
# ------------------------------

def draw_fold_line_horizontal(all_triangles, crease_y):
    xs = [v.pos.x for tri in all_triangles for v in [tri.v0, tri.v1, tri.v2] if abs(v.pos.y - crease_y) < 1e-6]
    if not xs:
        return
    pts = [vector(min(xs), crease_y, 0), vector(max(xs), crease_y, 0)]
    global fold_crease_dashes
    # Hide previous crease if any
    if fold_crease_dashes:
        for dash in fold_crease_dashes:
            dash.visible = False
    fold_crease_dashes = draw_dashed_line(pts, dash_length=0.5, gap_length=0.3, dash_color=color.black, dash_radius=0.03)

def draw_fold_line_vertical(all_triangles, crease_x):
    ys = [v.pos.y for tri in all_triangles for v in [tri.v0, tri.v1, tri.v2] if abs(v.pos.x - crease_x) < 1e-6]
    if not ys:
        return
    pts = [vector(crease_x, min(ys), 0), vector(crease_x, max(ys), 0)]
    global fold_crease_dashes
    if fold_crease_dashes:
        for dash in fold_crease_dashes:
            dash.visible = False
    fold_crease_dashes = draw_dashed_line(pts, dash_length=0.5, gap_length=0.3, dash_color=color.black, dash_radius=0.03)

def draw_fold_line_diagonal(p1, p2):
    pts = [p1, p2]
    global fold_crease_dashes
    if fold_crease_dashes:
        for dash in fold_crease_dashes:
            dash.visible = False
    fold_crease_dashes = draw_dashed_line(pts, dash_length=0.5, gap_length=0.3, dash_color=color.black, dash_radius=0.03)


def get_fixed_label_position(border_pts):
    if not border_pts:
        return vector(0, 0, 0)

    # Compute bounding box from the border points.
    xs = [p.x for p in border_pts]
    ys = [p.y for p in border_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Calculate offsets: for example, 15% above the top border and 5% from the left side.
    label_vertical_offset = (max_y - min_y) * 0.15
    label_horizontal_offset = (max_x - min_x) * 0.01
    label_pos = vector(min_x + label_horizontal_offset, max_y + label_vertical_offset, 0)
    return label_pos

def create_folding_label(text):
    """
    Creates a label at a fixed position
    """
    return label(pos=initial_label_pos, text=text, height=16, color=color.white, box=False, align='left')

# ------------------------------
# Recording Folding Operations
# ------------------------------
def record_fold_operation(fold_type, fold_to_front, steps, crease, moving, orig_positions, theta_factor):
    moving_tri_ids = [tri.tri_id for tri in moving]
    fold_history.append({
        'fold_type': fold_type,
        "original_fold_type": fold_type,
        'fold_to_front': fold_to_front,
        'steps': steps,
        'crease': crease,  # For horizontal/vertical: number; for diagonal: (p1, p2)
        'moving': moving,
        'moving_tri_ids': moving_tri_ids,
        'orig_positions': orig_positions,
        'theta_factor': theta_factor
    })


# ------------------------------
# Folding Functions
# ------------------------------
def fold_horizontal_top_to_bottom(all_triangles, steps=30, fold_to_front=True):
    direction = "forward" if fold_to_front else "backward"
    msg_label = create_folding_label(f"Folding horizontally from top to bottom edge, {direction}")
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    crease_y = (min_y + max_y) / 2
    draw_fold_line_horizontal(all_triangles, crease_y)
    moving = []
    orig_positions = {}
    for tri in all_triangles:
        centroid = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        if centroid.y > crease_y:
            moving.append(tri)
            orig_positions[tri] = [vector(tri.v0.pos), vector(tri.v1.pos), vector(tri.v2.pos)]
    theta_factor = 1 if fold_to_front else -1
    record_fold_operation('horizontal_top_to_bottom', fold_to_front, steps, crease_y, moving, orig_positions,
                          theta_factor)
    rot_axis = vector(1, 0, 0)
    for step in range(steps + 1):
        rate(10)
        t = step / steps
        theta = theta_factor * math.pi * t
        current_color = get_current_color(t, base_color, dark_color)
        for tri in moving:
            new_positions = []
            for P in orig_positions[tri]:
                pivot = vector(P.x, crease_y, 0)
                new_positions.append(pivot + (P - pivot).rotate(angle=theta, axis=rot_axis))
            tri.v0.pos, tri.v1.pos, tri.v2.pos = new_positions
            for v in [tri.v0, tri.v1, tri.v2]:
                v.color = current_color
        update_outer_border(all_triangles)
        # Capture the frame
        if step % 3 == 0:  # Save every 2nd frame only
            capture_frame_folding(bbox)
    update_outer_border(all_triangles)
    msg_label.visible = False

def fold_horizontal_bottom_to_top(all_triangles, steps=30, fold_to_front=True):
    direction = "forward" if fold_to_front else "backward"
    msg_label = create_folding_label(f"Folding horizontally from bottom to top edge, {direction}")
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    crease_y = (min_y + max_y) / 2
    draw_fold_line_horizontal(all_triangles, crease_y)
    moving = []
    orig_positions = {}
    for tri in all_triangles:
        centroid = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        if centroid.y < crease_y:
            moving.append(tri)
            orig_positions[tri] = [vector(tri.v0.pos), vector(tri.v1.pos), vector(tri.v2.pos)]
    theta_factor = -1 if fold_to_front else 1
    record_fold_operation('horizontal_bottom_to_top', fold_to_front, steps, crease_y, moving, orig_positions,
                          theta_factor)
    rot_axis = vector(1, 0, 0)
    for step in range(steps + 1):
        rate(10)
        t = step / steps
        theta = theta_factor * math.pi * t
        current_color = get_current_color(t, base_color, dark_color)
        for tri in moving:
            new_positions = []
            for P in orig_positions[tri]:
                pivot = vector(P.x, crease_y, 0)
                new_positions.append(pivot + (P - pivot).rotate(angle=theta, axis=rot_axis))
            tri.v0.pos, tri.v1.pos, tri.v2.pos = new_positions
            for v in [tri.v0, tri.v1, tri.v2]:
                v.color = current_color
        update_outer_border(all_triangles)
        if step % 3 == 0:  # Save every 2nd frame only
            capture_frame_folding(bbox)
    update_outer_border(all_triangles)
    msg_label.visible = False

def fold_vertical_left_to_right(all_triangles, steps=30, fold_to_front=True):
    direction = "forward" if fold_to_front else "backward"
    msg_label = create_folding_label(f"Folding vertically from left to right edge, {direction}")
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    crease_x = (min_x + max_x) / 2
    draw_fold_line_vertical(all_triangles, crease_x)
    moving = []
    orig_positions = {}
    for tri in all_triangles:
        centroid = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        if centroid.x < crease_x:
            moving.append(tri)
            orig_positions[tri] = [vector(tri.v0.pos), vector(tri.v1.pos), vector(tri.v2.pos)]
    theta_factor = 1 if fold_to_front else -1
    record_fold_operation('vertical_left_to_right', fold_to_front, steps, crease_x, moving, orig_positions,
                          theta_factor)
    rot_axis = vector(0, 1, 0)
    for step in range(steps + 1):
        rate(10)
        t = step / steps
        theta = theta_factor * math.pi * t
        current_color = get_current_color(t, base_color, dark_color)
        for tri in moving:
            new_positions = []
            for P in orig_positions[tri]:
                pivot = vector(crease_x, P.y, 0)
                new_positions.append(pivot + (P - pivot).rotate(angle=theta, axis=rot_axis))
            tri.v0.pos, tri.v1.pos, tri.v2.pos = new_positions
            for v in [tri.v0, tri.v1, tri.v2]:
                v.color = current_color
        update_outer_border(all_triangles)
        if step % 3 == 0:  # Save every 2nd frame only
            capture_frame_folding(bbox)
    update_outer_border(all_triangles)
    msg_label.visible = False


def fold_vertical_right_to_left(all_triangles, steps=30, fold_to_front=True):
    direction = "forward" if fold_to_front else "backward"
    msg_label = create_folding_label(f"Folding vertically from right to left edge, {direction}")
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    crease_x = (min_x + max_x) / 2
    draw_fold_line_vertical(all_triangles, crease_x)
    moving = []
    orig_positions = {}
    for tri in all_triangles:
        centroid = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        if centroid.x > crease_x:
            moving.append(tri)
            orig_positions[tri] = [vector(tri.v0.pos), vector(tri.v1.pos), vector(tri.v2.pos)]
    theta_factor = -1 if fold_to_front else 1
    record_fold_operation('vertical_right_to_left', fold_to_front, steps, crease_x, moving, orig_positions,
                          theta_factor)
    rot_axis = vector(0, 1, 0)
    for step in range(steps + 1):
        rate(10)
        t = step / steps
        theta = theta_factor * math.pi * t
        current_color = get_current_color(t, base_color, dark_color)
        for tri in moving:
            new_positions = []
            for P in orig_positions[tri]:
                pivot = vector(crease_x, P.y, 0)
                new_positions.append(pivot + (P - pivot).rotate(angle=theta, axis=rot_axis))
            tri.v0.pos, tri.v1.pos, tri.v2.pos = new_positions
            for v in [tri.v0, tri.v1, tri.v2]:
                v.color = current_color
        update_outer_border(all_triangles)
        if step % 3 == 0:  # Save every 2nd frame only
            capture_frame_folding(bbox)
    update_outer_border(all_triangles)
    msg_label.visible = False


def fold_triangles(all_triangles, crease_p1, crease_p2, steps, fold_to_front, theta_factor, fold_type):
    d = crease_p2 - crease_p1
    rot_axis = d.norm()
    moving = []
    orig_positions = {}
    for tri in all_triangles:
        centroid = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        sign = d.x * (centroid.y - crease_p1.y) - d.y * (centroid.x - crease_p1.x)
        if sign > 0:
            moving.append(tri)
            orig_positions[tri] = [vector(tri.v0.pos), vector(tri.v1.pos), vector(tri.v2.pos)]
    record_fold_operation(fold_type, fold_to_front, steps, (crease_p1, crease_p2), moving, orig_positions, theta_factor)
    for step in range(steps + 1):
        rate(10)
        t = step / steps
        theta = theta_factor * math.pi * t
        current_color = get_current_color(t, base_color, color.gray(0.8))
        for tri in moving:
            new_positions = []
            for P in orig_positions[tri]:
                pivot = crease_p1 + rot_axis * ((P - crease_p1).dot(rot_axis))
                new_positions.append(pivot + (P - pivot).rotate(angle=theta, axis=rot_axis))
            tri.v0.pos, tri.v1.pos, tri.v2.pos = new_positions
            for v in [tri.v0, tri.v1, tri.v2]:
                v.color = current_color
        update_outer_border(all_triangles)
        if step % 4 == 0:  # Save every 2nd frame only
            capture_frame_folding(bbox)
    update_outer_border(all_triangles)


def fold_diagonal_topRight_to_bottomLeft(all_triangles, steps=30, fold_to_front=True):
    direction = "forward" if fold_to_front else "backward"
    msg_label = create_folding_label(f"Folding diagonally from top-right to bottom-left edge, {direction}")
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    crease_p1 = vector(min_x, max_y, 0)
    crease_p2 = vector(max_x, min_y, 0)
    draw_fold_line_diagonal(crease_p1, crease_p2)
    theta_factor = 1 if fold_to_front else -1
    fold_triangles(all_triangles, crease_p1, crease_p2, steps, fold_to_front, theta_factor, "diagonal_topRight_to_bottomLeft")
    msg_label.visible = False

def fold_diagonal_bottomRight_to_topLeft(all_triangles, steps=30, fold_to_front=True):
    direction = "forward" if fold_to_front else "backward"
    msg_label = create_folding_label(f"Folding diagonally from bottom-right to top-left edge, {direction}")
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    crease_p1 = vector(max_x, max_y, 0)
    crease_p2 = vector(min_x, min_y, 0)
    draw_fold_line_diagonal(crease_p1, crease_p2)
    theta_factor = 1 if fold_to_front else -1
    fold_triangles(all_triangles, crease_p1, crease_p2, steps, fold_to_front, theta_factor, "diagonal_bottomRight_to_topLeft")
    msg_label.visible = False

def fold_diagonal_bottomLeft_to_topRight(all_triangles, steps=30, fold_to_front=True):
    direction = "forward" if fold_to_front else "backward"
    msg_label = create_folding_label(f"Folding diagonally from bottom-left to top-right edge, {direction}")
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    crease_p1 = vector(max_x, min_y, 0)
    crease_p2 = vector(min_x, max_y, 0)
    draw_fold_line_diagonal(crease_p1, crease_p2)
    theta_factor = 1 if fold_to_front else -1
    fold_triangles(all_triangles, crease_p1, crease_p2, steps, fold_to_front, theta_factor, "diagonal_bottomLeft_to_topRight")
    msg_label.visible = False

def fold_diagonal_topLeft_to_bottomRight(all_triangles, steps=30, fold_to_front=True):
    direction = "forward" if fold_to_front else "backward"
    msg_label = create_folding_label(f"Folding diagonally from top-left to bottom-right edge, {direction}")
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    crease_p1 = vector(min_x, min_y, 0)
    crease_p2 = vector(max_x, max_y, 0)
    draw_fold_line_diagonal(crease_p1, crease_p2)
    theta_factor = 1 if fold_to_front else -1
    fold_triangles(all_triangles, crease_p1, crease_p2, steps, fold_to_front, theta_factor, "diagonal_topLeft_to_bottomRight")
    msg_label.visible = False



def remove_fold_crease_line():
    """
    Remove any existing crease (fold) line from the scene.
    """
    global fold_crease_dashes
    if fold_crease_dashes:
        for dash in fold_crease_dashes:
            dash.visible = False
    fold_crease_dashes = []

def rotate(all_triangles, angle_degrees, steps=30):
    """
    Rotate the current paper configuration around its centroid in-plane
    and update every earlier fold-record so that subsequent UNFOLDs
    happen in the rotated coordinate frame.

    Parameters
    ----------
    all_triangles : list[vpython.triangle]
        Triangles that make up the sheet at the moment of rotation.
    angle_degrees : {90, 180, 270, 360}
        Counter-clockwise rotation amount (use multiples of 90° so the
        fold-type remapping table stays valid).
    steps : int, optional
        Animation steps for the visible spin.
    """
    remove_fold_crease_line()
    # find centre of the current sheet
    min_x, max_x, min_y, max_y = compute_bounding_box(all_triangles)
    center = vector((min_x + max_x) / 2, (min_y + max_y) / 2, 0)

    msg_label = create_folding_label(f"Rotating paper by {angle_degrees}°")

    # Store original vertex locations so the visual animation can interpolate from them
    moving = list(all_triangles)           # every triangle moves
    orig_positions = {
        tri: [vector(tri.v0.pos), vector(tri.v1.pos), vector(tri.v2.pos)]
        for tri in all_triangles
    }

    # Record the fact we are doing a rotation (kept for completeness,
    # but UNFOLD logic will ignore these “rotation-*” records).
    record_fold_operation(f"rotation-{angle_degrees}",
                          True, steps, center, moving,
                          orig_positions, angle_degrees)

    #ensure every historic fold record is expressed in the new coordinate frame so that when we UNFOLD later it “knows” the sheet was turned.
    angle_rad      = math.radians(angle_degrees)
    quarter_turns  = (angle_degrees // 90) % 4        # 0–3

    for rec in fold_history:
        # Skip other rotation records
        if rec['fold_type'].startswith('rotation'):
            continue

        # rotate stored vertex positions
        for tri, verts in rec['orig_positions'].items():
            rec['orig_positions'][tri] = [
                _rotate_point(v, center, angle_rad) for v in verts
            ]

        # rotate the crease description (H/V numeric, diagonals as points)
        if isinstance(rec['crease'], (float, int)):
            horiz = rec['fold_type'].startswith('horizontal')
            # point ON the old crease line, then spin it
            p_old = vector(center.x, rec['crease'], 0) if horiz else \
                vector(rec['crease'], center.y, 0)
            p_new = _rotate_point(p_old, center, angle_rad)

            swapped = (quarter_turns % 2 == 1)

            if horiz:
                rec['crease'] = p_new.x if swapped else p_new.y
            else:  # vertical
                rec['crease'] = p_new.y if swapped else p_new.x

        else:  # diagonal crease = (p1, p2)
            p1, p2 = rec['crease']
            rec['crease'] = (
                _rotate_point(p1, center, angle_rad),
                _rotate_point(p2, center, angle_rad)
            )

        # remap the symbolic fold type
        if 'original_fold_type' not in rec:
            rec['original_fold_type'] = rec['fold_type']
        rec['fold_type'] = remap_fold_type(rec['fold_type'], quarter_turns)

    # remember net orientation if you want to query it elsewhere
    global current_rotation_deg
    current_rotation_deg = (current_rotation_deg + angle_degrees) % 360

    for step in range(steps + 1):
        rate(10)                               # adjust for desired speed
        t          = step / steps
        cur_angle  = angle_rad * t
        for tri in moving:
            tri.v0.pos, tri.v1.pos, tri.v2.pos = [
                center + (P - center).rotate(angle=cur_angle,
                                             axis=vector(0, 0, 1))
                for P in orig_positions[tri]
            ]
        update_outer_border(all_triangles)
        if step % 3 == 0:  # Save every 2nd frame only
            capture_frame_folding(bbox)

    update_outer_border(all_triangles)
    msg_label.visible = False

# Create a mapping from folding action codes to functions with the required parameters.
def execute_fold(fold, all_triangles, steps=10):
    fold_mapping = {
        # Horizontal folds:
        "H1-F": lambda: fold_horizontal_top_to_bottom(all_triangles, steps=steps, fold_to_front=True),
        "H1-B": lambda: fold_horizontal_top_to_bottom(all_triangles, steps=steps, fold_to_front=False),
        "H2-F": lambda: fold_horizontal_bottom_to_top(all_triangles, steps=steps, fold_to_front=True),
        "H2-B": lambda: fold_horizontal_bottom_to_top(all_triangles, steps=steps, fold_to_front=False),

        # Vertical folds:
        "V1-F": lambda: fold_vertical_left_to_right(all_triangles, steps=steps, fold_to_front=True),
        "V1-B": lambda: fold_vertical_left_to_right(all_triangles, steps=steps, fold_to_front=False),
        "V2-F": lambda: fold_vertical_right_to_left(all_triangles, steps=steps, fold_to_front=True),
        "V2-B": lambda: fold_vertical_right_to_left(all_triangles, steps=steps, fold_to_front=False),

        # Diagonal folds:
        "D1-F": lambda: fold_diagonal_topLeft_to_bottomRight(all_triangles, steps=steps, fold_to_front=True),
        "D1-B": lambda: fold_diagonal_topLeft_to_bottomRight(all_triangles, steps=steps, fold_to_front=False),
        "D2-F": lambda: fold_diagonal_topRight_to_bottomLeft(all_triangles, steps=steps, fold_to_front=True),
        "D2-B": lambda: fold_diagonal_topRight_to_bottomLeft(all_triangles, steps=steps, fold_to_front=False),
        "D3-F": lambda: fold_diagonal_bottomLeft_to_topRight(all_triangles, steps=steps, fold_to_front=True),
        "D3-B": lambda: fold_diagonal_bottomLeft_to_topRight(all_triangles, steps=steps, fold_to_front=False),
        "D4-F": lambda: fold_diagonal_bottomRight_to_topLeft(all_triangles, steps=steps, fold_to_front=True),
        "D4-B": lambda: fold_diagonal_bottomRight_to_topLeft(all_triangles, steps=steps, fold_to_front=False),

        #Rotations"
        "R-90": lambda: rotate(all_triangles, angle_degrees=90, steps=15),
        "R-180": lambda: rotate(all_triangles, angle_degrees=180, steps=15),
        "R-270": lambda: rotate(all_triangles, angle_degrees=270, steps=15),
        "R-360": lambda: rotate(all_triangles, angle_degrees=360, steps=15),
    }

    if fold in fold_mapping:
        fold_mapping[fold]()
        return all_triangles
    else:
        raise ValueError(f"Unknown folding action: {fold}")



# ------------------------------
# Mirror Hole Function
# ------------------------------

def mirror_hole(P, rotation, crease, fold_type):
    if fold_type in ['horizontal_top_to_bottom', 'horizontal_bottom_to_top']:
        new_P = vector(P.x, 2 * crease - P.y, P.z)
        if rotation == 270:
            new_rotation = 270
        elif rotation == 90:
            new_rotation = 90
        elif rotation == 0:
            new_rotation = 180
        elif rotation == 180:
            new_rotation = 0
        else:
            new_rotation = rotation
    elif fold_type in ['vertical_left_to_right', 'vertical_right_to_left']:
        new_P = vector(2 * crease - P.x, P.y, P.z)
        if rotation == 270:
            new_rotation = 90
        elif rotation == 90:
            new_rotation = 270
        elif rotation == 0:
            new_rotation = 0
        elif rotation == 180:
            new_rotation = 180
        else:
            new_rotation = rotation
    elif fold_type in ['diagonal_topLeft_to_bottomRight', 'diagonal_bottomRight_to_topLeft']:
        p1, p2 = crease
        d = P - p1
        u = (p2 - p1).norm()
        proj = u * (d.dot(u))
        new_P = p1 + 2 * proj - d
        if rotation == 270:
            new_rotation = 0
        elif rotation == 90:
            new_rotation = 180
        elif rotation == 0:
            new_rotation = 270
        elif rotation == 180:
            new_rotation = 90
        else:
            new_rotation = rotation
    elif fold_type in ['diagonal_topRight_to_bottomLeft', 'diagonal_bottomLeft_to_topRight']:
        p1, p2 = crease
        d = P - p1
        u = (p2 - p1).norm()
        proj = u * (d.dot(u))
        new_P = p1 + 2 * proj - d
        if rotation == 270:
            new_rotation = 180
        elif rotation == 90:
            new_rotation = 0
        elif rotation == 0:
            new_rotation = 90
        elif rotation == 180:
            new_rotation = 270
        else:
            new_rotation = rotation
    else:
        new_P = P
        new_rotation = rotation
    return new_P, new_rotation


# ------------------------------
# Updating Hole Records After Unfolding
# ------------------------------

def update_hole_triangle_ids(all_triangles):
    global punched_holes
    valid_holes = []
    for hole_record in punched_holes:
        location = hole_record["base_pos"]
        tri_ids = []

        for tri in all_triangles:
            A = tri.v0.pos
            B = tri.v1.pos
            C = tri.v2.pos
            v0 = C - A
            v1 = B - A
            v2 = location - A
            dot00 = v0.dot(v0)
            dot01 = v0.dot(v1)
            dot02 = v0.dot(v2)
            dot11 = v1.dot(v1)
            dot12 = v1.dot(v2)
            invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            if (u >= -1e-3) and (v >= -1e-3) and (u + v <= 1 + 1e-3):
                tri_ids.append(tri.tri_id)

        if tri_ids:
            hole_record["tri_ids"] = tri_ids
            valid_holes.append(hole_record)
        else:
            # Hide visually even if it's an initial hole (but preserve in JSON)
            hole_record["stable"].visible = False
            hole_record["moving"].visible = False

            if hole_record.get("is_initial", False):
                # Keep it for JSON only
                hole_record["tri_ids"] = []
                valid_holes.append(hole_record)

    punched_holes = valid_holes

# ------------------------------
# Unfolding Functions
# ------------------------------

def unfold_operation(record, all_triangles, steps=40):
    fold_type = record['fold_type']
    moving = record['moving']
    moving_tri_ids = set(record['moving_tri_ids'])
    theta_factor = record['theta_factor']
    fold_to_front =  record['fold_to_front']

    if fold_type in ['horizontal_top_to_bottom', 'horizontal_bottom_to_top']:
        crease = record['crease']  # y-value
        draw_fold_line_horizontal(all_triangles, crease)
        rot_axis = vector(1, 0, 0)
        theta_max = (1 if fold_type == 'horizontal_top_to_bottom' else -1) * math.pi
        # Reverse the rotation if the fold was performed from behind.
        if not fold_to_front:
            theta_max = -theta_max
        for step in range(steps + 1):
            rate(7)
            t = step / steps
            theta_tri = theta_max * (1 - t)
            current_color = get_current_color(1 - t, base_color, dark_color)
            for tri in moving:
                new_positions = []
                for P in record['orig_positions'][tri]:
                    pivot = vector(P.x, crease, 0)
                    new_positions.append(pivot + (P - pivot).rotate(angle=theta_tri, axis=rot_axis))
                tri.v0.pos, tri.v1.pos, tri.v2.pos = new_positions
                for v in [tri.v0, tri.v1, tri.v2]:
                    v.color = current_color
            update_outer_border(all_triangles)
            if step % 3 == 0:
                capture_frame_unfolding(bbox)
        update_outer_border(all_triangles)

    elif fold_type in ['vertical_left_to_right', 'vertical_right_to_left']:
        crease = record['crease']  # x-value
        draw_fold_line_vertical(all_triangles, crease)
        rot_axis = vector(0, 1, 0)
        theta_max = (1 if fold_type == 'vertical_left_to_right' else -1) * math.pi
        # Reverse the rotation if the fold was performed from behind.
        if not fold_to_front:
            theta_max = -theta_max
        for step in range(steps + 1):
            rate(7)
            t = step / steps
            theta_tri = theta_max * (1 - t)
            current_color = get_current_color(1 - t, base_color, dark_color)
            for tri in moving:
                new_positions = []
                for P in record['orig_positions'][tri]:
                    pivot = vector(crease, P.y, 0)
                    new_positions.append(pivot + (P - pivot).rotate(angle=theta_tri, axis=rot_axis))
                tri.v0.pos, tri.v1.pos, tri.v2.pos = new_positions
                for v in [tri.v0, tri.v1, tri.v2]:
                    v.color = current_color
            update_outer_border(all_triangles)
            if step % 3 == 0:  # Save every 3rd frame only
                capture_frame_unfolding(bbox)
        update_outer_border(all_triangles)

    elif fold_type in ['diagonal_topLeft_to_bottomRight', 'diagonal_bottomRight_to_topLeft', 'diagonal_topRight_to_bottomLeft', 'diagonal_bottomLeft_to_topRight']:
        crease = record['crease']  # (p1, p2)
        p1, p2 = crease
        draw_fold_line_diagonal(crease[0], crease[1])
        rot_axis = (p2 - p1).norm()
        theta_max = theta_factor * math.pi
        for step in range(steps + 1):
            rate(10)
            t = step / steps
            theta_tri = theta_max * (1 - t)
            current_color = get_current_color(1 - t, base_color, color.gray(0.8))
            for tri in moving:
                new_positions = []
                for P in record['orig_positions'][tri]:
                    pivot = p1 + rot_axis * ((P - p1).dot(rot_axis))
                    new_positions.append(pivot + (P - pivot).rotate(angle=theta_tri, axis=rot_axis))
                tri.v0.pos, tri.v1.pos, tri.v2.pos = new_positions
                for v in [tri.v0, tri.v1, tri.v2]:
                    v.color = current_color
            update_outer_border(all_triangles)

            if step % 3 == 0:
                capture_frame_unfolding(bbox)
        update_outer_border(all_triangles)
    # --- After the unfolding animation, punch new holes on the unfolded (moving) part.
    # For each hole on the moving part, compute its mirror transformation and punch new stable and moving holes.
    if fold_type in [
        'horizontal_top_to_bottom', 'horizontal_bottom_to_top',
        'vertical_left_to_right', 'vertical_right_to_left',
        'diagonal_topLeft_to_bottomRight', 'diagonal_bottomRight_to_topLeft',
        'diagonal_topRight_to_bottomLeft', 'diagonal_bottomLeft_to_topRight'
    ]:

        new_holes = []
        for hole in punched_holes:
            if set(hole['tri_ids']).intersection(moving_tri_ids) and hole["stable"].visible:
                mirror_pos, mirror_rot = mirror_hole(hole['base_pos'], hole['direction'], record['crease'], fold_type)
                # Check if the hole is of type text and if its text has two letters.
                if hole['shape_type'].startswith("text"):
                    # Assume the original punched hole's stable object holds the text.
                    current_text = hole['stable'].text if hasattr(hole['stable'], 'text') else "TA"
                    if len(current_text) == 2:
                        new_text = current_text[1] + current_text[0]
                    else:
                        new_text = current_text
                    stable_new = punch_hole(mirror_pos, shape_type=hole['shape_type'],
                                            size_option=hole['size_option'], direction=mirror_rot, text_val=new_text)
                    moving_new = punch_hole(mirror_pos, shape_type=hole['shape_type'],
                                            size_option=hole['size_option'], direction=mirror_rot, text_val=new_text)
                else:
                    stable_new = punch_hole(mirror_pos, shape_type=hole['shape_type'],
                                            size_option=hole['size_option'], direction=mirror_rot)
                    moving_new = punch_hole(mirror_pos, shape_type=hole['shape_type'],
                                            size_option=hole['size_option'], direction=mirror_rot)
                new_hole = {
                    "stable": stable_new,
                    "moving": moving_new,
                    "base_pos": mirror_pos,
                    "shape_type": hole['shape_type'],
                    "size_option": hole['size_option'],
                    "direction": mirror_rot,
                    "tri_ids": hole['tri_ids'][:],
                    "initial_tri_id": hole.get("initial_tri_id"),
                    "initial_direction": hole.get("initial_direction", hole["direction"]),
                    "is_initial": False

                }
                new_holes.append(new_hole)
        punched_holes.extend(new_holes)
        update_hole_triangle_ids(all_triangles)
        for i in range(1):
            capture_frame_unfolding(bbox)

        remove_fold_crease_line()

def unfold_all(all_triangles, steps=40):
    # Simply perform each unfolding action in the fold history.
    for record in reversed(fold_history):
        unfold_operation(record, all_triangles, steps)
        # After final unfolding, print out the hole information.

    custom_ids = [
        26, 27, 30, 31, 25, 28, 29, 32,
        17, 20, 21, 24, 18, 19, 22, 23,
        10, 11, 14, 15, 9, 12, 13, 16,
        1, 4, 5, 8, 2, 3, 6, 7
    ]

    if any(record['fold_type'].startswith("rotation") for record in fold_history):

        # Sort triangles by centroid
        def centroid(tri):
            c = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
            return (round(c.y, 5), round(c.x, 5))  # Y before X

        sorted_tris = sorted(all_triangles, key=centroid)

        # build old→new mapping while assigning
        id_map = {}
        for new_id, tri in zip(custom_ids, sorted_tris):
            old_id = tri.tri_id
            id_map[old_id] = new_id
            tri.tri_id = new_id

    update_hole_triangle_ids(all_triangles)

def print_hole_info():
    print("Hole Information after final unfolding:")
    for hole in punched_holes:
        print("Triangle IDs:", hole["tri_ids"],
              "| Direction:", hole["direction"],
              "| Size Option:", hole["size_option"],
              "| Shape Type:", hole["shape_type"])
# ------------------------------
# Hole Punching Functions
# ------------------------------
def punch_hole(base_pos, shape_type="text", size_option="small", direction=0, text_val=None):
    size_multiplier = 1 if size_option == "small" else 2
    rotation_angle = math.radians(direction)
    pos = vector(base_pos.x, base_pos.y, HOLE_Z)
    if shape_type == "circle":
        hole = cylinder(pos=pos, axis=vector(0, 0, 0.02), radius=0.35 * size_multiplier,
                        color=color.black, visible=False)
    elif shape_type == "ellipse":
        hole = ellipsoid(pos=pos, length=0.75 * size_multiplier, height=0.33 * size_multiplier,
                         width=0.01, color=color.black, visible=False)
        hole.rotate(angle=rotation_angle, axis=vector(0, 0, 1))
    elif shape_type == "star":
        hole = extrusion(path=[pos, pos + vector(0, 0, 0.02)],
                         shape=shapes.star(n=5, radius=0.4 * size_multiplier),
                         color=color.black, visible=False)
        hole.rotate(angle=rotation_angle, axis=vector(0, 0, 1))
    elif shape_type == "triangle":
        hole = extrusion(path=[pos, pos + vector(0, 0, 0.02)],
                         shape=shapes.triangle(length=0.65 * size_multiplier),
                         color=color.black, visible=False)
        hole.rotate(angle=rotation_angle, axis=vector(0, 0, 1))
    elif shape_type == "rectangle":
        hole = extrusion(path=[pos, pos + vector(0, 0, 0.02)],
                         shape=shapes.rectangle(width=0.78 * size_multiplier, height=0.4 * size_multiplier),
                         color=color.black, visible=False)
        hole.rotate(angle=rotation_angle, axis=vector(0, 0, 1))
    elif shape_type == "trapezoid":
        hole = extrusion(path=[pos, pos + vector(0, 0, 0.02)],
                         shape=shapes.trapezoid(width=0.80 * size_multiplier, top=0.5 * size_multiplier,
                                                height=0.5 * size_multiplier),
                         color=color.black, visible=False)
        hole.rotate(angle=rotation_angle, axis=vector(0, 0, 1))
    elif shape_type == "letter":
        hole = text(pos=pos, text='T', align='center', height=0.5 * size_multiplier,
                    color=color.black, visible=False)
        hole.rotate(angle=rotation_angle, axis=vector(0, 0, 1))
    elif shape_type == "text":
        # Use the provided text_val if available; otherwise default to "TA"
        if text_val is None:
            text_val = "TA"
        hole = text(pos=pos, text=text_val, align='center', height=0.4 * size_multiplier,
                    color=color.black, visible=False)
        hole.rotate(angle=rotation_angle, axis=vector(0, 0, 1))
    elif shape_type == "square":
        hole = box(pos=pos, size=vector(0.5 * size_multiplier, 0.5 * size_multiplier, 0.02),
                   color=color.black, visible=False)
    hole.visible = True
    return hole


def calculate_hole_location(all_triangles):
    centers = []
    for tri in all_triangles:
        center = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        centers.append((center, tri.tri_id))
    unique = []
    for pos, tid in centers:
        if not any((pos - up[0]).mag < 1e-3 for up in unique):
            unique.append((pos, tid))
    return [up[0] for up in unique]

def hole_combinations(shape_types, sizes, directions, hole_locations, all_triangles,original_triangle):
    combinations = list(itertools.product(shape_types, sizes, directions, hole_locations))
    num_loc = len(hole_locations)
    # Max three hole can be punched
    if (num_loc > 8):
        num_holes = random.choice([1, 2, 3])
    elif (num_loc > 4):
        num_holes = random.choice([1, 2])
    else:
        num_holes = 1

    used_locations = []
    for i in range(num_holes):
        comb = random.choice(combinations)
        shape_type, size_option, direction, location = comb
        while location in used_locations:
            comb = random.choice(combinations)
            shape_type, size_option, direction, location = comb

        used_locations.append(location)
        stable_hole_obj = punch_hole(location, shape_type=shape_type, size_option=size_option, direction=direction)
        moving_hole_obj = punch_hole(location, shape_type=shape_type, size_option=size_option, direction=direction)
        tri_ids = []

        original_tri_id = None
        for tri_data in original_triangle:
            A = tri_data["A"]
            B = tri_data["B"]
            C = tri_data["C"]
            v0 = C - A
            v1 = B - A
            v2 = location - A
            dot00 = v0.dot(v0)
            dot01 = v0.dot(v1)
            dot02 = v0.dot(v2)
            dot11 = v1.dot(v1)
            dot12 = v1.dot(v2)
            invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            if (u >= -1e-3) and (v >= -1e-3) and (u + v <= 1 + 1e-3):
                original_tri_id = tri_data["tri_id"]

        for tri in all_triangles:
            A = tri.v0.pos
            B = tri.v1.pos
            C = tri.v2.pos
            v0 = C - A
            v1 = B - A
            v2 = location - A
            dot00 = v0.dot(v0)
            dot01 = v0.dot(v1)
            dot02 = v0.dot(v2)
            dot11 = v1.dot(v1)
            dot12 = v1.dot(v2)
            invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            if (u >= -1e-3) and (v >= -1e-3) and (u + v <= 1 + 1e-3):
                tri_ids.append(tri.tri_id)
        hole_record = {
            "stable": stable_hole_obj,
            "moving": moving_hole_obj,
            "base_pos": location,
            "shape_type": shape_type,
            "size_option": size_option,
            "direction": direction,
            "tri_ids": tri_ids,
            "initial_direction": direction,  # save the original direction here
            "initial_tri_id": original_tri_id,
            "is_initial": True
        }

        punched_holes.append(hole_record)
        capture_frame_folding(bbox)
    for i in range(3):
        capture_frame_folding(bbox)


def draw_paper_with_triangle_ids(all_triangles, next_id):
    """
    Draws the borders of each triangle and places a label with its ID
    at the center of the triangle.
    """

    for tri in all_triangles:
        # Create a list of points for the triangle edges, closing the triangle.
        pts = [tri.v0.pos, tri.v1.pos, tri.v2.pos, tri.v0.pos]
        # Draw dashed lines along the edges (using black dashes and a thin line).
        draw_dashed_line(pts, dash_length=0.1, gap_length=0.3, dash_color=color.black, dash_radius=0.01)

        # Compute the centroid of the triangle.
        center = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        # Place a label at the centroid showing the triangle's ID.

        # Adjust the height, offset, or color as needed.
        label(pos=center, text=str(tri.tri_id), height=12, box=False, opacity=0, color=color.red)
    time.sleep(2)
    for i in range(4):
        capture_image_result(bbox, next_id)

    flush_and_capture(capture_frame_unfolding, bbox)

# Automatically create next folder
def get_next_folder(base_dir, prefix):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    nums = [int(pattern.match(d).group(1)) for d in existing if pattern.match(d)]
    next_id = max(nums, default=0) + 1
    new_folder = f"{prefix}{next_id}"
    full_path = os.path.join(base_dir, new_folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path,next_id

def get_next_result_id(results_dir, prefix):
    os.makedirs(results_dir, exist_ok=True)
    existing = [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith("_result.png")]
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_result\\.png$")
    nums = [int(pattern.match(f).group(1)) for f in existing if pattern.match(f)]
    next_id = max(nums, default=0) + 1
    return next_id



bbox = (150, 150, 1400, 1100)
folding_frame_counter = 0
unfolding_frame_counter = 0
folding_base_dir = "folding_frames"
unfolding_base_dir = "unfolding_frames"
folding_frame_dir  = None   # will be set inside prediction()/planning()
unfolding_frame_dir = None
next_id            = None

def capture_frame_folding(bbox):
    global folding_frame_counter
    if folding_frame_dir is None: #planning
        return
    folding_frame_counter += 1
    filename = os.path.join(folding_frame_dir, f"frame_{folding_frame_counter:04d}.png")
    img = ImageGrab.grab(bbox= bbox)
    img.save(filename)


def capture_frame_unfolding(bbox):
    global unfolding_frame_counter
    if folding_frame_dir is None:  # planning
        return
    unfolding_frame_counter += 1
    filename = os.path.join(unfolding_frame_dir, f"frame_{unfolding_frame_counter:04d}.png")
    img = ImageGrab.grab(bbox= bbox)
    img.save(filename)

def capture_image_result(bbox, next_id):
    # Create the 'result' folder if it doesn't exist
    os.makedirs("prediction_results", exist_ok=True)
    filename = os.path.join("prediction_results", f"PFHP_{next_id}_result.png")
    img = ImageGrab.grab(bbox= bbox)
    img.save(filename)

def create_videos_for_all(
    frame_base_dir: str,
    fps: int = 1,    quality: int = 10
):
    """
    For each subfolder under `frame_base_dir`
    create a video in the matching videos directory.

    Parameters:
        frame_base_dir: str
            e.g. "folding_frames" or "unfolding_frames"
        fps: int
            frames per second
        quality: int
            1–10 (higher is better)
    """
    # Determine if we're in folding or unfolding mode
    if "unfolding" in frame_base_dir:
        video_base = frame_base_dir.replace("frames", "videos")
        prefix = "UP"
    elif "folding" in frame_base_dir:
        video_base = frame_base_dir.replace("frames", "videos")
        prefix = "PF"
    else:
        raise ValueError("frame_base_dir must contain 'folding' or 'unfolding'")

    os.makedirs(video_base, exist_ok=True)

    for sub in sorted(os.listdir(frame_base_dir)):
        subdir = os.path.join(frame_base_dir, sub)
        if not os.path.isdir(subdir):
            continue

        out_name = f"{sub}_animation.mp4"

        #out_name = f"{prefix}_{next_id}_animation.mp4"
        out_path = os.path.join(video_base, out_name)

        print(f"Creating video {out_path} from frames in {subdir}")

        writer = imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=quality,
            macro_block_size=None,  # disables resizing
            output_params = [
            "-probesize", "10000000",  # Increase probing buffer size
            "-analyzeduration", "10000000"  # Increase duration to analyze stream
            ]
        )

        # grab and sort all the .png frames
        frames = sorted(f for f in os.listdir(subdir) if f.endswith(".png"))
        for fname in frames:
            img = imageio.imread(os.path.join(subdir, fname))
            writer.append_data(img)

        writer.close()
        print(f" Video saved {out_path}")

def flush_and_capture(capture_fn, bbox):
    """
    Let VPython/OS finish painting, then run the supplied capture_fn.
    """
    rate(1)                 # one extra render cycle
    time.sleep(1)        #
    capture_fn(bbox)


def load_or_create_json(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)  # Start with an empty list of records
    with open(file_path, 'r') as f:
        return json.load(f)


def add_folding_record_json(task, JSON_FILE, next_id):


    # Process fold history
    folding_info = [
        {
            "foldType": record.get("original_fold_type", record["fold_type"]),
            "foldToFront": record['fold_to_front']
        }
        for record in fold_history

    ]


    # Step 2: Compute UNFOLDING sequence (reverse of remapped fold_types)
    unfold_types = []
    for rec in reversed(fold_history):
        ft = rec['fold_type']
        if ft.startswith("rotation"):
            continue  # skip rotations

        undo_type = _inverse_axis.get(ft, ft)
        undo_front = rec['fold_to_front']
        code = _fold2code.get((undo_type, undo_front))
        if code:
            unfold_types.append(code)

    def get_directions(hole, direction):
        # ellipse, rectangle two directions: 90-270 and 0-180 same
        # square and circle no direction, all of them same
        # letter, star, triangle, trapezoid and text have four different angle
        shape = hole["shape_type"]

        if shape in ("circle", "square"):
            return [""]
        elif shape in ("ellipse", "rectangle") and direction in (90, 270):
            return ["90, 270"]
        elif shape in ("ellipse", "rectangle") and direction in (0, 180):
            return ["0, 180"]
        else:
            return [direction]

    # Process initial holes (only the first num_holes entries)

    # ────────────────────────────────────────────────────────────────
    # 1. If any rotation occurred, realign *all* initial_tri_id fields
    # ────────────────────────────────────────────────────────────────
    if any(rec['fold_type'].startswith('rotation') for rec in fold_history):

        # Pass-1: rewrite the INITIAL holes and remember old→new mapping
        old2new = {}  # key = old id, value = new id
        for hole in punched_holes:
            if hole.get('is_initial', False) and hole.get('tri_ids'):
                old_id = hole.get('initial_tri_id')
                new_id = hole['tri_ids'][0]
                hole['initial_tri_id'] = new_id
                if old_id is not None:
                    old2new[old_id] = new_id

        # Pass-2: propagate that remap to every mirror derived from them
        for hole in punched_holes:
            if not hole.get('is_initial', False):
                oid = hole.get('initial_tri_id')
                if oid in old2new:
                    hole['initial_tri_id'] = old2new[oid]
    initial_holes = [
        {
            "shape": hole["shape_type"],
            "size": hole["size_option"],
            "direction": direction,
            "location": hole.get("initial_tri_id") or (hole["tri_ids"][0] if hole.get("tri_ids") else None),
        }
        for hole in punched_holes
        if hole.get("is_initial", False)
        for direction in get_directions(hole, hole["initial_direction"])
    ]

    # -----------------------------------------------------------
    #   Build the list of holes that are really visible at the end
    # -----------------------------------------------------------
    def mirror_is_visible(initial_hole):
        """Return True if any *visible* non-initial mirror of the same punch exists."""
        pid = initial_hole.get("initial_tri_id")
        return any(
            (not h.get("is_initial", False))  # must be a mirror
            and h.get("initial_tri_id") == pid  # derived from the same punch
            and h["stable"].visible  # and it actually shows
            for h in punched_holes
        )

    resulting_holes = []

    for hole in punched_holes:
        # 1) Ignore holes that are invisible after the final unfold
        if not hole["stable"].visible:
            continue

        # 2) If it's an invisible initial punch, skip it *unless it has a visible mirror*
        if not hole["stable"].visible:
            if hole.get("is_initial", False) and mirror_is_visible(hole):
                pass  # keep it
            else:
                continue  # skip invisible holes with no visible presence

        # 3) Everything that survived the filters above is written out
        for direct in get_directions(hole, hole["direction"]):
            resulting_holes.append({
                "shape": hole["shape_type"],
                "size": hole["size_option"],
                "direction": direct,
                "location": hole["tri_ids"][0]  # first triangle id is enough
            })

    # Build the record object
    if task == "prediction":
        record_obj = {
            "id": f"PFHP_{next_id}",
            "taskType": "Prediction",
            "numberofFoldingSteps": len(fold_history),
            "foldingTypes": folding_info,
            "unfoldingTypes": unfold_types,
            "foldingVideo": f"folding_videos/PF_{next_id}_animation.mp4",
            "foldingFramesDir": folding_frame_dir.replace("\\", "/"),
            "unfoldingVideo": f"unfolding_videos/UP_{next_id}_animation.mp4",
            "unfoldingFramesDir": unfolding_frame_dir.replace("\\", "/"),
            "resultImg": f"prediction_results/PFHP_{next_id}_result.png",
            "initialHoles": initial_holes,
            "resultHoles": resulting_holes,
            "totalNumberofHoles": len(resulting_holes),
        }
    elif task == "planning":
        record_obj = {
            "id": next_id,
            "taskType": "Planning",
            "numberofFoldingSteps": len(fold_history),
            "foldingTypes": folding_info,
            "unfoldingTypes": unfold_types,
            "initialHoles": initial_holes,
            "resultHoles": resulting_holes,
            "totalNumberofHoles": len(resulting_holes),
        }

    if record_obj["totalNumberofHoles"] % 2 == 1:
        raise ValueError(f"Total number of holes is odd ({record_obj['totalNumberofHoles']}). This is not allowed.")

    # Load existing data from the JSON file (or create a new list)
    data = load_or_create_json(JSON_FILE)

    # Append the new record and write back to the file
    data.append(record_obj)
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def _rotate_point(P, pivot, angle_rad):
    """Return P rotated CCW by angle_rad around Z through pivot."""
    return pivot + (P - pivot).rotate(angle=angle_rad, axis=vector(0, 0, 1))

def remap_fold_type(fold_type: str, quarter_turns: int) -> str:
    """
    Return the symbolic fold_type after rotating the whole sheet
    counter-clockwise by `quarter_turns` × 90°.

    quarter_turns must be 0‥3  (0 → 0°, 1 → 90°, 2 → 180°, 3 → 270°)
    """
    quarter_turns %= 4
    if quarter_turns == 0:                # no rotation
        return fold_type

    # --- 90° CCW -----------------------------------------------------------
    map90 = {
        # horizontals  →  verticals
        'horizontal_top_to_bottom' : 'vertical_left_to_right',
        'horizontal_bottom_to_top' : 'vertical_right_to_left',

        # verticals  →  horizontals
        'vertical_left_to_right'   : 'horizontal_bottom_to_top',
        'vertical_right_to_left'   : 'horizontal_top_to_bottom',

        # diagonals swap ↘ ↙
        'diagonal_topLeft_to_bottomRight'  : 'diagonal_bottomLeft_to_topRight',
        'diagonal_topRight_to_bottomLeft'  : 'diagonal_topLeft_to_bottomRight',
        'diagonal_bottomRight_to_topLeft'  : 'diagonal_topRight_to_bottomLeft',
        'diagonal_bottomLeft_to_topRight'  : 'diagonal_bottomRight_to_topLeft',
    }

    # --- 180° CCW  ------------------
    map180 = {
        # horizontals flip direction
        'horizontal_top_to_bottom' : 'horizontal_bottom_to_top',
        'horizontal_bottom_to_top' : 'horizontal_top_to_bottom',

        # verticals flip direction
        'vertical_left_to_right'   : 'vertical_right_to_left',
        'vertical_right_to_left'   : 'vertical_left_to_right',

        # diagonals halve-turn across same slope
        'diagonal_topLeft_to_bottomRight'  : 'diagonal_bottomRight_to_topLeft',
        'diagonal_bottomRight_to_topLeft'  : 'diagonal_topLeft_to_bottomRight',
        'diagonal_topRight_to_bottomLeft'  : 'diagonal_bottomLeft_to_topRight',
        'diagonal_bottomLeft_to_topRight'  : 'diagonal_topRight_to_bottomLeft',
    }

    # --- 270° CCW  ------------------------------------------
    map270 = {
        'horizontal_top_to_bottom' : 'vertical_right_to_left',
        'horizontal_bottom_to_top' : 'vertical_left_to_right',

        'vertical_left_to_right'   : 'horizontal_top_to_bottom',
        'vertical_right_to_left'   : 'horizontal_bottom_to_top',

        'diagonal_topLeft_to_bottomRight'  : 'diagonal_topRight_to_bottomLeft',
        'diagonal_topRight_to_bottomLeft'  : 'diagonal_bottomRight_to_topLeft',
        'diagonal_bottomRight_to_topLeft'  : 'diagonal_bottomLeft_to_topRight',
        'diagonal_bottomLeft_to_topRight'  : 'diagonal_topLeft_to_bottomRight',
    }

    # Pick correct lookup table
    if quarter_turns == 1:
        return map90.get(fold_type, fold_type)
    elif quarter_turns == 2:
        return map180.get(fold_type, fold_type)
    elif quarter_turns == 3:
        return map270.get(fold_type, fold_type)


_inverse_axis = {                    # opposite direction on SAME axis
    'horizontal_top_to_bottom'   : 'horizontal_bottom_to_top',
    'horizontal_bottom_to_top'   : 'horizontal_top_to_bottom',
    'vertical_left_to_right'     : 'vertical_right_to_left',
    'vertical_right_to_left'     : 'vertical_left_to_right',
    'diagonal_topLeft_to_bottomRight'  : 'diagonal_bottomRight_to_topLeft',
    'diagonal_bottomRight_to_topLeft'  : 'diagonal_topLeft_to_bottomRight',
    'diagonal_topRight_to_bottomLeft'  : 'diagonal_bottomLeft_to_topRight',
    'diagonal_bottomLeft_to_topRight'  : 'diagonal_topRight_to_bottomLeft',
}

_fold2code = {   # (type, front?) ➜ terse command
    ('horizontal_top_to_bottom',   True)  : 'H1-F',
    ('horizontal_top_to_bottom',   False) : 'H1-B',
    ('horizontal_bottom_to_top',   True)  : 'H2-F',
    ('horizontal_bottom_to_top',   False) : 'H2-B',
    ('vertical_left_to_right',     True)  : 'V1-F',
    ('vertical_left_to_right',     False) : 'V1-B',
    ('vertical_right_to_left',     True)  : 'V2-F',
    ('vertical_right_to_left',     False) : 'V2-B',
    ('diagonal_topLeft_to_bottomRight',   True)  : 'D1-F',
    ('diagonal_topLeft_to_bottomRight',   False) : 'D1-B',
    ('diagonal_topRight_to_bottomLeft',   True)  : 'D2-F',
    ('diagonal_topRight_to_bottomLeft',   False) : 'D2-B',
    ('diagonal_bottomLeft_to_topRight',   True)  : 'D3-F',
    ('diagonal_bottomLeft_to_topRight',   False) : 'D3-B',
    ('diagonal_bottomRight_to_topLeft',   True)  : 'D4-F',
    ('diagonal_bottomRight_to_topLeft',   False) : 'D4-B',
}




# ------------------------------
# Main Routine
# ------------------------------
def prediction(fold):
    scene.title = "Paper Folding & Hole Punching"
    scene.width = 1200
    scene.height = 800
    scene.background = color.black
    scene.camera.pos = vector(0, 0, 30)
    scene.camera.axis = vector(0, 0, -30)
    task = "prediction"

    global initial_label_pos
    global folding_frame_dir, unfolding_frame_dir, next_id
    folding_frame_dir, next_id = get_next_folder("folding_frames", "PF_")
    unfolding_frame_dir, _ = get_next_folder("unfolding_frames", "UP_")

    paper_size = 16

    JSON_FILE = "MentalBlackboard_Prediction_Data.json"

    # If there is a comma in fold_arg, then it contains multiple fold codes.
    if "," in fold:
        fold_codes = fold.split(',')
    else:
        fold_codes = [fold]

    all_triangles = create_paper_triangles(paper_size)

    original_triangle = []
    for tri in all_triangles:
        A = tri.v0.pos
        B = tri.v1.pos
        C = tri.v2.pos
        original_triangle.append({
            "tri_id": tri.tri_id,
            "A": vector(A.x, A.y, A.z),
            "B": vector(B.x, B.y, B.z),
            "C": vector(C.x, C.y, C.z)
        })

    # Store the original ID position map before any fold/run rotation
    triangle_label_positions = []
    for tri in all_triangles:
        center = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        triangle_label_positions.append((tri.tri_id, vector(center.x, center.y, center.z)))

    initial_border_points = get_ordered_border_vertices(all_triangles)

    # Compute and store the fixed label position
    initial_label_pos = get_fixed_label_position(initial_border_points)
    update_outer_border(all_triangles)

    for fcode in fold_codes:
        #print(f"Executing fold: {fcode}")
        all_triangles = execute_fold(fcode, all_triangles, steps=20)

    # Punch holes after folding:
    if "R-" in fold:
        #Circle and square does not work for rotation.
        hole_shape = ['ellipse', 'star', 'triangle', 'rectangle', 'trapezoid', 'letter', 'text']
    else:
        # Text is not represented accurate in image and text map format.
        # Also square and rectangle can be placed out of the triangle position in image and text format.
        #hole_shape_2D = ['ellipse', 'star', 'triangle', 'circle', 'trapezoid', 'letter']
        hole_shape = ['circle', 'square','ellipse', 'star', 'triangle', 'rectangle', 'trapezoid', 'letter', 'text']

    sizes = ['small','large']
    directions = [0, 90, 180, 270]
    hole_locations = calculate_hole_location(all_triangles)
    if hole_locations:
        hole_combinations(hole_shape, sizes, directions, hole_locations, all_triangles,original_triangle)

    num_holes = len(punched_holes)

    # Unfold all recorded folds.
    unfold_all(all_triangles, steps= 15)

    # draw IDs – they match the initial arrangement even if rotations happened
    draw_paper_with_triangle_ids(all_triangles, next_id)

    add_folding_record_json(task, JSON_FILE, next_id)


def planning(fold: str, hole_specs: list[dict], id, input_file):
    scene.title = "Paper Folding & Hole Punching - Planning"
    scene.width = 1200
    scene.height = 800
    scene.background = color.black
    scene.camera.pos = vector(0, 0, 30)
    scene.camera.axis = vector(0, 0, -30)
    task = "planning"

    global initial_label_pos
    global folding_frame_dir, unfolding_frame_dir, next_id
    folding_frame_dir = None
    unfolding_frame_dir = None

    paper_size = 16

    base_name = os.path.splitext(input_file)[0]
    JSON_FILE = base_name + "_results.json"

    if "," in fold:
        fold_codes = [f.strip() for f in fold.split(',')]
    else:
        fold_codes = [fold.strip()]

    all_triangles = create_paper_triangles(paper_size)

    original_triangle = []
    for tri in all_triangles:
        A = tri.v0.pos
        B = tri.v1.pos
        C = tri.v2.pos
        original_triangle.append({
            "tri_id": tri.tri_id,
            "A": vector(A.x, A.y, A.z),
            "B": vector(B.x, B.y, B.z),
            "C": vector(C.x, C.y, C.z)
        })

    flat_centroid = {
        tri["tri_id"]: (tri["A"] + tri["B"] + tri["C"]) / 3
        for tri in original_triangle
    }

    triangle_label_positions = []
    for tri in all_triangles:
        center = (tri.v0.pos + tri.v1.pos + tri.v2.pos) / 3
        triangle_label_positions.append((tri.tri_id, vector(center.x, center.y, center.z)))

    initial_border_points = get_ordered_border_vertices(all_triangles)
    initial_label_pos = get_fixed_label_position(initial_border_points)
    update_outer_border(all_triangles)

    next_id = id

    # Perform folding
    for fcode in fold_codes:
        all_triangles = execute_fold(fcode, all_triangles, steps=20)

    for spec in hole_specs:
        shape_type = spec["shape"]
        size_option = spec["size"]
        direction = spec["direction"]
        tri_id = spec["location"]

        center = flat_centroid.get(tri_id)
        if center is None:
            print(f"Triangle {tri_id} not recognised in flat sheet.")
            continue

        stable_hole_obj = punch_hole(center, shape_type=shape_type, size_option=size_option, direction=direction)
        moving_hole_obj = punch_hole(center, shape_type=shape_type, size_option=size_option, direction=direction)

        # --- Find all triangles that contain this point ---
        tri_ids = []
        for tri in all_triangles:
            A, B, C = tri.v0.pos, tri.v1.pos, tri.v2.pos
            v0 = C - A
            v1 = B - A
            v2 = center - A
            dot00 = v0.dot(v0)
            dot01 = v0.dot(v1)
            dot02 = v0.dot(v2)
            dot11 = v1.dot(v1)
            dot12 = v1.dot(v2)
            invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            if (u >= -1e-3) and (v >= -1e-3) and (u + v <= 1 + 1e-3):
                tri_ids.append(tri.tri_id)

        hole_record = {
            "stable": stable_hole_obj,
            "moving": moving_hole_obj,
            "base_pos": center,
            "shape_type": shape_type,
            "size_option": size_option,
            "direction": direction,
            "tri_ids": tri_ids,
            "initial_direction": direction,
            "initial_tri_id": tri_id,
            "is_initial": True
        }

        punched_holes.append(hole_record)

    num_holes = len(punched_holes)

    # Important: update tri_ids before unfolding so mirror detection works
    update_hole_triangle_ids(all_triangles)

    # Unfold all folds and update visuals
    unfold_all(all_triangles, steps=15)
    add_folding_record_json(task, JSON_FILE, next_id)



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage for prediction: python run.py prediction [structure_group] [count]")
        print("Usage for planning:   python run.py planning <folds> <hole_specs>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    fold_arg = sys.argv[2]

    if mode == "prediction":
        print(f"Running prediction with folds: {fold_arg}")
        prediction(fold_arg)

    elif mode == "planning":
        if len(sys.argv) < 5:
            print("Missing hole_specs.")
            sys.exit(1)

        hole_specs_str = sys.argv[3]
        id_ = sys.argv[4]
        input_file = sys.argv[5]
        try:
            hole_specs = ast.literal_eval(hole_specs_str)
            print(f"Running planning with folds: {fold_arg}")
            print(f"Hole specs: {hole_specs}")
            planning(fold_arg, hole_specs, id_, input_file)
        except Exception as e:
            print("Error parsing hole_specs:", e)
            sys.exit(1)

    else:
        print(f"Unknown mode '{mode}'. Use 'prediction' or 'planning'.")
        sys.exit(1)
