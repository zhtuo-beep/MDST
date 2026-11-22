"""
CityGrid -> a closed, printable 3D mesh, in plain numpy. No Blender needed.

Same geometry as scripts/export_to_blender.py:
  * terrain: the height grid as a surface, a vertical skirt around the edge,
    and a flat base underneath -- one closed solid (watertight)
  * buildings: one closed box per building cell, sunk slightly into the
    ground so nothing floats

Two outputs:
  write_stl(city, path)  binary STL, one colour, for 3D printing / slicers
  write_glb(city, path)  coloured GLB, for viewing in a browser (the app)

Every triangle is wound counter-clockwise seen from OUTSIDE the solid, so
its normal points outwards. Blender fixed that for us with
recalc_face_normals; here we have to get it right by construction.
"""
from __future__ import annotations

import json
import struct

import numpy as np

from .city_grid import CityGrid, LAND_BUILDING, LAND_GREEN, LAND_ROAD

CELL_SIZE = 1.0             # world size of one grid cell
HEIGHT_SCALE = 8.0          # terrain_height is 0-1; scale it up to be visible
BUILDING_HEIGHT_SCALE = 1.0
BASE_THICKNESS = 2.0        # solid slab under the lowest point of the terrain
BUILDING_EMBED = 0.3        # how far each building sinks into the ground
BUILDING_HALF_WIDTH = 0.4   # boxes are 0.8 of a cell wide, so neighbours don't fuse

COLORS = {                  # same palette as export_to_blender.py
    "ground":         (0.78, 0.72, 0.55),
    "green":          (0.30, 0.55, 0.25),
    "road":           (0.15, 0.15, 0.16),
    "water":          (0.10, 0.35, 0.55),
    "building_short": (0.55, 0.55, 0.55),
    "building_mid":   (0.70, 0.62, 0.45),
    "building_tall":  (0.35, 0.42, 0.52),
}


def surface_heights(city: CityGrid) -> np.ndarray:
    """z of every grid point: water sits at 0, land at its scaled terrain height."""
    z = city.terrain_height.astype(np.float64) * HEIGHT_SCALE
    return np.where(city.is_water, 0.0, z)


def _perimeter(size: int) -> list:
    """Edge grid points in counter-clockwise order (seen from above), corners once."""
    p = [(0, x) for x in range(size)]
    p += [(y, size - 1) for y in range(1, size)]
    p += [(size - 1, x) for x in range(size - 2, -1, -1)]
    p += [(y, 0) for y in range(size - 2, 0, -1)]
    return p


def terrain_mesh(city: CityGrid):
    """
    The terrain as ONE closed solid. Returns (vertices (V, 3), triangles (T, 3), colors (T, 3)).

    Vertex y * size + x is grid point (x, y). Each grid cell is a quad between
    four neighbouring points, split into two triangles.
    """
    size = city.size
    z = surface_heights(city)
    ys, xs = np.mgrid[0:size, 0:size]
    top = np.column_stack([xs.ravel() * CELL_SIZE, ys.ravel() * CELL_SIZE, z.ravel()])

    # --- top surface: (x,y) -> (x+1,y) -> (x+1,y+1) -> (x,y+1) is CCW from above
    cy, cx = np.mgrid[0:size - 1, 0:size - 1]
    a = (cy * size + cx).ravel()
    b = a + 1
    c = a + size + 1
    d = a + size
    surface = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])

    cell = np.full((size - 1, size - 1, 3), COLORS["ground"])
    lu = city.land_use[:-1, :-1]
    cell[lu == LAND_GREEN] = COLORS["green"]
    cell[lu == LAND_ROAD] = COLORS["road"]
    cell[city.is_water[:-1, :-1]] = COLORS["water"]
    cell = cell.reshape(-1, 3)
    surface_colors = np.concatenate([cell, cell])

    # --- skirt and base. Bottom vertices are created ONCE and shared by the
    # skirt and the base; creating them twice would leave an invisible seam.
    perim = _perimeter(size)
    n = len(perim)
    base_z = z.min() - BASE_THICKNESS
    top_idx = np.array([y * size + x for y, x in perim])
    bottom = np.column_stack([top[top_idx, 0], top[top_idx, 1], np.full(n, base_z)])
    bot_idx = len(top) + np.arange(n)
    center_idx = len(top) + n
    center = [[(size - 1) * CELL_SIZE / 2, (size - 1) * CELL_SIZE / 2, base_z]]

    i = np.arange(n)
    j = (i + 1) % n
    # walking the edge CCW from above, the outside is to the right:
    # (top_i, bottom_i, bottom_j) and (top_i, bottom_j, top_j) face outwards
    skirt = np.concatenate([
        np.column_stack([top_idx[i], bot_idx[i], bot_idx[j]]),
        np.column_stack([top_idx[i], bot_idx[j], top_idx[j]]),
    ])
    # base: a fan from the centre, wound clockwise from above so it faces DOWN.
    # (A fan from a corner would create zero-area slivers along the straight edges.)
    base = np.column_stack([np.full(n, center_idx), bot_idx[j], bot_idx[i]])

    vertices = np.vstack([top, bottom, center])
    triangles = np.vstack([surface, skirt, base])
    colors = np.vstack([surface_colors, np.tile(COLORS["ground"], (len(skirt) + len(base), 1))])
    return vertices, triangles, colors


# one box: 8 corners, 12 outward-facing triangles
_BOX_CORNERS = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
_BOX_TRIANGLES = np.array([
    [0, 2, 1], [0, 3, 2],          # bottom (faces down)
    [4, 5, 6], [4, 6, 7],          # top    (faces up)
    [0, 1, 5], [0, 5, 4],          # side y = -1
    [1, 2, 6], [1, 6, 5],          # side x = +1
    [2, 3, 7], [2, 7, 6],          # side y = +1
    [3, 0, 4], [3, 4, 7],          # side x = -1
])


def buildings_mesh(city: CityGrid):
    """One closed box per building cell. Returns (vertices, triangles, colors)."""
    size = city.size
    z = surface_heights(city)
    mask = (city.land_use == LAND_BUILDING) & (city.building_height > 0)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3))

    h = city.building_height[ys, xs].astype(np.float64) * BUILDING_HEIGHT_SCALE
    # start from the LOWEST of the four surrounding grid points, then sink a
    # little further -- otherwise boxes hover on slopes and print as floating blocks
    y1 = np.minimum(ys + 1, size - 1)
    x1 = np.minimum(xs + 1, size - 1)
    base_z = np.minimum.reduce([z[ys, xs], z[ys, x1], z[y1, xs], z[y1, x1]]) - BUILDING_EMBED
    top_z = z[ys, xs] + h

    k = len(ys)
    corners = np.broadcast_to(_BOX_CORNERS, (k, 8, 3)).copy()
    corners[:, :, 0] = xs[:, None] * CELL_SIZE + corners[:, :, 0] * BUILDING_HALF_WIDTH * CELL_SIZE
    corners[:, :, 1] = ys[:, None] * CELL_SIZE + corners[:, :, 1] * BUILDING_HALF_WIDTH * CELL_SIZE
    corners[:, :, 2] = np.where(_BOX_CORNERS[None, :, 2] > 0, top_z[:, None], base_z[:, None])
    vertices = corners.reshape(-1, 3)
    triangles = (_BOX_TRIANGLES[None, :, :] + 8 * np.arange(k)[:, None, None]).reshape(-1, 3)

    low_cut, high_cut = np.percentile(h, [40, 75])
    tone = np.where(h < low_cut, 0, np.where(h < high_cut, 1, 2))
    palette = np.array([COLORS["building_short"], COLORS["building_mid"], COLORS["building_tall"]])
    colors = np.repeat(palette[tone], 12, axis=0)
    return vertices, triangles, colors


def city_mesh(city: CityGrid):
    """Terrain + buildings combined. Returns (vertices, triangles, colors)."""
    tv, tt, tc = terrain_mesh(city)
    bv, bt, bc = buildings_mesh(city)
    return np.vstack([tv, bv]), np.vstack([tt, bt + len(tv)]), np.vstack([tc, bc])


def _face_normals(vertices, triangles):
    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    n = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(length == 0, 1, length)


def write_stl(city: CityGrid, path) -> int:
    """Binary STL of the whole city. Returns the number of triangles."""
    vertices, triangles, _ = city_mesh(city)
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(triangles), dtype=record)
    data["normal"] = _face_normals(vertices, triangles)
    data["v"] = vertices[triangles]
    with open(path, "wb") as f:
        f.write(b"citygen city".ljust(80, b" "))
        f.write(struct.pack("<I", len(triangles)))
        f.write(data.tobytes())
    return len(triangles)


def write_glb(city: CityGrid, path) -> int:
    """
    Coloured GLB (binary glTF) for the browser viewer. Returns the triangle count.

    Each triangle gets its own three vertices so it can carry its own colour
    (flat, per-cell colours, like the Blender materials). No normals are
    stored: glTF viewers must then compute flat normals, which is the look
    we want for a blocky city.
    """
    vertices, triangles, colors = city_mesh(city)
    # glTF is Y-up; our city is Z-up. Rotate -90 deg about x: (x, y, z) -> (x, z, -y).
    # A rotation keeps every triangle's winding, so faces still point outwards.
    v = vertices[triangles].reshape(-1, 3)
    v = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])
    v -= (v.min(axis=0) + v.max(axis=0)) / 2          # centre the model on the origin
    v /= np.abs(v).max() / 5.0                        # and fit it in a ~10-unit box
    positions = v.astype("<f4")
    rgba = np.column_stack([np.repeat(colors, 3, axis=0), np.ones(len(positions))]).astype("<f4")

    pos_bytes, col_bytes = positions.tobytes(), rgba.tobytes()
    binary = pos_bytes + col_bytes
    count = len(positions)
    gltf = {
        "asset": {"version": "2.0", "generator": "citygen"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "city"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                 "metallicFactor": 0.0, "roughnessFactor": 0.9},
                       "doubleSided": False}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(col_bytes), "target": 34962},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": count, "type": "VEC4"},
        ],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)       # chunks must be 4-byte aligned
    binary += b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes)
        f.write(struct.pack("<I4s", len(binary), b"BIN\x00") + binary)
    return len(triangles)
