"""Check whether binary STL files contain closed, outward-facing solids.

Usage: python scripts/check_stl.py data/gallery/city_001_x.stl [other.stl ...]
"""
from __future__ import annotations

import argparse
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def check(path: str | Path) -> bool:
    """Report mesh integrity for one binary STL. Return False on any issue."""
    path = Path(path)
    try:
        raw = path.read_bytes()
        if len(raw) < 84:
            raise ValueError("file is too short to be a binary STL")
        n = struct.unpack_from("<I", raw, 80)[0]
        if n == 0:
            raise ValueError("STL contains no triangles")
        expected = 84 + 50 * n
        if len(raw) != expected:
            raise ValueError(f"file length is {len(raw)} bytes; triangle count requires {expected}")
        rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        tri = np.frombuffer(raw, dtype=rec, count=n, offset=84)["v"].astype(np.float64)
        if not np.isfinite(tri).all():
            raise ValueError("triangle vertices contain NaN or infinity")
    except (OSError, ValueError, struct.error) as exc:
        print(f"{path}: invalid STL: {exc}")
        return False

    pts, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2

    directed = Counter()
    undirected = Counter()
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            directed[(int(x), int(y))] += 1
            undirected[tuple(sorted((int(x), int(y))))] += 1
    bad_direction_counts = sum(count != 1 for count in directed.values())
    bad_edges = sum(
        uses != 2 or directed[(a, b)] != 1 or directed[(b, a)] != 1
        for (a, b), uses in undirected.items()
    )

    # Join faces into shells by their shared vertices.
    parent = list(range(len(pts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces:
        root = find(int(a))
        for other in (int(b), int(c)):
            other_root = find(other)
            if other_root != root:
                parent[other_root] = root

    face_shell = [find(int(face[0])) for face in faces]
    shell_faces = Counter(face_shell)
    shell_vertices: dict[int, set[int]] = defaultdict(set)
    shell_edges: dict[int, set[tuple[int, int]]] = defaultdict(set)
    signed_volume = Counter()
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6
    for face, shell, volume in zip(faces, face_shell, signed):
        a, b, c = map(int, face)
        shell_vertices[shell].update((a, b, c))
        shell_edges[shell].update((tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))))
        signed_volume[shell] += float(volume)

    # Closed orientable shells may have handles (Euler characteristic 2-2g),
    # so accept any even value <= 2 instead of assuming every shell is a sphere.
    bad_euler = 0
    for shell, face_count in shell_faces.items():
        chi = len(shell_vertices[shell]) - len(shell_edges[shell]) + face_count
        if chi > 2 or chi % 2 != 0:
            bad_euler += 1
    inward_or_flat = sum(volume <= 0 for volume in signed_volume.values())
    zero_area = int((area < 1e-9).sum())

    print(f"{path}: {n} triangles, {len(shell_faces)} shells")
    print(f"  directed edges used !=1 time:             {bad_direction_counts}")
    print(f"  edges without exactly one opposite pair:  {bad_edges}")
    print(f"  shells with impossible Euler characteristic: {bad_euler}")
    print(f"  shells with non-positive signed volume:   {inward_or_flat}")
    print(f"  zero-area triangles:                      {zero_area}")
    ok = bad_direction_counts == bad_edges == bad_euler == inward_or_flat == zero_area == 0
    print("  WATERTIGHT, outward-facing" if ok else "  PROBLEMS FOUND")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", nargs="+", type=Path, help="binary STL file(s) to inspect")
    args = parser.parse_args()
    results = [check(path) for path in args.stl]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
