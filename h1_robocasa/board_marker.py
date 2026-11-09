"""Opt-in visual fiducial for the static manipulation-net board.

The board frame is the asset's authored frame (metres, top at z=0.025).
The marker frame is centred on its black square, x right, y up, z out of
the printed face, matching OpenCV's centred ArUco object-point convention.
No ROS publisher or scene registration occurs on import.
"""

import math
import json
import struct
import tempfile
import xml.etree.ElementTree as ET
import zlib
from functools import lru_cache
from pathlib import Path


DICTIONARY = "DICT_4X4_50"
MARKER_ID = 7
# Canonical row-major payloads, checked against OpenCV's predefined dictionary.
_PAYLOADS = {
    7: ((1, 1, 0, 0), (0, 1, 0, 0), (1, 1, 1, 1), (0, 0, 1, 0)),
    8: ((1, 1, 1, 1), (1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 0)),
}


def board_marker_layout():
    """Read the same rigid calibration installed with the perception node."""
    path = (Path(__file__).resolve().parent.parent / "core_ws/src/model_server/config/pegboard_layout.json")
    return json.loads(path.read_text())


def marker_pixels(cell_pixels=32, quiet_cells=1, marker_id=MARKER_ID):
    """Return square grayscale rows; six cells span the measured black edge.

    One black border cell surrounds the 4x4 code. The white quiet zone is
    separate from the marker size used for pose estimation.
    """
    if type(cell_pixels) is not int or cell_pixels < 1:
        raise ValueError("cell_pixels must be a positive integer")
    if type(quiet_cells) is not int or quiet_cells < 1:
        raise ValueError("quiet_cells must be a positive integer")
    if type(marker_id) is not int or marker_id not in _PAYLOADS:
        raise ValueError("supported marker IDs are 7 and 8")
    payload = _PAYLOADS[marker_id]
    cells = 6 + 2 * quiet_cells
    rows = []
    for y in range(cells):
        row = bytearray()
        for x in range(cells):
            mx, my = x - quiet_cells, y - quiet_cells
            value = 255
            if 0 <= mx < 6 and 0 <= my < 6:
                value = (255 * payload[my - 1][mx - 1]
                         if 1 <= mx <= 4 and 1 <= my <= 4 else 0)
            row.extend(bytes([value]) * cell_pixels)
        rows.extend([bytes(row)] * cell_pixels)
    return tuple(rows)


def marker_png(cell_pixels=32, quiet_cells=1, marker_id=MARKER_ID):
    """Encode the deterministic texture as an 8-bit grayscale PNG, without cv2."""
    rows = marker_pixels(cell_pixels, quiet_cells, marker_id)

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", len(rows), len(rows), 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row for row in rows), 9))
            + chunk(b"IEND", b""))


@lru_cache(maxsize=2)
def _texture_file(marker_id):
    # Keep the directory alive while MuJoCo/robosuite may recompile the XML.
    directory = tempfile.TemporaryDirectory(prefix="golem_board_marker_")
    path = Path(directory.name) / f"aruco_4x4_50_id{marker_id}.png"
    path.write_bytes(marker_png(marker_id=marker_id))
    return directory, str(path)


def mount_board_marker(board_body, asset, *, name="pegboard_aruco",
                       black_size_m=None, mount_pos=None, marker_id=MARKER_ID):
    """Append a massless, non-contact print on the board and return calibration data.

    Pass the static board body's Element and the scene's asset Element. The
    default 4 cm paper sits flush with the top and outer side board edges
    (x right, y up). The entire quiet zone must fit on the board and clear
    every catalog hole's bounding box. Geometry axes match the board. The top face is
    at mount_pos; the 0.1 mm visual thickness extends below it. Nothing is
    added to the board's collision geometry or inertial specification.
    """
    if type(marker_id) is not int or marker_id not in _PAYLOADS:
        raise ValueError("supported marker IDs are 7 and 8")
    spec = next(m for m in board_marker_layout()["markers"] if m["id"] == marker_id)
    black_size_m = spec["black_size_m"] if black_size_m is None else black_size_m
    if not math.isfinite(black_size_m) or black_size_m <= 0:
        raise ValueError("black_size_m must be finite and positive")
    if not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for ch in name):
        raise ValueError("name must contain only letters, digits and underscores")
    outer = black_size_m * 8 / 6
    half = outer / 2
    pos = tuple(mount_pos) if mount_pos is not None else tuple(spec["position_board_m"])
    if len(pos) != 3 or not all(math.isfinite(v) for v in pos):
        raise ValueError("mount_pos must contain three finite board-local metres")
    catalog = json.loads((Path(__file__).resolve().parent.parent / "CL_Assets" /
                          "mujoco_assets/manipulation_net/manipulation_catalog.json").read_text())
    if any(abs(pos[i]) + half > catalog["board"]["footprint"][i] / 2 + 1e-12
           for i in (0, 1)):
        raise ValueError("marker quiet zone must fit entirely inside the board footprint")
    for hole, spec in catalog["holes"].items():
        profile = spec["profile_xy"]
        if all(pos[i] - half < max(p[i] for p in profile) and
               pos[i] + half > min(p[i] for p in profile) for i in (0, 1)):
            raise ValueError(f"marker quiet zone overlaps hole bounds: {hole}")
    names = {node.get("name") for section in (board_body, asset) for node in section.iter()}
    if any(name + suffix in names for suffix in ("", "_mesh", "_texture", "_material")):
        raise ValueError(f"marker name already present: {name}")

    def fmt(values):
        return " ".join(f"{v:.12g}" for v in values)

    ET.SubElement(asset, "texture", name=name + "_texture", type="2d", file=_texture_file(marker_id)[1])
    ET.SubElement(asset, "material", name=name + "_material", texture=name + "_texture",
                  texrepeat="1 1", texuniform="false", rgba="1 1 1 1",
                  specular="0", shininess="0", reflectance="0")
    # MuJoCo's PNG row zero is v=0; put that row at the face's +Y edge.
    vertices = [(x * half, y * half, z) for z in (-0.0001, 0.0)
                for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ET.SubElement(asset, "mesh", name=name + "_mesh",
                  vertex=fmt(v for vertex in vertices for v in vertex),
                  texcoord="0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0",
                  face="4 5 6 4 6 7 0 2 1 0 3 2 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7")
    ET.SubElement(board_body, "geom", name=name, type="mesh", mesh=name + "_mesh",
                  material=name + "_material", pos=fmt(pos), group="1", mass="0",
                  contype="0", conaffinity="0", rgba="1 1 1 1")
    return {"dictionary": DICTIONARY, "marker_id": marker_id,
            "black_size_m": black_size_m, "quiet_zone_m": black_size_m / 6,
             "outer_size_m": outer, "geom_name": name,
            "position_board_m": list(pos),
            "T_board_marker": [[1., 0., 0., pos[0]], [0., 1., 0., pos[1]],
                               [0., 0., 1., pos[2]], [0., 0., 0., 1.]]}
