#!/usr/bin/env python3
"""
Patched e57_to_nav2_costmap.py

Additions vs original:
- Free/Occupied slope thresholds with non-linear ramp (GAMMA_RAMP)
- Dual YAML outputs: raw (direct costs) + trinary (free_thresh/occupied_thresh)
- "Sure overlays": force regions to FREE or OBSTACLE using world XY points and a radius
- Pillow deprecation fix (no `mode` arg)

Edit the CONFIG CONSTANTS below to customize paths/params.
"""

import os
from pathlib import Path
import math
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# --------------------------
# CONFIG CONSTANTS (EDIT ME)
# --------------------------
PROJECT_ROOT = Path("~/Desktop/projectred/map_converter_pj/").expanduser()

INPUT_E57   = PROJECT_ROOT / "data" / "MarsYard_local.e57"   # put your .e57 here
OUTPUT_DIR  = PROJECT_ROOT / "output"                         # results (created if missing)

OUTPUT_STEM = "map"                    # produces map.png + map.yaml (+ map_trinary.yaml)

RESOLUTION_M      = 0.10               # meters per pixel
MIN_SAMPLES_CELL  = 3                  # min points per cell; else unknown
GAUSS_SIGMA_CELLS = 1.0                # masked smoothing (0 disables)
CELL_REDUCER      = "median"           # "median" | "min" | "mean"
FLIP_Y            = False              # mirror across XZ plane (fix inverted Y)

# Optional crop after flip (set to None to disable). Example: (xmin, xmax, ymin, ymax).
CROP_BOUNDS: Optional[Tuple[float, float, float, float]] = None
# CROP_BOUNDS = (0.0, 50.0, -20.0, 30.0)

# ---- Slope -> Cost thresholds (new) ----
FREE_SLOPE_DEG = 20.0     # <= this is fully free (cost 0)
OCC_SLOPE_DEG  = 50.0    # >= this is lethal obstacle (cost 254)
GAMMA_RAMP     = 1.0     # >1 steeper ramp, <1 softer; 1.0 = linear

# ---- Trinary YAML thresholds (new) ----
TRINARY_FREE_THRESH     = 0.196  # fraction: <= free
TRINARY_OCCUPIED_THRESH = 0.65   # fraction: >= occupied

# ---- Sure overlays (new) ----
USE_SURE_OVERLAYS   = True          # master switch
SURE_OBS_RADIUS_M   = 0.10          # obstacle radius (m)
SURE_FREE_RADIUS_M  = 0.10          # free radius (m)

# Provide world XY points (meters) for hard constraints
# NOTE: Fill these with your table values. Defaults left empty; set to the values from your sheet.
# Landmarks (L1..): forced OBSTACLE within SURE_OBS_RADIUS_M
SURE_OBS_POINTS: List[Tuple[float,float]] = [
    # (x, y),  # e.g., (3.1374, 4.3246)
    (3.1374, +4.3246),   # L1, H=204.9754
    (9.0888, -4.5555),   # L2, H=204.9893
    (8.2731, +2.2478),   # L3, H=205.0730
    (13.5552,+3.3260),   # L4, H=204.7818
    (17.6623,-2.7646),   # L5, H=204.8333
    (23.8746,-2.3014),   # L6, H=204.4606
    (27.7097,+2.7192),   # L7, H=204.8636
    (28.3320,+8.6813),   # L8, H=204.9092
    (25.8693,+7.3461),   # L9, H=204.4087
    (18.6570,+4.5163),   # L10, H=204.9571
    (14.9031,+6.1368),   # L11, H=204.9864
    (13.2623,+11.3769),  # L12, H=204.9394
    (10.0015,+5.6827),   # L13, H=205.1797
    (8.0354, +12.9120),  # L14, H=204.9494
    (2.7876, +13.5601),  # L15, H=205.0208
]
# Starting points (S1..), Waypoints (W1..): forced FREE within SURE_FREE_RADIUS_M
SURE_FREE_POINTS: Dict[str, List[Tuple[float,float]]] = {
    "starts": [
        # (x, y),
        (0.0000, +0.0000),  # S1, H=204.6042
        (0.0393, +7.1697),  # S2, H=204.5313
        (3.8015, -9.1614),  # S3, H=204.4902
        (18.6625,+10.8159), # S4, H=204.8022
        (37.3445,+10.4773), # S5, H=204.6558
        (23.7990,-8.1839),  # S6, H=204.5222
        (13.0403,-3.3190),  # S7, H=204.8725
        (21.8293,+2.3056),  # S8, H=204.5733
    ],
    "waypoints": [
        # (x, y),
        (15.1159,-3.0854),  # W1, H=204.7734
        (6.8073, +10.3746), # W2, H=204.6294
        (12.3282,+6.7797),  # W3, H=204.6860
        (19.7272,+5.2386),  # W4, H=204.6205
        (25.2349,+2.0235),  # W5, H=204.2558
        (10.0126,+0.0000),  # W6, H=204.6875
        (20.1270,+0.0000),  # W7, H=204.8407
        (24.4818,+7.9578),  # W8, H=204.6665
        (17.9612,+2.8924),  # W9, H=205.1249
    ],
}

# --------------------------
# END CONFIG
# --------------------------

try:
    import pdal
except ImportError as e:
    raise SystemExit(
        "PDAL Python bindings not found.\n"
        "Did you activate the env? -> micromamba activate ~/Desktop/projectred/map_converter_pj/e57nav2"
    ) from e


@dataclass
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def read_e57_xyz(e57_path: Path, flip_y: bool) -> np.ndarray:
    """Read XYZ from E57 with PDAL. Optionally flip Y axis."""
    pipe = [{"type": "readers.e57", "filename": str(e57_path)}]
    p = pdal.Pipeline(json.dumps(pipe))
    p.execute()
    arr = p.arrays[0]
    pts = np.vstack([arr["X"], arr["Y"], arr["Z"]]).T.astype(np.float32)
    if flip_y:
        pts[:, 1] *= -1.0
    return pts


def compute_bounds(points: np.ndarray, pad: float = 0.0) -> Bounds:
    xmin, ymin, _ = points.min(axis=0)
    xmax, ymax, _ = points.max(axis=0)

    return Bounds(xmin - pad, xmax + pad, ymin - pad, ymax + pad)


def bin_to_dem(points: np.ndarray, res: float, b: Bounds,min_samples: int, reducer: str) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize to DEM; returns (dem, mask_valid). NaN for unknown."""
    assert reducer in ("median", "min", "mean")
    W = int(math.ceil((b.xmax - b.xmin) / res))
    H = int(math.ceil((b.ymax - b.ymin) / res))

    ix = np.floor((points[:, 0] - b.xmin) / res).astype(np.int32)
    iy = np.floor((points[:, 1] - b.ymin) / res).astype(np.int32)

    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy, z = ix[inside], iy[inside], points[inside, 2]

    buckets = {}
    for i, j, zz in zip(ix, iy, z):
        buckets.setdefault((i, j), []).append(float(zz))

    dem = np.full((H, W), np.nan, dtype=np.float32)
    for (i, j), vals in tqdm(buckets.items(), desc="Reducing cells", leave=False):
        if len(vals) >= min_samples:
            if reducer == "median":
                dem[j, i] = float(np.median(vals))
            elif reducer == "min":
                dem[j, i] = float(np.min(vals))
            else:
                dem[j, i] = float(np.mean(vals))
        # else: NaN stays (unknown)

    mask = np.isfinite(dem)
    return dem, mask


def masked_gaussian(dem: np.ndarray, mask: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian blur without bleeding through unknowns."""
    if sigma_cells <= 0:
        return dem
    valid = mask.astype(np.float32)
    dem_filled = dem.copy()
    dem_filled[~mask] = 0.0
    blur_dem = gaussian_filter(dem_filled, sigma=sigma_cells, mode="nearest")
    blur_m  = gaussian_filter(valid,      sigma=sigma_cells, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        dem_blur = blur_dem / np.maximum(blur_m, 1e-6)
    dem_blur[~mask] = np.nan
    return dem_blur


def slope_degrees(dem: np.ndarray, mask: np.ndarray, res: float) -> np.ndarray:
    """Central differences (with fwd/bwd fallback) → slope in degrees."""
    H, W = dem.shape
    dzdx = np.full_like(dem, np.nan, dtype=np.float32)
    dzdy = np.full_like(dem, np.nan, dtype=np.float32)

    def cdiff_x():
        left  = np.roll(dem, 1, axis=1); mL = np.roll(mask, 1, axis=1)
        right = np.roll(dem,-1, axis=1); mR = np.roll(mask,-1, axis=1)
        out = np.full_like(dem, np.nan, dtype=np.float32)
        central = mask & mL & mR
        out[central] = (right[central] - left[central]) / (2*res)
        fwd = mask & mR & ~mL
        out[fwd] = (right[fwd] - dem[fwd]) / res
        bwd = mask & mL & ~mR
        out[bwd] = (dem[bwd] - left[bwd]) / res
        return out

    def cdiff_y():
        up    = np.roll(dem, 1, axis=0); mU = np.roll(mask, 1, axis=0)
        down  = np.roll(dem,-1, axis=0); mD = np.roll(mask,-1, axis=0)
        out = np.full_like(dem, np.nan, dtype=np.float32)
        central = mask & mU & mD
        out[central] = (down[central] - up[central]) / (2*res)
        fwd = mask & mD & ~mU
        out[fwd] = (down[fwd] - dem[fwd]) / res
        bwd = mask & mU & ~mD
        out[bwd] = (dem[bwd] - up[bwd]) / res
        return out

    dzdx = cdiff_x()
    dzdy = cdiff_y()
    grad = np.sqrt(dzdx**2 + dzdy**2)
    slope = np.degrees(np.arctan(grad)).astype(np.float32)
    slope[~mask] = np.nan
    return slope


def slope_to_cost(slope_deg: np.ndarray,
                  free_deg: float = FREE_SLOPE_DEG,
                  occ_deg: float  = OCC_SLOPE_DEG,
                  gamma: float    = GAMMA_RAMP) -> np.ndarray:
    """
    Map slope(deg) -> Nav2 cost:
      - slope <= free_deg          -> 0
      - free_deg < slope < occ_deg -> 1..253 (ramped)
      - slope >= occ_deg           -> 254
      - NaN                        -> 255 (unknown)
    """
    if not (0.0 <= free_deg < occ_deg <= 90.0):
        raise ValueError(f"Bad thresholds: FREE={free_deg}, OCC={occ_deg} (require 0 <= free < occ <= 90)")

    cost = np.full(slope_deg.shape, 255, dtype=np.uint8)
    v = np.isfinite(slope_deg)
    if not np.any(v):
        return cost

    s = np.clip(slope_deg[v], 0.0, 90.0)

    below = s <= free_deg
    above = s >= occ_deg
    mid = ~(below | above)

    t = (s[mid] - free_deg) / max(occ_deg - free_deg, 1e-6)  # 0..1
    if gamma != 1.0:
        t = np.power(t, gamma)
    ramp = np.rint(1 + t * 252.0).astype(np.uint8)  # 1..253 inclusive

    out = np.zeros_like(s, dtype=np.uint8)
    out[below] = 0
    out[mid] = ramp
    out[above] = 254

    cost[v] = out
    return cost


def save_png(cost: np.ndarray, path: Path) -> None:
    img = Image.fromarray(cost)  # let Pillow infer mode; avoids deprecation
    img.save(path, optimize=True)


def save_yaml(image_path: Path, yaml_path: Path, res: float, b: Bounds,
              mode: str = "raw",
              free_thresh: Optional[float] = None,
              occ_thresh: Optional[float] = None) -> None:
    """
    Save a map_server YAML file.
    mode: "raw" or "trinary"
    free_thresh / occ_thresh: only used if mode="trinary"
    """
    meta = {
        "image": image_path.name,
        "resolution": float(res),
        "origin": [float(b.xmin), float(b.ymin), 0.0],  # bottom-left corner in world coords
        "mode": mode,
        "negate": 0,
    }
    if mode == "trinary":
        meta["free_thresh"] = float(free_thresh if free_thresh is not None else 0.196)
        meta["occupied_thresh"] = float(occ_thresh if occ_thresh is not None else 0.65)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


# ---------- Overlays helpers ----------

def _xy_to_ij(x: float, y: float, b: Bounds, res: float) -> Tuple[int, int]:
    i = int(math.floor((x - b.xmin) / res))
    j = int(math.floor((y - b.ymin) / res))
    return i, j


def _draw_disks(points: List[Tuple[float, float]], radius_m: float,
                shape: Tuple[int, int], b: Bounds, res: float) -> np.ndarray:
    """Return a boolean mask with filled disks at given world XY points."""
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if not points:
        return mask
    r_px = max(1, int(math.ceil(radius_m / res)))
    rr2 = r_px * r_px
    for (x, y) in points:
        i, j = _xy_to_ij(x, y, b, res)
        if i < -r_px or j < -r_px or i >= W + r_px or j >= H + r_px:
            continue  # way outside
        i0, i1 = max(0, i - r_px), min(W - 1, i + r_px)
        j0, j1 = max(0, j - r_px), min(H - 1, j + r_px)
        ys = np.arange(j0, j1 + 1)
        xs = np.arange(i0, i1 + 1)
        XX, YY = np.meshgrid(xs, ys)
        d2 = (XX - i) ** 2 + (YY - j) ** 2
        mask[j0:j1+1, i0:i1+1] |= (d2 <= rr2)
    return mask


def apply_overlays(cost: np.ndarray, b: Bounds, res: float) -> np.ndarray:
    """Apply sure FREE then sure OBSTACLE overlays to costmap."""
    if not USE_SURE_OVERLAYS:
        return cost

    H, W = cost.shape

    # Free regions: starts + waypoints
    free_points: List[Tuple[float,float]] = []
    for cat in ("starts", "waypoints"):
        free_points.extend(SURE_FREE_POINTS.get(cat, []))
    free_mask = _draw_disks(free_points, SURE_FREE_RADIUS_M, (H, W), b, res)

    # Obstacle regions: landmarks
    obs_mask = _draw_disks(SURE_OBS_POINTS, SURE_OBS_RADIUS_M, (H, W), b, res)

    out = cost.copy()
    # Free first, then obstacle (so landmarks win if overlapping)
    out[free_mask] = 0
    out[obs_mask]  = 254
    return out


# ---------- Main ----------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[+] Reading: {INPUT_E57}")
    pts = read_e57_xyz(INPUT_E57, flip_y=FLIP_Y)

    # Optional crop (after flip)
    if CROP_BOUNDS is not None:
        xmin, xmax, ymin, ymax = CROP_BOUNDS
        m = (pts[:,0] >= xmin) & (pts[:,0] <= xmax) & (pts[:,1] >= ymin) & (pts[:,1] <= ymax)
        pts = pts[m]
        print(f"[i] Applied crop {CROP_BOUNDS}, kept {pts.shape[0]} points")

    b = compute_bounds(pts)
    print(f"[i] Bounds: xmin={b.xmin:.3f} xmax={b.xmax:.3f}  ymin={b.ymin:.3f} ymax={b.ymax:.3f}")

    dem, mask = bin_to_dem(
        pts, res=RESOLUTION_M, b=b,
        min_samples=MIN_SAMPLES_CELL, reducer=CELL_REDUCER
    )

    if GAUSS_SIGMA_CELLS > 0:
        dem_s = masked_gaussian(dem, mask, GAUSS_SIGMA_CELLS)
    else:
        dem_s = dem

    slope = slope_degrees(dem_s, np.isfinite(dem_s), res=RESOLUTION_M)
    cost  = slope_to_cost(slope)

    # Apply overlays (forced free/obstacle regions)
    cost = apply_overlays(cost, b=b, res=RESOLUTION_M)

    out_png  = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    out_yaml_raw = OUTPUT_DIR / f"{OUTPUT_STEM}.yaml"
    out_yaml_tri = OUTPUT_DIR / f"{OUTPUT_STEM}_trinary.yaml"

    save_png(cost, out_png)

    # Raw mode (direct costs)
    save_yaml(out_png, out_yaml_raw, res=RESOLUTION_M, b=b, mode="raw")

    # Trinary mode (Nav2 thresholds)
    save_yaml(out_png, out_yaml_tri, res=RESOLUTION_M, b=b,
              mode="trinary",
              free_thresh=TRINARY_FREE_THRESH,
              occ_thresh=TRINARY_OCCUPIED_THRESH)

    total = cost.size
    unknown = int((cost == 255).sum())
    print(f"[+] Wrote {out_png} and {out_yaml_raw} (+ {out_yaml_tri})")
    print(f"[i] Resolution: {RESOLUTION_M} m/px  |  Size: {cost.shape[1]} x {cost.shape[0]} px")
    print(f"[i] Unknown cells: {unknown}/{total} ({unknown*100.0/total:.2f}%)")
    print(f"[i] Slope thresholds: free<= {FREE_SLOPE_DEG}°, lethal>= {OCC_SLOPE_DEG}°, gamma={GAMMA_RAMP}")
    print("[i] Nav2 costmap semantics: 0=free, 254=lethal, 255=unknown (mode: raw).")


if __name__ == "__main__":
    main()
