#!/usr/bin/env python3
"""
ept_to_nav2_costmap.py

Convert an EPT or E57 point cloud into Nav2-compatible costmap images/YAML files.

Main outputs:
  - map.png: raw Nav2 cost image, where 0=free, 254=lethal, 255=unknown
  - map.yaml: Nav2 map_server YAML in raw mode
  - map_trinary.yaml: Nav2 map_server YAML in trinary mode
  - diagnostics/*.png: optional debug images for DEM, slope, density, roughness, etc.

Notes:
  - EPT input should point to the ept.json file.
  - E57 input should point to the .e57 file.
  - For large EPT datasets, use READ_BOUNDS or CROP_BOUNDS to avoid loading too much data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation
from tqdm import tqdm

try:
    import pdal
except ImportError as e:
    raise SystemExit(
        "PDAL Python bindings not found.\n"
        "Activate your environment first, for example:\n"
        "  micromamba activate ~/Desktop/projectred/map_converter_pj/e57nav2\n"
        "Then install/check PDAL Python bindings."
    ) from e


# =============================================================================
# CONFIGURATION - EDIT HERE
# =============================================================================

# --------------------------
# Paths
# --------------------------
PROJECT_ROOT = Path("~/Desktop/projectred/sirio_ws/src/map_converter_pj/").expanduser()

# Choose one input type: "ept" or "e57".
INPUT_TYPE: Literal["ept", "e57"] = "ept"

# For EPT, this must point to the ept.json file, not only to the folder.
INPUT_EPT_JSON = PROJECT_ROOT / "data" / "entwine_pointcloud" / "ept.json"

# Optional fallback/reuse for old workflow.
INPUT_E57 = PROJECT_ROOT / "data" / "MarsYard_local.e57"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_STEM = "map"

# --------------------------
# Reading / filtering
# --------------------------
# Coordinate transform applied after reading.
FLIP_Y = False
FLIP_X = False
SWAP_XY = False
Z_OFFSET_M = 0.0

# EPT read bounds applied directly inside PDAL, before loading all points.
# Format: (xmin, xmax, ymin, ymax). Set None to disable.
# Strongly recommended for large EPT datasets.
READ_BOUNDS: Optional[Tuple[float, float, float, float]] = None
# READ_BOUNDS = (0.0, 40.0, -12.0, 16.0)

# Optional crop after reading and after coordinate flips/swaps.
# Format: (xmin, xmax, ymin, ymax). Set None to disable.
CROP_BOUNDS: Optional[Tuple[float, float, float, float]] = None
# CROP_BOUNDS = (0.0, 40.0, -12.0, 16.0)

# Optional Z crop after reading. Useful to remove ceiling/noise/outliers.
# Format: (zmin, zmax). Set None to disable.
Z_CROP_BOUNDS: Optional[Tuple[float, float]] = None
# Z_CROP_BOUNDS = (203.5, 206.5)

# If True, prints available point dimensions from PDAL.
PRINT_POINT_DIMENSIONS = True

# --------------------------
# Raster/map parameters
# --------------------------
RESOLUTION_M = 0.40
MAP_PADDING_M = 0.0
MIN_SAMPLES_CELL = 1

# Reducer for DEM terrain elevation.
# "median" is robust, "min" can better approximate ground, "mean" can blur obstacles.
DEM_REDUCER: Literal["median", "min", "mean"] = "median"

# Additional per-cell height stats used for obstacle detection.
COMPUTE_HEIGHT_STATS = True

# Unknown handling.
UNKNOWN_COST = 255
FREE_COST = 0
LETHAL_COST = 254

# --------------------------
# DEM smoothing and gap policy
# --------------------------
GAUSS_SIGMA_CELLS = 3.0

# If True, slope is computed only where original/smoothed DEM is valid.
# If False, very small gaps can optionally be filled before slope computation.
STRICT_VALID_SLOPE = True

# Optional unknown inflation: expands unknown regions by N cells.
UNKNOWN_INFLATION_CELLS = 0

# --------------------------
# Traversability from slope
# --------------------------
USE_SLOPE_COST = True
FREE_SLOPE_DEG = 20.0
OCC_SLOPE_DEG = 50.0
GAMMA_RAMP = 1.0

# --------------------------
# Obstacle detection from local height range
# --------------------------
# This catches vertical objects/rocks that slope-only DEM can miss.
USE_HEIGHT_OBSTACLES = True

# If z_max - z_min in a cell exceeds this value, mark as obstacle.
HEIGHT_OBS_THRESHOLD_M = 0.25

# Also mark obstacles if local height range after neighborhood max/min exceeds this.
# This is useful when an obstacle occupies neighboring cells rather than the same cell.
USE_NEIGHBOR_HEIGHT_OBSTACLES = True
NEIGHBOR_RADIUS_CELLS = 1
NEIGHBOR_HEIGHT_OBS_THRESHOLD_M = 0.30

# If True, obstacle height constraints override slope costs.
HEIGHT_OBSTACLES_OVERRIDE_COST = True

# --------------------------
# Cost post-processing
# --------------------------
# Inflate lethal obstacles by N cells. This is image-level inflation, not Nav2 inflation layer.
LETHAL_INFLATION_CELLS = 0

# Optional smoothing of non-lethal cost field. Disabled by default because it may blur obstacles.
SMOOTH_COST_FIELD = False
COST_GAUSS_SIGMA_CELLS = 1.0

# --------------------------
# Image orientation and YAML
# --------------------------
# PNG row 0 is top in image viewers, while raster j=0 corresponds to ymin.
# For many ROS map workflows, flipping vertically before saving is easier to interpret.
# The YAML origin is adjusted accordingly only when needed.
SAVE_IMAGE_FLIPPED_Y = False
ROTATE_DEG = -90.0

# For map_server raw mode.
RAW_MODE = "raw"

# For trinary mode.
TRINARY_FREE_THRESH = 0.196
TRINARY_OCCUPIED_THRESH = 0.65

# --------------------------
# Diagnostics
# --------------------------
SAVE_DIAGNOSTICS = True
DIAGNOSTICS_DIRNAME = "diagnostics"
SAVE_NUMPY_ARRAYS = True

# --------------------------
# Sure overlays: hard constraints
# --------------------------
USE_SURE_OVERLAYS = True
SURE_OBS_RADIUS_M = 0.10
SURE_FREE_RADIUS_M = 0.10

# Landmarks: forced OBSTACLE within SURE_OBS_RADIUS_M.
SURE_OBS_POINTS: List[Tuple[float, float]] = [
    # (3.1374, +4.3246),
    # (9.0888, -4.5555),
    # (8.2731, +2.2478),
    # (13.5552, +3.3260),
    # (17.6623, -2.7646),
    # (23.8746, -2.3014),
    # (27.7097, +2.7192),
    # (28.3320, +8.6813),
    # (25.8693, +7.3461),
    # (18.6570, +4.5163),
    # (14.9031, +6.1368),
    # (13.2623, +11.3769),
    # (10.0015, +5.6827),
    # (8.0354, +12.9120),
    # (2.7876, +13.5601),
    (0,0),
    (10.24,1.22),
    (2.04,7.47),
    (-3.35,15.07),
]

# Starting points and waypoints: forced FREE within SURE_FREE_RADIUS_M.
SURE_FREE_POINTS: Dict[str, List[Tuple[float, float]]] = {
    "starts": [
        (0.0000, +0.0000),
        # (0.0393, +7.1697),
        # (3.8015, -9.1614),
        # (18.6625, +10.8159),
        # (37.3445, +10.4773),
        # (23.7990, -8.1839),
        # (13.0403, -3.3190),
        # (21.8293, +2.3056),
    ],
    "waypoints": [
        # (15.1159, -3.0854),
        # (6.8073, +10.3746),
        # (12.3282, +6.7797),
        # (19.7272, +5.2386),
        # (25.2349, +2.0235),
        # (10.0126, +0.0000),
        # (20.1270, +0.0000),
        # (24.4818, +7.9578),
        # (17.9612, +2.8924),
        (10.83, 3.9),
        (0.99,1.53),
        (1.01,10.11),
        (-5.84,17.30),
    ],
}

# =============================================================================
# END CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class Bounds:
    """2D map bounds in world coordinates."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class CellStats:
    """Rasterized point-cloud statistics per grid cell."""

    dem: np.ndarray
    mask_valid: np.ndarray
    count: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    height_range: np.ndarray


def _pdal_bounds_2d(bounds: Tuple[float, float, float, float]) -> str:
    """Return PDAL bounds string from (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = bounds
    return f"([{xmin},{xmax}],[{ymin},{ymax}])"


def read_point_cloud(input_type: str, flip_x: bool, flip_y: bool, swap_xy: bool) -> np.ndarray:
    """Read XYZ points from EPT or E57 using PDAL and apply simple coordinate transforms."""
    if input_type == "ept":
        if not INPUT_EPT_JSON.exists():
            raise FileNotFoundError(f"EPT ept.json not found: {INPUT_EPT_JSON}")
        reader = {"type": "readers.ept", "filename": str(INPUT_EPT_JSON)}
        if READ_BOUNDS is not None:
            reader["bounds"] = _pdal_bounds_2d(READ_BOUNDS)
        pipe = [reader]
    elif input_type == "e57":
        if not INPUT_E57.exists():
            raise FileNotFoundError(f"E57 file not found: {INPUT_E57}")
        pipe = [{"type": "readers.e57", "filename": str(INPUT_E57)}]
    else:
        raise ValueError(f"Unsupported INPUT_TYPE={input_type!r}. Use 'ept' or 'e57'.")

    pipeline = pdal.Pipeline(json.dumps(pipe))
    pipeline.execute()
    arr = pipeline.arrays[0]

    if PRINT_POINT_DIMENSIONS:
        print(f"[i] Point dimensions: {arr.dtype.names}")

    pts = np.vstack([arr["X"], arr["Y"], arr["Z"]]).T.astype(np.float32)

    if swap_xy:
        pts[:, [0, 1]] = pts[:, [1, 0]]
    if flip_x:
        pts[:, 0] *= -1.0
    if flip_y:
        pts[:, 1] *= -1.0
    if Z_OFFSET_M != 0.0:
        pts[:, 2] += float(Z_OFFSET_M)

    return pts


def crop_points(points: np.ndarray) -> np.ndarray:
    """Apply optional XY and Z crops after coordinate transforms."""
    keep = np.ones(points.shape[0], dtype=bool)

    if CROP_BOUNDS is not None:
        xmin, xmax, ymin, ymax = CROP_BOUNDS
        keep &= (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
        keep &= (points[:, 1] >= ymin) & (points[:, 1] <= ymax)

    if Z_CROP_BOUNDS is not None:
        zmin, zmax = Z_CROP_BOUNDS
        keep &= (points[:, 2] >= zmin) & (points[:, 2] <= zmax)

    cropped = points[keep]
    if cropped.size == 0:
        raise ValueError("No points left after crop. Check READ_BOUNDS, CROP_BOUNDS, and Z_CROP_BOUNDS.")
    return cropped


def compute_bounds(points: np.ndarray, pad: float = 0.0) -> Bounds:
    """Compute XY bounds from points with optional padding."""
    xmin = float(np.min(points[:, 0])) - pad
    xmax = float(np.max(points[:, 0])) + pad
    ymin = float(np.min(points[:, 1])) - pad
    ymax = float(np.max(points[:, 1])) + pad
    return Bounds(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def _xy_to_ij_array(points: np.ndarray, b: Bounds, res: float) -> Tuple[np.ndarray, np.ndarray]:
    """Map point XY coordinates to integer grid indices."""
    ix = np.floor((points[:, 0] - b.xmin) / res).astype(np.int32)
    iy = np.floor((points[:, 1] - b.ymin) / res).astype(np.int32)
    return ix, iy


def rasterize_cell_stats(points: np.ndarray, res: float, b: Bounds, min_samples: int, reducer: str) -> CellStats:
    """Rasterize XYZ points into DEM and auxiliary per-cell height statistics."""
    if reducer not in {"median", "min", "mean"}:
        raise ValueError(f"Bad DEM_REDUCER={reducer!r}. Use 'median', 'min', or 'mean'.")

    width = int(math.ceil((b.xmax - b.xmin) / res))
    height = int(math.ceil((b.ymax - b.ymin) / res))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster size: width={width}, height={height}")

    ix, iy = _xy_to_ij_array(points, b, res)
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    ix = ix[inside]
    iy = iy[inside]
    z = points[inside, 2].astype(np.float32)

    linear = iy.astype(np.int64) * width + ix.astype(np.int64)
    order = np.argsort(linear)
    linear = linear[order]
    z = z[order]

    unique, start, counts = np.unique(linear, return_index=True, return_counts=True)

    dem = np.full((height, width), np.nan, dtype=np.float32)
    count_grid = np.zeros((height, width), dtype=np.uint32)
    z_min = np.full((height, width), np.nan, dtype=np.float32)
    z_max = np.full((height, width), np.nan, dtype=np.float32)
    z_mean = np.full((height, width), np.nan, dtype=np.float32)

    for lin, st, cnt in tqdm(zip(unique, start, counts), total=len(unique), desc="Reducing cells", leave=False):
        j = int(lin // width)
        i = int(lin % width)
        vals = z[st: st + cnt]
        count_grid[j, i] = int(cnt)
        z_min[j, i] = float(np.min(vals))
        z_max[j, i] = float(np.max(vals))
        z_mean[j, i] = float(np.mean(vals))

        if cnt >= min_samples:
            if reducer == "median":
                dem[j, i] = float(np.median(vals))
            elif reducer == "min":
                dem[j, i] = float(np.min(vals))
            else:
                dem[j, i] = float(np.mean(vals))

    mask_valid = np.isfinite(dem)
    height_range = z_max - z_min
    height_range[~np.isfinite(height_range)] = np.nan

    return CellStats(
        dem=dem,
        mask_valid=mask_valid,
        count=count_grid,
        z_min=z_min,
        z_max=z_max,
        z_mean=z_mean,
        height_range=height_range.astype(np.float32),
    )


def masked_gaussian(dem: np.ndarray, mask: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Apply Gaussian blur to a DEM without bleeding values through unknown cells."""
    if sigma_cells <= 0.0:
        return dem.copy()

    valid = mask.astype(np.float32)
    dem_filled = dem.copy()
    dem_filled[~mask] = 0.0

    blur_dem = gaussian_filter(dem_filled, sigma=sigma_cells, mode="nearest")
    blur_mask = gaussian_filter(valid, sigma=sigma_cells, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        out = blur_dem / np.maximum(blur_mask, 1e-6)
    out[~mask] = np.nan
    return out.astype(np.float32)


def slope_degrees(dem: np.ndarray, mask: np.ndarray, res: float) -> np.ndarray:
    """Compute DEM slope in degrees using central differences with forward/backward fallback."""
    left = np.roll(dem, 1, axis=1)
    right = np.roll(dem, -1, axis=1)
    up = np.roll(dem, 1, axis=0)
    down = np.roll(dem, -1, axis=0)

    m_left = np.roll(mask, 1, axis=1)
    m_right = np.roll(mask, -1, axis=1)
    m_up = np.roll(mask, 1, axis=0)
    m_down = np.roll(mask, -1, axis=0)

    # Prevent wraparound across image borders.
    m_left[:, 0] = False
    m_right[:, -1] = False
    m_up[0, :] = False
    m_down[-1, :] = False

    dzdx = np.full_like(dem, np.nan, dtype=np.float32)
    dzdy = np.full_like(dem, np.nan, dtype=np.float32)

    central_x = mask & m_left & m_right
    dzdx[central_x] = (right[central_x] - left[central_x]) / (2.0 * res)
    fwd_x = mask & m_right & ~m_left
    dzdx[fwd_x] = (right[fwd_x] - dem[fwd_x]) / res
    bwd_x = mask & m_left & ~m_right
    dzdx[bwd_x] = (dem[bwd_x] - left[bwd_x]) / res

    central_y = mask & m_up & m_down
    dzdy[central_y] = (down[central_y] - up[central_y]) / (2.0 * res)
    fwd_y = mask & m_down & ~m_up
    dzdy[fwd_y] = (down[fwd_y] - dem[fwd_y]) / res
    bwd_y = mask & m_up & ~m_down
    dzdy[bwd_y] = (dem[bwd_y] - up[bwd_y]) / res

    grad = np.sqrt(dzdx ** 2 + dzdy ** 2)
    slope = np.degrees(np.arctan(grad)).astype(np.float32)
    slope[~mask] = np.nan
    return slope


def slope_to_cost(slope_deg: np.ndarray, free_deg: float, occ_deg: float, gamma: float) -> np.ndarray:
    """Map slope values in degrees to Nav2 raw cost values."""
    if not (0.0 <= free_deg < occ_deg <= 90.0):
        raise ValueError(f"Bad slope thresholds: free={free_deg}, occupied={occ_deg}. Require 0 <= free < occupied <= 90.")

    cost = np.full(slope_deg.shape, UNKNOWN_COST, dtype=np.uint8)
    valid = np.isfinite(slope_deg)
    if not np.any(valid):
        return cost

    s = np.clip(slope_deg[valid], 0.0, 90.0)
    below = s <= free_deg
    above = s >= occ_deg
    mid = ~(below | above)

    out = np.zeros_like(s, dtype=np.uint8)
    out[below] = FREE_COST
    out[above] = LETHAL_COST

    t = (s[mid] - free_deg) / max(occ_deg - free_deg, 1e-6)
    if gamma != 1.0:
        t = np.power(t, gamma)
    out[mid] = np.rint(1.0 + t * 252.0).astype(np.uint8)

    cost[valid] = out
    return cost


def height_obstacle_mask(stats: CellStats) -> np.ndarray:
    """Build obstacle mask from per-cell and neighborhood height variation."""
    mask = np.zeros(stats.dem.shape, dtype=bool)

    if USE_HEIGHT_OBSTACLES:
        mask |= np.isfinite(stats.height_range) & (stats.height_range >= HEIGHT_OBS_THRESHOLD_M)

    if USE_NEIGHBOR_HEIGHT_OBSTACLES:
        radius = max(1, int(NEIGHBOR_RADIUS_CELLS))
        size = 2 * radius + 1

        # Approximate neighborhood max/min using dilation-like operations.
        finite_min = np.where(np.isfinite(stats.z_min), stats.z_min, np.inf)
        finite_max = np.where(np.isfinite(stats.z_max), stats.z_max, -np.inf)

        from scipy.ndimage import minimum_filter, maximum_filter

        local_min = minimum_filter(finite_min, size=size, mode="nearest")
        local_max = maximum_filter(finite_max, size=size, mode="nearest")
        local_range = local_max - local_min
        local_valid = np.isfinite(local_range) & (local_min < np.inf) & (local_max > -np.inf)
        mask |= local_valid & (local_range >= NEIGHBOR_HEIGHT_OBS_THRESHOLD_M)

    return mask


def smooth_cost(cost: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Smooth valid non-lethal cost values while preserving unknown and lethal cells."""
    if sigma_cells <= 0.0:
        return cost

    unknown = cost == UNKNOWN_COST
    lethal = cost == LETHAL_COST
    valid = ~(unknown | lethal)

    values = cost.astype(np.float32)
    values[~valid] = 0.0
    weights = valid.astype(np.float32)

    blur_values = gaussian_filter(values, sigma=sigma_cells, mode="nearest")
    blur_weights = gaussian_filter(weights, sigma=sigma_cells, mode="nearest")

    out = cost.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = blur_values / np.maximum(blur_weights, 1e-6)
    out[valid] = np.clip(np.rint(smoothed[valid]), 0, 253).astype(np.uint8)
    out[unknown] = UNKNOWN_COST
    out[lethal] = LETHAL_COST
    return out


def inflate_masks(cost: np.ndarray) -> np.ndarray:
    """Apply optional image-level inflation to unknown and lethal cells."""
    out = cost.copy()

    if UNKNOWN_INFLATION_CELLS > 0:
        structure = np.ones((2 * UNKNOWN_INFLATION_CELLS + 1, 2 * UNKNOWN_INFLATION_CELLS + 1), dtype=bool)
        unknown = binary_dilation(cost == UNKNOWN_COST, structure=structure)
        out[unknown] = UNKNOWN_COST

    if LETHAL_INFLATION_CELLS > 0:
        structure = np.ones((2 * LETHAL_INFLATION_CELLS + 1, 2 * LETHAL_INFLATION_CELLS + 1), dtype=bool)
        lethal = binary_dilation(cost == LETHAL_COST, structure=structure)
        out[lethal] = LETHAL_COST

    return out


def _xy_to_ij(x: float, y: float, b: Bounds, res: float) -> Tuple[int, int]:
    """Map a world XY point to raster cell indices."""
    i = int(math.floor((x - b.xmin) / res))
    j = int(math.floor((y - b.ymin) / res))
    return i, j


def _draw_disks(points: List[Tuple[float, float]], radius_m: float, shape: Tuple[int, int], b: Bounds, res: float) -> np.ndarray:
    """Return a boolean raster mask with filled disks at given world XY points."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if not points:
        return mask

    r_px = max(1, int(math.ceil(radius_m / res)))
    rr2 = r_px * r_px

    for x, y in points:
        i, j = _xy_to_ij(x, y, b, res)
        if i < -r_px or j < -r_px or i >= width + r_px or j >= height + r_px:
            continue

        i0 = max(0, i - r_px)
        i1 = min(width - 1, i + r_px)
        j0 = max(0, j - r_px)
        j1 = min(height - 1, j + r_px)

        xs = np.arange(i0, i1 + 1)
        ys = np.arange(j0, j1 + 1)
        xx, yy = np.meshgrid(xs, ys)
        mask[j0: j1 + 1, i0: i1 + 1] |= ((xx - i) ** 2 + (yy - j) ** 2) <= rr2

    return mask


def apply_overlays(cost: np.ndarray, b: Bounds, res: float) -> np.ndarray:
    """Apply hard free and obstacle overlays. Obstacles win over free when overlapping."""
    if not USE_SURE_OVERLAYS:
        return cost

    height, width = cost.shape

    free_points: List[Tuple[float, float]] = []
    for category in ("starts", "waypoints"):
        free_points.extend(SURE_FREE_POINTS.get(category, []))

    free_mask = _draw_disks(free_points, SURE_FREE_RADIUS_M, (height, width), b, res)
    obs_mask = _draw_disks(SURE_OBS_POINTS, SURE_OBS_RADIUS_M, (height, width), b, res)

    out = cost.copy()
    out[free_mask] = FREE_COST
    out[obs_mask] = LETHAL_COST
    return out


def maybe_flip_for_image(arr: np.ndarray) -> np.ndarray:
    """Optionally flip raster vertically before saving to image."""
    if SAVE_IMAGE_FLIPPED_Y:
        return np.flipud(arr)
    return arr


def save_png_uint8(arr: np.ndarray, path: Path) -> None:
    """Save a uint8 raster as PNG."""
    Image.fromarray(arr.astype(np.uint8)).save(path, optimize=True)


def save_float_debug(arr: np.ndarray, path: Path, invalid_value: int = 255) -> None:
    """Save a floating-point array as contrast-stretched uint8 PNG for diagnostics."""
    out = np.full(arr.shape, invalid_value, dtype=np.uint8)
    finite = np.isfinite(arr)
    if np.any(finite):
        vals = arr[finite].astype(np.float32)
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
        out[finite] = np.rint(norm * 254.0).astype(np.uint8)
    save_png_uint8(maybe_flip_for_image(out), path)


def save_count_debug(count: np.ndarray, path: Path) -> None:
    """Save a count/density image as logarithmic uint8 PNG."""
    out = np.zeros(count.shape, dtype=np.uint8)
    if np.any(count > 0):
        vals = np.log1p(count.astype(np.float32))
        vmax = float(np.percentile(vals[count > 0], 98.0))
        vmax = max(vmax, 1e-6)
        out = np.clip(np.rint(vals / vmax * 254.0), 0, 254).astype(np.uint8)
        out[count == 0] = 255
    save_png_uint8(maybe_flip_for_image(out), path)


def save_yaml(image_path: Path, yaml_path: Path, res: float, b: Bounds, mode: str, free_thresh: Optional[float] = None, occ_thresh: Optional[float] = None) -> None:
    """Save Nav2/map_server YAML metadata for the generated image."""
    # If the image is flipped vertically, the lower-left world origin is still xmin/ymin
    # for map_server semantics. The flip affects only how pixels are arranged in the PNG.
    meta = {
        "image": image_path.name,
        "resolution": float(res),
        "origin": [float(b.xmin), float(b.ymin), 0.0],
        "mode": mode,
        "negate": 0,
    }
    if mode == "trinary":
        meta["free_thresh"] = float(TRINARY_FREE_THRESH if free_thresh is None else free_thresh)
        meta["occupied_thresh"] = float(TRINARY_OCCUPIED_THRESH if occ_thresh is None else occ_thresh)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def save_diagnostics(output_dir: Path, stats: CellStats, dem_smooth: np.ndarray, slope: np.ndarray, cost: np.ndarray, obstacle_mask: np.ndarray) -> None:
    """Save optional diagnostic images and arrays."""
    diag_dir = output_dir / DIAGNOSTICS_DIRNAME
    diag_dir.mkdir(parents=True, exist_ok=True)

    save_float_debug(stats.dem, diag_dir / "dem_raw.png")
    save_float_debug(dem_smooth, diag_dir / "dem_smooth.png")
    save_float_debug(slope, diag_dir / "slope_deg.png")
    save_count_debug(stats.count, diag_dir / "density_count.png")
    save_float_debug(stats.height_range, diag_dir / "height_range.png")
    save_png_uint8(maybe_flip_for_image(cost), diag_dir / "final_cost_debug.png")
    save_png_uint8(maybe_flip_for_image(obstacle_mask.astype(np.uint8) * 254), diag_dir / "height_obstacle_mask.png")

    if SAVE_NUMPY_ARRAYS:
        np.save(diag_dir / "dem_raw.npy", stats.dem)
        np.save(diag_dir / "dem_smooth.npy", dem_smooth)
        np.save(diag_dir / "slope_deg.npy", slope)
        np.save(diag_dir / "density_count.npy", stats.count)
        np.save(diag_dir / "height_range.npy", stats.height_range)
        np.save(diag_dir / "final_cost.npy", cost)
        np.save(diag_dir / "height_obstacle_mask.npy", obstacle_mask)


def save_limited_diagnostics(output_dir: Path, cost: np.ndarray) -> None:
    """Save only the limited-map diagnostic image."""
    diag_dir = output_dir / DIAGNOSTICS_DIRNAME
    diag_dir.mkdir(parents=True, exist_ok=True)
    save_png_uint8(maybe_flip_for_image(cost), diag_dir / "limited_map.png")


def summarize_cost(cost: np.ndarray) -> None:
    """Print a compact summary of final costmap cell classes."""
    total = int(cost.size)
    unknown = int(np.sum(cost == UNKNOWN_COST))
    lethal = int(np.sum(cost == LETHAL_COST))
    free = int(np.sum(cost == FREE_COST))
    intermediate = total - unknown - lethal - free

    print(f"[i] Cells total:        {total}")
    print(f"[i] Free cells:         {free} ({100.0 * free / total:.2f}%)")
    print(f"[i] Intermediate cells: {intermediate} ({100.0 * intermediate / total:.2f}%)")
    print(f"[i] Lethal cells:       {lethal} ({100.0 * lethal / total:.2f}%)")
    print(f"[i] Unknown cells:      {unknown} ({100.0 * unknown / total:.2f}%)")


def rotate_cost_and_bounds(cost: np.ndarray, b: Bounds, res: float, angle_deg: float) -> Tuple[np.ndarray, Bounds]:
    """Rotate the cost image (degrees, CCW) and expand bounds to keep map center fixed."""
    if abs(float(angle_deg)) < 1e-9:
        return cost, b

    # Keep class values exact (nearest-neighbor) and mark new outside area as unknown.
    img = Image.fromarray(cost.astype(np.uint8))
    rot = img.rotate(float(angle_deg), resample=Image.Resampling.NEAREST, expand=True, fillcolor=UNKNOWN_COST)
    out = np.array(rot, dtype=np.uint8)

    cx = 0.5 * (b.xmin + b.xmax)
    cy = 0.5 * (b.ymin + b.ymax)
    new_h, new_w = out.shape
    half_w_m = 0.5 * new_w * res
    half_h_m = 0.5 * new_h * res
    b_out = Bounds(
        xmin=cx - half_w_m,
        xmax=cx + half_w_m,
        ymin=cy - half_h_m,
        ymax=cy + half_h_m,
    )
    return out, b_out


def square_bounds_from_max_axis(b: Bounds) -> Bounds:
    """Return square bounds by expanding the shorter axis around center."""
    cx = 0.5 * (b.xmin + b.xmax)
    cy = 0.5 * (b.ymin + b.ymax)
    side = max(b.xmax - b.xmin, b.ymax - b.ymin)
    half = 0.5 * side
    return Bounds(xmin=cx - half, xmax=cx + half, ymin=cy - half, ymax=cy + half)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert EPT/E57 point clouds to Nav2 costmaps.")
    parser.add_argument(
        "--limit",
        action="store_true",
        help="If set, skip slope/height processing and build a binary map: free where points exist, lethal elsewhere.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the point-cloud to Nav2 costmap conversion."""
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[+] Reading input type: {INPUT_TYPE}")
    if INPUT_TYPE == "ept":
        print(f"[+] EPT: {INPUT_EPT_JSON}")
        if READ_BOUNDS is None:
            print("[!] READ_BOUNDS is None. Large EPT datasets may consume a lot of memory.")
    else:
        print(f"[+] E57: {INPUT_E57}")

    points = read_point_cloud(INPUT_TYPE, flip_x=FLIP_X, flip_y=FLIP_Y, swap_xy=SWAP_XY)
    print(f"[i] Loaded points: {points.shape[0]}")

    points = crop_points(points)
    print(f"[i] Points after crop: {points.shape[0]}")

    b = compute_bounds(points, pad=MAP_PADDING_M)
    if args.limit:
        b = square_bounds_from_max_axis(b)
        print("[i] LIMIT mode enabled: using square bounds based on max axis.")
    print(f"[i] Bounds: xmin={b.xmin:.3f}, xmax={b.xmax:.3f}, ymin={b.ymin:.3f}, ymax={b.ymax:.3f}")

    stats = rasterize_cell_stats(points, RESOLUTION_M, b, MIN_SAMPLES_CELL, DEM_REDUCER)
    print(f"[i] Raster size: {stats.dem.shape[1]} x {stats.dem.shape[0]} px at {RESOLUTION_M:.3f} m/px")
    if args.limit:
        # Binary occupancy from point support only:
        # free where at least one point exists, lethal elsewhere.
        cost = np.full(stats.dem.shape, LETHAL_COST, dtype=np.uint8)
        cost[stats.count > 0] = FREE_COST
        dem_smooth = np.full(stats.dem.shape, np.nan, dtype=np.float32)
        slope = np.full(stats.dem.shape, np.nan, dtype=np.float32)
        obstacle_mask = np.zeros(stats.dem.shape, dtype=bool)
    else:
        dem_smooth = masked_gaussian(stats.dem, stats.mask_valid, GAUSS_SIGMA_CELLS)
        slope_mask = np.isfinite(dem_smooth) if STRICT_VALID_SLOPE else stats.mask_valid
        slope = slope_degrees(dem_smooth, slope_mask, RESOLUTION_M)

        if USE_SLOPE_COST:
            cost = slope_to_cost(slope, FREE_SLOPE_DEG, OCC_SLOPE_DEG, GAMMA_RAMP)
        else:
            cost = np.full(stats.dem.shape, UNKNOWN_COST, dtype=np.uint8)
            cost[stats.mask_valid] = FREE_COST

        obstacle_mask = height_obstacle_mask(stats)
        if HEIGHT_OBSTACLES_OVERRIDE_COST:
            cost[obstacle_mask] = LETHAL_COST

        cost = inflate_masks(cost)

        if SMOOTH_COST_FIELD:
            cost = smooth_cost(cost, COST_GAUSS_SIGMA_CELLS)

        cost = apply_overlays(cost, b, RESOLUTION_M)

    out_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    out_yaml_raw = OUTPUT_DIR / f"{OUTPUT_STEM}.yaml"
    out_yaml_tri = OUTPUT_DIR / f"{OUTPUT_STEM}_trinary.yaml"
    cost_out, b_out = rotate_cost_and_bounds(cost, b, RESOLUTION_M, ROTATE_DEG)

    save_png_uint8(maybe_flip_for_image(cost_out), out_png)
    save_yaml(out_png, out_yaml_raw, RESOLUTION_M, b_out, mode=RAW_MODE)
    save_yaml(out_png, out_yaml_tri, RESOLUTION_M, b_out, mode="trinary", free_thresh=TRINARY_FREE_THRESH, occ_thresh=TRINARY_OCCUPIED_THRESH)

    if SAVE_DIAGNOSTICS:
        if args.limit:
            save_limited_diagnostics(OUTPUT_DIR, cost_out)
        else:
            save_diagnostics(OUTPUT_DIR, stats, dem_smooth, slope, cost_out, obstacle_mask)

    print(f"[+] Wrote: {out_png}")
    print(f"[+] Wrote: {out_yaml_raw}")
    print(f"[+] Wrote: {out_yaml_tri}")
    print(f"[i] Image rotation: {ROTATE_DEG:.3f} deg")
    if SAVE_DIAGNOSTICS:
        print(f"[+] Diagnostics: {OUTPUT_DIR / DIAGNOSTICS_DIRNAME}")

    summarize_cost(cost)
    if args.limit:
        print("[i] LIMIT mode: free where points exist, lethal elsewhere.")
    else:
        print(f"[i] Slope thresholds: free <= {FREE_SLOPE_DEG:.1f} deg, lethal >= {OCC_SLOPE_DEG:.1f} deg, gamma={GAMMA_RAMP}")
        print(f"[i] Height obstacle threshold: {HEIGHT_OBS_THRESHOLD_M:.3f} m")
    print("[i] Nav2 raw semantics: 0=free, 254=lethal, 255=unknown.")


if __name__ == "__main__":
    main()
