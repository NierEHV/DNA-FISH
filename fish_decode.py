"""DNA-FISH multi-round decoding pipeline.

This script reads multi-round TIFF images, segments nuclei from DAPI,
detects fluorescent spots, decodes base calls across rounds, and maps
decoded spots back onto the DAPI image.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import exposure, filters, measure, morphology, segmentation
from skimage.feature import blob_log
import matplotlib.pyplot as plt


CHANNEL_ORDER = ["DAPI", "FAM", "CY3", "TXR", "Y5"]
CHANNEL_TO_BASE = {
    "FAM": "T",
    "CY3": "G",
    "TXR": "C",
    "Y5": "A",
}
CHANNEL_SUFFIXES = ["_ch00.tif", "_ch01.tif", "_ch02.tif", "_ch03.tif", "_ch04.tif"]


@dataclass
class RoundImages:
    round_id: str
    images: Dict[str, np.ndarray]


def parse_round_id(path: Path) -> str:
    match = re.search(r"(.+)_ch\d{2}\.tif$", path.name)
    if not match:
        raise ValueError(f"Unexpected file name: {path.name}")
    return match.group(1)


def load_round_images(input_dir: Path) -> List[RoundImages]:
    files = []
    for suffix in CHANNEL_SUFFIXES:
        files.extend(sorted(input_dir.glob(f"*{suffix}")))

    if not files:
        raise FileNotFoundError("No TIFF files found with expected suffixes.")

    grouped: Dict[str, Dict[str, Path]] = {}
    for file_path in files:
        round_id = parse_round_id(file_path)
        grouped.setdefault(round_id, {})
        channel_index = int(re.search(r"_ch(\d{2})\.tif$", file_path.name).group(1))
        channel_name = CHANNEL_ORDER[channel_index]
        grouped[round_id][channel_name] = file_path

    rounds: List[RoundImages] = []
    for round_id in sorted(grouped.keys()):
        channel_paths = grouped[round_id]
        images = {name: tifffile.imread(path) for name, path in channel_paths.items()}
        rounds.append(RoundImages(round_id=round_id, images=images))
    return rounds


def segment_nuclei(dapi_image: np.ndarray) -> np.ndarray:
    smooth = filters.gaussian(dapi_image, sigma=1.0)
    thresh = filters.threshold_otsu(smooth)
    binary = smooth > thresh
    binary = morphology.remove_small_objects(binary, 64)
    distance = ndi.distance_transform_edt(binary)
    local_maxi = morphology.h_maxima(distance, 0.1)
    markers = measure.label(local_maxi)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return labels


def detect_spots(channel_stack: np.ndarray) -> np.ndarray:
    max_proj = np.max(channel_stack, axis=0)
    norm = exposure.rescale_intensity(max_proj, out_range=(0, 1))
    blobs = blob_log(norm, min_sigma=1, max_sigma=3, num_sigma=3, threshold=0.05)
    if blobs.size == 0:
        return np.empty((0, 2), dtype=float)
    return blobs[:, :2]


def assign_base_calls(
    spot_coords: np.ndarray,
    channel_images: Dict[str, np.ndarray],
    window: int = 1,
) -> List[str]:
    bases = []
    for y, x in spot_coords:
        y = int(round(y))
        x = int(round(x))
        y0, y1 = max(0, y - window), min(channel_images["FAM"].shape[0], y + window + 1)
        x0, x1 = max(0, x - window), min(channel_images["FAM"].shape[1], x + window + 1)
        intensities = {}
        for channel, base in CHANNEL_TO_BASE.items():
            patch = channel_images[channel][y0:y1, x0:x1]
            intensities[channel] = np.mean(patch)
        best_channel = max(intensities, key=intensities.get)
        bases.append(CHANNEL_TO_BASE[best_channel])
    return bases


def match_spots(
    reference: np.ndarray,
    target: np.ndarray,
    max_distance: float = 3.0,
) -> List[Optional[int]]:
    if reference.size == 0:
        return []
    if target.size == 0:
        return [None] * len(reference)
    tree = cKDTree(target)
    distances, indices = tree.query(reference, distance_upper_bound=max_distance)
    matched = []
    for dist, idx in zip(distances, indices):
        if np.isinf(dist):
            matched.append(None)
        else:
            matched.append(int(idx))
    return matched


def load_codebook(codebook_path: Optional[Path]) -> Dict[str, str]:
    if codebook_path is None:
        return {}
    mapping = {}
    with codebook_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            barcode = row.get("barcode")
            gene = row.get("gene")
            if barcode and gene:
                mapping[barcode] = gene
    return mapping


def decode_rounds(
    rounds: List[RoundImages],
    max_distance: float = 3.0,
    codebook_path: Optional[Path] = None,
) -> pd.DataFrame:
    if not rounds:
        raise ValueError("No rounds provided for decoding.")

    base_round = rounds[0]
    channel_stack = np.stack([base_round.images[ch] for ch in CHANNEL_TO_BASE.keys()], axis=0)
    reference_spots = detect_spots(channel_stack)
    base_calls = assign_base_calls(reference_spots, base_round.images)

    barcodes = [base for base in base_calls]
    for round_images in rounds[1:]:
        channel_stack = np.stack([round_images.images[ch] for ch in CHANNEL_TO_BASE.keys()], axis=0)
        spots = detect_spots(channel_stack)
        base_calls = assign_base_calls(spots, round_images.images)
        matches = match_spots(reference_spots, spots, max_distance=max_distance)
        for idx, match in enumerate(matches):
            if match is None:
                barcodes[idx] += "N"
            else:
                barcodes[idx] += base_calls[match]

    codebook = load_codebook(codebook_path)
    genes = [codebook.get(barcode, "Unknown") for barcode in barcodes]
    return pd.DataFrame(
        {
            "y": reference_spots[:, 0],
            "x": reference_spots[:, 1],
            "barcode": barcodes,
            "gene": genes,
        }
    )


def plot_overlay(dapi: np.ndarray, decoded: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(dapi, cmap="gray")
    plt.scatter(decoded["x"], decoded["y"], s=12, c="lime", alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    codebook_path: Optional[Path],
    max_distance: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = load_round_images(input_dir)
    dapi = rounds[0].images["DAPI"]
    nuclei_labels = segment_nuclei(dapi)

    decoded = decode_rounds(rounds, max_distance=max_distance, codebook_path=codebook_path)
    decoded["nucleus_id"] = ndi.map_coordinates(
        nuclei_labels,
        [decoded["y"].to_numpy(), decoded["x"].to_numpy()],
        order=0,
        mode="nearest",
    )

    decoded_csv = output_dir / "decoded_spots.csv"
    decoded.to_csv(decoded_csv, index=False)

    overlay_path = output_dir / "decoded_overlay.png"
    plot_overlay(dapi, decoded, overlay_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode multi-round DNA-FISH images.")
    parser.add_argument("input_dir", type=Path, help="Folder containing TIFF images.")
    parser.add_argument("output_dir", type=Path, help="Folder to write outputs.")
    parser.add_argument(
        "--codebook",
        type=Path,
        default=None,
        help="Optional CSV with columns: gene, barcode.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=3.0,
        help="Maximum distance for matching spots across rounds.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        codebook_path=args.codebook,
        max_distance=args.max_distance,
    )


if __name__ == "__main__":
    main()
