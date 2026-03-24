from __future__ import annotations

import itertools
import logging
import math
import re
import struct
import subprocess
import tempfile
from pathlib import Path

from .acceleration import bitmap_rgb_rows, rgb_pixel
from .analyzer import COLOR_SATURATION_THRESHOLD, COLOR_VALUE_THRESHOLD, rgb_distance, rgb_to_hsv
from .imaging import Bitmap
from .models import DetectedColor, ImageCandidate
from .workbook import normalize_text, open_workbook, parse_shared_strings, read_workbook_sheets

LOGGER = logging.getLogger(__name__)

LEGEND_CROP_X_RATIO = 0.22
LEGEND_CROP_Y_RATIO = 0.28
LEGEND_SWATCH_X_RATIO = 0.22
LEGEND_ROW_HALF_HEIGHT = 8
LEGEND_COLOR_DISTANCE_MAX = 120.0
AZIMUTH_CONFIRM_MAX_DEG = 65.0
AZIMUTH_CONFIRM_BONUS = 40.0
OCR_BINARY_PATH = Path(__file__).resolve().parent.parent / ".cache" / "legend_ocr"

OCR_SOURCE = r"""
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: legend_ocr <image-path>\n");
            return 1;
        }
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSURL *url = [NSURL fileURLWithPath:path];
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = NO;
        request.recognitionLanguages = @[@"en-US", @"fr-FR"];
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithURL:url options:@{}];
        NSError *error = nil;
        BOOL ok = [handler performRequests:@[request] error:&error];
        if (!ok) {
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String ?: "perform failed");
            return 4;
        }
        for (VNRecognizedTextObservation *obs in request.results) {
            VNRecognizedText *top = [[obs topCandidates:1] firstObject];
            if (!top) { continue; }
            CGRect box = obs.boundingBox;
            printf("%f\t%f\t%f\t%f\t%s\n",
                box.origin.x,
                box.origin.y,
                box.origin.x + box.size.width,
                box.origin.y + box.size.height,
                top.string.UTF8String ?: "");
        }
    }
    return 0;
}
""".strip()


def enrich_warning_messages(
    file_bytes: bytes,
    candidate: ImageCandidate,
    bitmap: Bitmap,
    detected_colors: list[DetectedColor],
    warning_details: list[dict[str, object]],
) -> list[str]:
    if not warning_details:
        return []

    identifier_lookup = extract_identifier_lookup(file_bytes, candidate)
    if not identifier_lookup:
        return [str(detail["message"]) for detail in warning_details]

    sector_name_map = map_sector_names(bitmap, detected_colors, identifier_lookup)
    warnings: list[str] = []
    for detail in warning_details:
        warnings.append(str(detail["message"]))
        source_index = int(detail.get("source_index", -1))
        target_index = int(detail.get("target_index", -1))
        source_entry = sector_name_map.get(source_index)
        target_entry = sector_name_map.get(target_index)
        if source_entry and target_entry:
            warnings.append(f"Possible late HO from {source_entry['name']} sector to {target_entry['name']} sector")

    return warnings


def extract_identifier_lookup(file_bytes: bytes, candidate: ImageCandidate) -> dict[str, dict[str, object]]:
    identifier_headers = expected_identifier_headers(candidate.target_key)
    with open_workbook(file_bytes) as archive:
        shared_strings = parse_shared_strings(archive)
        sheets = read_workbook_sheets(archive, shared_strings)

    ordered_sheets = sorted(
        sheets,
        key=lambda sheet: (
            sheet.name != candidate.sheet_name,
            0 if "donnees" in normalize_text(sheet.name) else 1,
            sheet.name,
        ),
    )

    lookup: dict[str, dict[str, object]] = {}
    for sheet in ordered_sheets:
        lookup.update(extract_identifier_lookup_from_sheet(sheet.cells, identifier_headers))

    return lookup


def expected_identifier_headers(target_key: str) -> tuple[str, ...]:
    if target_key == "serving_cell_id":
        return ("cell id",)
    if target_key == "best_server":
        return ("psc", "cell id")
    return ("pci", "cell id", "psc")


def extract_identifier_lookup_from_sheet(
    cells: dict[tuple[int, int], str],
    identifier_headers: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    normalized_headers = {normalize_text(header) for header in identifier_headers}

    for (row, col), value in cells.items():
        if normalize_text(value) != "cell name":
            continue

        identifier_col = None
        azimuth_col = None
        for offset in range(1, 4):
            candidate_header = cells.get((row, col + offset), "")
            normalized_header = normalize_text(candidate_header)
            if normalized_header in normalized_headers:
                identifier_col = col + offset
            if normalized_header == "azimuth":
                azimuth_col = col + offset

        if identifier_col is None:
            continue

        blank_rows = 0
        next_row = row + 1
        while blank_rows < 2:
            cell_name = str(cells.get((next_row, col), "")).strip()
            identifier = normalize_identifier(cells.get((next_row, identifier_col), ""))
            azimuth = parse_azimuth(cells.get((next_row, azimuth_col), "")) if azimuth_col is not None else None

            if cell_name and identifier and any(character.isdigit() for character in identifier):
                lookup[identifier] = {
                    "name": cell_name,
                    "azimuth": azimuth,
                }
                blank_rows = 0
            elif cell_name or identifier:
                blank_rows = 0
            else:
                blank_rows += 1

            next_row += 1

    return lookup


def normalize_identifier(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", str(value or "")).upper()


def map_sector_names(
    bitmap: Bitmap,
    detected_colors: list[DetectedColor],
    identifier_lookup: dict[str, dict[str, object]],
) -> dict[int, dict[str, object]]:
    if not identifier_lookup:
        return {}

    rows = read_legend_rows(bitmap, list(identifier_lookup.keys()))
    if not rows:
        return {}

    sector_name_map: dict[int, dict[str, object]] = {}
    for row in rows:
        row_rgb = row.get("rgb")
        identifier = row.get("identifier")
        if not row_rgb or not identifier:
            continue

        best_index = None
        best_distance = None
        for index, detected_color in enumerate(detected_colors):
            distance = rgb_distance(tuple(row_rgb), detected_color.rgb)
            distance -= azimuth_confirmation_bonus(detected_color, identifier_lookup.get(identifier))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index is None or best_distance is None or best_distance > LEGEND_COLOR_DISTANCE_MAX:
            continue

        lookup_entry = identifier_lookup[identifier]
        sector_name_map[best_index] = {
            "identifier": identifier,
            "name": str(lookup_entry["name"]),
            "azimuth": lookup_entry.get("azimuth"),
        }

    return sector_name_map


def read_legend_rows(bitmap: Bitmap, candidate_ids: list[str]) -> list[dict[str, object]]:
    ocr_rows = ocr_legend_rows(bitmap)
    if not ocr_rows:
        return []

    matched_rows = assign_identifiers_to_rows(ocr_rows, candidate_ids)
    for row in matched_rows:
        row["rgb"] = sample_legend_row_color(bitmap, row)

    return matched_rows


def ocr_legend_rows(bitmap: Bitmap) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="ssv-legend-") as temp_dir:
        temp_path = Path(temp_dir)
        crop_bmp = temp_path / "legend.bmp"
        crop_png = temp_path / "legend.png"
        write_legend_crop_bmp(bitmap, crop_bmp)
        subprocess.run(
            ["sips", "-s", "format", "png", str(crop_bmp), "--out", str(crop_png)],
            check=True,
            capture_output=True,
            text=True,
        )
        output = run_ocr_binary(crop_png)

    rows: list[dict[str, object]] = []
    for raw_line in output.splitlines():
        parts = raw_line.strip().split("\t")
        if len(parts) != 5:
            continue

        try:
            x1, y1, x2, y2 = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            continue

        text = parts[4].strip()
        if not text:
            continue

        rows.append(
            {
                "text": text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "y_center": (y1 + y2) / 2.0,
            }
        )

    rows.sort(key=lambda row: row["y_center"], reverse=True)
    return rows


def run_ocr_binary(image_path: Path) -> str:
    binary_path = ensure_ocr_binary()
    result = subprocess.run(
        [str(binary_path), str(image_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def ensure_ocr_binary() -> Path:
    OCR_BINARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_path = OCR_BINARY_PATH.with_suffix(".m")
    if not OCR_BINARY_PATH.exists():
        source_path.write_text(OCR_SOURCE + "\n", encoding="utf-8")
        subprocess.run(
            [
                "clang",
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Vision",
                str(source_path),
                "-o",
                str(OCR_BINARY_PATH),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    return OCR_BINARY_PATH


def write_legend_crop_bmp(bitmap: Bitmap, output_path: Path) -> None:
    crop_width = max(1, int(bitmap.width * LEGEND_CROP_X_RATIO))
    crop_height = max(1, int(bitmap.height * LEGEND_CROP_Y_RATIO))
    rows = bitmap_rgb_rows(bitmap)
    pixels = [row[:crop_width] for row in rows[:crop_height]]

    row_stride = ((crop_width * 3 + 3) // 4) * 4
    image_size = row_stride * crop_height
    header_size = 14 + 40
    file_size = header_size + image_size

    with output_path.open("wb") as file_handle:
        file_handle.write(b"BM")
        file_handle.write(struct.pack("<IHHI", file_size, 0, 0, header_size))
        file_handle.write(
            struct.pack(
                "<IIIHHIIIIII",
                40,
                crop_width,
                crop_height,
                1,
                24,
                0,
                image_size,
                2835,
                2835,
                0,
                0,
            )
        )
        padding = b"\x00" * (row_stride - (crop_width * 3))
        for row in reversed(pixels):
            for red, green, blue in row:
                file_handle.write(bytes((blue, green, red)))
            file_handle.write(padding)


def assign_identifiers_to_rows(rows: list[dict[str, object]], candidate_ids: list[str]) -> list[dict[str, object]]:
    if not rows or not candidate_ids:
        return []

    candidate_ids = [normalize_identifier(identifier) for identifier in candidate_ids]
    usable_rows = rows[: min(len(rows), len(candidate_ids))]
    best_assignment = None
    best_cost = None

    for ids_subset in itertools.permutations(candidate_ids, len(usable_rows)):
        total_cost = 0
        for row, identifier in zip(usable_rows, ids_subset):
            total_cost += row_identifier_cost(str(row["text"]), identifier)
        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_assignment = ids_subset

    if best_assignment is None:
        return []

    matched_rows = []
    for row, identifier in zip(usable_rows, best_assignment):
        matched_row = dict(row)
        matched_row["identifier"] = identifier
        matched_rows.append(matched_row)
    return matched_rows


def row_identifier_cost(text: str, identifier: str) -> int:
    normalized_identifier = normalize_identifier(identifier)
    digit_groups = re.findall(r"\d+", text)
    windows = list(digit_groups)
    digit_stream = "".join(digit_groups)
    if len(digit_stream) >= len(normalized_identifier):
        for start in range(0, len(digit_stream) - len(normalized_identifier) + 1):
            windows.append(digit_stream[start : start + len(normalized_identifier)])

    if normalized_identifier in windows or normalized_identifier in digit_stream:
        return 0
    if not windows:
        return len(normalized_identifier) + 10

    return min(edit_distance(window, normalized_identifier) for window in windows)


def edit_distance(left: str, right: str) -> int:
    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if left[row - 1] == right[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + cost,
            )

    return dp[-1][-1]


def sample_legend_row_color(bitmap: Bitmap, row: dict[str, object]) -> tuple[int, int, int] | None:
    crop_width = max(1, int(bitmap.width * LEGEND_CROP_X_RATIO))
    crop_height = max(1, int(bitmap.height * LEGEND_CROP_Y_RATIO))
    center_y = int((1.0 - float(row["y_center"])) * crop_height)
    x_end = max(1, int(crop_width * LEGEND_SWATCH_X_RATIO))
    y0 = max(0, center_y - LEGEND_ROW_HALF_HEIGHT)
    y1 = min(crop_height, center_y + LEGEND_ROW_HALF_HEIGHT + 1)

    samples: list[tuple[int, int, int]] = []
    for y in range(y0, y1):
        for x in range(x_end):
            rgb = rgb_pixel(bitmap, x, y)
            hue, saturation, value = rgb_to_hsv(*rgb)
            if saturation >= COLOR_SATURATION_THRESHOLD and value >= COLOR_VALUE_THRESHOLD:
                samples.append(rgb)

    if not samples:
        return None

    total = len(samples)
    return (
        sum(sample[0] for sample in samples) // total,
        sum(sample[1] for sample in samples) // total,
        sum(sample[2] for sample in samples) // total,
    )


def parse_azimuth(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def azimuth_confirmation_bonus(detected_color: DetectedColor, lookup_entry: dict[str, object] | None) -> float:
    if not lookup_entry:
        return 0.0

    azimuth = lookup_entry.get("azimuth")
    if azimuth is None or detected_color.site_angle is None:
        return 0.0

    detected_compass = image_angle_to_compass(float(detected_color.site_angle))
    difference = circular_degree_distance(detected_compass, float(azimuth))
    if difference > AZIMUTH_CONFIRM_MAX_DEG:
        return 0.0

    return ((AZIMUTH_CONFIRM_MAX_DEG - difference) / AZIMUTH_CONFIRM_MAX_DEG) * AZIMUTH_CONFIRM_BONUS


def image_angle_to_compass(angle_degrees: float) -> float:
    return (90.0 - angle_degrees) % 360.0


def circular_degree_distance(left: float, right: float) -> float:
    distance = abs(left - right) % 360.0
    return min(distance, 360.0 - distance)
