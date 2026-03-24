from __future__ import annotations

import base64
import math
import time
import statistics
from collections.abc import Sequence
from typing import Any

from .acceleration import bitmap_hsv_arrays, bitmap_rgb_array, cv2, extract_binary_components, np, rgb_pixel
from .analyzer import (
    COLOR_SATURATION_THRESHOLD,
    COLOR_VALUE_THRESHOLD,
    LEGEND_X_RATIO,
    LEGEND_Y_RATIO,
    rgb_to_hex,
    rgb_to_hsv,
)
from .models import AnalysisOutcome, Bitmap, DotComponent, LegendSwatch

RED_HUE_WINDOW_DEG = 35.0
LEGEND_HUE_WINDOW_DEG = 10.0
LEGEND_DISTANCE_MARGIN_RATIO = 0.82
# Orange-label guard: reject components where avg green/red > this threshold.
# True red route dots have very little green (green/red ≈ 0.10); orange map
# labels have substantial green (green/red ≈ 0.45–0.60).
ORANGE_REJECT_GREEN_RATIO = 0.45
# Degraded-route clusters must span at least this fraction of the image diagonal.
# Compact clusters (map text / legend artefacts) are rejected by this check.
MIN_ROUTE_CLUSTER_DIAGONAL_RATIO = 0.08
MIN_CONTINUOUS_RED_POINTS = 6
MIN_SPARSE_KPI_POINTS = 6
MIN_TOTAL_KPI_POINTS = 20
# Pixel-ratio fallback thresholds used when route dots form blobs too large for
# component-based detection (e.g. very dense routes where dots overlap and merge
# into a single connected region spanning hundreds of pixels).
DEGRADED_PIXEL_RATIO_THRESHOLD = 0.15
DEGRADED_PIXEL_MIN_DIAGONAL_RATIO = 0.12
DEGRADED_PIXEL_HUE_TOLERANCE = 15.0
CLUSTER_LINK_DISTANCE = 18.0
RED_RUN_LINK_DISTANCE = 84.0
RED_POINT_GAP_DISTANCE = 115.0
ADAPTIVE_LINK_DISTANCE_MULTIPLIER = 1.6
ADAPTIVE_LINK_DISTANCE_MIN = 18.0
ADAPTIVE_LINK_DISTANCE_MAX = 96.0
DOT_CHAIN_LINK_DISTANCE_MULTIPLIER = 1.85
DOT_CHAIN_LINK_DISTANCE_MIN = 18.0
DOT_CHAIN_LINK_DISTANCE_MAX = 120.0
DOT_CHAIN_NEIGHBOR_RANK_LIMIT = 3
DOT_CHAIN_MIN_PAIR_ANGLE_DEG = 140.0
DOT_CHAIN_MERGE_DISTANCE_MULTIPLIER = 1.75
DOT_CHAIN_MERGE_DISTANCE_MAX = 210.0
DOT_CHAIN_MERGE_ALIGNMENT_MIN = 0.3
DOT_CHAIN_MERGE_ALIGNMENT_GOOD = 0.6
HOTSPOT_PADDING = 18.0
RED_RUN_COMPONENT_MIN_PIXELS = 20
RED_RUN_COMPONENT_MAX_PIXELS = 2200
RED_RUN_COMPONENT_MIN_SPAN = 5
RED_RUN_COMPONENT_MAX_SPAN = 48
RED_RUN_COMPONENT_MIN_FILL = 0.35
RED_RUN_COMPONENT_ASPECT_MIN = 0.55
RED_RUN_COMPONENT_ASPECT_MAX = 1.8
RED_RUN_DOMINANT_AREA_RATIO_MIN = 0.45
RED_RUN_DOMINANT_AREA_RATIO_MAX = 1.85
KPI_COMPONENT_MIN_PIXELS = 20
KPI_COMPONENT_MAX_PIXELS = 2200
KPI_COMPONENT_MIN_SPAN = 5
KPI_COMPONENT_MAX_SPAN = 48
KPI_COMPONENT_MIN_FILL = 0.28
KPI_COMPONENT_ASPECT_MIN = 0.45
KPI_COMPONENT_ASPECT_MAX = 2.2
KPI_COMPONENT_DOMINANT_AREA_RATIO_MIN = 0.35
KPI_COMPONENT_DOMINANT_AREA_RATIO_MAX = 2.1


class SsvKpiError(ValueError):
    """Raised when a KPI map cannot be analyzed reliably."""


def analyze_kpi_bitmap(
    bitmap: Bitmap,
    preview_image_uri: str,
    metric_name: str | None,
    metric_group: str | None,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> AnalysisOutcome:
    stage_started = time.perf_counter()
    point_components = extract_kpi_point_components(bitmap)
    point_extraction_s = time.perf_counter() - stage_started
    total_points = len(point_components)
    if total_points < MIN_SPARSE_KPI_POINTS:
        raise SsvKpiError("The extracted KPI image does not contain enough colored measurement points for degradation analysis.")

    warnings: list[str] = []
    stage_started = time.perf_counter()
    red_component_indexes = extract_visual_red_dot_indexes(
        bitmap,
        point_components,
        legend_swatches_override=legend_swatches_override,
        degraded_swatch_override=degraded_swatch_override,
    )
    degraded_classification_s = time.perf_counter() - stage_started
    red_components = [point_components[index] for index in red_component_indexes]
    stage_started = time.perf_counter()
    chain_link_distance = estimate_dot_chain_link_distance(point_components)
    dot_chain_indexes = build_ordered_dot_chain_indexes(point_components, chain_link_distance)
    chain_build_s = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    qualifying_run_indexes = extract_qualifying_degraded_run_indexes(
        dot_chain_indexes,
        point_components,
        red_component_indexes,
        chain_link_distance,
    )
    run_extraction_s = time.perf_counter() - stage_started
    continuity_strategy = "ordered_chain"
    red_point_count = len(red_components)
    red_point_ratio = (red_point_count / total_points) if total_points else 0.0
    highlighted_clusters: list[list[dict[str, object] | DotComponent]] = []
    hotspot_circles: list[tuple[float, float, float]] = []
    continuous_red_count = 0

    stage_started = time.perf_counter()
    fallback_red_components: list[DotComponent] = []
    if qualifying_run_indexes:
        run_summaries = [build_run_summary_from_indexes(run, point_components) for run in qualifying_run_indexes]
    elif (metric_group or "").lower() in {"quality", "coverage"} and (
        degraded_swatch_override is not None or bitmap_has_degraded_legend_swatch(bitmap)
    ):
        fallback_red_components = extract_red_run_components(
            bitmap,
            point_components,
            legend_swatches_override=legend_swatches_override,
            degraded_swatch_override=degraded_swatch_override,
        )
        fallback_gap_distance = max(RED_POINT_GAP_DISTANCE, estimate_red_run_link_distance(fallback_red_components))
        fallback_clusters = [
            cluster
            for cluster in cluster_components_by_bbox_gap(fallback_red_components, gap_distance=fallback_gap_distance)
            if len(cluster) >= MIN_CONTINUOUS_RED_POINTS
            and not is_text_like_label_cluster(list(cluster))
            and is_degraded_route_cluster(list(cluster), bitmap.width, bitmap.height)
        ]
        run_summaries = [build_run_summary(cluster) for cluster in fallback_clusters]
        if run_summaries:
            continuity_strategy = "red_component_bbox_gap"
        else:
            # Pixel-ratio fallback: handles maps where route dots are so densely
            # packed that they merge into blobs larger than KPI_COMPONENT_MAX_SPAN
            # and cannot be separated by any component-based method.
            pixel_result = measure_degraded_pixel_ratio_result(
                bitmap,
                legend_swatches_override=legend_swatches_override,
                degraded_swatch_override=degraded_swatch_override,
            )
            if pixel_result is not None:
                run_summaries = [pixel_result]
                continuity_strategy = "degraded_pixel_ratio"
                warnings.append(
                    f"Degraded pixel ratio {pixel_result['degraded_pixel_ratio'] * 100.0:.1f}% "
                    "detected across a broad route area (dots too dense for individual detection)."
                )
    else:
        run_summaries = []
    if run_summaries:
        run_summaries.sort(
            key=lambda summary: (
                -len(summary["indexes"]),
                -summary["total_area"],
                -summary["max_extent"],
                summary["sort_key"][0],
                summary["sort_key"][1],
            ),
        )
        highlighted_clusters = [summary["components"] for summary in run_summaries]
        hotspot_circles = [summary["circle"] for summary in run_summaries if summary["circle"] is not None]
        continuous_red_count = len(highlighted_clusters[0])
        if fallback_red_components:
            red_components = fallback_red_components
            red_point_count = len(red_components)
            red_point_ratio = (red_point_count / total_points) if total_points else 0.0
    summary_sort_s = time.perf_counter() - stage_started

    degradation_detected = bool(highlighted_clusters)

    if continuous_red_count >= MIN_CONTINUOUS_RED_POINTS:
        warnings.append(f"Continuous red points detected ({continuous_red_count}).")

    verdict = "SSV NOK" if degradation_detected else "SSV OK"
    stage_started = time.perf_counter()
    annotated_preview = (
        build_kpi_annotated_preview(
            bitmap=bitmap,
            preview_image_uri=preview_image_uri,
            hotspot_circles=hotspot_circles,
        )
        if degradation_detected and hotspot_circles
        else ""
    )
    annotation_s = time.perf_counter() - stage_started

    pixel_ratio_summary = run_summaries[0] if continuity_strategy == "degraded_pixel_ratio" and run_summaries else None
    metrics = {
        "metric_name": metric_name or "KPI",
        "metric_group": metric_group or "kpi",
        "total_point_count": total_points,
        "red_point_count": red_point_count,
        "red_point_ratio": round(red_point_ratio, 4),
        "continuous_red_count": continuous_red_count,
        "degradation_run_count": len(highlighted_clusters),
        "red_link_distance": round(chain_link_distance, 2),
        "red_cluster_strategy": continuity_strategy,
        "degraded_pixel_ratio": round(float(pixel_ratio_summary["degraded_pixel_ratio"]), 4) if pixel_ratio_summary else None,
        "degraded_pixel_count": int(pixel_ratio_summary["degraded_pixel_count"]) if pixel_ratio_summary else None,
        "degradation_detected": degradation_detected,
        "warnings": warnings,
        "stage_timings": {
            "point_extraction_s": round(point_extraction_s, 4),
            "degraded_classification_s": round(degraded_classification_s, 4),
            "chain_build_s": round(chain_build_s, 4),
            "run_extraction_s": round(run_extraction_s, 4),
            "summary_sort_s": round(summary_sort_s, 4),
            "annotation_s": round(annotation_s, 4),
        },
    }

    return AnalysisOutcome(
        cross=False,
        verdict=verdict,
        detected_colors=[],
        metrics=metrics,
        site_center={"x": 0.0, "y": 0.0},
        annotated_preview=annotated_preview,
        analysis_kind="degradation",
        is_failure=degradation_detected,
        warnings=warnings,
        warning_details=[],
    )


def extract_kpi_point_components(bitmap: Bitmap) -> list[DotComponent]:
    return extract_candidate_dot_components(bitmap, exclude_legend=True)


def extract_candidate_dot_components(
    bitmap: Bitmap,
    exclude_legend: bool,
) -> list[DotComponent]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    hsv_arrays = bitmap_hsv_arrays(bitmap)
    if hsv_arrays is not None:
        _hue, saturation, value = hsv_arrays
        mask = (saturation >= COLOR_SATURATION_THRESHOLD) & (value >= COLOR_VALUE_THRESHOLD)
        if exclude_legend:
            mask = mask.copy()
            mask[:legend_y, :legend_x] = False
        mask_array = mask.astype(np.uint8)
        mask_rows = None
    else:
        mask_rows = [[0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                if exclude_legend and x < legend_x and y < legend_y:
                    continue
                red, green, blue = rgb_pixel(bitmap, x, y)
                _hue, saturation, value = rgb_to_hsv(red, green, blue)
                if saturation >= COLOR_SATURATION_THRESHOLD and value >= COLOR_VALUE_THRESHOLD:
                    mask_rows[y][x] = 1
        mask_array = None

    if mask_array is not None and np is not None and cv2 is not None:
        accelerated_components = extract_candidate_dot_components_accelerated(bitmap, mask_array)
        if accelerated_components:
            return accelerated_components

    raw_components = extract_binary_components(mask_rows=mask_rows, mask_array=mask_array)
    raw_components = [component for component in raw_components if is_kpi_measurement_component_stats(component)]
    if not raw_components:
        return []

    dominant_area = estimate_dominant_component_area(raw_components)
    min_area = max(KPI_COMPONENT_MIN_PIXELS, dominant_area * KPI_COMPONENT_DOMINANT_AREA_RATIO_MIN)
    max_area = max(min_area, dominant_area * KPI_COMPONENT_DOMINANT_AREA_RATIO_MAX)
    filtered = [component for component in raw_components if min_area <= int(component["area"]) <= max_area]
    selected = filtered or raw_components
    return [build_dot_component(bitmap, component) for component in selected]


def extract_candidate_dot_components_accelerated(
    bitmap: Bitmap,
    mask_array: Any,
) -> list[DotComponent]:
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_array.astype(np.uint8, copy=False), connectivity=8)
    if labels_count <= 1:
        return []

    raw_components: list[dict[str, object]] = []
    for label_index in range(1, labels_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        left = int(stats[label_index, cv2.CC_STAT_LEFT])
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        component = {
            "label": label_index,
            "area": area,
            "width": width,
            "height": height,
            "bbox": (left, top, left + width - 1, top + height - 1),
            "center": (float(centroids[label_index][0]), float(centroids[label_index][1])),
        }
        if is_kpi_measurement_component_stats(component):
            raw_components.append(component)

    if not raw_components:
        return []

    dominant_area = estimate_dominant_component_area(raw_components)
    min_area = max(KPI_COMPONENT_MIN_PIXELS, dominant_area * KPI_COMPONENT_DOMINANT_AREA_RATIO_MIN)
    max_area = max(min_area, dominant_area * KPI_COMPONENT_DOMINANT_AREA_RATIO_MAX)
    selected = [component for component in raw_components if min_area <= int(component["area"]) <= max_area] or raw_components

    rgb_array = bitmap_rgb_array(bitmap)
    materialized: list[DotComponent] = []
    for component in selected:
        materialized.append(build_accelerated_dot_component(bitmap, component, labels, rgb_array))
    return materialized


def build_accelerated_dot_component(
    bitmap: Bitmap,
    component: dict[str, object],
    labels: Any,
    rgb_array: Any,
) -> DotComponent:
    label_index = int(component["label"])
    min_x, min_y, max_x, max_y = component["bbox"]
    width = int(component["width"])
    height = int(component["height"])
    area = int(component["area"])

    label_roi = labels[min_y : max_y + 1, min_x : max_x + 1]
    local_mask = label_roi == label_index
    local_ys, local_xs = np.where(local_mask)
    xs = local_xs.astype(np.int32) + int(min_x)
    ys = local_ys.astype(np.int32) + int(min_y)

    if rgb_array is not None:
        roi_rgb = rgb_array[min_y : max_y + 1, min_x : max_x + 1]
        component_rgb = roi_rgb[local_mask]
        red_green_blue = component_rgb.mean(axis=0)
        mean_red = float(red_green_blue[0])
        mean_green = float(red_green_blue[1])
        mean_blue = float(red_green_blue[2])
    else:
        mean_red = sum(rgb_pixel(bitmap, int(x), int(y))[0] for x, y in zip(xs.tolist(), ys.tolist())) / area
        mean_green = sum(rgb_pixel(bitmap, int(x), int(y))[1] for x, y in zip(xs.tolist(), ys.tolist())) / area
        mean_blue = sum(rgb_pixel(bitmap, int(x), int(y))[2] for x, y in zip(xs.tolist(), ys.tolist())) / area

    fill_ratio = area / float(max(1, width * height))
    center = (float(component["center"][0]), float(component["center"][1]))
    mean_lab = rgb_to_lab(mean_red, mean_green, mean_blue)
    pixels = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
    return DotComponent(
        pixels=pixels,
        area=area,
        bbox=(int(min_x), int(min_y), int(max_x), int(max_y)),
        center=center,
        width=width,
        height=height,
        fill_ratio=fill_ratio,
        mean_rgb=(mean_red, mean_green, mean_blue),
        mean_lab=mean_lab,
    )


def build_dot_component(bitmap: Bitmap, component: dict[str, object] | DotComponent) -> DotComponent:
    if isinstance(component, DotComponent):
        return component

    pixels = component["pixels"]
    bbox = component.get("bbox")
    area = int(component.get("area", len(pixels)))
    if bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        width = int(component.get("width", max_x - min_x + 1))
        height = int(component.get("height", max_y - min_y + 1))
    else:
        xs = [pixel[0] for pixel in pixels]
        ys = [pixel[1] for pixel in pixels]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x + 1
        height = max_y - min_y + 1

    fill_ratio = area / float(max(1, width * height))
    xs_array = component.get("xs")
    ys_array = component.get("ys")
    rgb_array = bitmap_rgb_array(bitmap)
    if rgb_array is not None and xs_array is not None and ys_array is not None and len(xs_array) == area:
        red_green_blue = rgb_array[ys_array, xs_array].mean(axis=0)
        mean_red = float(red_green_blue[0])
        mean_green = float(red_green_blue[1])
        mean_blue = float(red_green_blue[2])
        center = (float(xs_array.mean()), float(ys_array.mean()))
    else:
        mean_red = sum(rgb_pixel(bitmap, x, y)[0] for x, y in pixels) / area
        mean_green = sum(rgb_pixel(bitmap, x, y)[1] for x, y in pixels) / area
        mean_blue = sum(rgb_pixel(bitmap, x, y)[2] for x, y in pixels) / area
        center = (
            sum(pixel[0] for pixel in pixels) / area,
            sum(pixel[1] for pixel in pixels) / area,
        )
    mean_lab = rgb_to_lab(mean_red, mean_green, mean_blue)
    return DotComponent(
        pixels=pixels,
        area=area,
        bbox=(min_x, min_y, max_x, max_y),
        center=center,
        width=width,
        height=height,
        fill_ratio=fill_ratio,
        mean_rgb=(mean_red, mean_green, mean_blue),
        mean_lab=mean_lab,
    )


def is_kpi_measurement_component(component: dict[str, object] | DotComponent) -> bool:
    component = build_component_like(component)
    width = component.width
    height = component.height
    area = component.area
    if area < KPI_COMPONENT_MIN_PIXELS or area > KPI_COMPONENT_MAX_PIXELS:
        return False
    if width < KPI_COMPONENT_MIN_SPAN or height < KPI_COMPONENT_MIN_SPAN:
        return False
    if width > KPI_COMPONENT_MAX_SPAN or height > KPI_COMPONENT_MAX_SPAN:
        return False

    fill_ratio = component.fill_ratio
    if fill_ratio < KPI_COMPONENT_MIN_FILL:
        return False

    aspect_ratio = width / float(max(1, height))
    return KPI_COMPONENT_ASPECT_MIN <= aspect_ratio <= KPI_COMPONENT_ASPECT_MAX


def is_kpi_measurement_component_stats(component: dict[str, object] | DotComponent) -> bool:
    if isinstance(component, DotComponent):
        return is_kpi_measurement_component(component)

    area = int(component.get("area", 0))
    width = int(component.get("width", 0))
    height = int(component.get("height", 0))
    if area < KPI_COMPONENT_MIN_PIXELS or area > KPI_COMPONENT_MAX_PIXELS:
        return False
    if width < KPI_COMPONENT_MIN_SPAN or height < KPI_COMPONENT_MIN_SPAN:
        return False
    if width > KPI_COMPONENT_MAX_SPAN or height > KPI_COMPONENT_MAX_SPAN:
        return False

    fill_ratio = area / float(max(1, width * height))
    if fill_ratio < KPI_COMPONENT_MIN_FILL:
        return False

    aspect_ratio = width / float(max(1, height))
    return KPI_COMPONENT_ASPECT_MIN <= aspect_ratio <= KPI_COMPONENT_ASPECT_MAX


def build_component_like(component: dict[str, object] | DotComponent) -> DotComponent:
    if isinstance(component, DotComponent):
        return component

    pixels = component.get("pixels", [])
    bbox = component.get("bbox")
    area = int(component.get("area", len(pixels)))
    if bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        width = int(component.get("width", max_x - min_x + 1))
        height = int(component.get("height", max_y - min_y + 1))
    else:
        xs = [pixel[0] for pixel in pixels]
        ys = [pixel[1] for pixel in pixels]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
    fill_ratio = area / float(max(1, width * height))
    xs_array = component.get("xs")
    ys_array = component.get("ys")
    raw_center = component.get("center")
    if raw_center is not None:
        center = (float(raw_center[0]), float(raw_center[1]))
    elif xs_array is not None and ys_array is not None and len(xs_array) == area:
        center = (float(xs_array.mean()), float(ys_array.mean()))
    else:
        center = (
            sum(pixel[0] for pixel in pixels) / area,
            sum(pixel[1] for pixel in pixels) / area,
        )
    return DotComponent(
        pixels=pixels,
        area=area,
        bbox=(min_x, min_y, max_x, max_y),
        center=center,
        width=width,
        height=height,
        fill_ratio=fill_ratio,
        mean_rgb=(0.0, 0.0, 0.0),
        mean_lab=(0.0, 0.0, 0.0),
    )


def is_red_component(bitmap: Bitmap, component: dict[str, object] | DotComponent) -> bool:
    component = build_dot_component(bitmap, component) if not isinstance(component, DotComponent) or component.mean_rgb == (0.0, 0.0, 0.0) else component
    red, green, blue = component.mean_rgb
    hue, saturation, value = rgb_to_hsv(int(red), int(green), int(blue))

    if saturation < COLOR_SATURATION_THRESHOLD or value < COLOR_VALUE_THRESHOLD:
        return False

    hue_degrees = hue * 360.0
    red_dominant = red >= (green * 1.12) and red >= (blue * 1.35)
    return red_dominant and (hue_degrees <= RED_HUE_WINDOW_DEG or hue_degrees >= (360.0 - RED_HUE_WINDOW_DEG))


def extract_red_run_components(
    bitmap: Bitmap,
    point_components: Sequence[DotComponent] | None = None,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> list[DotComponent]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    legend_swatches = list(legend_swatches_override) if legend_swatches_override is not None else detect_legend_swatches(bitmap, legend_x, legend_y)
    degraded_swatch = degraded_swatch_override if degraded_swatch_override is not None else resolve_degraded_swatch(legend_swatches)
    candidates = list(point_components) if point_components is not None else extract_kpi_point_components(bitmap)

    min_fill_ratio = RED_RUN_COMPONENT_MIN_FILL if degraded_swatch_override is None else max(KPI_COMPONENT_MIN_FILL, RED_RUN_COMPONENT_MIN_FILL - 0.07)
    # When a legend swatch is available, use hue proximity to the swatch as the
    # pre-filter instead of assuming the degraded color is always red-dominant.
    # This allows yellow, orange, and other non-red degraded colours to be found.
    # When no swatch is available, fall back to the red-dominance heuristic.
    swatch_hue_tolerance = LEGEND_HUE_WINDOW_DEG * 2.5  # ≈ 25°

    raw_components = []
    for component in candidates:
        red, green, blue = component.mean_rgb
        hue, saturation, value = rgb_to_hsv(int(red), int(green), int(blue))
        hue_degrees = hue * 360.0

        if degraded_swatch is not None:
            # Legend-aware path: require hue close to the degraded swatch.
            if circular_hue_distance(hue_degrees, degraded_swatch.hue_degrees) > swatch_hue_tolerance:
                continue
            # For red-like swatches (hue ≤ 20° or ≥ 340°) additionally reject
            # orange-leaning components (map labels) via the green/red ratio.
            swatch_is_red = degraded_swatch.hue_degrees <= 20.0 or degraded_swatch.hue_degrees >= 340.0
            if swatch_is_red and red > 0 and green / red > ORANGE_REJECT_GREEN_RATIO:
                continue
        else:
            # No legend: fall back to red-dominance + orange rejection.
            red_dominant = red >= (green * 1.08) and red >= (blue * 1.25)
            if not red_dominant:
                continue
            if red > 0 and green / red > ORANGE_REJECT_GREEN_RATIO:
                continue

        if saturation < 0.35 or value < 0.30:
            continue
        if not is_degraded_component_color(component, degraded_swatch, legend_swatches, hue_degrees):
            continue
        if is_red_run_component(component, min_fill_ratio=min_fill_ratio) and not is_top_ui_noise_component(component, legend_y):
            raw_components.append(component)
    if not raw_components:
        return []

    dominant_area = estimate_dominant_red_area(raw_components)
    min_area = max(RED_RUN_COMPONENT_MIN_PIXELS, dominant_area * RED_RUN_DOMINANT_AREA_RATIO_MIN)
    max_area = max(min_area, dominant_area * RED_RUN_DOMINANT_AREA_RATIO_MAX)

    filtered = [component for component in raw_components if min_area <= component.area <= max_area]
    return filtered or raw_components


def extract_visual_red_dot_components(
    bitmap: Bitmap,
    point_components: Sequence[DotComponent] | None = None,
) -> list[DotComponent]:
    candidates = list(point_components) if point_components is not None else extract_kpi_point_components(bitmap)
    indexes = extract_visual_red_dot_indexes(bitmap, candidates)
    return [candidates[index] for index in indexes]


def extract_visual_red_dot_indexes(
    bitmap: Bitmap,
    point_components: Sequence[DotComponent] | None = None,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> list[int]:
    raw_components = extract_red_run_components(
        bitmap,
        point_components,
        legend_swatches_override=legend_swatches_override,
        degraded_swatch_override=degraded_swatch_override,
    )
    if not raw_components:
        return []

    dominant_area = estimate_dominant_red_area(raw_components)
    min_area = max(RED_RUN_COMPONENT_MIN_PIXELS, dominant_area * 0.72)
    max_area = max(min_area, dominant_area * 1.28)
    filtered = [component for component in raw_components if min_area <= component.area <= max_area]
    selected = filtered or raw_components
    if point_components is None:
        return list(range(len(selected)))

    component_indexes = {id(component): index for index, component in enumerate(point_components)}
    return [component_indexes[id(component)] for component in selected if id(component) in component_indexes]


def detect_legend_degraded_hue(bitmap: Bitmap, legend_x: int, legend_y: int) -> float | None:
    swatches = detect_legend_swatches(bitmap, legend_x, legend_y)
    if not swatches:
        return None
    degraded_swatch = resolve_degraded_swatch(swatches)
    if degraded_swatch is None:
        return None
    return degraded_swatch.hue_degrees


def bitmap_has_degraded_legend_swatch(bitmap: Bitmap) -> bool:
    legend_x = int(bitmap.width * LEGEND_X_RATIO)
    legend_y = int(bitmap.height * LEGEND_Y_RATIO)
    swatches = detect_legend_swatches(bitmap, legend_x, legend_y)
    return resolve_degraded_swatch(swatches) is not None


def extract_bitmap_legend_reference(bitmap: Bitmap) -> tuple[list[LegendSwatch], LegendSwatch | None]:
    legend_x = int(bitmap.width * LEGEND_X_RATIO)
    legend_y = int(bitmap.height * LEGEND_Y_RATIO)
    swatches = detect_legend_swatches(bitmap, legend_x, legend_y)
    return swatches, resolve_degraded_swatch(swatches)


def detect_legend_swatches(bitmap: Bitmap, legend_x: int, legend_y: int) -> list[LegendSwatch]:
    swatch_x_limit = min(legend_x, max(24, int(bitmap.width * 0.08)))
    hsv_arrays = bitmap_hsv_arrays(bitmap)
    if hsv_arrays is not None:
        _hue, saturation, value = hsv_arrays
        roi_mask_array = (
            (saturation[:legend_y, :swatch_x_limit] >= COLOR_SATURATION_THRESHOLD)
            & (value[:legend_y, :swatch_x_limit] >= COLOR_VALUE_THRESHOLD)
        ).astype(np.uint8)
        roi_mask_rows = None
    else:
        roi_mask_rows = [[0] * swatch_x_limit for _ in range(legend_y)]
        for y in range(legend_y):
            for x in range(swatch_x_limit):
                red, green, blue = rgb_pixel(bitmap, x, y)
                hue, saturation, value = rgb_to_hsv(red, green, blue)
                if saturation >= COLOR_SATURATION_THRESHOLD and value >= COLOR_VALUE_THRESHOLD:
                    roi_mask_rows[y][x] = 1
        roi_mask_array = None

    swatches: list[LegendSwatch] = []
    swatch_component_count = 0
    for raw_component in extract_binary_components(mask_rows=roi_mask_rows, mask_array=roi_mask_array):
        component = build_dot_component(bitmap, raw_component)
        min_x, min_y, max_x, max_y = component.bbox
        area = component.area
        width = component.width
        height = component.height
        if area < 20 or area > 200:
            continue
        if width < 4 or width > 16 or height < 4 or height > 16:
            continue
        swatch_component_count += 1

        red, green, blue = component.mean_rgb
        lab = component.mean_lab
        hue, saturation, value = rgb_to_hsv(int(red), int(green), int(blue))
        hue_degrees = hue * 360.0
        if saturation < 0.5 or value < 0.5:
            continue
        center_y = (min_y + max_y) / 2.0
        swatches.append(
            LegendSwatch(
                bbox=component.bbox,
                center_y=center_y,
                rgb=(red, green, blue),
                lab=lab,
                hue_degrees=hue_degrees,
                saturation=saturation,
                value=value,
            )
        )

    if swatch_component_count < 3 or not swatches:
        return []

    swatches.sort(key=lambda item: item.center_y)
    return swatches


def resolve_degraded_swatch(swatches: Sequence[LegendSwatch]) -> LegendSwatch | None:
    if not swatches:
        return None
    return max(swatches, key=lambda swatch: swatch.center_y)


def is_degraded_component_color(
    component: DotComponent,
    degraded_swatch: LegendSwatch | None,
    swatches: Sequence[LegendSwatch],
    hue_degrees: float,
) -> bool:
    if degraded_swatch is None or not swatches:
        return matches_degraded_hue(hue_degrees, None)

    distances = [
        (
            color_distance_lab(component.mean_lab, swatch.lab),
            swatch,
        )
        for swatch in swatches
    ]
    distances.sort(key=lambda item: item[0])
    nearest_distance, nearest_swatch = distances[0]
    if color_distance_lab(nearest_swatch.lab, degraded_swatch.lab) > 1.0:
        return False
    non_degraded_distances = [
        distance
        for distance, swatch in distances[1:]
        if color_distance_lab(swatch.lab, degraded_swatch.lab) > 1.0
    ]
    if not non_degraded_distances:
        return True

    second_distance = non_degraded_distances[0]
    return nearest_distance <= second_distance * LEGEND_DISTANCE_MARGIN_RATIO


def matches_degraded_hue(hue_degrees: float, legend_degraded_hue: float | None) -> bool:
    if legend_degraded_hue is None:
        # Without a legend reference use a tighter hue window (35° instead of 40°)
        # so that orange map-label text (hue ≈ 25–35°) is not included when the
        # legend is unavailable (e.g. raw low-resolution image).
        return hue_degrees <= 35.0 or hue_degrees >= 345.0
    return circular_hue_distance(hue_degrees, legend_degraded_hue) <= LEGEND_HUE_WINDOW_DEG


def is_degraded_route_cluster(
    cluster: list[dict[str, object] | DotComponent],
    image_width: int,
    image_height: int,
) -> bool:
    """Return True if *cluster* spans enough of the image to be a real drive-test route.

    Map text labels form compact clusters whose component centres are packed
    into a small area.  A genuine degraded route spans a meaningful fraction of
    the map image.  Clusters that are too spatially compact are rejected.
    """
    if not cluster:
        return False
    centers = [component_center(component) for component in cluster]
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    span = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
    image_diagonal = math.sqrt(image_width ** 2 + image_height ** 2)
    return span >= image_diagonal * MIN_ROUTE_CLUSTER_DIAGONAL_RATIO


def circular_hue_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def color_distance_lab(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def rgb_to_lab(red: float, green: float, blue: float) -> tuple[float, float, float]:
    def to_linear(channel: float) -> float:
        channel = channel / 255.0
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red_linear = to_linear(red)
    green_linear = to_linear(green)
    blue_linear = to_linear(blue)

    x = (0.4124564 * red_linear) + (0.3575761 * green_linear) + (0.1804375 * blue_linear)
    y = (0.2126729 * red_linear) + (0.7151522 * green_linear) + (0.0721750 * blue_linear)
    z = (0.0193339 * red_linear) + (0.1191920 * green_linear) + (0.9503041 * blue_linear)

    x /= 0.95047
    z /= 1.08883

    def f(value: float) -> float:
        if value > 0.008856:
            return value ** (1.0 / 3.0)
        return (7.787 * value) + (16.0 / 116.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return (l, a, b)


def is_top_ui_noise_component(component: dict[str, object] | DotComponent, legend_y: int) -> bool:
    min_x, min_y, max_x, max_y = component_bbox(component)
    _ = min_x, min_y
    return max_y < legend_y


def is_red_run_component(
    component: dict[str, object] | DotComponent,
    min_fill_ratio: float = RED_RUN_COMPONENT_MIN_FILL,
) -> bool:
    component = build_component_like(component)
    width = component.width
    height = component.height
    area = component.area
    if area < RED_RUN_COMPONENT_MIN_PIXELS or area > RED_RUN_COMPONENT_MAX_PIXELS:
        return False
    if width < RED_RUN_COMPONENT_MIN_SPAN or height < RED_RUN_COMPONENT_MIN_SPAN:
        return False
    if width > RED_RUN_COMPONENT_MAX_SPAN or height > RED_RUN_COMPONENT_MAX_SPAN:
        return False

    fill_ratio = component.fill_ratio
    if fill_ratio < min_fill_ratio:
        return False

    aspect_ratio = width / float(max(1, height))
    return RED_RUN_COMPONENT_ASPECT_MIN <= aspect_ratio <= RED_RUN_COMPONENT_ASPECT_MAX


def estimate_dominant_red_area(components: Sequence[dict[str, object] | DotComponent]) -> float:
    return estimate_dominant_component_area(components)


def estimate_dominant_component_area(components: Sequence[dict[str, object] | DotComponent]) -> float:
    if not components:
        return float(RED_RUN_COMPONENT_MIN_PIXELS)

    areas = [build_component_like(component).area for component in components]
    if not areas:
        return float(RED_RUN_COMPONENT_MIN_PIXELS)
    largest_areas = sorted(areas, reverse=True)[:10]
    if np is not None:
        return float(np.median(np.asarray(largest_areas, dtype=np.float32)))
    return float(statistics.median(largest_areas))


def estimate_red_run_link_distance(components: Sequence[dict[str, object] | DotComponent]) -> float:
    if len(components) < 2:
        return ADAPTIVE_LINK_DISTANCE_MIN

    centers = [component_center(component) for component in components]
    nearest_neighbor_distances: list[float] = []
    for index, (x1, y1) in enumerate(centers):
        distances = [
            math.hypot(x2 - x1, y2 - y1)
            for other_index, (x2, y2) in enumerate(centers)
            if other_index != index
        ]
        if not distances:
            continue
        nearest_neighbor_distances.append(min(distances))

    if not nearest_neighbor_distances:
        return ADAPTIVE_LINK_DISTANCE_MIN

    adaptive_distance = statistics.median(nearest_neighbor_distances) * ADAPTIVE_LINK_DISTANCE_MULTIPLIER
    return max(ADAPTIVE_LINK_DISTANCE_MIN, min(ADAPTIVE_LINK_DISTANCE_MAX, adaptive_distance))


def estimate_dot_chain_link_distance(components: Sequence[dict[str, object] | DotComponent]) -> float:
    if len(components) < 2:
        return DOT_CHAIN_LINK_DISTANCE_MIN

    centers = [component_center(component) for component in components]
    nearest_neighbor_distances = nearest_neighbor_distances_for_centers(centers)
    if not nearest_neighbor_distances:
        return DOT_CHAIN_LINK_DISTANCE_MIN

    adaptive_distance = statistics.median(nearest_neighbor_distances) * DOT_CHAIN_LINK_DISTANCE_MULTIPLIER
    return max(DOT_CHAIN_LINK_DISTANCE_MIN, min(DOT_CHAIN_LINK_DISTANCE_MAX, adaptive_distance))


def build_ordered_dot_chains(
    components: Sequence[dict[str, object] | DotComponent],
    link_distance: float,
) -> list[list[DotComponent]]:
    normalized = [build_component_like(component) for component in components]
    chain_indexes = build_ordered_dot_chain_indexes(normalized, link_distance)
    return [[normalized[index] for index in chain] for chain in chain_indexes]


def build_ordered_dot_chain_indexes(
    components: Sequence[dict[str, object] | DotComponent],
    link_distance: float,
) -> list[list[int]]:
    normalized = [build_component_like(component) for component in components]
    if not normalized:
        return []

    centers = [component.center for component in normalized]
    distance_matrix = pairwise_distance_matrix(centers)
    neighbor_lists: list[list[int]] = []
    for index in range(len(normalized)):
        candidates = [
            other_index
            for other_index, distance in enumerate(distance_matrix[index])
            if other_index != index and distance <= link_distance
        ]
        candidates.sort(key=lambda other_index: distance_matrix[index][other_index])
        neighbor_lists.append(candidates)

    reciprocal_neighbors = build_reciprocal_neighbor_sets(neighbor_lists)
    adjacency = prune_chain_adjacency(reciprocal_neighbors, distance_matrix, centers)

    visited = [False] * len(normalized)
    chains: list[list[int]] = []
    for index in range(len(normalized)):
        if visited[index]:
            continue
        stack = [index]
        component_indexes: list[int] = []
        visited[index] = True
        while stack:
            current = stack.pop()
            component_indexes.append(current)
            for neighbor in adjacency[current]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)

        chains.append(order_dot_chain_indexes(component_indexes, adjacency, normalized))

    chains = merge_chain_endpoint_indexes(chains, normalized, link_distance)
    chains.sort(key=lambda chain: (-len(chain), chain_sort_key_from_indexes(chain, normalized)[0], chain_sort_key_from_indexes(chain, normalized)[1]))
    return chains


def build_reciprocal_neighbor_sets(neighbor_lists: Sequence[Sequence[int]]) -> list[set[int]]:
    reciprocal: list[set[int]] = [set() for _ in neighbor_lists]
    for index, neighbors in enumerate(neighbor_lists):
        rank_lookup = {neighbor: rank for rank, neighbor in enumerate(neighbors[:DOT_CHAIN_NEIGHBOR_RANK_LIMIT])}
        for neighbor, rank in rank_lookup.items():
            if index not in neighbor_lists[neighbor][:DOT_CHAIN_NEIGHBOR_RANK_LIMIT]:
                continue
            reciprocal[index].add(neighbor)
            reciprocal[neighbor].add(index)
    return reciprocal


def prune_chain_adjacency(
    reciprocal_neighbors: Sequence[set[int]],
    distance_matrix: Sequence[Sequence[float]],
    centers: Sequence[tuple[float, float]],
) -> list[set[int]]:
    selected_neighbors: list[set[int]] = [set() for _ in reciprocal_neighbors]
    for index, neighbors in enumerate(reciprocal_neighbors):
        if not neighbors:
            continue
        neighbor_list = sorted(neighbors, key=lambda other_index: distance_matrix[index][other_index])
        best_score = -math.inf
        best_subset: tuple[int, ...] = ()
        candidate_subsets: list[tuple[int, ...]] = [()]
        candidate_subsets.extend((neighbor,) for neighbor in neighbor_list)
        for left_pos in range(len(neighbor_list)):
            for right_pos in range(left_pos + 1, len(neighbor_list)):
                candidate_subsets.append((neighbor_list[left_pos], neighbor_list[right_pos]))

        for subset in candidate_subsets:
            score = score_neighbor_subset(index, subset, distance_matrix, centers)
            if score > best_score:
                best_score = score
                best_subset = subset
        selected_neighbors[index] = set(best_subset)

    adjacency: list[set[int]] = [set() for _ in reciprocal_neighbors]
    for index, neighbors in enumerate(selected_neighbors):
        for neighbor in neighbors:
            if index not in selected_neighbors[neighbor]:
                continue
            adjacency[index].add(neighbor)
            adjacency[neighbor].add(index)
    return adjacency


def score_neighbor_subset(
    index: int,
    subset: Sequence[int],
    distance_matrix: Sequence[Sequence[float]],
    centers: Sequence[tuple[float, float]],
) -> float:
    if not subset:
        return 0.0

    distance_score = sum(1.0 / max(1.0, distance_matrix[index][neighbor]) for neighbor in subset)
    if len(subset) == 1:
        return distance_score

    left_neighbor, right_neighbor = subset
    angle_degrees = abs(
        vector_angle_degrees(
            vector_between(centers[index], centers[left_neighbor]),
            vector_between(centers[index], centers[right_neighbor]),
        )
    )
    if angle_degrees < DOT_CHAIN_MIN_PAIR_ANGLE_DEG:
        return -math.inf
    angle_bonus = (angle_degrees - DOT_CHAIN_MIN_PAIR_ANGLE_DEG) / max(1.0, 180.0 - DOT_CHAIN_MIN_PAIR_ANGLE_DEG)
    return distance_score + angle_bonus


def order_dot_chain(
    component_indexes: Sequence[int],
    adjacency: Sequence[set[int]],
    components: Sequence[DotComponent],
) -> list[DotComponent]:
    ordered_indexes = order_dot_chain_indexes(component_indexes, adjacency, components)
    return [components[index] for index in ordered_indexes]


def order_dot_chain_indexes(
    component_indexes: Sequence[int],
    adjacency: Sequence[set[int]],
    components: Sequence[DotComponent],
) -> list[int]:
    local_indexes = set(component_indexes)
    endpoint_candidates = [index for index in component_indexes if len(adjacency[index] & local_indexes) <= 1]
    if endpoint_candidates:
        start_index = min(endpoint_candidates, key=lambda index: (components[index].center[0], components[index].center[1]))
    else:
        start_index = min(component_indexes, key=lambda index: (components[index].center[0], components[index].center[1]))

    ordered_indexes: list[int] = []
    visited: set[int] = set()
    previous_index: int | None = None
    current_index: int | None = start_index

    while current_index is not None and current_index not in visited:
        ordered_indexes.append(current_index)
        visited.add(current_index)
        next_candidates = [index for index in adjacency[current_index] if index in local_indexes and index not in visited]
        if not next_candidates:
            current_index = None
            continue
        if previous_index is None or len(next_candidates) == 1:
            next_index = min(
                next_candidates,
                key=lambda index: math.hypot(
                    components[index].center[0] - components[current_index].center[0],
                    components[index].center[1] - components[current_index].center[1],
                ),
            )
        else:
            previous_vector = (
                components[current_index].center[0] - components[previous_index].center[0],
                components[current_index].center[1] - components[previous_index].center[1],
            )
            next_index = max(
                next_candidates,
                key=lambda index: chain_direction_score(previous_vector, components[current_index].center, components[index].center),
            )
        previous_index, current_index = current_index, next_index

    if len(ordered_indexes) != len(component_indexes):
        remaining_indexes = sorted(
            (index for index in component_indexes if index not in visited),
            key=lambda index: (components[index].center[0], components[index].center[1]),
        )
        ordered_indexes.extend(remaining_indexes)

    return ordered_indexes


def chain_direction_score(
    previous_vector: tuple[float, float],
    current_center: tuple[float, float],
    next_center: tuple[float, float],
) -> float:
    next_vector = (next_center[0] - current_center[0], next_center[1] - current_center[1])
    previous_length = math.hypot(*previous_vector)
    next_length = math.hypot(*next_vector)
    if previous_length == 0 or next_length == 0:
        return 0.0
    dot = (previous_vector[0] * next_vector[0]) + (previous_vector[1] * next_vector[1])
    return dot / (previous_length * next_length)


def merge_chain_endpoints(
    chains: Sequence[Sequence[DotComponent]],
    link_distance: float,
) -> list[list[DotComponent]]:
    if not chains:
        return []
    flattened: list[DotComponent] = []
    indexed_chains: list[list[int]] = []
    offset = 0
    for chain in chains:
        flattened.extend(chain)
        indexed_chains.append(list(range(offset, offset + len(chain))))
        offset += len(chain)
    merged_indexes = merge_chain_endpoint_indexes(
        indexed_chains,
        flattened,
        link_distance,
    )
    return [[flattened[index] for index in chain] for chain in merged_indexes]


def merge_chain_endpoint_indexes(
    chains: Sequence[Sequence[int]],
    components: Sequence[DotComponent],
    link_distance: float,
) -> list[list[int]]:
    merged = [list(chain) for chain in chains if chain]
    if len(merged) < 2:
        return merged

    merge_distance = min(DOT_CHAIN_MERGE_DISTANCE_MAX, max(link_distance * DOT_CHAIN_MERGE_DISTANCE_MULTIPLIER, link_distance + 32.0))

    while True:
        best_candidate: tuple[float, int, int, list[int]] | None = None
        for left_index in range(len(merged)):
            for right_index in range(left_index + 1, len(merged)):
                candidate = evaluate_chain_merge_indexes(merged[left_index], merged[right_index], components, merge_distance)
                if candidate is None:
                    continue
                score, combined = candidate
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, left_index, right_index, combined)

        if best_candidate is None:
            break

        _score, left_index, right_index, combined = best_candidate
        merged[left_index] = combined
        merged.pop(right_index)

    return merged


def evaluate_chain_merge(
    left_chain: Sequence[DotComponent],
    right_chain: Sequence[DotComponent],
    merge_distance: float,
) -> tuple[float, list[DotComponent]] | None:
    if not left_chain or not right_chain:
        return None
    combined_components = [component for component in left_chain] + [component for component in right_chain]
    left_indexes = list(range(len(left_chain)))
    right_indexes = list(range(len(left_chain), len(combined_components)))
    result = evaluate_chain_merge_indexes(left_indexes, right_indexes, combined_components, merge_distance)
    if result is None:
        return None
    score, combined = result
    return score, [combined_components[index] for index in combined]


def evaluate_chain_merge_indexes(
    left_chain: Sequence[int],
    right_chain: Sequence[int],
    components: Sequence[DotComponent],
    merge_distance: float,
) -> tuple[float, list[int]] | None:
    best: tuple[float, list[int]] | None = None
    orientations = [
        (list(left_chain), list(right_chain)),
        (list(left_chain), list(reversed(right_chain))),
        (list(reversed(left_chain)), list(right_chain)),
        (list(reversed(left_chain)), list(reversed(right_chain))),
    ]
    for oriented_left, oriented_right in orientations:
        candidate = score_chain_merge_orientation_indexes(oriented_left, oriented_right, components, merge_distance)
        if candidate is None:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def score_chain_merge_orientation(
    left_chain: Sequence[DotComponent],
    right_chain: Sequence[DotComponent],
    merge_distance: float,
) -> tuple[float, list[DotComponent]] | None:
    if not left_chain or not right_chain:
        return None
    combined_components = [component for component in left_chain] + [component for component in right_chain]
    left_indexes = list(range(len(left_chain)))
    right_indexes = list(range(len(left_chain), len(combined_components)))
    result = score_chain_merge_orientation_indexes(left_indexes, right_indexes, combined_components, merge_distance)
    if result is None:
        return None
    score, combined = result
    return score, [combined_components[index] for index in combined]


def score_chain_merge_orientation_indexes(
    left_chain: Sequence[int],
    right_chain: Sequence[int],
    components: Sequence[DotComponent],
    merge_distance: float,
) -> tuple[float, list[int]] | None:
    if not left_chain or not right_chain:
        return None

    left_end = components[left_chain[-1]].center
    right_start = components[right_chain[0]].center
    bridge_vector = vector_between(left_end, right_start)
    bridge_distance = math.hypot(*bridge_vector)
    if bridge_distance == 0.0 or bridge_distance > merge_distance:
        return None

    left_outward = chain_endpoint_outward_vector_indexes(left_chain, components, at_start=False)
    right_outward = chain_endpoint_outward_vector_indexes(right_chain, components, at_start=True)
    left_alignment = vector_cosine(left_outward, bridge_vector) if left_outward is not None else 1.0
    right_alignment = vector_cosine(right_outward, (-bridge_vector[0], -bridge_vector[1])) if right_outward is not None else 1.0
    if left_alignment < DOT_CHAIN_MERGE_ALIGNMENT_MIN or right_alignment < DOT_CHAIN_MERGE_ALIGNMENT_MIN:
        return None

    bridge_alignment = min(left_alignment, right_alignment)
    if bridge_alignment < DOT_CHAIN_MERGE_ALIGNMENT_GOOD and bridge_distance > (merge_distance * 0.7):
        return None

    merged_chain = list(left_chain) + list(right_chain)
    score = (
        bridge_alignment * 10.0
        + min(left_alignment, 1.0)
        + min(right_alignment, 1.0)
        + (len(merged_chain) / 100.0)
        - (bridge_distance / max(1.0, merge_distance))
    )
    return score, merged_chain


def chain_endpoint_outward_vector(
    chain: Sequence[DotComponent],
    at_start: bool,
) -> tuple[float, float] | None:
    if not chain:
        return None
    return chain_endpoint_outward_vector_indexes(list(range(len(chain))), chain, at_start)


def chain_endpoint_outward_vector_indexes(
    chain: Sequence[int],
    components: Sequence[DotComponent],
    at_start: bool,
) -> tuple[float, float] | None:
    if len(chain) < 2:
        return None
    sample_count = min(3, len(chain) - 1)
    if at_start:
        anchor = components[chain[0]].center
        neighbor = components[chain[sample_count]].center
        return (anchor[0] - neighbor[0], anchor[1] - neighbor[1])
    anchor = components[chain[-1]].center
    neighbor = components[chain[-1 - sample_count]].center
    return (anchor[0] - neighbor[0], anchor[1] - neighbor[1])


def vector_between(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    return (end[0] - start[0], end[1] - start[1])


def vector_cosine(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    left_length = math.hypot(*left)
    right_length = math.hypot(*right)
    if left_length == 0.0 or right_length == 0.0:
        return 0.0
    return ((left[0] * right[0]) + (left[1] * right[1])) / (left_length * right_length)


def vector_angle_degrees(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    cosine = max(-1.0, min(1.0, vector_cosine(left, right)))
    return math.degrees(math.acos(cosine))


def extract_qualifying_degraded_runs(
    dot_chains: Sequence[Sequence[DotComponent]],
    degraded_component_ids: set[int],
    link_distance: float,
) -> list[list[DotComponent]]:
    flat_components: list[DotComponent] = []
    chains_as_indexes: list[list[int]] = []
    offset = 0
    for chain in dot_chains:
        flat_components.extend(chain)
        chains_as_indexes.append(list(range(offset, offset + len(chain))))
        offset += len(chain)
    component_positions = {id(component): index for index, component in enumerate(flat_components)}
    degraded_indexes = {component_positions[component_id] for component_id in degraded_component_ids if component_id in component_positions}
    run_indexes = extract_qualifying_degraded_run_indexes(chains_as_indexes, flat_components, degraded_indexes, link_distance)
    return [[flat_components[index] for index in run] for run in run_indexes]


def extract_qualifying_degraded_run_indexes(
    dot_chains: Sequence[Sequence[int]],
    components: Sequence[DotComponent],
    degraded_component_indexes: Sequence[int] | set[int],
    link_distance: float,
) -> list[list[int]]:
    degraded_flags = [False] * len(components)
    for index in degraded_component_indexes:
        if 0 <= index < len(degraded_flags):
            degraded_flags[index] = True

    raw_runs: list[list[int]] = []
    for chain in dot_chains:
        current_run: list[int] = []
        for component_index in chain:
            if degraded_flags[component_index]:
                current_run.append(component_index)
                continue
            if current_run:
                raw_runs.append(current_run.copy())
                current_run.clear()
        if current_run:
            raw_runs.append(current_run.copy())

    merged_runs = merge_chain_endpoint_indexes(raw_runs, components, link_distance)
    qualifying_runs: list[list[int]] = []
    for run in merged_runs:
        if len(run) < MIN_CONTINUOUS_RED_POINTS:
            continue
        if is_text_like_label_cluster_from_indexes(run, components):
            continue
        qualifying_runs.append(run)
    return qualifying_runs


def is_text_like_label_cluster_from_indexes(
    cluster_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> bool:
    return is_text_like_label_cluster([components[index] for index in cluster_indexes])


def chain_sort_key_from_indexes(
    chain_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> tuple[float, float]:
    centers = [components[index].center for index in chain_indexes]
    min_x = min(point[0] for point in centers)
    mean_y = sum(point[1] for point in centers) / len(centers)
    return (min_x, mean_y)


def cluster_total_area_from_indexes(
    cluster_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> int:
    return sum(components[index].area for index in cluster_indexes)


def cluster_max_extent_from_indexes(
    cluster_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> float:
    centers = [components[index].center for index in cluster_indexes]
    if len(centers) < 2:
        return 0.0
    if np is not None:
        matrix = pairwise_distance_matrix(centers)
        if matrix:
            return max(max(row) for row in matrix if row)
    max_extent = 0.0
    for index, (x1, y1) in enumerate(centers):
        for x2, y2 in centers[index + 1:]:
            max_extent = max(max_extent, math.hypot(x2 - x1, y2 - y1))
    return max_extent

    raw_runs: list[list[DotComponent]] = []
    for chain in dot_chains:
        current_run: list[DotComponent] = []
        for component in chain:
            if id(component) in degraded_component_ids:
                current_run.append(component)
                continue
            if current_run:
                raw_runs.append(current_run.copy())
            current_run.clear()

        if current_run:
            raw_runs.append(current_run.copy())

    merged_runs = merge_chain_endpoints(raw_runs, link_distance)
    return [
        run
        for run in merged_runs
        if len(run) >= MIN_CONTINUOUS_RED_POINTS and not is_text_like_label_cluster(list(run))
    ]


def choose_best_red_clusters(
    adaptive_clusters: list[list[dict[str, object] | DotComponent]],
    bbox_clusters: list[list[dict[str, object] | DotComponent]],
) -> tuple[list[list[dict[str, object] | DotComponent]], str]:
    filtered_adaptive = [cluster for cluster in adaptive_clusters if not is_text_like_label_cluster(cluster)]
    filtered_bbox = [cluster for cluster in bbox_clusters if not is_text_like_label_cluster(cluster)]

    adaptive_score = red_cluster_strategy_score(filtered_adaptive)
    bbox_score = red_cluster_strategy_score(filtered_bbox)
    if bbox_score > adaptive_score:
        return filtered_bbox, "bbox_gap"
    return filtered_adaptive, "adaptive_center"


def red_cluster_strategy_score(clusters: list[list[dict[str, object] | DotComponent]]) -> tuple[int, int, float]:
    qualifying = [cluster for cluster in clusters if len(cluster) >= MIN_CONTINUOUS_RED_POINTS]
    if not qualifying:
        return (0, 0, 0.0)
    largest = max(len(cluster) for cluster in qualifying)
    qualifying_count = len(qualifying)
    total_extent = sum(cluster_max_extent(cluster) for cluster in qualifying)
    return (largest, qualifying_count, total_extent)


def cluster_components(
    components: Sequence[dict[str, object] | DotComponent],
    link_distance: float = CLUSTER_LINK_DISTANCE,
) -> list[list[dict[str, object] | DotComponent]]:
    if not components:
        return []

    centers = [component_center(component) for component in components]
    components = list(components)
    visited = [False] * len(components)
    clusters: list[list[dict[str, object] | DotComponent]] = []

    for index in range(len(components)):
        if visited[index]:
            continue

        stack = [index]
        visited[index] = True
        cluster: list[dict[str, object] | DotComponent] = []

        while stack:
            current = stack.pop()
            cluster.append(components[current])
            current_x, current_y = centers[current]

            for other_index in range(len(components)):
                if visited[other_index]:
                    continue
                other_x, other_y = centers[other_index]
                if math.hypot(current_x - other_x, current_y - other_y) > link_distance:
                    continue
                visited[other_index] = True
                stack.append(other_index)

        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


def cluster_components_by_bbox_gap(
    components: Sequence[dict[str, object] | DotComponent],
    gap_distance: float = RED_POINT_GAP_DISTANCE,
) -> list[list[dict[str, object] | DotComponent]]:
    if not components:
        return []

    components = list(components)
    visited = [False] * len(components)
    boxes = [component_bbox(component) for component in components]
    clusters: list[list[dict[str, object] | DotComponent]] = []

    for index in range(len(components)):
        if visited[index]:
            continue

        stack = [index]
        visited[index] = True
        cluster: list[dict[str, object] | DotComponent] = []

        while stack:
            current = stack.pop()
            cluster.append(components[current])

            for other_index in range(len(components)):
                if visited[other_index]:
                    continue
                if component_bbox_gap_distance(boxes[current], boxes[other_index]) > gap_distance:
                    continue
                visited[other_index] = True
                stack.append(other_index)

        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters




def component_bbox(component: dict[str, object] | DotComponent) -> tuple[int, int, int, int]:
    return build_component_like(component).bbox


def component_bbox_gap_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_min_x, left_min_y, left_max_x, left_max_y = left
    right_min_x, right_min_y, right_max_x, right_max_y = right
    dx = max(0, max(left_min_x - right_max_x, right_min_x - left_max_x))
    dy = max(0, max(left_min_y - right_max_y, right_min_y - left_max_y))
    return math.hypot(dx, dy)


def component_center(component: dict[str, object] | DotComponent) -> tuple[float, float]:
    return build_component_like(component).center


def cluster_sort_key(cluster: list[dict[str, object] | DotComponent]) -> tuple[float, float]:
    centers = [component_center(component) for component in cluster]
    min_x = min(point[0] for point in centers)
    mean_y = sum(point[1] for point in centers) / len(centers)
    return (min_x, mean_y)


def cluster_total_area(cluster: list[dict[str, object] | DotComponent]) -> int:
    return sum(build_component_like(component).area for component in cluster)


def cluster_max_extent(cluster: list[dict[str, object] | DotComponent]) -> float:
    centers = [component_center(component) for component in cluster]
    if len(centers) < 2:
        return 0.0
    if np is not None:
        matrix = pairwise_distance_matrix(centers)
        if matrix:
            return max(max(row) for row in matrix if row)
    max_extent = 0.0
    for index, (x1, y1) in enumerate(centers):
        for x2, y2 in centers[index + 1:]:
            max_extent = max(max_extent, math.hypot(x2 - x1, y2 - y1))
    return max_extent


def is_text_like_label_cluster(cluster: list[dict[str, object] | DotComponent]) -> bool:
    if len(cluster) < MIN_CONTINUOUS_RED_POINTS:
        return False

    min_x, min_y, max_x, max_y = cluster_bbox(cluster)
    cluster_height = max_y - min_y + 1
    if cluster_height > 32:
        return False

    centers = [component_center(component) for component in cluster]
    row_groups = group_cluster_rows(centers, tolerance=5.0)
    if len(row_groups) != 2:
        return False

    row_groups.sort(key=lambda group: statistics.mean(point[1] for point in group))
    upper_row, lower_row = row_groups
    if len(upper_row) < 2 or len(lower_row) < 2:
        return False

    upper_y = statistics.mean(point[1] for point in upper_row)
    lower_y = statistics.mean(point[1] for point in lower_row)
    row_gap = lower_y - upper_y
    if row_gap < 8.0 or row_gap > 24.0:
        return False

    upper_spread = max(abs(point[1] - upper_y) for point in upper_row)
    lower_spread = max(abs(point[1] - lower_y) for point in lower_row)
    if upper_spread > 4.0 or lower_spread > 4.0:
        return False

    upper_min_x = min(point[0] for point in upper_row)
    upper_max_x = max(point[0] for point in upper_row)
    lower_min_x = min(point[0] for point in lower_row)
    lower_max_x = max(point[0] for point in lower_row)
    overlap = min(upper_max_x, lower_max_x) - max(upper_min_x, lower_min_x)
    smaller_width = min(upper_max_x - upper_min_x, lower_max_x - lower_min_x)
    if smaller_width <= 0:
        return False

    overlap_ratio = overlap / smaller_width
    return overlap_ratio >= 0.45


def group_cluster_rows(
    centers: list[tuple[float, float]],
    tolerance: float,
) -> list[list[tuple[float, float]]]:
    groups: list[list[tuple[float, float]]] = []
    for center in sorted(centers, key=lambda point: point[1]):
        placed = False
        for group in groups:
            group_mean_y = statistics.mean(point[1] for point in group)
            if abs(center[1] - group_mean_y) <= tolerance:
                group.append(center)
                placed = True
                break
        if not placed:
            groups.append([center])
    return groups


def cluster_bbox(cluster: list[dict[str, object] | DotComponent]) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for component in cluster:
        for pixel_x, pixel_y in build_component_like(component).pixels:
            xs.append(pixel_x)
            ys.append(pixel_y)
    return min(xs), min(ys), max(xs), max(ys)


def hotspot_circle(components: list[dict[str, object] | DotComponent]) -> tuple[float, float, float] | None:
    if not components:
        return None

    pixels = [pixel for component in components for pixel in build_component_like(component).pixels]
    xs = [pixel[0] for pixel in pixels]
    ys = [pixel[1] for pixel in pixels]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    base_radius = max(math.hypot(pixel_x - center_x, pixel_y - center_y) for pixel_x, pixel_y in pixels)
    radius = max(18.0, base_radius + HOTSPOT_PADDING)
    return center_x, center_y, radius


def build_run_summary_from_indexes(
    run_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> dict[str, object]:
    run_components = [components[index] for index in run_indexes]
    return {
        "indexes": list(run_indexes),
        "components": run_components,
        "total_area": cluster_total_area_from_indexes(run_indexes, components),
        "max_extent": cluster_max_extent_from_indexes(run_indexes, components),
        "sort_key": chain_sort_key_from_indexes(run_indexes, components),
        "circle": hotspot_circle(run_components),
    }


def build_run_summary(
    run_components: Sequence[dict[str, object] | DotComponent],
) -> dict[str, object]:
    components = [build_component_like(component) for component in run_components]
    return {
        "indexes": list(range(len(components))),
        "components": components,
        "total_area": cluster_total_area(list(components)),
        "max_extent": cluster_max_extent(list(components)),
        "sort_key": cluster_sort_key(list(components)),
        "circle": hotspot_circle(list(components)),
    }


def nearest_neighbor_distances_for_centers(
    centers: Sequence[tuple[float, float]],
) -> list[float]:
    if len(centers) < 2:
        return []
    if np is not None:
        center_array = np.asarray(centers, dtype=np.float32)
        diff = center_array[:, None, :] - center_array[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        return nearest.astype(float).tolist()

    nearest_neighbor_distances: list[float] = []
    for index, (x1, y1) in enumerate(centers):
        distances = [
            math.hypot(x2 - x1, y2 - y1)
            for other_index, (x2, y2) in enumerate(centers)
            if other_index != index
        ]
        if distances:
            nearest_neighbor_distances.append(min(distances))
    return nearest_neighbor_distances


def pairwise_distance_matrix(
    centers: Sequence[tuple[float, float]],
) -> list[list[float]]:
    count = len(centers)
    if count == 0:
        return []
    if np is not None:
        center_array = np.asarray(centers, dtype=np.float32)
        diff = center_array[:, None, :] - center_array[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
        return distances.astype(float).tolist()

    matrix = [[math.inf] * count for _ in range(count)]
    for left_index in range(count):
        matrix[left_index][left_index] = 0.0
        left_x, left_y = centers[left_index]
        for right_index in range(left_index + 1, count):
            right_x, right_y = centers[right_index]
            distance = math.hypot(right_x - left_x, right_y - left_y)
            matrix[left_index][right_index] = distance
            matrix[right_index][left_index] = distance
    return matrix


def measure_degraded_pixel_ratio_result(
    bitmap: Bitmap,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> dict[str, object] | None:
    """Pixel-level degradation check for images where route dots merge into blobs.

    When dots are so densely packed that they overlap and form connected regions
    larger than KPI_COMPONENT_MAX_SPAN, all component-based detection paths
    fail.  This fallback scans every pixel in the route area (outside the legend
    box) and measures what fraction of *coloured* pixels match the degraded-swatch
    hue.  If the ratio exceeds DEGRADED_PIXEL_RATIO_THRESHOLD **and** the
    matching pixels span enough of the image diagonal, degradation is declared.

    Returns a run_summary-compatible dict (with a synthetic hotspot circle) or
    None if degradation cannot be confirmed.
    """
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)

    if degraded_swatch_override is not None:
        swatch = degraded_swatch_override
    else:
        swatches = (
            list(legend_swatches_override)
            if legend_swatches_override is not None
            else detect_legend_swatches(bitmap, legend_x, legend_y)
        )
        swatch = resolve_degraded_swatch(swatches)

    if swatch is None:
        return None

    swatch_hue = swatch.hue_degrees
    tol = DEGRADED_PIXEL_HUE_TOLERANCE
    image_diagonal = math.sqrt(width * width + height * height)
    min_span = image_diagonal * DEGRADED_PIXEL_MIN_DIAGONAL_RATIO

    hsv_arrays = bitmap_hsv_arrays(bitmap)
    if np is not None and hsv_arrays is not None:
        hue_arr, sat_arr, val_arr = hsv_arrays
        hue_deg = hue_arr * 360.0

        route_mask = np.ones((height, width), dtype=bool)
        route_mask[:legend_y, :legend_x] = False

        colored_mask = (
            route_mask
            & (sat_arr >= COLOR_SATURATION_THRESHOLD)
            & (val_arr >= COLOR_VALUE_THRESHOLD)
        )
        hue_dist = np.minimum(
            np.abs(hue_deg - swatch_hue),
            360.0 - np.abs(hue_deg - swatch_hue),
        )
        degraded_mask = colored_mask & (hue_dist <= tol)

        total_colored = int(np.count_nonzero(colored_mask))
        total_degraded = int(np.count_nonzero(degraded_mask))
        if total_colored == 0 or total_degraded == 0:
            return None
        ratio = total_degraded / total_colored
        if ratio < DEGRADED_PIXEL_RATIO_THRESHOLD:
            return None

        ys_deg, xs_deg = np.where(degraded_mask)
        if len(xs_deg) == 0:
            return None
        min_x = int(xs_deg.min())
        max_x = int(xs_deg.max())
        min_y = int(ys_deg.min())
        max_y = int(ys_deg.max())
    else:
        # Pure-Python fallback (no numpy / no HSV cache)
        total_colored = 0
        total_degraded = 0
        min_x, max_x = width, 0
        min_y, max_y = height, 0
        for y in range(height):
            for x in range(width):
                if x < legend_x and y < legend_y:
                    continue
                r, g, b = rgb_pixel(bitmap, x, y)
                _h, s, v = rgb_to_hsv(r, g, b)
                if s < COLOR_SATURATION_THRESHOLD or v < COLOR_VALUE_THRESHOLD:
                    continue
                total_colored += 1
                hue_d = _h * 360.0
                dist = min(abs(hue_d - swatch_hue), 360.0 - abs(hue_d - swatch_hue))
                if dist <= tol:
                    total_degraded += 1
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
        if total_colored == 0 or total_degraded == 0:
            return None
        ratio = total_degraded / total_colored
        if ratio < DEGRADED_PIXEL_RATIO_THRESHOLD:
            return None

    span = math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    if span < min_span:
        return None

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    radius = max(HOTSPOT_PADDING, span / 2.0 + HOTSPOT_PADDING)
    circle = (center_x, center_y, radius)
    return {
        "indexes": [],
        "components": [],
        "total_area": total_degraded,
        "max_extent": span,
        "sort_key": (float(min_x), float(center_y)),
        "circle": circle,
        "degraded_pixel_ratio": ratio,
        "degraded_pixel_count": total_degraded,
    }


def build_kpi_annotated_preview(
    bitmap: Bitmap,
    preview_image_uri: str,
    hotspot_circles: Sequence[tuple[float, float, float]],
) -> str:
    width = bitmap.width
    height = bitmap.height
    circle_markup = ""
    if hotspot_circles:
        scale = max(1.0, width / 1400.0)
        outer_stroke = min(10.0, 6.0 * scale)
        inner_stroke = min(5.0, 3.0 * scale)
        circles: list[str] = []
        for circle in hotspot_circles:
            center_x, center_y, radius = circle
            circles.append(
                f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{radius:.2f}" '
                f'fill="#ff7c87" fill-opacity="0.16" stroke="#fff4f4" stroke-width="{inner_stroke:.2f}" />'
                f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{radius:.2f}" '
                f'fill="none" stroke="#ff7684" stroke-width="{outer_stroke:.2f}" stroke-dasharray="10 7" />'
            )
        circle_markup = "".join(circles)

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image href="{preview_image_uri}" width="{width}" height="{height}" />
  {circle_markup}
</svg>
""".strip()

    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
