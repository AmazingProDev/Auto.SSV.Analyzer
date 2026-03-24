from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import re

from .analyzer import SsvAnalysisError, analyze_bitmap
from .imaging import (
    SsvImageError,
    convert_image_to_bmp,
    decode_bmp,
    decode_image_bytes_for_analysis,
    prepare_image_bytes_for_analysis_cached,
    prepare_image_for_analysis,
    prepare_kpi_analysis_bytes,
    supports_direct_embedded_image_processing,
)
from .kpi_analyzer import SsvKpiError, analyze_kpi_bitmap, bitmap_has_degraded_legend_swatch, extract_bitmap_legend_reference
from .legend_mapping import enrich_warning_messages
from .throughput import evaluate_avg_throughput
from .workbook import SsvWorkbookError, extract_avg_throughput_metrics, row_col_to_ref, select_target_images

LOGGER = logging.getLogger(__name__)


class SsvValidationError(ValueError):
    """Top-level validation error surfaced by the API."""


def should_keep_temp_workspace() -> bool:
    return os.environ.get("SSV_KEEP_TEMP_WORKSPACE", "").strip().lower() in {"1", "true", "yes", "on"}


def debug_workspace_root() -> Path:
    configured = os.environ.get("SSV_DEBUG_WORKSPACE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / ".tmp_ssv_debug"


def persist_debug_workspace(temp_path: Path, filename: str) -> Path:
    root = debug_workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    file_slug = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-") or "ssv"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = root / f"{timestamp}-{file_slug}"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(temp_path, destination)
    LOGGER.info("Kept SSV debug workspace at %s", destination)
    return destination


def candidate_workspace_name(index: int, media_path: str, target_key: str, sheet_name: str) -> str:
    sheet_slug = re.sub(r"[^a-z0-9]+", "-", sheet_name.lower()).strip("-")[:40] or "sheet"
    media_slug = Path(media_path).stem[:24] or "image"
    target_slug = re.sub(r"[^a-z0-9]+", "-", target_key.lower()).strip("-") or "target"
    return f"{index:02d}-{sheet_slug}-{target_slug}-{media_slug}"


def image_data_uri(image_bytes: bytes, mime_type: str | None) -> str:
    resolved_mime = mime_type or "application/octet-stream"
    return "data:{};base64,{}".format(
        resolved_mime,
        base64.b64encode(image_bytes).decode("ascii"),
    )


def should_retry_prepared_kpi(candidate, outcome) -> bool:
    metric_group = (candidate.metric_group or "").lower()
    if candidate.analysis_kind != "degradation" or metric_group != "quality":
        return False
    if outcome.is_failure:
        return False
    # Retry if some red/degraded dots were seen but no qualifying run was found.
    if outcome.metrics.get("red_point_count", 0):
        return True
    # Also retry when the raw image was too small for reliable dot separation:
    # fewer than 20 coloured components means individual route dots may be below
    # the minimum-pixel threshold and need the nearest-neighbour upscaled image.
    from .kpi_analyzer import MIN_TOTAL_KPI_POINTS
    total = outcome.metrics.get("total_point_count", 0)
    return total < MIN_TOTAL_KPI_POINTS


def prepared_bitmap_is_meaningfully_larger(raw_bitmap, prepared_bitmap) -> bool:
    raw_area = raw_bitmap.width * raw_bitmap.height
    prepared_area = prepared_bitmap.width * prepared_bitmap.height
    return prepared_area > (raw_area * 1.5)


def validate_ssv_workbook(file_bytes: bytes, filename: str, include_all_previews: bool = False) -> dict[str, object]:
    try:
        selected_images = select_target_images(file_bytes)
    except SsvWorkbookError as exc:
        raise SsvValidationError(str(exc)) from exc

    analyses = []
    kept_workspace_path: str | None = None
    keep_workspace = should_keep_temp_workspace()
    prepared_preview_cache: dict[tuple[str, str], tuple[bytes, str]] = {}
    pending_entries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="ssv-validation-") as temp_dir:
        temp_path = Path(temp_dir)
        LOGGER.debug("Using SSV request workspace: %s", temp_path)

        for index, (candidate, image_bytes, mime_type) in enumerate(selected_images):
            suffix = Path(candidate.media_path).suffix or ".img"
            candidate_path = temp_path / candidate_workspace_name(index, candidate.media_path, candidate.target_key, candidate.sheet_name)
            candidate_path.mkdir(parents=True, exist_ok=True)

            source_path = candidate_path / f"source{suffix}"
            prepared_path = candidate_path / "prepared-preview.png"
            bitmap_path = candidate_path / "analysis.bmp"

            try:
                raw_legend_swatches = None
                raw_degraded_swatch = None
                if supports_direct_embedded_image_processing():
                    bitmap = decode_image_bytes_for_analysis(image_bytes)
                    initial_preview_image = image_data_uri(image_bytes, mime_type)
                else:
                    source_path.write_bytes(image_bytes)
                    prepared_bytes, prepared_mime_type = prepare_image_for_analysis(source_path, prepared_path)
                    convert_image_to_bmp(source_path, bitmap_path)
                    bitmap = decode_bmp(bitmap_path)
                    initial_preview_image = image_data_uri(prepared_bytes, prepared_mime_type)
                if candidate.analysis_kind == "degradation":
                    if (candidate.metric_group or "").lower() in {"quality", "coverage"}:
                        raw_legend_swatches, raw_degraded_swatch = extract_bitmap_legend_reference(bitmap)
                    outcome = analyze_kpi_bitmap(
                        bitmap=bitmap,
                        preview_image_uri=initial_preview_image,
                        metric_name=candidate.metric_name,
                        metric_group=candidate.metric_group,
                    )
                else:
                    outcome = analyze_bitmap(bitmap, initial_preview_image)

                if supports_direct_embedded_image_processing() and should_retry_prepared_kpi(candidate, outcome):
                    prepared_bytes, prepared_mime_type = prepare_image_bytes_for_analysis_cached(
                        image_bytes,
                        prepared_preview_cache,
                        prepared_path if keep_workspace else None,
                    )
                    if prepared_bytes != image_bytes:
                        prepared_bitmap = decode_image_bytes_for_analysis(prepared_bytes)
                        prepared_has_legend = bitmap_has_degraded_legend_swatch(prepared_bitmap)
                        if prepared_bitmap_is_meaningfully_larger(bitmap, prepared_bitmap) and (
                            prepared_has_legend or raw_degraded_swatch is not None
                        ):
                            # Use Lanczos-prepared image as the UI preview (better visual quality).
                            prepared_preview_image = image_data_uri(prepared_bytes, prepared_mime_type)
                            # Use nearest-neighbour upscale for the analysis bitmap so that
                            # orange map-label pixels keep their exact colour and are not
                            # shifted toward red by Lanczos blending.
                            kpi_analysis = prepare_kpi_analysis_bytes(image_bytes)
                            if kpi_analysis is not None:
                                analysis_bytes, _analysis_mime = kpi_analysis
                                analysis_bitmap = decode_image_bytes_for_analysis(analysis_bytes)
                                analysis_has_legend = bitmap_has_degraded_legend_swatch(analysis_bitmap)
                            else:
                                analysis_bitmap = prepared_bitmap
                                analysis_has_legend = prepared_has_legend
                            prepared_outcome = analyze_kpi_bitmap(
                                bitmap=analysis_bitmap,
                                preview_image_uri=prepared_preview_image,
                                metric_name=candidate.metric_name,
                                metric_group=candidate.metric_group,
                                legend_swatches_override=None if analysis_has_legend else raw_legend_swatches,
                                degraded_swatch_override=None if analysis_has_legend else raw_degraded_swatch,
                            )
                            if prepared_outcome.is_failure:
                                bitmap = analysis_bitmap
                                outcome = prepared_outcome
                                initial_preview_image = prepared_preview_image
            except (SsvImageError, SsvAnalysisError, SsvKpiError) as exc:
                raise SsvValidationError(f"{candidate.target_label}: {exc}") from exc

            pending_entries.append(
                {
                    "candidate": candidate,
                    "image_bytes": image_bytes,
                    "mime_type": mime_type,
                    "bitmap": bitmap,
                    "outcome": outcome,
                    "initial_preview_image": initial_preview_image,
                }
            )

        overall_failure = any(
            (entry["outcome"].is_failure if entry["outcome"].is_failure is not None else entry["outcome"].cross)
            for entry in pending_entries
        )
        visible_indices = {
            index
            for index, entry in enumerate(pending_entries)
            if not overall_failure
            or bool(entry["outcome"].is_failure if entry["outcome"].is_failure is not None else entry["outcome"].cross)
        }

        for index, entry in enumerate(pending_entries):
            candidate = entry["candidate"]
            image_bytes = entry["image_bytes"]
            mime_type = entry["mime_type"]
            bitmap = entry["bitmap"]
            outcome = entry["outcome"]
            initial_preview_image = entry["initial_preview_image"]
            preview_image = None
            annotated_preview = None
            should_materialize_preview = include_all_previews or index in visible_indices or not supports_direct_embedded_image_processing()

            if should_materialize_preview:
                if supports_direct_embedded_image_processing():
                    prepared_bytes, prepared_mime_type = prepare_image_bytes_for_analysis_cached(
                        image_bytes,
                        prepared_preview_cache,
                        prepared_path if keep_workspace else None,
                    )
                    preview_image = image_data_uri(prepared_bytes, prepared_mime_type)
                    if preview_image != initial_preview_image:
                        if candidate.analysis_kind == "degradation":
                            outcome = analyze_kpi_bitmap(
                                bitmap=bitmap,
                                preview_image_uri=preview_image,
                                metric_name=candidate.metric_name,
                                metric_group=candidate.metric_group,
                            )
                        else:
                            outcome = analyze_bitmap(bitmap, preview_image)
                    annotated_preview = outcome.annotated_preview
                else:
                    preview_image = initial_preview_image
                    annotated_preview = outcome.annotated_preview

            warnings = outcome.warnings
            if candidate.analysis_kind != "degradation":
                try:
                    warnings = enrich_warning_messages(
                        file_bytes=file_bytes,
                        candidate=candidate,
                        bitmap=bitmap,
                        detected_colors=outcome.detected_colors,
                        warning_details=outcome.warning_details,
                    )
                except Exception as exc:  # pragma: no cover - best effort enrichment
                    LOGGER.warning("Legend warning enrichment failed for %s: %s", candidate.sheet_name, exc)
                    warnings = outcome.warnings

            is_failure = outcome.is_failure if outcome.is_failure is not None else outcome.cross
            analyses.append(
                {
                    "label": candidate.target_label,
                    "targetKey": candidate.target_key,
                    "analysisKind": candidate.analysis_kind,
                    "cross": outcome.cross,
                    "isFailure": is_failure,
                    "verdict": outcome.verdict,
                    "warnings": warnings,
                    "detected_colors": [
                        {
                            "name": detected_color.name,
                            "rgb": list(detected_color.rgb),
                            "hex": detected_color.hex,
                            "dominant_angle": round(detected_color.dominant_angle, 2),
                            "site_angle": round(detected_color.site_angle, 2) if detected_color.site_angle is not None else None,
                            "point_count": detected_color.point_count,
                        }
                        for detected_color in outcome.detected_colors
                    ],
                    "metrics": outcome.metrics,
                    "siteCenter": outcome.site_center,
                    "previewImage": preview_image,
                    "annotatedPreview": annotated_preview,
                    "previewDeferred": supports_direct_embedded_image_processing() and not should_materialize_preview,
                    "selection": {
                        "sheetName": candidate.sheet_name,
                        "sheetPath": candidate.sheet_path,
                        "drawingPath": candidate.drawing_path,
                        "mediaPath": candidate.media_path,
                        "metricGroup": candidate.metric_group,
                        "metricName": candidate.metric_name,
                        "anchor": {
                            "row": candidate.anchor_row + 1,
                            "col": candidate.anchor_col + 1,
                            "ref": row_col_to_ref(candidate.anchor_row, candidate.anchor_col),
                        },
                        "captionCell": candidate.caption_ref,
                        "nearbyText": candidate.nearby_text,
                    },
                }
            )

        if keep_workspace:
            kept_workspace_path = str(persist_debug_workspace(temp_path, filename))

    throughput_metrics = extract_avg_throughput_metrics(file_bytes, filename)
    if throughput_metrics is not None:
        throughput_analysis = evaluate_avg_throughput(throughput_metrics)
        throughput_analysis["selection"] = {
            "sheetName": throughput_metrics["sheet_name"],
            "sheetPath": None,
            "drawingPath": None,
            "mediaPath": None,
            "metricGroup": "throughput",
            "metricName": "Avg Throughput",
            "anchor": None,
            "captionCell": throughput_metrics.get("dl_label_ref"),
            "nearbyText": [
                f"{throughput_metrics['ul_label_ref']}: Avg UL Throughput (Mbps)",
                f"{throughput_metrics['ul_value_ref']}: {throughput_metrics['ul_value_mbps']:.2f} Mbps",
                f"{throughput_metrics['dl_label_ref']}: Avg DL Throughput (Mbps)",
                f"{throughput_metrics['dl_value_ref']}: {throughput_metrics['dl_value_mbps']:.2f} Mbps",
            ],
        }
        analyses.append(throughput_analysis)

    if not analyses:
        raise SsvValidationError("No analyzable SSV images were extracted from the workbook.")

    primary = analyses[0]
    overall_failure = any(bool(analysis.get("isFailure")) for analysis in analyses)
    has_ssv_checks = any(analysis.get("analysisKind") in {"degradation", "throughput_average"} for analysis in analyses)
    overall_cross = any(analysis["cross"] for analysis in analyses)
    if any(bool(analysis.get("isFailure")) and analysis.get("analysisKind") in {"degradation", "throughput_average"} for analysis in analyses):
        overall_verdict = "SSV NOK"
    elif overall_cross:
        overall_verdict = "Cross detected"
    elif has_ssv_checks:
        overall_verdict = "SSV OK"
    else:
        overall_verdict = "No cross detected"

    return {
        "success": True,
        "isFailure": overall_failure,
        "cross": overall_cross,
        "verdict": overall_verdict,
        "analysisCount": len(analyses),
        "analyses": analyses,
        "summary": {
            "crossedImages": sum(1 for analysis in analyses if analysis["cross"]),
            "failedImages": sum(1 for analysis in analyses if analysis.get("isFailure")),
            "clearImages": sum(1 for analysis in analyses if not analysis.get("isFailure")),
            "labels": [analysis["label"] for analysis in analyses],
            "warnings": [warning for analysis in analyses for warning in analysis.get("warnings", [])],
        },
        "detected_colors": primary["detected_colors"],
        "metrics": primary["metrics"],
        "siteCenter": primary["siteCenter"],
        "previewImage": primary["previewImage"],
        "annotatedPreview": primary["annotatedPreview"],
        "selection": primary["selection"],
        "warnings": primary.get("warnings", []),
        "debugWorkspace": kept_workspace_path,
        "includesAllPreviews": include_all_previews,
    }
