#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssv_validation.acceleration import cv2, np
from ssv_validation.analyzer import analyze_bitmap
from ssv_validation.imaging import (
    convert_image_to_bmp,
    decode_bmp,
    decode_image_bytes_for_analysis,
    empty_image_prep_stage_timings,
    prepare_image_bytes_for_analysis_cached,
    prepare_image_bytes_for_analysis_cached_profiled,
    prepare_image_for_analysis,
    supports_direct_embedded_image_processing,
)
from ssv_validation.kpi_analyzer import analyze_kpi_bitmap
from ssv_validation.throughput import evaluate_avg_throughput
from ssv_validation.workbook import extract_avg_throughput_metrics, select_target_images


def now() -> float:
    return time.perf_counter()


def timed_call(func, *args, **kwargs):
    started = now()
    value = func(*args, **kwargs)
    elapsed = now() - started
    return value, elapsed


def benchmark_workbook(workbook_path: Path) -> dict[str, Any]:
    file_bytes = workbook_path.read_bytes()

    selected_images, workbook_parse_s = timed_call(select_target_images, file_bytes)
    throughput_metrics, throughput_extract_s = timed_call(extract_avg_throughput_metrics, file_bytes, workbook_path.name)

    stage_totals = {
        "workbook_parse_s": workbook_parse_s,
        "throughput_extract_s": throughput_extract_s,
        "image_prep_s": 0.0,
        "bitmap_convert_s": 0.0,
        "bitmap_decode_s": 0.0,
        "cross_analysis_s": 0.0,
        "kpi_analysis_s": 0.0,
    }
    image_prep_stage_totals = empty_image_prep_stage_timings()
    analyses: list[dict[str, Any]] = []
    prepared_preview_cache: dict[tuple[str, str], tuple[bytes, str]] = {}

    with tempfile.TemporaryDirectory(prefix="ssv-benchmark-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, (candidate, image_bytes, _mime_type) in enumerate(selected_images):
            suffix = Path(candidate.media_path).suffix or ".img"
            source_path = temp_root / f"{index:02d}-source{suffix}"
            prepared_path = temp_root / f"{index:02d}-prepared.png"
            bitmap_path = temp_root / f"{index:02d}-analysis.bmp"

            if supports_direct_embedded_image_processing():
                prepared_result, image_prep_s = timed_call(
                    prepare_image_bytes_for_analysis_cached_profiled,
                    image_bytes,
                    prepared_preview_cache,
                    None,
                )
                (prepared_bytes, prepared_mime_type), image_prep_stage_timings = prepared_result
                stage_totals["image_prep_s"] += image_prep_s
                for stage_name, stage_value in image_prep_stage_timings.items():
                    image_prep_stage_totals[stage_name] += stage_value
                preview_image = "data:{};base64,{}".format(
                    prepared_mime_type,
                    base64.b64encode(prepared_bytes).decode("ascii"),
                )
                bitmap, bitmap_decode_s = timed_call(decode_image_bytes_for_analysis, image_bytes)
                bitmap_convert_s = 0.0
                stage_totals["bitmap_decode_s"] += bitmap_decode_s
            else:
                image_prep_stage_timings = empty_image_prep_stage_timings()
                source_path.write_bytes(image_bytes)
                prepared_result, image_prep_s = timed_call(prepare_image_for_analysis, source_path, prepared_path)
                prepared_bytes, prepared_mime_type = prepared_result
                stage_totals["image_prep_s"] += image_prep_s
                preview_image = "data:{};base64,{}".format(
                    prepared_mime_type,
                    base64.b64encode(prepared_bytes).decode("ascii"),
                )
                _, bitmap_convert_s = timed_call(convert_image_to_bmp, source_path, bitmap_path)
                stage_totals["bitmap_convert_s"] += bitmap_convert_s
                bitmap, bitmap_decode_s = timed_call(decode_bmp, bitmap_path)
                stage_totals["bitmap_decode_s"] += bitmap_decode_s

            if candidate.analysis_kind == "degradation":
                outcome, analysis_s = timed_call(
                    analyze_kpi_bitmap,
                    bitmap=bitmap,
                    preview_image_uri=preview_image,
                    metric_name=candidate.metric_name,
                    metric_group=candidate.metric_group,
                )
                stage_totals["kpi_analysis_s"] += analysis_s
                analysis_stage = "kpi"
            else:
                outcome, analysis_s = timed_call(analyze_bitmap, bitmap, preview_image)
                stage_totals["cross_analysis_s"] += analysis_s
                analysis_stage = "cross"

            analyses.append(
                {
                    "sheet_name": candidate.sheet_name,
                    "label": candidate.target_label,
                    "analysis_kind": candidate.analysis_kind,
                    "analysis_stage": analysis_stage,
                    "verdict": outcome.verdict,
                    "continuous_red_count": outcome.metrics.get("continuous_red_count"),
                    "red_cluster_strategy": outcome.metrics.get("red_cluster_strategy"),
                    "image_prep_s": image_prep_s,
                    "bitmap_convert_s": bitmap_convert_s,
                    "bitmap_decode_s": bitmap_decode_s,
                    "analysis_s": analysis_s,
                    "image_prep_stage_timings": image_prep_stage_timings,
                    "stage_timings": outcome.metrics.get("stage_timings", {}),
                }
            )

    throughput_average_s = 0.0
    throughput_average_summary: dict[str, Any] | None = None
    if throughput_metrics is not None:
        throughput_result, throughput_average_s = timed_call(evaluate_avg_throughput, throughput_metrics)
        throughput_average_summary = {
            "verdict": throughput_result["verdict"],
            "band": throughput_metrics["band"],
            "dl_value_mbps": throughput_metrics["dl_value_mbps"],
            "ul_value_mbps": throughput_metrics["ul_value_mbps"],
            "analysis_s": throughput_average_s,
        }

    total_runtime_s = sum(stage_totals.values()) + throughput_average_s
    return {
        "workbook": str(workbook_path),
        "environment": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy": getattr(np, "__version__", None),
            "cv2": getattr(cv2, "__version__", None),
            "accelerated_backend": bool(np is not None and cv2 is not None),
        },
        "stage_totals": {
            **{key: round(value, 4) for key, value in stage_totals.items()},
            "throughput_average_s": round(throughput_average_s, 4),
            "total_runtime_s": round(total_runtime_s, 4),
        },
        "image_prep_stage_totals": {key: round(value, 4) for key, value in image_prep_stage_totals.items()},
        "analysis_count": len(analyses),
        "analyses": analyses,
        "throughput_average": throughput_average_summary,
    }


def run_worker(workbook_paths: list[Path]) -> int:
    payload = {
        "results": [benchmark_workbook(path) for path in workbook_paths],
    }
    print(json.dumps(payload, indent=2))
    return 0


def run_controller(
    workbook_paths: list[Path],
    fallback_python: str,
    accelerated_python: str,
) -> int:
    script_path = Path(__file__).resolve()
    results_by_label: list[tuple[str, dict[str, Any]]] = []
    for label, python_exec in (("fallback", fallback_python), ("accelerated", accelerated_python)):
        command = [python_exec, str(script_path), "--worker", *[str(path) for path in workbook_paths]]
        completed = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
        results_by_label.append((label, json.loads(completed.stdout)))

    print("")
    print("SSV Benchmark")
    print("=============")
    for workbook_index, workbook_path in enumerate(workbook_paths):
        print("")
        print(workbook_path)
        fallback_result = results_by_label[0][1]["results"][workbook_index]
        accelerated_result = results_by_label[1][1]["results"][workbook_index]
        print(
            "  fallback   : {:.4f}s total | accel backend={}".format(
                fallback_result["stage_totals"]["total_runtime_s"],
                fallback_result["environment"]["accelerated_backend"],
            )
        )
        print(
            "  accelerated: {:.4f}s total | accel backend={}".format(
                accelerated_result["stage_totals"]["total_runtime_s"],
                accelerated_result["environment"]["accelerated_backend"],
            )
        )
        print("  stages:")
        for stage_name in (
            "workbook_parse_s",
            "throughput_extract_s",
            "image_prep_s",
            "bitmap_convert_s",
            "bitmap_decode_s",
            "cross_analysis_s",
            "kpi_analysis_s",
            "throughput_average_s",
        ):
            print(
                "    {:<20} fallback={:>8.4f}s accelerated={:>8.4f}s".format(
                    stage_name,
                    fallback_result["stage_totals"][stage_name],
                    accelerated_result["stage_totals"][stage_name],
                )
            )
        print("  image prep sub-stages:")
        for stage_name in (
            "open_decode_s",
            "normalize_composite_s",
            "upscale_s",
            "png_encode_s",
        ):
            print(
                "    {:<20} fallback={:>8.4f}s accelerated={:>8.4f}s".format(
                    stage_name,
                    fallback_result["image_prep_stage_totals"][stage_name],
                    accelerated_result["image_prep_stage_totals"][stage_name],
                )
            )

        print("  per-analysis:")
        fallback_by_label = {entry["label"]: entry for entry in fallback_result["analyses"]}
        accelerated_by_label = {entry["label"]: entry for entry in accelerated_result["analyses"]}
        for label_name in sorted(set(fallback_by_label) | set(accelerated_by_label)):
            left = fallback_by_label.get(label_name)
            right = accelerated_by_label.get(label_name)
            if left is None or right is None:
                continue
            print(
                "    {:<24} fallback={:>7.4f}s accelerated={:>7.4f}s verdict={}".format(
                    label_name,
                    left["analysis_s"],
                    right["analysis_s"],
                    right["verdict"],
                )
            )
            if right.get("stage_timings"):
                timings = right["stage_timings"]
                if "point_extraction_s" in timings:
                    print(
                        "      sub-stages: points={point_extraction_s:.4f}s classify={degraded_classification_s:.4f}s chain={chain_build_s:.4f}s runs={run_extraction_s:.4f}s sort={summary_sort_s:.4f}s annot={annotation_s:.4f}s".format(
                            **timings,
                        )
                    )
                elif "segment_point_clouds_s" in timings:
                    print(
                        "      sub-stages: cache={color_cache_s:.4f}s hint={site_hint_s:.4f}s hues={sector_hues_s:.4f}s center={site_center_s:.4f}s signatures={sector_signatures_s:.4f}s points={segment_point_clouds_s:.4f}s metrics={cross_metrics_s:.4f}s annot={annotation_s:.4f}s".format(
                            **timings,
                        )
                    )
            if right.get("image_prep_stage_timings"):
                prep = right["image_prep_stage_timings"]
                print(
                    "      prep: open={open_decode_s:.4f}s normalize={normalize_composite_s:.4f}s upscale={upscale_s:.4f}s encode={png_encode_s:.4f}s".format(
                        **prep,
                    )
                )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SSV validation stages across Python runtimes.")
    parser.add_argument("workbooks", nargs="+", help="Workbook paths to benchmark.")
    parser.add_argument("--worker", action="store_true", help="Run benchmark worker mode and emit JSON.")
    parser.add_argument("--fallback-python", default="python3", help="Python executable for the fallback runtime.")
    parser.add_argument("--accelerated-python", default=str(ROOT / ".venv" / "bin" / "python"), help="Python executable for the accelerated runtime.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workbook_paths = [Path(path).expanduser().resolve() for path in args.workbooks]
    if args.worker:
        return run_worker(workbook_paths)
    return run_controller(workbook_paths, args.fallback_python, args.accelerated_python)


if __name__ == "__main__":
    raise SystemExit(main())
