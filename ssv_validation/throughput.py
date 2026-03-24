from __future__ import annotations

import base64

from .workbook import THROUGHPUT_THRESHOLDS


def evaluate_avg_throughput(metrics: dict[str, object]) -> dict[str, object]:
    band = str(metrics.get("band") or "")
    dl_value = float(metrics.get("dl_value_mbps") or 0.0)
    ul_value = float(metrics.get("ul_value_mbps") or 0.0)
    thresholds = THROUGHPUT_THRESHOLDS.get(band)

    warnings: list[str] = []
    if thresholds is None:
        warnings.append("Unknown LTE band for throughput thresholds.")
        verdict = "SSV NOK"
        is_failure = True
        dl_threshold = None
        ul_threshold = None
    else:
        dl_threshold = thresholds["dl"]
        ul_threshold = thresholds["ul"]
        below_dl = dl_value < dl_threshold
        below_ul = ul_value < ul_threshold
        if below_dl:
            warnings.append(f"Avg DL Throughput {dl_value:.2f} Mbps is below the {band} minimum of {dl_threshold:.0f} Mbps.")
        if below_ul:
            warnings.append(f"Avg UL Throughput {ul_value:.2f} Mbps is below the {band} minimum of {ul_threshold:.0f} Mbps.")
        is_failure = below_dl or below_ul
        verdict = "SSV NOK" if is_failure else "SSV OK"

    return {
        "analysisKind": "throughput_average",
        "label": "Avg Throughput",
        "cross": False,
        "isFailure": is_failure,
        "verdict": verdict,
        "warnings": warnings,
        "detected_colors": [],
        "metrics": {
            "metric_name": "Avg Throughput",
            "metric_group": "throughput",
            "band": band or "Unknown",
            "dl_average_mbps": round(dl_value, 2),
            "ul_average_mbps": round(ul_value, 2),
            "dl_threshold_mbps": dl_threshold,
            "ul_threshold_mbps": ul_threshold,
        },
        "siteCenter": {"x": 0.0, "y": 0.0},
        "previewImage": None,
        "annotatedPreview": build_throughput_summary_preview(
            band=band or "Unknown",
            dl_value=dl_value,
            ul_value=ul_value,
            dl_threshold=dl_threshold,
            ul_threshold=ul_threshold,
            verdict=verdict,
        ),
    }


def build_throughput_summary_preview(
    band: str,
    dl_value: float,
    ul_value: float,
    dl_threshold: float | None,
    ul_threshold: float | None,
    verdict: str,
) -> str:
    verdict_bg = "#FF2900" if verdict == "SSV NOK" else "#00FF37"
    verdict_fg = "#fff5f2" if verdict == "SSV NOK" else "#001f08"
    dl_threshold_text = f"{dl_threshold:.0f} Mbps" if dl_threshold is not None else "n/a"
    ul_threshold_text = f"{ul_threshold:.0f} Mbps" if ul_threshold is not None else "n/a"
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420">
  <rect width="900" height="420" rx="32" fill="#101a29" />
  <rect x="38" y="34" width="200" height="62" rx="16" fill="{verdict_bg}" />
  <text x="62" y="74" fill="{verdict_fg}" font-size="30" font-weight="700" font-family="Inter, sans-serif">{verdict}</text>
  <text x="40" y="150" fill="#dbe8ff" font-size="34" font-weight="700" font-family="Inter, sans-serif">Throughput summary</text>
  <text x="40" y="196" fill="#9fc3f6" font-size="24" font-family="Inter, sans-serif">Band: {band}</text>
  <text x="40" y="266" fill="#dbe8ff" font-size="28" font-family="Inter, sans-serif">Avg DL Throughput: {dl_value:.2f} Mbps</text>
  <text x="40" y="306" fill="#9fc3f6" font-size="22" font-family="Inter, sans-serif">Minimum: {dl_threshold_text}</text>
  <text x="40" y="366" fill="#dbe8ff" font-size="28" font-family="Inter, sans-serif">Avg UL Throughput: {ul_value:.2f} Mbps</text>
  <text x="40" y="406" fill="#9fc3f6" font-size="22" font-family="Inter, sans-serif">Minimum: {ul_threshold_text}</text>
</svg>
""".strip()
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
