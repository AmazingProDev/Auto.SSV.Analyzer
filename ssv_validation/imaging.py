from __future__ import annotations

import hashlib
import logging
import os
import struct
import subprocess
import threading
import time
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

from .acceleration import np
from .models import Bitmap

try:  # pragma: no cover - environment dependent
    from PIL import Image
except Exception:  # pragma: no cover - graceful fallback
    Image = None

LOGGER = logging.getLogger(__name__)
DEFAULT_PREP_MODE = "upscale"
UPSCALE_TARGET_MAX_DIMENSION = 2048
UPSCALE_ENABLE_BELOW_DIMENSION = 1600
PREPARED_IMAGE_CACHE_SIZE = 32

try:  # pragma: no cover - Pillow version dependent
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if Image is not None else None
except Exception:  # pragma: no cover - graceful fallback
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", None) if Image is not None else None

try:  # pragma: no cover - Pillow version dependent
    RESAMPLE_NEAREST = Image.Resampling.NEAREST if Image is not None else None
except Exception:  # pragma: no cover - graceful fallback
    RESAMPLE_NEAREST = getattr(Image, "NEAREST", None) if Image is not None else None

_PREPARED_IMAGE_CACHE: "OrderedDict[tuple[str, str], tuple[bytes, str]]" = OrderedDict()
_PREPARED_IMAGE_CACHE_LOCK = threading.Lock()


class SsvImageError(ValueError):
    """Raised when the SSV image cannot be prepared for analysis."""


def empty_image_prep_stage_timings() -> dict[str, float]:
    return {
        "open_decode_s": 0.0,
        "normalize_composite_s": 0.0,
        "upscale_s": 0.0,
        "png_encode_s": 0.0,
    }


def supports_direct_embedded_image_processing() -> bool:
    return Image is not None


def current_prep_mode() -> str:
    mode = os.environ.get("SSV_IMAGE_PREP_MODE", DEFAULT_PREP_MODE).strip().lower() or DEFAULT_PREP_MODE
    if mode == "excel":
        LOGGER.info("Excel-rendered SSV export is not enabled yet; falling back to high-quality upscale.")
        mode = "upscale"
    if mode not in {"raw", "upscale"}:
        raise SsvImageError(f"Unsupported SSV image preparation mode: {mode}")
    return mode


def prepared_image_cache_key(image_bytes: bytes, mode: str) -> tuple[str, str]:
    digest = hashlib.sha1(image_bytes).hexdigest()
    return mode, digest


def cached_prepared_image(key: tuple[str, str]) -> tuple[bytes, str] | None:
    with _PREPARED_IMAGE_CACHE_LOCK:
        cached = _PREPARED_IMAGE_CACHE.get(key)
        if cached is None:
            return None
        _PREPARED_IMAGE_CACHE.move_to_end(key)
        return cached


def store_prepared_image_cache(key: tuple[str, str], prepared_bytes: bytes, mime_type: str) -> None:
    with _PREPARED_IMAGE_CACHE_LOCK:
        _PREPARED_IMAGE_CACHE[key] = (prepared_bytes, mime_type)
        _PREPARED_IMAGE_CACHE.move_to_end(key)
        while len(_PREPARED_IMAGE_CACHE) > PREPARED_IMAGE_CACHE_SIZE:
            _PREPARED_IMAGE_CACHE.popitem(last=False)


def clear_prepared_image_cache() -> None:
    with _PREPARED_IMAGE_CACHE_LOCK:
        _PREPARED_IMAGE_CACHE.clear()


def write_prepared_bytes(prepared_bytes: bytes, prepared_path: Path | None) -> None:
    if prepared_path is not None:
        prepared_path.write_bytes(prepared_bytes)


def can_reuse_original_preview(image: "Image.Image", mode: str, image_bytes: bytes) -> tuple[bytes, str] | None:
    image_format = (image.format or "").upper()
    if image_format not in {"PNG", "JPEG"}:
        return None
    if mode == "upscale" and max(image.size) < UPSCALE_ENABLE_BELOW_DIMENSION:
        return None

    mime_type = image.get_format_mimetype() or ("image/png" if image_format == "PNG" else "image/jpeg")
    return image_bytes, mime_type


def normalize_embedded_image(image: "Image.Image") -> "Image.Image":
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def preview_encode_format(image: "Image.Image") -> tuple[str, str]:
    image_format = (image.format or "").upper()
    if image_format == "JPEG":
        return "JPEG", "image/jpeg"
    return "JPEG", "image/jpeg"


def decode_image_bytes_for_analysis(image_bytes: bytes) -> Bitmap:
    if Image is None:
        raise SsvImageError("Direct embedded-image decoding requires Pillow.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = normalize_embedded_image(image)
            width, height = rgb_image.size
            if np is not None:
                rgb_array = np.asarray(rgb_image, dtype=np.uint8)
                rgb_pixels = None
            else:
                rgb_array = None
                rgb_pixels = list(rgb_image.getdata())
    except Exception as exc:
        raise SsvImageError(f"The embedded image could not be decoded directly: {exc}") from exc

    rows = None
    if rgb_pixels is not None:
        rows = [
            [tuple(pixel) for pixel in rgb_pixels[row_start : row_start + width]]
            for row_start in range(0, len(rgb_pixels), width)
        ]
    bitmap = Bitmap(width=width, height=height, pixels=rows)
    if rgb_array is not None:
        setattr(bitmap, "_numpy_rgb_array", rgb_array)
    return bitmap


def prepare_kpi_analysis_bytes(image_bytes: bytes) -> tuple[bytes, str] | None:
    """Return nearest-neighbour upscaled image bytes for KPI dot analysis, or None.

    Lanczos upscaling (used for the UI preview) blends neighbouring pixels,
    which can shift orange map-label colours toward the red hue range and cause
    false degraded-dot detections.  Nearest-neighbour resampling maps each
    source pixel to an integer-sized block so every dot keeps its exact colour.

    Returns None when Pillow is unavailable, the image is already large enough,
    or decoding fails — the caller should fall back to the original bytes.
    """
    if Image is None or RESAMPLE_NEAREST is None:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            if max(width, height) >= UPSCALE_ENABLE_BELOW_DIMENSION:
                return None
            rgb_image = normalize_embedded_image(image)
            scale = UPSCALE_TARGET_MAX_DIMENSION / float(max(width, height))
            target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            rgb_image = rgb_image.resize(target_size, RESAMPLE_NEAREST)
            buffer = BytesIO()
            rgb_image.save(buffer, format="PNG", compress_level=1)
            return buffer.getvalue(), "image/png"
    except Exception:
        return None


def prepare_image_bytes_for_analysis(image_bytes: bytes, prepared_path: Path | None = None) -> tuple[bytes, str]:
    prepared_result, _stage_timings = prepare_image_bytes_for_analysis_profiled(image_bytes, prepared_path)
    return prepared_result


def prepare_image_bytes_for_analysis_profiled(
    image_bytes: bytes,
    prepared_path: Path | None = None,
) -> tuple[tuple[bytes, str], dict[str, float]]:
    if Image is None:
        raise SsvImageError("Direct embedded-image preparation requires Pillow.")

    stage_timings = empty_image_prep_stage_timings()
    mode = current_prep_mode()
    cache_key = prepared_image_cache_key(image_bytes, mode)
    cached = cached_prepared_image(cache_key)
    if cached is not None:
        prepared_bytes, mime_type = cached
        write_prepared_bytes(prepared_bytes, prepared_path)
        return (prepared_bytes, mime_type), stage_timings

    try:
        decode_started = time.perf_counter()
        with Image.open(BytesIO(image_bytes)) as image:
            stage_timings["open_decode_s"] += time.perf_counter() - decode_started
            passthrough = can_reuse_original_preview(image, mode, image_bytes)
            if passthrough is not None:
                prepared_bytes, mime_type = passthrough
                store_prepared_image_cache(cache_key, prepared_bytes, mime_type)
                write_prepared_bytes(prepared_bytes, prepared_path)
                return (prepared_bytes, mime_type), stage_timings

            normalize_started = time.perf_counter()
            rgb_image = normalize_embedded_image(image)
            stage_timings["normalize_composite_s"] += time.perf_counter() - normalize_started
    except Exception as exc:
        raise SsvImageError(f"The embedded image could not be prepared directly: {exc}") from exc

    if mode == "upscale":
        width, height = rgb_image.size
        max_dimension = max(width, height)
        if max_dimension < UPSCALE_ENABLE_BELOW_DIMENSION:
            upscale_started = time.perf_counter()
            scale = UPSCALE_TARGET_MAX_DIMENSION / float(max_dimension)
            target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            rgb_image = rgb_image.resize(target_size, RESAMPLE_LANCZOS)
            stage_timings["upscale_s"] += time.perf_counter() - upscale_started

    encode_started = time.perf_counter()
    buffer = BytesIO()
    output_format, mime_type = preview_encode_format(image)
    if output_format == "JPEG":
        rgb_image.save(buffer, format="JPEG", quality=90, subsampling=0, optimize=False)
    else:
        rgb_image.save(buffer, format="PNG", compress_level=1)
    prepared_bytes = buffer.getvalue()
    stage_timings["png_encode_s"] += time.perf_counter() - encode_started
    store_prepared_image_cache(cache_key, prepared_bytes, mime_type)
    write_prepared_bytes(prepared_bytes, prepared_path)
    return (prepared_bytes, mime_type), stage_timings


def prepare_image_bytes_for_analysis_cached(
    image_bytes: bytes,
    request_cache: dict[tuple[str, str], tuple[bytes, str]] | None,
    prepared_path: Path | None = None,
) -> tuple[bytes, str]:
    prepared_result, _stage_timings = prepare_image_bytes_for_analysis_cached_profiled(
        image_bytes,
        request_cache,
        prepared_path,
    )
    return prepared_result


def prepare_image_bytes_for_analysis_cached_profiled(
    image_bytes: bytes,
    request_cache: dict[tuple[str, str], tuple[bytes, str]] | None,
    prepared_path: Path | None = None,
) -> tuple[tuple[bytes, str], dict[str, float]]:
    mode = current_prep_mode()
    cache_key = prepared_image_cache_key(image_bytes, mode)
    stage_timings = empty_image_prep_stage_timings()
    if request_cache is not None:
        cached = request_cache.get(cache_key)
        if cached is not None:
            prepared_bytes, mime_type = cached
            write_prepared_bytes(prepared_bytes, prepared_path)
            return (prepared_bytes, mime_type), stage_timings

    (prepared_bytes, mime_type), stage_timings = prepare_image_bytes_for_analysis_profiled(image_bytes, prepared_path)
    if request_cache is not None:
        request_cache[cache_key] = (prepared_bytes, mime_type)
    return (prepared_bytes, mime_type), stage_timings


def prepare_image_for_analysis(source_path: Path, prepared_path: Path) -> tuple[bytes, str]:
    mode = current_prep_mode()

    if mode == "raw":
        completed = subprocess.run(
            ["sips", "-s", "format", "png", str(source_path), "--out", str(prepared_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        LOGGER.debug("sips output: %s", completed.stdout.strip())
        return prepared_path.read_bytes(), "image/png"

    if mode != "upscale":
        raise SsvImageError(f"Unsupported SSV image preparation mode: {mode}")

    width, height = read_image_dimensions(source_path)
    max_dimension = max(width, height)
    command = ["sips", "-s", "format", "png", "-s", "formatOptions", "best"]
    if max_dimension < UPSCALE_ENABLE_BELOW_DIMENSION:
        command.extend(["--resampleHeightWidthMax", str(UPSCALE_TARGET_MAX_DIMENSION)])
    command.extend([str(source_path), "--out", str(prepared_path)])

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        LOGGER.debug("sips output: %s", completed.stdout.strip())
    except FileNotFoundError as exc:
        raise SsvImageError("The macOS 'sips' tool is required to analyze SSV images on this machine.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise SsvImageError(f"The embedded image could not be prepared for analysis: {stderr or 'sips failed'}") from exc

    return prepared_path.read_bytes(), "image/png"


def convert_image_to_bmp(source_path: Path, output_path: Path) -> None:
    try:
        completed = subprocess.run(
            ["sips", "-s", "format", "bmp", str(source_path), "--out", str(output_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        LOGGER.debug("sips output: %s", completed.stdout.strip())
    except FileNotFoundError as exc:
        raise SsvImageError("The macOS 'sips' tool is required to analyze SSV images on this machine.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise SsvImageError(f"The embedded image could not be normalized for analysis: {stderr or 'sips failed'}") from exc


def read_image_dimensions(source_path: Path) -> tuple[int, int]:
    try:
        completed = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(source_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SsvImageError("The macOS 'sips' tool is required to analyze SSV images on this machine.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise SsvImageError(f"The embedded image dimensions could not be read: {stderr or 'sips failed'}") from exc

    width = None
    height = None
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("pixelWidth:"):
            width = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("pixelHeight:"):
            height = int(stripped.split(":", 1)[1].strip())

    if not width or not height:
        raise SsvImageError("The embedded image dimensions could not be determined.")

    return width, height


def decode_bmp(bitmap_path: Path) -> Bitmap:
    data = bitmap_path.read_bytes()
    if data[:2] != b"BM":
        raise SsvImageError("The normalized SSV image is not a BMP file.")

    pixel_offset = struct.unpack("<I", data[10:14])[0]
    dib_size = struct.unpack("<I", data[14:18])[0]
    width, height = struct.unpack("<ii", data[18:26])
    top_down = height < 0
    height = abs(height)
    bits_per_pixel = struct.unpack("<H", data[28:30])[0]
    compression = struct.unpack("<I", data[30:34])[0]

    if bits_per_pixel not in (24, 32):
        raise SsvImageError(f"Unsupported BMP pixel depth: {bits_per_pixel}")

    pixels: list[list[tuple[int, int, int]]] = []

    if bits_per_pixel == 32:
        row_stride = width * 4
        for y in range(height):
            row_index = y if top_down else (height - 1 - y)
            base = pixel_offset + (row_index * row_stride)
            row: list[tuple[int, int, int]] = []
            for x in range(width):
                blue, green, red, _alpha = data[base + x * 4 : base + x * 4 + 4]
                row.append((red, green, blue))
            pixels.append(row)
    else:
        row_stride = ((width * 3) + 3) & ~3
        for y in range(height):
            row_index = y if top_down else (height - 1 - y)
            base = pixel_offset + (row_index * row_stride)
            row = []
            for x in range(width):
                blue, green, red = data[base + x * 3 : base + x * 3 + 3]
                row.append((red, green, blue))
            pixels.append(row)

    if compression not in (0, 3):
        raise SsvImageError(f"Unsupported BMP compression mode: {compression}")

    if dib_size < 40:
        raise SsvImageError("The normalized BMP header is incomplete.")

    return Bitmap(width=width, height=height, pixels=pixels)
