from __future__ import annotations

from typing import Any

from .models import Bitmap

try:  # pragma: no cover - environment dependent
    import numpy as np
except Exception:  # pragma: no cover - graceful fallback
    np = None

try:  # pragma: no cover - environment dependent
    import cv2
except BaseException:  # pragma: no cover - graceful fallback (KeyboardInterrupt during macOS AVFoundation bootstrap)
    cv2 = None


def bitmap_rgb_array(bitmap: Bitmap) -> Any:
    if np is None:
        return None
    cached = getattr(bitmap, "_numpy_rgb_array", None)
    if cached is None:
        if bitmap.pixels is None:
            return None
        cached = np.asarray(bitmap.pixels, dtype=np.uint8)
        setattr(bitmap, "_numpy_rgb_array", cached)
    return cached


def bitmap_rgb_rows(bitmap: Bitmap) -> list[list[tuple[int, int, int]]]:
    if bitmap.pixels is not None:
        return bitmap.pixels
    rgb_array = bitmap_rgb_array(bitmap)
    if rgb_array is None:
        return []
    rows = [
        [tuple(int(channel) for channel in rgb_array[y, x]) for x in range(bitmap.width)]
        for y in range(bitmap.height)
    ]
    bitmap.pixels = rows
    return rows


def rgb_pixel(bitmap: Bitmap, x: int, y: int) -> tuple[int, int, int]:
    rgb_array = bitmap_rgb_array(bitmap)
    if rgb_array is not None:
        pixel = rgb_array[y, x]
        return int(pixel[0]), int(pixel[1]), int(pixel[2])
    if bitmap.pixels is None:
        return (0, 0, 0)
    return bitmap.pixels[y][x]


def bitmap_hsv_array(bitmap: Bitmap) -> Any:
    if np is None:
        return None
    cached = getattr(bitmap, "_numpy_hsv_array", None)
    if cached is not None:
        return cached

    rgb = bitmap_rgb_array(bitmap).astype(np.float32) / 255.0
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    delta = max_channel - min_channel

    hue = np.zeros_like(max_channel, dtype=np.float32)
    nonzero = delta > 1e-6
    red_mask = nonzero & (max_channel == red)
    green_mask = nonzero & (max_channel == green)
    blue_mask = nonzero & (max_channel == blue)

    hue[red_mask] = np.mod((green[red_mask] - blue[red_mask]) / delta[red_mask], 6.0)
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / delta[green_mask]) + 2.0
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / delta[blue_mask]) + 4.0
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(max_channel, dtype=np.float32)
    positive_value = max_channel > 1e-6
    saturation[positive_value] = delta[positive_value] / max_channel[positive_value]
    value = max_channel

    cached = np.stack((hue, saturation, value), axis=2)
    setattr(bitmap, "_numpy_hsv_array", cached)
    return cached


def bitmap_hsv_arrays(bitmap: Bitmap) -> tuple[Any, Any, Any] | None:
    hsv_array = bitmap_hsv_array(bitmap)
    if hsv_array is None:
        return None
    return hsv_array[..., 0], hsv_array[..., 1], hsv_array[..., 2]


def hsv_pixel(hsv_cache: Any, x: int, y: int) -> tuple[float, float, float]:
    pixel = hsv_cache[y][x]
    if np is not None and isinstance(pixel, np.ndarray):
        return float(pixel[0]), float(pixel[1]), float(pixel[2])
    if pixel is None:
        return (0.0, 0.0, 0.0)
    return pixel


def shift_component_pixels(
    component: dict[str, object],
    offset_x: int,
    offset_y: int,
) -> dict[str, object]:
    return {
        "pixels": [(pixel_x + offset_x, pixel_y + offset_y) for pixel_x, pixel_y in component["pixels"]],
    }


def extract_binary_components(
    mask_rows: list[list[int]] | None = None,
    mask_array: Any = None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict[str, object]]:
    if np is not None and cv2 is not None:
        if mask_array is None:
            if mask_rows is None:
                return []
            mask_array = np.asarray(mask_rows, dtype=np.uint8)
        else:
            mask_array = mask_array.astype(np.uint8, copy=False)

        if mask_array.size == 0:
            return []

        labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_array, connectivity=8)
        components: list[dict[str, object]] = []
        for label_index in range(1, labels_count):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            left = int(stats[label_index, cv2.CC_STAT_LEFT]) + offset_x
            top = int(stats[label_index, cv2.CC_STAT_TOP]) + offset_y
            width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            ys, xs = np.where(labels == label_index)
            xs_offset = xs.astype(np.int32) + offset_x
            ys_offset = ys.astype(np.int32) + offset_y
            pixels = [(int(x) + offset_x, int(y) + offset_y) for y, x in zip(ys.tolist(), xs.tolist())]
            components.append(
                {
                    "pixels": pixels,
                    "area": area,
                    "width": width,
                    "height": height,
                    "bbox": (left, top, left + width - 1, top + height - 1),
                    "xs": xs_offset,
                    "ys": ys_offset,
                }
            )
        return components

    if mask_rows is None:
        if mask_array is None:
            return []
        mask_rows = mask_array.astype(int).tolist()
    components = extract_components_python(mask_rows)
    if offset_x == 0 and offset_y == 0:
        return components
    return [shift_component_pixels(component, offset_x, offset_y) for component in components]


def extract_components_python(mask: list[list[int]]) -> list[dict[str, object]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False] * width for _ in range(height)]
    components: list[dict[str, object]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y][x] or visited[y][x]:
                continue

            stack = [(x, y)]
            visited[y][x] = True
            pixels: list[tuple[int, int]] = []
            min_x = max_x = x
            min_y = max_y = y

            while stack:
                current_x, current_y = stack.pop()
                pixels.append((current_x, current_y))
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)

                for offset_y_inner in (-1, 0, 1):
                    for offset_x_inner in (-1, 0, 1):
                        if offset_x_inner == 0 and offset_y_inner == 0:
                            continue
                        next_x = current_x + offset_x_inner
                        next_y = current_y + offset_y_inner
                        if not (0 <= next_x < width and 0 <= next_y < height):
                            continue
                        if visited[next_y][next_x] or not mask[next_y][next_x]:
                            continue
                        visited[next_y][next_x] = True
                        stack.append((next_x, next_y))

            components.append(
                {
                    "pixels": pixels,
                    "area": len(pixels),
                    "width": (max_x - min_x) + 1,
                    "height": (max_y - min_y) + 1,
                    "bbox": (min_x, min_y, max_x, max_y),
                }
            )

    return components


def build_integral_image(mask: Any) -> Any:
    if np is not None:
        mask_array = np.asarray(mask, dtype=np.int32)
        if mask_array.ndim != 2:
            mask_array = mask_array.reshape((0, 0))
        height, width = mask_array.shape if mask_array.size else (0, 0)
        integral = np.zeros((height + 1, width + 1), dtype=np.int32)
        if height and width:
            integral[1:, 1:] = mask_array.cumsum(axis=0).cumsum(axis=1)
        return integral

    height = len(mask)
    width = len(mask[0]) if height else 0
    integral = [[0] * (width + 1) for _ in range(height + 1)]

    for y in range(height):
        row_sum = 0
        for x in range(width):
            row_sum += mask[y][x]
            integral[y + 1][x + 1] = integral[y][x + 1] + row_sum

    return integral


def neighborhood_sum(
    integral: Any,
    x: int,
    y: int,
    radius: int,
    width: int,
    height: int,
) -> int:
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(width, x + radius + 1)
    y1 = min(height, y + radius + 1)
    return int(integral[y1][x1] - integral[y0][x1] - integral[y1][x0] + integral[y0][x0])
