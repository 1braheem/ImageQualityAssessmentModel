"""Transparent, deterministic image diagnostics complementary to IQA inference.

These checks measure pixel-level signals; they are not additional learned model
outputs. In particular, occlusion and framing cannot be established from simple
image statistics, so those results are explicitly reported as advisory risks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

import numpy as np
from PIL import Image, ImageOps


JSONScalar: TypeAlias = bool | int | float | str
AssessmentType: TypeAlias = Literal[
    "deterministic_signal", "advisory_risk_heuristic"
]

REVIEW_SCORE = 0.50
ANALYSIS_MAX_SIDE = 1024


@dataclass(frozen=True)
class DiagnosticResult:
    """Serializable result for one defect signal or advisory risk."""

    score: float
    passes: bool
    status: str
    assessment_type: AssessmentType
    explanation: str
    measured: dict[str, JSONScalar]
    thresholds: dict[str, JSONScalar]


@dataclass(frozen=True)
class ImageDiagnostics:
    """All eight diagnostics and a small integration-friendly summary."""

    image_width: int
    image_height: int
    flagged_count: int
    review_recommended: bool
    checks: dict[str, DiagnosticResult]

    def to_dict(self) -> dict[str, object]:
        """Return a recursively JSON-serializable dictionary."""

        return asdict(self)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _severity_below(value: float, *, good: float, bad: float) -> float:
    """Map values at/below ``bad`` to 1 and at/above ``good`` to 0."""

    return _clip01((good - value) / (good - bad))


def _severity_above(value: float, *, good: float, bad: float) -> float:
    """Map values at/below ``good`` to 0 and at/above ``bad`` to 1."""

    return _clip01((value - good) / (bad - good))


def _result(
    score: float,
    *,
    pass_status: str,
    review_status: str,
    assessment_type: AssessmentType,
    explanation: str,
    measured: dict[str, JSONScalar],
    thresholds: dict[str, JSONScalar],
) -> DiagnosticResult:
    score = _clip01(score)
    passes = score < REVIEW_SCORE
    return DiagnosticResult(
        score=_rounded(score),
        passes=passes,
        status=pass_status if passes else review_status,
        assessment_type=assessment_type,
        explanation=explanation,
        measured=measured,
        thresholds={"review_score": REVIEW_SCORE, **thresholds},
    )


def _analysis_rgb(image: Image.Image) -> tuple[Image.Image, int, int]:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    oriented = ImageOps.exif_transpose(image)
    width, height = oriented.size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")

    rgb = oriented.convert("RGB")
    if max(width, height) > ANALYSIS_MAX_SIDE:
        scale = ANALYSIS_MAX_SIDE / max(width, height)
        resized = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        rgb = rgb.resize(resized, Image.Resampling.LANCZOS)
    return rgb, width, height


def _grid_fractions(mask: np.ndarray, rows: int = 8, columns: int = 8) -> list[float]:
    fractions: list[float] = []
    row_edges = np.linspace(0, mask.shape[0], rows + 1, dtype=int)
    column_edges = np.linspace(0, mask.shape[1], columns + 1, dtype=int)
    for row in range(rows):
        for column in range(columns):
            tile = mask[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            if tile.size:
                fractions.append(float(tile.mean()))
    return fractions


def _tile_information(
    luminance: np.ndarray, gradient: np.ndarray, rows: int = 6, columns: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = min(rows, luminance.shape[0])
    columns = min(columns, luminance.shape[1])
    standard_deviations = np.zeros((rows, columns), dtype=np.float32)
    gradient_means = np.zeros((rows, columns), dtype=np.float32)
    tone_means = np.zeros((rows, columns), dtype=np.float32)
    row_edges = np.linspace(0, luminance.shape[0], rows + 1, dtype=int)
    column_edges = np.linspace(0, luminance.shape[1], columns + 1, dtype=int)
    for row in range(rows):
        for column in range(columns):
            row_slice = slice(row_edges[row], row_edges[row + 1])
            column_slice = slice(column_edges[column], column_edges[column + 1])
            tile = luminance[row_slice, column_slice]
            gradient_tile = gradient[row_slice, column_slice]
            if tile.size:
                standard_deviations[row, column] = float(tile.std())
                gradient_means[row, column] = float(gradient_tile.mean())
                tone_means[row, column] = float(tile.mean())
    return standard_deviations, gradient_means, tone_means


def _adjacent_correlation(values: np.ndarray, *, axis: int) -> float:
    """Return Pearson correlation for adjacent pixels along one axis."""

    if values.shape[axis] < 2:
        return 0.0
    if axis == 1:
        first, second = values[:, :-1], values[:, 1:]
    else:
        first, second = values[:-1, :], values[1:, :]
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(
        np.sqrt(np.mean(first_centered**2) * np.mean(second_centered**2))
    )
    if denominator <= 1e-8:
        return 0.0
    return float(np.mean(first_centered * second_centered) / denominator)


def analyze_diagnostics(
    image: Image.Image,
    minimum_width: int = 224,
    minimum_height: int = 224,
) -> ImageDiagnostics:
    """Measure eight transparent image-quality signals.

    Every score is in ``[0, 1]`` and higher means more severe or risky. A check
    is flagged for review at a score of 0.50, except low resolution, whose pass
    result is the exact stated pixel-dimension rule. Thresholds are intentionally
    exposed in each result so consumers do not mistake heuristics for ground truth.
    """

    if minimum_width < 1 or minimum_height < 1:
        raise ValueError("minimum image dimensions must be positive")

    rgb_image, original_width, original_height = _analysis_rgb(image)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    luminance = (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    )

    horizontal_gradient = np.abs(np.diff(luminance, axis=1))
    vertical_gradient = np.abs(np.diff(luminance, axis=0))
    gradient = np.zeros_like(luminance)
    if luminance.shape[1] > 1:
        gradient[:, 1:] += horizontal_gradient
    if luminance.shape[0] > 1:
        gradient[1:, :] += vertical_gradient
    gradient *= 0.5

    if luminance.shape[0] >= 3 and luminance.shape[1] >= 3:
        center = luminance[1:-1, 1:-1]
        laplacian = (
            luminance[:-2, 1:-1]
            + luminance[2:, 1:-1]
            + luminance[1:-1, :-2]
            + luminance[1:-1, 2:]
            - 4.0 * center
        )
        laplacian_variance = float(laplacian.var())
    else:
        laplacian_variance = 0.0
    mean_gradient = float(gradient.mean())

    blur_good = 0.0025
    blur_bad = 0.00015
    blur_score = _severity_below(
        laplacian_variance, good=blur_good, bad=blur_bad
    )
    blur = _result(
        blur_score,
        pass_status="Pass: sufficient high-frequency detail detected",
        review_status="Review: weak high-frequency detail suggests blur",
        assessment_type="deterministic_signal",
        explanation=(
            "Uses variance of a four-neighbour luminance Laplacian. Low variance "
            "indicates weak fine detail, but intentionally smooth scenes can also score poorly."
        ),
        measured={
            "laplacian_variance": _rounded(laplacian_variance),
            "mean_gradient": _rounded(mean_gradient),
        },
        thresholds={
            "bad_laplacian_variance": blur_bad,
            "good_laplacian_variance": blur_good,
        },
    )

    channel_maximum = rgb.max(axis=2)
    channel_minimum = rgb.min(axis=2)
    saturation = np.divide(
        channel_maximum - channel_minimum,
        channel_maximum,
        out=np.zeros_like(channel_maximum),
        where=channel_maximum > 1e-6,
    )
    neutral_highlights = (luminance >= 0.985) & (saturation <= 0.15)
    global_highlight_fraction = float(neutral_highlights.mean())
    peak_grid_highlight_fraction = max(_grid_fractions(neutral_highlights), default=0.0)
    localized_highlight_excess = max(
        0.0, peak_grid_highlight_fraction - global_highlight_fraction
    )
    peak_component = _severity_above(
        peak_grid_highlight_fraction, good=0.15, bad=0.70
    )
    excess_component = _severity_above(
        localized_highlight_excess, good=0.10, bad=0.55
    )
    broad_highlight_discount = 1.0 - _clip01(global_highlight_fraction / 0.75)
    glare_score = max(excess_component, peak_component * broad_highlight_discount)
    glare = _result(
        glare_score,
        pass_status="Pass: no concentrated neutral highlight detected",
        review_status="Review: concentrated bright highlight suggests glare",
        assessment_type="deterministic_signal",
        explanation=(
            "Finds low-saturation pixels above 98.5% luminance in an 8 by 8 grid. "
            "Localized concentrations indicate glare; broad brightness is left to the "
            "overexposure check. Bright white objects can trigger this signal."
        ),
        measured={
            "highlight_pixel_fraction": _rounded(global_highlight_fraction),
            "peak_grid_highlight_fraction": _rounded(peak_grid_highlight_fraction),
            "localized_highlight_excess": _rounded(localized_highlight_excess),
        },
        thresholds={
            "highlight_luminance": 0.985,
            "maximum_highlight_saturation": 0.15,
            "peak_fraction_good": 0.15,
            "peak_fraction_bad": 0.70,
            "localized_excess_good": 0.10,
            "localized_excess_bad": 0.55,
        },
    )

    mean_luminance = float(luminance.mean())
    median_luminance = float(np.median(luminance))
    dark_pixel_fraction = float((luminance <= 0.15).mean())
    darkness_score = 0.65 * _severity_below(
        mean_luminance, good=0.30, bad=0.08
    ) + 0.35 * _severity_above(dark_pixel_fraction, good=0.25, bad=0.85)
    darkness = _result(
        darkness_score,
        pass_status="Pass: luminance distribution is not strongly dark",
        review_status="Review: image appears substantially dark",
        assessment_type="deterministic_signal",
        explanation=(
            "Combines mean luminance with the fraction of pixels at or below 15% "
            "luminance. Night scenes may be intentionally dark."
        ),
        measured={
            "mean_luminance": _rounded(mean_luminance),
            "median_luminance": _rounded(median_luminance),
            "dark_pixel_fraction": _rounded(dark_pixel_fraction),
        },
        thresholds={
            "dark_pixel_luminance": 0.15,
            "mean_luminance_good": 0.30,
            "mean_luminance_bad": 0.08,
            "dark_fraction_good": 0.25,
            "dark_fraction_bad": 0.85,
        },
    )

    bright_pixel_fraction = float((luminance >= 0.95).mean())
    clipped_channel_fraction = float((channel_maximum >= 0.995).mean())
    overexposure_score = 0.55 * _severity_above(
        mean_luminance, good=0.72, bad=0.95
    ) + 0.45 * _severity_above(bright_pixel_fraction, good=0.10, bad=0.80)
    overexposure = _result(
        overexposure_score,
        pass_status="Pass: no strong global overexposure signal",
        review_status="Review: broad bright or clipped regions suggest overexposure",
        assessment_type="deterministic_signal",
        explanation=(
            "Combines global mean luminance with the fraction of pixels above 95% "
            "luminance. High-key scenes or white backgrounds can trigger this signal."
        ),
        measured={
            "mean_luminance": _rounded(mean_luminance),
            "bright_pixel_fraction": _rounded(bright_pixel_fraction),
            "clipped_channel_fraction": _rounded(clipped_channel_fraction),
        },
        thresholds={
            "bright_pixel_luminance": 0.95,
            "mean_luminance_good": 0.72,
            "mean_luminance_bad": 0.95,
            "bright_fraction_good": 0.10,
            "bright_fraction_bad": 0.80,
        },
    )

    horizontal_energy = float(horizontal_gradient.mean()) if horizontal_gradient.size else 0.0
    vertical_energy = float(vertical_gradient.mean()) if vertical_gradient.size else 0.0
    total_directional_energy = horizontal_energy + vertical_energy
    directional_anisotropy = (
        abs(horizontal_energy - vertical_energy) / total_directional_energy
        if total_directional_energy > 1e-8
        else 0.0
    )
    detail_confidence = _clip01(total_directional_energy / 0.02)
    if horizontal_energy < vertical_energy:
        suppressed_axis = "horizontal"
        suppressed_axis_correlation = _adjacent_correlation(luminance, axis=1)
        perpendicular_axis_correlation = _adjacent_correlation(luminance, axis=0)
    elif vertical_energy < horizontal_energy:
        suppressed_axis = "vertical"
        suppressed_axis_correlation = _adjacent_correlation(luminance, axis=0)
        perpendicular_axis_correlation = _adjacent_correlation(luminance, axis=1)
    else:
        suppressed_axis = "none"
        suppressed_axis_correlation = 0.0
        perpendicular_axis_correlation = 0.0
    correlation_contrast = max(
        0.0, suppressed_axis_correlation - perpendicular_axis_correlation
    )
    anisotropy_component = _severity_above(
        directional_anisotropy, good=0.25, bad=0.75
    )
    correlation_component = _severity_above(
        correlation_contrast, good=0.15, bad=0.70
    )
    motion_score = (
        anisotropy_component
        * (0.20 + 0.80 * correlation_component)
        * detail_confidence
    )
    motion_artifacts = _result(
        motion_score,
        pass_status="Pass: no strong directional-smearing signal",
        review_status="Review: directional detail imbalance suggests motion smearing",
        assessment_type="deterministic_signal",
        explanation=(
            "Combines horizontal-versus-vertical gradient imbalance with excess adjacent-"
            "pixel correlation along the smoother axis. It is a motion-smear signal, not "
            "proof of camera motion; strongly directional scene content can trigger it."
        ),
        measured={
            "horizontal_gradient_energy": _rounded(horizontal_energy),
            "vertical_gradient_energy": _rounded(vertical_energy),
            "directional_anisotropy": _rounded(directional_anisotropy),
            "detail_confidence": _rounded(detail_confidence),
            "suppressed_axis": suppressed_axis,
            "suppressed_axis_correlation": _rounded(suppressed_axis_correlation),
            "perpendicular_axis_correlation": _rounded(
                perpendicular_axis_correlation
            ),
            "directional_correlation_contrast": _rounded(correlation_contrast),
        },
        thresholds={
            "detail_confidence_energy": 0.02,
            "anisotropy_good": 0.25,
            "anisotropy_bad": 0.75,
            "correlation_contrast_good": 0.15,
            "correlation_contrast_bad": 0.70,
        },
    )

    tile_std, tile_gradient, tile_tone = _tile_information(luminance, gradient)
    low_information_tiles = (tile_std < 0.025) & (tile_gradient < 0.012)
    extreme_tone_tiles = low_information_tiles & (
        (tile_tone < 0.08) | (tile_tone > 0.96)
    )
    low_information_fraction = float(low_information_tiles.mean())
    extreme_uniform_fraction = float(extreme_tone_tiles.mean())
    center_row_start = low_information_tiles.shape[0] // 3
    center_row_end = low_information_tiles.shape[0] - center_row_start
    center_column_start = low_information_tiles.shape[1] // 3
    center_column_end = low_information_tiles.shape[1] - center_column_start
    center_low_information_fraction = float(
        low_information_tiles[
            center_row_start:center_row_end,
            center_column_start:center_column_end,
        ].mean()
    )
    occlusion_score = (
        0.30
        * _severity_above(low_information_fraction, good=0.20, bad=0.75)
        + 0.55 * center_low_information_fraction
        + 0.15
        * _severity_above(extreme_uniform_fraction, good=0.05, bad=0.50)
    )
    occlusion = _result(
        occlusion_score,
        pass_status="Pass (advisory): no large low-information blockage pattern",
        review_status="Review (advisory): low-information regions may indicate occlusion",
        assessment_type="advisory_risk_heuristic",
        explanation=(
            "Advisory only: measures uniform, low-gradient tiles, with extra weight for "
            "the center and extreme tones. It cannot recognize an occluding object; skies, "
            "walls, graphics, and shallow-depth-of-field scenes can produce false alarms."
        ),
        measured={
            "low_information_tile_fraction": _rounded(low_information_fraction),
            "center_low_information_tile_fraction": _rounded(
                center_low_information_fraction
            ),
            "extreme_uniform_tile_fraction": _rounded(extreme_uniform_fraction),
            "analysis_grid_rows": int(low_information_tiles.shape[0]),
            "analysis_grid_columns": int(low_information_tiles.shape[1]),
        },
        thresholds={
            "maximum_tile_luminance_std": 0.025,
            "maximum_tile_gradient": 0.012,
            "extreme_dark_tile_mean": 0.08,
            "extreme_bright_tile_mean": 0.96,
            "low_information_fraction_good": 0.20,
            "low_information_fraction_bad": 0.75,
        },
    )

    total_gradient = float(gradient.sum())
    if total_gradient > 1e-8:
        normalized_x = np.linspace(0.0, 1.0, gradient.shape[1], dtype=np.float32)
        normalized_y = np.linspace(0.0, 1.0, gradient.shape[0], dtype=np.float32)
        centroid_x = float((gradient * normalized_x[None, :]).sum() / total_gradient)
        centroid_y = float((gradient * normalized_y[:, None]).sum() / total_gradient)
        centroid_offset = min(
            1.0,
            ((centroid_x - 0.5) ** 2 + (centroid_y - 0.5) ** 2) ** 0.5
            / (0.5 * 2**0.5),
        )

        border_y = max(1, round(gradient.shape[0] * 0.06))
        border_x = max(1, round(gradient.shape[1] * 0.06))
        border_mask = np.zeros(gradient.shape, dtype=bool)
        border_mask[:border_y, :] = True
        border_mask[-border_y:, :] = True
        border_mask[:, :border_x] = True
        border_mask[:, -border_x:] = True
        border_mean = float(gradient[border_mask].mean())
        interior_mean = (
            float(gradient[~border_mask].mean()) if (~border_mask).any() else 0.0
        )
        border_to_interior_ratio = border_mean / max(interior_mean, 1e-8)
    else:
        centroid_x = 0.5
        centroid_y = 0.5
        centroid_offset = 0.0
        border_to_interior_ratio = 0.0
    border_component = _severity_above(
        border_to_interior_ratio, good=1.25, bad=3.00
    )
    centroid_component = _severity_above(centroid_offset, good=0.18, bad=0.45)
    framing_score = max(
        0.60 * border_component,
        0.55 * centroid_component,
        0.50 * border_component + 0.50 * centroid_component,
    )
    poor_framing = _result(
        framing_score,
        pass_status="Pass (advisory): detail distribution shows no strong framing risk",
        review_status="Review (advisory): border detail or imbalance suggests framing risk",
        assessment_type="advisory_risk_heuristic",
        explanation=(
            "Advisory only: uses excess detail near the outer 6% border and displacement "
            "of the gradient-energy centroid. It cannot identify the intended subject or "
            "composition, so deliberate off-center framing can trigger it."
        ),
        measured={
            "detail_centroid_x": _rounded(centroid_x),
            "detail_centroid_y": _rounded(centroid_y),
            "normalized_centroid_offset": _rounded(centroid_offset),
            "border_to_interior_gradient_ratio": _rounded(
                border_to_interior_ratio
            ),
            "border_fraction_per_side": 0.06,
        },
        thresholds={
            "centroid_offset_good": 0.18,
            "centroid_offset_bad": 0.45,
            "border_ratio_good": 1.25,
            "border_ratio_bad": 3.00,
        },
    )

    width_ratio = original_width / minimum_width
    height_ratio = original_height / minimum_height
    resolution_score = 1.0 - min(1.0, width_ratio, height_ratio)
    resolution_passes = (
        original_width >= minimum_width and original_height >= minimum_height
    )
    low_resolution = DiagnosticResult(
        score=_rounded(resolution_score),
        passes=resolution_passes,
        status=(
            "Pass: original dimensions meet the minimum"
            if resolution_passes
            else "Review: one or both original dimensions are below the minimum"
        ),
        assessment_type="deterministic_signal",
        explanation=(
            "Directly compares the EXIF-oriented original dimensions with the stated "
            "minimums. This rule is independent of the learned quality score."
        ),
        measured={
            "image_width": original_width,
            "image_height": original_height,
            "width_ratio_to_minimum": _rounded(width_ratio),
            "height_ratio_to_minimum": _rounded(height_ratio),
        },
        thresholds={
            "minimum_width": minimum_width,
            "minimum_height": minimum_height,
        },
    )

    checks = {
        "blur": blur,
        "glare": glare,
        "darkness": darkness,
        "overexposure": overexposure,
        "motion_artifacts": motion_artifacts,
        "occlusion": occlusion,
        "poor_framing": poor_framing,
        "low_resolution": low_resolution,
    }
    flagged_count = sum(not result.passes for result in checks.values())
    return ImageDiagnostics(
        image_width=original_width,
        image_height=original_height,
        flagged_count=flagged_count,
        review_recommended=flagged_count > 0,
        checks=checks,
    )


__all__ = [
    "AssessmentType",
    "DiagnosticResult",
    "ImageDiagnostics",
    "analyze_diagnostics",
]
