"""Request-independent response schemas for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ServiceResponse(BaseModel):
    """Basic machine-readable service information."""

    name: str
    status: Literal["running"]
    documentation: str
    analyze_endpoint: str


class HealthResponse(BaseModel):
    """Service and checkpoint readiness without forcing model loading."""

    status: Literal["ready", "model_unavailable"]
    model_available: bool
    model_path: str


class ImageDetails(BaseModel):
    """Decoded image properties measured before model resizing."""

    width: int = Field(ge=1, description="Original displayed width in pixels.")
    height: int = Field(ge=1, description="Original displayed height in pixels.")
    format: str | None = Field(
        default=None, description="Image format detected from the uploaded bytes."
    )


class ModelQualityCheck(BaseModel):
    """Learned quality decision, separate from deterministic rules."""

    threshold: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Heuristic suitability threshold applied to the normalized model score; "
            "the default is 0.60. This is not a separately trained classifier."
        ),
    )
    passes: bool = Field(
        description="True when quality_score is greater than or equal to threshold."
    )


class ResolutionCheck(BaseModel):
    """Transparent pixel-dimension rule, not a model prediction."""

    minimum_width: int = Field(ge=1)
    minimum_height: int = Field(ge=1)
    passes: bool = Field(
        description="True when both original dimensions meet their stated minimum."
    )


DiagnosticValue = bool | int | float | str


class DiagnosticResultResponse(BaseModel):
    """One transparent defect signal or advisory risk estimate."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Defect severity from 0 (minimal) to 1 (high).",
    )
    passes: bool = Field(
        description="True when the measured severity remains within its threshold."
    )
    status: str = Field(description="Short human-readable assessment.")
    assessment_type: Literal[
        "deterministic_signal", "advisory_risk_heuristic"
    ] = Field(
        description=(
            "Whether this is a direct image signal or a context-limited risk heuristic."
        )
    )
    explanation: str = Field(
        description="What was measured and how the result should be interpreted."
    )
    measured: dict[str, DiagnosticValue] = Field(
        description="Measured image statistics used by the diagnostic."
    )
    thresholds: dict[str, DiagnosticValue] = Field(
        description="Thresholds used to convert the measurements into a review decision."
    )


class ImageDiagnosticsResponse(BaseModel):
    """The eight checks requested by the project brief."""

    blur: DiagnosticResultResponse
    glare: DiagnosticResultResponse
    darkness: DiagnosticResultResponse
    overexposure: DiagnosticResultResponse
    motion_artifacts: DiagnosticResultResponse
    occlusion: DiagnosticResultResponse
    poor_framing: DiagnosticResultResponse
    low_resolution: DiagnosticResultResponse


class DiagnosticSummary(BaseModel):
    """Compact interpretation of the eight diagnostic results."""

    flagged_count: int = Field(ge=0, le=8)
    review_recommended: bool


class QualityAnalysisResponse(BaseModel):
    """Overall IQA prediction plus eight transparent image diagnostics."""

    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="EfficientNet-B0 overall perceptual-quality prediction.",
    )
    mos_equivalent: float = Field(
        description="Quality score mapped to the MOS range stored in the checkpoint."
    )
    suitable: bool = Field(
        description=(
            "True only when the learned quality check and all eight diagnostic "
            "checks pass."
        )
    )
    model_check: ModelQualityCheck
    resolution_check: ResolutionCheck
    diagnostics: ImageDiagnosticsResponse
    diagnostics_summary: DiagnosticSummary
    image: ImageDetails
