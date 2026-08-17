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


class QualityAnalysisResponse(BaseModel):
    """Overall IQA score and the two checks used for suitability."""

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
            "True only when the model-quality check and the separate resolution "
            "check both pass."
        )
    )
    model_check: ModelQualityCheck
    resolution_check: ResolutionCheck
    image: ImageDetails
