"""FastAPI application for no-reference image-quality analysis."""

from __future__ import annotations

import io
import warnings
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from api.schemas import (
    HealthResponse,
    ImageDetails,
    ModelQualityCheck,
    QualityAnalysisResponse,
    ResolutionCheck,
    ServiceResponse,
)
from src.iqa.inference import IQAPredictor, resolve_model_path


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_CONTENT_TYPES = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"


app = FastAPI(
    title="Image Quality Assessment API",
    version="1.0.0",
    description=(
        "Predicts an overall perceptual-quality score for one image. The model "
        "does not produce individual blur, glare, exposure, or occlusion labels."
    ),
)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


@lru_cache(maxsize=4)
def _cached_predictor(model_path: str) -> IQAPredictor:
    """Create one predictor per resolved checkpoint path, only when first needed."""

    return IQAPredictor(model_path=Path(model_path))


def get_predictor() -> IQAPredictor:
    """Return the cached predictor while honoring the current IQA_MODEL_PATH."""

    return _cached_predictor(str(resolve_model_path()))


def _decode_image(data: bytes, content_type: str) -> Image.Image:
    expected_format = SUPPORTED_CONTENT_TYPES[content_type]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            detected_format = (image.format or "").upper()
            if detected_format != expected_format:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=(
                        f"Content-Type {content_type!r} does not match the detected "
                        f"image format {detected_format or 'unknown'!r}."
                    ),
                )
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Decoded image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit.",
                )
            image.load()
            return image
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Decoded image is too large to process safely.",
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The upload is not a valid, readable image.",
        ) from exc


@app.get("/", include_in_schema=False)
def user_interface() -> FileResponse:
    """Serve the local image-upload interface."""

    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/api-info", response_model=ServiceResponse, tags=["service"])
def service_info() -> ServiceResponse:
    """Return machine-readable service information."""

    return ServiceResponse(
        name="Image Quality Assessment API",
        status="running",
        documentation="/docs",
        analyze_endpoint="/analyze-quality",
    )


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Report whether the configured local model checkpoint is present."""

    model_path = resolve_model_path()
    model_available = model_path.is_file()
    return HealthResponse(
        status="ready" if model_available else "model_unavailable",
        model_available=model_available,
        model_path=str(model_path),
    )


@app.post(
    "/analyze-quality",
    response_model=QualityAnalysisResponse,
    tags=["analysis"],
    responses={
        400: {"description": "Empty or corrupted image"},
        413: {"description": "Upload or decoded image is too large"},
        415: {"description": "Unsupported or mismatched image type"},
        503: {"description": "Model checkpoint is unavailable or invalid"},
    },
)
async def analyze_quality(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP image (maximum 10 MiB)."),
) -> QualityAnalysisResponse:
    """Run overall quality inference for one validated image upload."""

    content_type = (file.content_type or "").lower().split(";", maxsplit=1)[0].strip()
    if content_type not in SUPPORTED_CONTENT_TYPES:
        await file.close()
        supported = ", ".join(sorted(SUPPORTED_CONTENT_TYPES))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported Content-Type. Use one of: {supported}.",
        )

    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB upload limit.",
        )

    image = _decode_image(data, content_type)
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model checkpoint is unavailable. Train the model or set "
                f"IQA_MODEL_PATH to a local checkpoint. {exc}"
            ),
        ) from exc
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The configured model checkpoint could not be loaded: {exc}",
        ) from exc

    try:
        prediction = predictor.predict(image)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image quality inference failed: {exc}",
        ) from exc

    return QualityAnalysisResponse(
        quality_score=prediction.quality_score,
        mos_equivalent=prediction.mos_equivalent,
        suitable=prediction.suitable,
        model_check=ModelQualityCheck(
            threshold=prediction.quality_threshold,
            passes=prediction.passes_quality_threshold,
        ),
        resolution_check=ResolutionCheck(
            minimum_width=prediction.minimum_width,
            minimum_height=prediction.minimum_height,
            passes=prediction.passes_resolution_check,
        ),
        image=ImageDetails(
            width=prediction.image_width,
            height=prediction.image_height,
            format=prediction.image_format,
        ),
    )


__all__ = ["app", "get_predictor"]
