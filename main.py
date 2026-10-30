"""
backend/main.py
----------------
FastAPI entrypoint for RestoreAI HYPIR Restoration Service.
Provides RESTful endpoints for health, model metadata, image restoration,
and file downloads.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
    Security,
    status,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery

from backend.config import settings
from backend.hypir_service import HypirService

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-grade REST API for RestoreAI image restoration powered by "
        "HYPIR (Harnessing Diffusion-Yielded Score Priors) and Real-ESRGAN GAN."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
):
    """Verifies API key if configured in settings."""
    if not settings.API_KEY:
        return True  # Authentication disabled if no key configured
    key = header_key or query_key
    if key == settings.API_KEY:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key. Pass 'X-API-Key' header or 'api_key' query parameter.",
    )


@app.get("/", tags=["Info"])
async def root():
    """Returns basic service information and endpoint links."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "restore": "POST /api/restore",
            "info": "GET /api/restore/info",
            "download": "GET /api/download/{filename}",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Returns service health status, model availability, and hardware details.
    """
    status_info = HypirService.get_status()
    http_code = (
        status.HTTP_200_OK
        if status_info["status"] == "ready"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=http_code, content=status_info)


@app.get("/api/restore/info", tags=["Restoration"])
async def model_info():
    """Returns detailed model metadata, supported parameters, and config."""
    status_info = HypirService.get_status()
    return {
        "service": settings.PROJECT_NAME,
        "active_engine": status_info["active_engine"],
        "supported_upscale_factors": [2, 3, 4],
        "default_patch_size": 512,
        "supported_formats": ["image/jpeg", "image/png", "image/webp"],
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        "device": status_info["hardware"]["device"],
        "gpu_name": status_info["hardware"]["gpu_name"],
    }


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}


@app.post("/api/restore", tags=["Restoration"])
async def restore_image_endpoint(
    request: Request,
    image: UploadFile = File(..., description="Degraded input image (JPG, PNG, WEBP)"),
    upscale: int = Form(4, description="Upscaling factor (2, 3, or 4)"),
    patch_size: int = Form(512, description="HYPIR processing patch size"),
    stride: int = Form(256, description="HYPIR processing stride"),
    seed: int = Form(231, description="Random seed"),
    prompt: Optional[str] = Form(None, description="Optional text guidance prompt"),
    download: bool = Form(False, description="Directly stream file download"),
    authenticated: bool = Security(verify_api_key),
):
    """
    Upload an image for generative AI restoration.
    Returns JSON with restoration metadata and download URL, or streams file directly.
    """
    if upscale not in (2, 3, 4):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported upscale factor: {upscale}. Allowed values: [2, 3, 4].",
        )

    # Validate file extension
    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {list(ALLOWED_EXTENSIONS)}",
        )

    # Save uploaded file safely with unique token
    token = f"{os.urandom(6).hex()}_{Path(image.filename).name}"
    safe_upload_path = settings.UPLOAD_DIR / token

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_bytes = 0

    try:
        with open(safe_upload_path, "wb") as buffer:
            while chunk := await image.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Image exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                    )
                buffer.write(chunk)

        # Run inference through service layer
        result = await HypirService.restore_image(
            input_image_path=safe_upload_path,
            upscale=upscale,
            patch_size=patch_size,
            stride=stride,
            seed=seed,
            prompt=prompt,
        )

        base_url = str(request.base_url).rstrip("/")
        download_url = f"{base_url}/api/download/{result['filename']}"
        result["download_url"] = download_url
        result["original_filename"] = image.filename

        if download:
            return FileResponse(
                path=result["output_path"],
                filename=result["filename"],
                media_type="image/png",
            )

        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(fe))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    finally:
        # Clean up temporary uploaded file
        if safe_upload_path.exists():
            try:
                safe_upload_path.unlink()
            except Exception:
                pass


@app.get("/api/download/{filename}", tags=["Restoration"])
async def download_file(filename: str):
    """
    Secure file download endpoint preventing directory traversal attacks.
    """
    # Strict path traversal protection
    safe_name = os.path.basename(filename)
    target_path = (settings.OUTPUT_DIR / safe_name).resolve()

    if not target_path.exists() or not str(target_path).startswith(str(settings.OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Restored file not found.")

    return FileResponse(path=target_path, filename=safe_name, media_type="image/png")
