"""
backend/config.py
-----------------
Configuration management for RestoreAI HYPIR Backend API.
Reads configuration from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Base project directory is parent of backend/
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)


class Settings:
    PROJECT_NAME: str = "RestoreAI HYPIR API"
    VERSION: str = "1.0.0"

    # Server settings
    HOST: str = os.getenv("RESTORE_AI_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("RESTORE_AI_PORT", "8000"))
    API_URL: Optional[str] = os.getenv("RESTORE_AI_API_URL", None)
    API_KEY: Optional[str] = os.getenv("RESTORE_AI_API_KEY", None)

    # CORS
    _origins = os.getenv("RESTORE_AI_ALLOWED_ORIGINS", "*")
    ALLOWED_ORIGINS: List[str] = (
        [o.strip() for o in _origins.split(",") if o.strip()]
        if _origins != "*"
        else ["*"]
    )

    # Storage paths
    UPLOAD_DIR: Path = Path(os.getenv("RESTORE_AI_UPLOAD_DIR", str(BASE_DIR / "uploads")))
    OUTPUT_DIR: Path = Path(os.getenv("RESTORE_AI_OUTPUT_DIR", str(BASE_DIR / "outputs")))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))

    # HYPIR Model configuration
    HYPIR_REPO_DIR: Path = Path(os.getenv("HYPIR_REPO_DIR", str(BASE_DIR / "HYPIR")))
    HYPIR_CONFIG_PATH: Path = Path(
        os.getenv("HYPIR_CONFIG_PATH", str(BASE_DIR / "HYPIR" / "configs" / "sd2_gradio.yaml"))
    )
    HYPIR_WEIGHT_PATH: Path = Path(
        os.getenv("HYPIR_WEIGHT_PATH", str(BASE_DIR / "weights" / "HYPIR_sd2.pth"))
    )

    # Fallback Edge AI model (Real-ESRGAN ONNX)
    REAL_ESRGAN_PATH: Path = BASE_DIR / "weights" / "realesr-general-x4v3.onnx"

    @classmethod
    def ensure_directories(cls):
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.HYPIR_WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)


def update_env_file(key: str, value: str):
    """
    Safely updates or adds a key in the .env file without overwriting
    unrelated configuration variables or comments.
    """
    lines = []
    found = False
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"# {key}="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{key}={value}\n")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


settings = Settings()
settings.ensure_directories()
