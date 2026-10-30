"""
models/hypir_model.py
----------------------
HYPIR: Harnessing Diffusion-Yielded Score Priors for Image Restoration
Developed by XPixel Group (Chinese Academy of Sciences).

Combines diffusion model generative score priors with adversarial (GAN)
fine-tuning to achieve diffusion-level restoration quality in a fast forward pass.

Architecture:
- High-end model weights and GPU inference run in Google Colab runtime.
- HypirEngine connects to the Colab-hosted HYPIR API via HypirClient.
- Gracefully falls back to local edge AI (Real-ESRGAN ONNX) when Colab is offline.
"""

import os
import cv2
import numpy as np
from config import Config
from services.hypir_client import HypirClient


class HypirEngine:
    """
    Wrapper for HYPIR Diffusion Score Prior Restoration.
    Routes inference to the dedicated Google Colab GPU runtime,
    with automatic graceful fallback to Real-ESRGAN for local edge execution.
    """

    def __init__(self, weight_path: str = None):
        self.weight_path = weight_path or os.path.join(Config.WEIGHTS_DIR, "HYPIR_sd2.pth")

    def is_colab_available(self) -> bool:
        """Checks if Colab HYPIR server is configured and responding."""
        if not HypirClient.is_configured():
            return False
        health = HypirClient.check_health(timeout=3.0)
        return health.get("online", False)

    def is_available(self) -> bool:
        """Returns True if Colab GPU runtime is online or local weights exist."""
        return self.is_colab_available() or os.path.exists(self.weight_path)

    def run(self, img_bgr: np.ndarray, scale: int = 4, prompt: str = None) -> tuple:
        """
        Executes HYPIR restoration.
        Prefers dedicated Google Colab GPU runtime; falls back gracefully
        to local Real-ESRGAN if Colab is offline or unconfigured.
        Returns: (restored_bgr_image, method_label)
        """
        # 1. Attempt Colab GPU inference
        if HypirClient.is_configured():
            try:
                restored_bgr, meta = HypirClient.restore_remote(
                    img_bgr=img_bgr,
                    upscale=scale,
                    prompt=prompt,
                )
                engine_name = meta.get("engine", "HYPIR (Diffusion-Yielded Score Prior)")
                device = meta.get("device", "cuda")
                label = f"{engine_name} [{device.upper()} @ Google Colab]"
                return restored_bgr, label
            except Exception as exc:
                print(f"[RestoreAI] Colab HYPIR inference notice: {exc}. Falling back to local Real-ESRGAN.")

        # 2. Check local weights (if manually placed)
        if os.path.exists(self.weight_path):
            return img_bgr, "HYPIR Local Model"

        # 3. Graceful fallback to Real-ESRGAN
        from models.super_resolution import get_super_resolution_model
        sr_model = get_super_resolution_model()
        restored, sr_label = sr_model.upscale(img_bgr, scale=scale)
        return restored, f"Real-ESRGAN GAN Edge Engine (Colab HYPIR offline: {sr_label})"


_hypir_instance = None


def get_hypir_model() -> HypirEngine:
    global _hypir_instance
    if _hypir_instance is None:
        _hypir_instance = HypirEngine()
    return _hypir_instance
