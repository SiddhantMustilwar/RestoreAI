"""
models/super_resolution.py
----------------------------
High-performance Super-Resolution and Clarity Engine for RestoreAI.

Supports:
  1. Real-ESRGAN (Generative Adversarial Network / ONNX)
     - SOTA generative restoration: reconstructs lost micro-textures, sharpens edges,
       and removes artifacts.
     - Tiled inference engine for low memory footprint and arbitrary image sizes.
     - Anti-aliased super-sampling for 2x, 3x, and 4x scales.
  2. FSRCNN (Fast Super-Resolution CNN)
     - OpenCV `dnn_superres` with pretrained FSRCNN weights.
  3. ESPCN (Efficient Sub-Pixel CNN)
     - Classical pretrained fallback weights.
  4. Adaptive High-Fidelity Edge Upscaler
     - Zero-dependency mathematical fallback ensuring images are always crisp.
"""

import os
import cv2
import numpy as np
from config import Config

_SUPPORTED_SCALES = (2, 3, 4)


class RealEsrganEngine:
    """Tiled inference engine for Real-ESRGAN ONNX."""

    def __init__(self, weight_path: str):
        self.weight_path = weight_path
        self._net = None

    def is_available(self) -> bool:
        return os.path.exists(self.weight_path)

    def _load(self):
        if self._net is None:
            if not self.is_available():
                raise FileNotFoundError(f"Real-ESRGAN weights not found at {self.weight_path}")
            self._net = cv2.dnn.readNetFromONNX(self.weight_path)
        return self._net

    def run_tiled(self, img_bgr: np.ndarray, tile_size: int = 256, tile_pad: int = 16) -> np.ndarray:
        """Runs 4x Real-ESRGAN inference using overlapping tiles to avoid RAM spikes."""
        net = self._load()
        h, w = img_bgr.shape[:2]
        model_scale = 4

        out_h = h * model_scale
        out_w = w * model_scale
        out_bgr = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # Real-ESRGAN expects normalized RGB in [0.0, 1.0]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y1 = max(0, y - tile_pad)
                y2 = min(h, y + tile_size + tile_pad)
                x1 = max(0, x - tile_pad)
                x2 = min(w, x + tile_size + tile_pad)

                tile = rgb[y1:y2, x1:x2]
                blob = np.transpose(tile, (2, 0, 1))[np.newaxis, ...]
                net.setInput(blob)
                tile_out = net.forward()[0]
                tile_out = np.transpose(tile_out, (1, 2, 0))
                tile_out = np.clip(tile_out * 255.0, 0, 255).astype(np.uint8)
                tile_out_bgr = cv2.cvtColor(tile_out, cv2.COLOR_RGB2BGR)

                crop_top = (y - y1) * model_scale
                crop_bottom = crop_top + min(tile_size, h - y) * model_scale
                crop_left = (x - x1) * model_scale
                crop_right = crop_left + min(tile_size, w - x) * model_scale

                tile_crop = tile_out_bgr[crop_top:crop_bottom, crop_left:crop_right]

                out_y1 = y * model_scale
                out_y2 = out_y1 + tile_crop.shape[0]
                out_x1 = x * model_scale
                out_x2 = out_x1 + tile_crop.shape[1]

                out_bgr[out_y1:out_y2, out_x1:out_x2] = tile_crop

        return out_bgr


class SuperResolutionModel:
    def __init__(self):
        self._realesrgan = RealEsrganEngine(Config.REAL_ESRGAN_MODEL_PATH)
        self._dnn_engines = {}  # key: (model_name, scale)

    def _get_dnn_engine(self, model_name: str, scale: int):
        key = (model_name, scale)
        if key in self._dnn_engines:
            return self._dnn_engines[key]

        if model_name == "fsrcnn":
            path = Config.FSRCNN_MODEL_PATH.format(scale=scale)
        elif model_name == "espcn":
            path = Config.SR_MODEL_PATH.format(scale=scale)
        else:
            raise ValueError(f"Unknown DNN model: {model_name}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"{model_name.upper()} weights not found at {path}")

        engine = cv2.dnn_superres.DnnSuperResImpl_create()
        engine.readModel(path)
        engine.setModel(model_name, scale)
        self._dnn_engines[key] = engine
        return engine

    def is_available(self, scale: int = 2) -> bool:
        if self._realesrgan.is_available():
            return True
        fsrcnn_path = Config.FSRCNN_MODEL_PATH.format(scale=scale)
        if os.path.exists(fsrcnn_path):
            return True
        espcn_path = Config.SR_MODEL_PATH.format(scale=scale)
        if os.path.exists(espcn_path):
            return True
        return True  # Adaptive fallback is always available

    def upscale(self, img: np.ndarray, scale: int = 2, prefer: str = "auto") -> tuple:
        """
        Upscales image by scale factor.
        Returns: (upscaled_image, method_label)
        """
        if scale not in _SUPPORTED_SCALES:
            scale = 2

        # 0. Check HYPIR (Diffusion-Yielded Score Priors) if available
        try:
            from models.hypir_model import get_hypir_model
            hypir = get_hypir_model()
            if hypir.is_available():
                return hypir.run(img, scale=scale), f"HYPIR Diffusion Score Prior ({scale}x)"
        except Exception:
            pass

        # 1. Try Real-ESRGAN (Generative AI / GAN)
        if prefer in ("auto", "realesrgan") and self._realesrgan.is_available():
            try:
                # Real-ESRGAN generates a full 4x super-resolved output
                hr_4x = self._realesrgan.run_tiled(img)
                if scale == 4:
                    return hr_4x, "Real-ESRGAN Generative AI (GAN 4x)"
                elif scale == 2:
                    # Super-sampling down to 2x gives exceptional sharpness and zero aliasing
                    target_w = img.shape[1] * 2
                    target_h = img.shape[0] * 2
                    hr_2x = cv2.resize(hr_4x, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    return hr_2x, "Real-ESRGAN Generative AI (Super-Sampled 2x)"
                elif scale == 3:
                    target_w = img.shape[1] * 3
                    target_h = img.shape[0] * 3
                    hr_3x = cv2.resize(hr_4x, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    return hr_3x, "Real-ESRGAN Generative AI (Super-Sampled 3x)"
            except Exception as e:
                print(f"[RestoreAI] Real-ESRGAN inference notice: {e}, falling back.")

        # 2. Try FSRCNN
        fsrcnn_path = Config.FSRCNN_MODEL_PATH.format(scale=scale)
        if os.path.exists(fsrcnn_path):
            try:
                engine = self._get_dnn_engine("fsrcnn", scale)
                return engine.upsample(img), f"FSRCNN deep super-resolution ({scale}x)"
            except Exception as e:
                print(f"[RestoreAI] FSRCNN error: {e}, trying next fallback.")

        # 3. Try ESPCN
        espcn_path = Config.SR_MODEL_PATH.format(scale=scale)
        if os.path.exists(espcn_path):
            try:
                engine = self._get_dnn_engine("espcn", scale)
                return engine.upsample(img), f"Pretrained ESPCN ({scale}x)"
            except Exception as e:
                print(f"[RestoreAI] ESPCN error: {e}, using adaptive upscaler.")

        # 4. High-Fidelity Edge-Adaptive Upscaler (Ensures crisp results always)
        target_w = img.shape[1] * scale
        target_h = img.shape[0] * scale
        base = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Detail injection via Laplacian high-pass
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap = np.clip(lap, -25, 25)
        for c in range(3):
            base[:, :, c] = np.clip(base[:, :, c].astype(np.float32) + 0.35 * lap, 0, 255).astype(np.uint8)

        return base, f"High-fidelity adaptive super-resolution ({scale}x)"


_model_instance = None


def get_super_resolution_model() -> SuperResolutionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = SuperResolutionModel()
    return _model_instance
