"""
models/trained_models.py
--------------------------
Wrappers for the team's own trained models (denoising + 2x super-resolution),
trained from scratch on BSDS300 in Colab and exported to ONNX — see
RestoreAI_Train_Models.ipynb for the training code.

Both are loaded through OpenCV's DNN module (`cv2.dnn.readNetFromONNX`),
the same inference mechanism already used for the pretrained ESPCN model
in super_resolution.py, so no extra production dependency (e.g. PyTorch)
is needed to run them.

If the ONNX files aren't present in weights/, these wrappers report
themselves unavailable and the pipeline falls back to the classical
methods (Non-Local Means denoising, pretrained ESPCN) automatically —
see models/restoration_model.py.
"""

import os
import cv2
import numpy as np
from config import Config


class OnnxImageModel:
    """Shared loading/inference logic for a fully-convolutional BGR-in,
    BGR-out image model exported from PyTorch (NCHW, float32, [0,1]).
    """

    def __init__(self, weight_path: str, label: str):
        self.weight_path = weight_path
        self.label = label
        self._net = None

    def is_available(self) -> bool:
        return os.path.exists(self.weight_path)

    def _load(self):
        if self._net is None:
            if not self.is_available():
                raise FileNotFoundError(
                    f"{self.label} weights not found at {self.weight_path}. "
                    f"Train it with RestoreAI_Train_Models.ipynb and place the "
                    f".onnx file in weights/, or it will be skipped automatically."
                )
            self._net = cv2.dnn.readNetFromONNX(self.weight_path)
        return self._net

    def run(self, img_bgr: np.ndarray) -> np.ndarray:
        """img_bgr: HxWx3 uint8. Returns HxWx3 (or upscaled) uint8 BGR."""
        net = self._load()
        blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0 / 255.0, swapRB=False, crop=False)
        net.setInput(blob)
        out = net.forward()  # 1 x 3 x H x W, float32 in [0,1]
        out_img = out[0].transpose(1, 2, 0)
        out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
        return out_img


_denoise_instance = None
_sr_instance = None


def get_trained_denoise_model() -> OnnxImageModel:
    global _denoise_instance
    if _denoise_instance is None:
        _denoise_instance = OnnxImageModel(Config.TRAINED_DENOISE_MODEL_PATH, "Trained denoising model")
    return _denoise_instance


def get_trained_sr_model() -> OnnxImageModel:
    global _sr_instance
    if _sr_instance is None:
        _sr_instance = OnnxImageModel(Config.TRAINED_SR_MODEL_PATH, "Trained super-resolution model")
    return _sr_instance
