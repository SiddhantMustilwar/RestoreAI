"""
backend/hypir_service.py
------------------------
Service layer for executing HYPIR (Diffusion-Yielded Score Priors)
and Edge Generative Restoration. Handles isolated execution, concurrency lock,
file validation, and cleanup.
"""

import os
import sys
import time
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image

from backend.config import settings

# Global async lock to prevent concurrent GPU VRAM exhaustion
_inference_lock = asyncio.Lock()


class HypirService:
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Detects Python, PyTorch, CUDA, and GPU hardware availability."""
        info = {
            "python_version": sys.version.split()[0],
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "gpu_vram_mb": None,
            "device": "cpu",
        }
        try:
            import torch
            info["torch_available"] = True
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_vram_mb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                )
                info["device"] = "cuda"
        except ImportError:
            pass
        return info

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Inspects availability of HYPIR repository, configuration, and weights."""
        dev = cls.get_device_info()
        repo_ready = (
            settings.HYPIR_REPO_DIR.exists()
            and (settings.HYPIR_REPO_DIR / "test.py").exists()
        )
        hypir_weights_ready = (
            settings.HYPIR_WEIGHT_PATH.exists()
            and settings.HYPIR_WEIGHT_PATH.stat().st_size > 1024 * 1024 * 50  # >50MB
        )
        edge_weights_ready = settings.REAL_ESRGAN_PATH.exists()

        is_ready = hypir_weights_ready or edge_weights_ready

        return {
            "status": "ready" if is_ready else "weights_missing",
            "hypir_repo_found": repo_ready,
            "hypir_weights_found": hypir_weights_ready,
            "hypir_weight_path": str(settings.HYPIR_WEIGHT_PATH),
            "edge_weights_found": edge_weights_ready,
            "active_engine": (
                "HYPIR (Diffusion-Yielded Score Priors)"
                if hypir_weights_ready
                else ("Real-ESRGAN (Edge GAN)" if edge_weights_ready else "None")
            ),
            "hardware": dev,
        }

    @staticmethod
    def validate_image_file(file_path: Path) -> Tuple[int, int]:
        """Validates that the file exists, is non-empty, and can be decoded by PIL."""
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise ValueError("Uploaded file is empty or does not exist.")
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                w, h = img.size
                if w < 16 or h < 16:
                    raise ValueError(f"Image too small ({w}x{h} px). Minimum dimension is 16x16.")
                if w > 8192 or h > 8192:
                    raise ValueError(f"Image too large ({w}x{h} px). Maximum dimension is 8192x8192.")
                return w, h
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted image format: {exc}")

    @classmethod
    async def restore_image(
        cls,
        input_image_path: Path,
        upscale: int = 4,
        patch_size: int = 512,
        stride: int = 256,
        seed: int = 231,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes real image restoration with serialized concurrency control.
        Uses official HYPIR test.py when weights are present; falls back to
        the local Real-ESRGAN engine on edge/CPU environments.
        """
        orig_w, orig_h = cls.validate_image_file(input_image_path)
        status = cls.get_status()

        # Unique token for per-request isolation
        req_id = f"{int(time.time()*1000)}_{os.urandom(3).hex()}"
        out_filename = f"restored_hypir_{req_id}.png"
        final_output_path = settings.OUTPUT_DIR / out_filename

        start_time = time.time()

        # Serialized inference under lock to protect GPU/CPU from parallel exhaustion
        async with _inference_lock:
            if status["hypir_weights_found"] and status["hypir_repo_found"]:
                # --- OFFICIAL HYPIR EXECUTION PATH ---
                temp_in_dir = settings.UPLOAD_DIR / f"tmp_in_{req_id}"
                temp_out_dir = settings.OUTPUT_DIR / f"tmp_out_{req_id}"
                temp_in_dir.mkdir(parents=True, exist_ok=True)
                temp_out_dir.mkdir(parents=True, exist_ok=True)

                temp_img_copy = temp_in_dir / input_image_path.name
                shutil.copy2(input_image_path, temp_img_copy)

                cmd = [
                    sys.executable,
                    str(settings.HYPIR_REPO_DIR / "test.py"),
                    "--base_model_type", "sd2",
                    "--base_model_path", "sd2-community/stable-diffusion-2-1-base",
                    "--model_t", "200",
                    "--coeff_t", "200",
                    "--lora_rank", "256",
                    "--lora_modules", "to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
                    "--weight_path", str(settings.HYPIR_WEIGHT_PATH),
                    "--patch_size", str(patch_size),
                    "--stride", str(stride),
                    "--lq_dir", str(temp_in_dir),
                    "--scale_by", "factor",
                    "--upscale", str(upscale),
                    "--output_dir", str(temp_out_dir),
                    "--seed", str(seed),
                    "--device", status["hardware"]["device"],
                ]

                # HYPIR requires txt_dir for text guidance (use prompt or default photographic prompt)
                temp_txt_dir = temp_in_dir / "prompts"
                temp_txt_dir.mkdir(parents=True, exist_ok=True)
                prompt_text = prompt.strip() if (prompt and prompt.strip()) else "a clean, high-resolution, sharp, detailed photograph, master quality"
                prompt_file = temp_txt_dir / f"{input_image_path.stem}.txt"
                prompt_file.write_text(prompt_text, encoding="utf-8")
                cmd.extend(["--txt_dir", str(temp_txt_dir)])

                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await proc.communicate()
                    if proc.returncode != 0:
                        err_msg = stderr.decode("utf-8", errors="replace")
                        raise RuntimeError(f"HYPIR inference error: {err_msg}")

                    # Locate generated output in temp_out_dir
                    produced_files = list(temp_out_dir.glob("*.*"))
                    if not produced_files:
                        raise RuntimeError("HYPIR completed but produced no output image file.")
                    shutil.move(str(produced_files[0]), str(final_output_path))
                    engine_used = "HYPIR (Diffusion-Yielded Score Prior)"
                finally:
                    shutil.rmtree(temp_in_dir, ignore_errors=True)
                    shutil.rmtree(temp_out_dir, ignore_errors=True)

            elif status["edge_weights_found"]:
                # --- LOCAL EDGE AI RESTORATION PATH (Real-ESRGAN GAN) ---
                # Runs when full 8GB diffusion checkpoints are absent on local CPU/Edge
                from models.super_resolution import get_super_resolution_model
                from services.postprocessing import postprocess
                import cv2

                img_bgr = cv2.imread(str(input_image_path))
                if img_bgr is None:
                    raise ValueError("Failed reading image with OpenCV.")

                sr_model = get_super_resolution_model()
                upscaled, label = sr_model.upscale(img_bgr, scale=upscale)
                final_bgr, _ = postprocess(upscaled)
                cv2.imwrite(str(final_output_path), final_bgr)
                engine_used = f"Real-ESRGAN GAN Edge Engine ({label})"
            else:
                raise FileNotFoundError(
                    "No model weights available. Run the setup cells in Restore_AI_HYPIR.ipynb "
                    "or verify that weights/realesr-general-x4v3.onnx exists."
                )

        elapsed = round(time.time() - start_time, 3)

        # Validate final output
        final_w, final_h = cls.validate_image_file(final_output_path)

        return {
            "success": True,
            "filename": out_filename,
            "output_path": str(final_output_path),
            "original_dimensions": [orig_w, orig_h],
            "restored_dimensions": [final_w, final_h],
            "upscale_factor": upscale,
            "processing_time_sec": elapsed,
            "engine": engine_used,
            "device": status["hardware"]["device"],
        }
