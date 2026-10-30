"""
app.py
------
RestoreAI — Flask application entrypoint.
"""

import os
import traceback

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from utils.image_utils import (
    allowed_file, make_unique_filename, read_image, write_image,
    image_info, ensure_dirs,
)
from services.restoration import run_pipeline
from services.metrics import compute_metrics, metrics_unavailable_response
import database


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    ensure_dirs()

    try:
        database.init_db()
        app.config["DB_READY"] = True
    except RuntimeError as exc:
        app.config["DB_READY"] = False
        print(f"[RestoreAI] WARNING: database not ready — {exc}")
        print("[RestoreAI] History features will be unavailable until MySQL is configured. "
              "See README 'MySQL Setup'.")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/restore")
    def restore_page():
        return render_template("restore.html")

    @app.route("/result")
    def result_page():
        return render_template("result.html")

    @app.route("/history")
    def history_page():
        return render_template("history.html")

    @app.route("/about")
    def about_page():
        return render_template("about.html")

    @app.route("/api/hypir/status", methods=["GET"])
    def api_hypir_status():
        """Returns the live connection status of the Google Colab HYPIR server."""
        from services.hypir_client import HypirClient
        health = HypirClient.check_health(timeout=3.5)
        return jsonify(health)

    @app.route("/api/hypir/config", methods=["POST"])
    def api_hypir_config():
        """Updates the Colab tunnel URL and persists it to .env safely."""
        data = request.get_json(force=True, silent=True) or {}
        new_url = (data.get("api_url") or "").strip().rstrip("/")
        new_key = (data.get("api_key") or "").strip()

        if new_url and not (new_url.startswith("http://") or new_url.startswith("https://")):
            return jsonify({"error": "Invalid API URL. Must start with http:// or https://"}), 400

        Config.RESTORE_AI_API_URL = new_url
        os.environ["RESTORE_AI_API_URL"] = new_url

        # Persist to .env safely
        env_path = os.path.join(Config.BASE_DIR, ".env")
        if os.path.exists(env_path):
            lines = []
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = []
            found_url = False
            found_key = False
            for line in lines:
                if line.strip().startswith("RESTORE_AI_API_URL=") or line.strip().startswith("# RESTORE_AI_API_URL="):
                    new_lines.append(f"RESTORE_AI_API_URL={new_url}\n")
                    found_url = True
                elif new_key and (line.strip().startswith("RESTORE_AI_API_KEY=") or line.strip().startswith("# RESTORE_AI_API_KEY=")):
                    new_lines.append(f"RESTORE_AI_API_KEY={new_key}\n")
                    found_key = True
                else:
                    new_lines.append(line)
            if not found_url:
                new_lines.append(f"RESTORE_AI_API_URL={new_url}\n")
            if new_key and not found_key:
                new_lines.append(f"RESTORE_AI_API_KEY={new_key}\n")
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

        if new_key:
            Config.RESTORE_AI_API_KEY = new_key
            os.environ["RESTORE_AI_API_KEY"] = new_key

        from services.hypir_client import HypirClient
        health = HypirClient.check_health(timeout=4.0) if new_url else {
            "online": False,
            "configured": False,
            "status": "cleared",
            "message": "Colab HYPIR URL removed. System will use local edge engine.",
        }
        return jsonify({"success": True, "health": health})

    @app.route("/api/restore", methods=["POST"])
    def api_restore():
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type. Use JPG, JPEG, PNG, or WEBP."}), 400

        mode = request.form.get("mode", "automatic")
        quality = request.form.get("quality", "standard")

        upload_name = make_unique_filename(file.filename)
        upload_path = os.path.join(Config.UPLOAD_FOLDER, upload_name)
        file.save(upload_path)

        try:
            original_img = read_image(upload_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            result = run_pipeline(original_img, mode=mode, quality=quality)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Restoration failed: {exc}"}), 500

        restored_name = "restored_" + upload_name.rsplit(".", 1)[0] + ".png"
        restored_path = os.path.join(Config.OUTPUT_FOLDER, restored_name)
        write_image(restored_path, result["restored_image"])

        metrics = metrics_unavailable_response()
        if "reference" in request.files and request.files["reference"].filename:
            ref_file = request.files["reference"]
            if allowed_file(ref_file.filename):
                ref_name = make_unique_filename(ref_file.filename)
                ref_path = os.path.join(Config.UPLOAD_FOLDER, ref_name)
                ref_file.save(ref_path)
                try:
                    ref_img = read_image(ref_path)
                    metrics = compute_metrics(ref_img, result["restored_image"])
                except ValueError as exc:
                    metrics = metrics_unavailable_response()
                    metrics["reason"] = f"Reference image could not be read: {exc}"

        original_info = image_info(upload_path, original_img)
        restored_info = image_info(restored_path, result["restored_image"])

        return jsonify({
            "original_url": f"/api/download/{upload_name}?folder=uploads",
            "restored_url": f"/api/download/{restored_name}?folder=outputs",
            "original_filename": file.filename,
            "original_path": upload_path,
            "restored_path": restored_path,
            "original_info": original_info,
            "restored_info": restored_info,
            "mode": mode,
            "quality": quality,
            "engine": result.get("engine", "RestoreAI Hybrid Engine"),
            "steps": result["steps"],
            "analysis": result["analysis"],
            "processing_time_sec": result["processing_time_sec"],
            "metrics": metrics,
        })

    @app.route("/api/evaluate", methods=["POST"])
    def api_evaluate():
        if "reference" not in request.files or "restored_path" not in request.form:
            return jsonify({"error": "reference image and restored_path are required."}), 400

        ref_file = request.files["reference"]
        restored_path = request.form["restored_path"]

        if not allowed_file(ref_file.filename):
            return jsonify({"error": "Unsupported reference image type."}), 400
        if not os.path.exists(restored_path):
            return jsonify({"error": "Restored image not found. Please restore an image first."}), 404

        ref_name = make_unique_filename(ref_file.filename)
        ref_path = os.path.join(Config.UPLOAD_FOLDER, ref_name)
        ref_file.save(ref_path)

        try:
            ref_img = read_image(ref_path)
            restored_img = read_image(restored_path)
            metrics = compute_metrics(ref_img, restored_img)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify({"metrics": metrics})

    @app.route("/api/history", methods=["GET"])
    def api_history_list():
        if not app.config.get("DB_READY"):
            return jsonify({"error": "Database not configured. See README 'MySQL Setup'."}), 503
        search = request.args.get("search")
        sort = request.args.get("sort", "created_at")
        order = request.args.get("order", "desc")
        try:
            rows = database.get_history(search=search, sort=sort, order=order)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        return jsonify({"history": rows})

    @app.route("/api/history", methods=["POST"])
    def api_history_create():
        if not app.config.get("DB_READY"):
            return jsonify({"error": "Database not configured. See README 'MySQL Setup'."}), 503
        data = request.get_json(force=True, silent=True) or {}
        required = ["original_filename", "original_path", "restored_path", "restoration_mode",
                    "original_width", "original_height", "restored_width", "restored_height",
                    "original_size_kb", "restored_size_kb", "processing_time_sec"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        try:
            new_id = database.insert_history(data)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        return jsonify({"id": new_id}), 201

    @app.route("/api/history/<int:item_id>", methods=["GET"])
    def api_history_get(item_id):
        if not app.config.get("DB_READY"):
            return jsonify({"error": "Database not configured. See README 'MySQL Setup'."}), 503
        row = database.get_history_item(item_id)
        if not row:
            return jsonify({"error": "Not found."}), 404
        return jsonify(row)

    @app.route("/api/history/<int:item_id>", methods=["DELETE"])
    def api_history_delete(item_id):
        if not app.config.get("DB_READY"):
            return jsonify({"error": "Database not configured. See README 'MySQL Setup'."}), 503
        deleted = database.delete_history_item(item_id)
        if not deleted:
            return jsonify({"error": "Not found."}), 404
        return jsonify({"deleted": True})

    @app.route("/api/download/<path:filename>")
    def api_download(filename):
        folder = request.args.get("folder", "outputs")
        directory = Config.OUTPUT_FOLDER if folder == "outputs" else Config.UPLOAD_FOLDER
        as_attachment = request.args.get("attachment", "false") == "true"
        return send_from_directory(directory, filename, as_attachment=as_attachment)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(e):
        return jsonify({"error": "File too large. Maximum upload size is 16 MB."}), 413

    @app.errorhandler(404)
    def handle_404(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Endpoint not found."}), 404
        return render_template("index.html"), 404

    @app.errorhandler(500)
    def handle_500(e):
        return jsonify({"error": "Internal server error."}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=Config.DEBUG)
