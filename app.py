"""
padel-alpha-clean  (v10 - dynamic showcase_bg via garment_set, light/dark only)
------------------------------------------------------------------------------
POST /clean  (JSON body)
Limpa/upscale um PNG transparente e devolve um showcase_bg coerente com o
resultado esperado do design:
  - garment_set = light -> fundo BRANCO
  - garment_set = dark  -> fundo ESCURO

Body:
{
  "url":       "https://...png",
  "imgbb_key": "....",
  "scale":     4,
  "threshold": 0,
  "erode":     0,
  "keyline":   0,
  "garment_set": "light" | "dark"
}
Resposta: {"url": "https://i.ibb.co/...", "showcase_bg": "#ffffff"}  ou
          {"url": "https://i.ibb.co/...", "showcase_bg": "#1a1a1a"}
"""
import io, gc
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
app = Flask(__name__)
Image.MAX_IMAGE_PIXELS = None

MAX_OUTPUT_PX = 4000
MAX_INPUT_PX = 3000


def pick_showcase_bg(garment_set):
    """light -> branco, dark -> escuro; fallback seguro -> branco."""
    gs = (garment_set or "").strip().lower()
    if gs == "dark":
        return "#1a1a1a"
    return "#ffffff"


def _fit_within(w, h, max_side):
    longest = max(w, h)
    if longest <= max_side:
        return w, h
    f = max_side / float(longest)
    return max(1, round(w * f)), max(1, round(h * f))


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean", "ver": 10})


@app.route("/clean", methods=["POST"])
def clean():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    imgbb_key = data.get("imgbb_key")
    scale = float(data.get("scale", 1))
    threshold = int(data.get("threshold", 0))
    erode_px = int(data.get("erode", 0))
    keyline_px = int(data.get("keyline", 0))
    garment_set = data.get("garment_set", "")
    if not url or not imgbb_key:
        return jsonify({"error": "faltam 'url' e/ou 'imgbb_key'"}), 400
    try:
        r = requests.get(url, timeout=120); r.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"download falhou: {e}"}), 502
    try:
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception as e:
        return jsonify({"error": f"abrir imagem falhou: {e}"}), 422
    finally:
        r = None

    showcase_bg = pick_showcase_bg(garment_set)

    iw, ih = _fit_within(img.width, img.height, MAX_INPUT_PX)
    if (iw, ih) != img.size:
        img = img.resize((iw, ih), Image.LANCZOS)
        gc.collect()

    if scale and scale != 1.0:
        target_w, target_h = round(img.width * scale), round(img.height * scale)
        target_w, target_h = _fit_within(target_w, target_h, MAX_OUTPUT_PX)
        if (target_w, target_h) != (img.width, img.height):
            img = img.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)
            gc.collect()

    if threshold > 0 or erode_px > 0 or keyline_px > 0:
        import cv2
        arr = np.array(img); rgb = arr[:, :, :3]; alpha = arr[:, :, 3]; del arr
        if threshold > 0:
            opaque = (alpha >= threshold).astype(np.uint8)
        else:
            opaque = (alpha > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        if erode_px > 0:
            opaque = cv2.erode(opaque, kernel, iterations=erode_px)
        if keyline_px > 0:
            outer = cv2.dilate(opaque, kernel, iterations=keyline_px)
            ring = (outer & (1 - opaque)).astype(bool)
            out_rgb = rgb.copy(); out_rgb[ring] = (255, 255, 255)
            out_alpha = (outer * 255).astype(np.uint8)
        else:
            out_rgb = rgb
            out_alpha = (opaque * 255).astype(np.uint8) if threshold > 0 else alpha
        img = Image.fromarray(np.dstack([out_rgb, out_alpha]), mode="RGBA")
        del rgb, alpha, opaque
        gc.collect()

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    img = None
    gc.collect()
    buf.seek(0)
    try:
        up = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": imgbb_key},
            files={"image": ("design.png", buf, "image/png")},
            timeout=120,
        )
        up.raise_for_status()
        out_url = up.json()["data"]["url"]
    except Exception as e:
        return jsonify({"error": f"upload ImgBB falhou: {e}"}), 502
    finally:
        buf = None
        gc.collect()

    return jsonify({"url": out_url, "showcase_bg": showcase_bg})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
