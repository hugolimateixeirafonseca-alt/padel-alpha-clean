"""
padel-alpha-clean  (v14 - extra safe margins for mockup crop avoidance)
----------------------------------------------------------------------------
POST /clean  (JSON body)
Limpa/upscale um PNG transparente e devolve um showcase_bg coerente com o
resultado esperado do design. Agora tambem faz auto-trim da transparencia e
reaplica uma margem pequena controlada para que o design ocupe mais area util.

Body:
{
  "url": "https://...png",
  "imgbb_key": "....",
  "scale": 2,
  "threshold": 0,
  "erode": 0,
  "keyline": 0,
  "garment_set": "light" | "dark",
  "auto_trim": true,
  "padding_pct": 14,
  "padding_x_pct": 22,
  "padding_y_pct": 14
}

Regras visuais:
- garment_set = light -> fundo BRANCO
- garment_set = dark  -> fundo ESCURO
- auto_trim = true    -> corta transparencia exterior e reaplica padding
- padding_pct         -> margem uniforme de fallback (%)
- padding_x_pct       -> margem horizontal (%), recomendado 20-24 para evitar corte lateral
- padding_y_pct       -> margem vertical (%), recomendado 12-16
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
DEFAULT_PADDING_PCT = 14.0
DEFAULT_PADDING_X_PCT = 22.0
DEFAULT_PADDING_Y_PCT = 14.0


def pick_showcase_bg(garment_set):
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


def _alpha_bbox(img, alpha_threshold=1):
    arr = np.asarray(img)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha >= alpha_threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _trim_and_pad(img, padding_pct=DEFAULT_PADDING_PCT, padding_x_pct=None, padding_y_pct=None, alpha_threshold=1):
    bbox = _alpha_bbox(img, alpha_threshold=alpha_threshold)
    if bbox is None:
        return img
    left, top, right, bottom = bbox
    cropped = img.crop((left, top, right, bottom))
    cw, ch = cropped.size
    longest = max(cw, ch)
    if padding_x_pct is None:
        padding_x_pct = padding_pct
    if padding_y_pct is None:
        padding_y_pct = padding_pct
    pad_x = max(1, round(longest * max(0.0, float(padding_x_pct)) / 100.0))
    pad_y = max(1, round(longest * max(0.0, float(padding_y_pct)) / 100.0))
    new_w = cw + pad_x * 2
    new_h = ch + pad_y * 2
    out = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    out.paste(cropped, (pad_x, pad_y), cropped)
    return out


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean", "ver": 14})


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
    auto_trim = bool(data.get("auto_trim", True))
    try:
        padding_pct = float(data.get("padding_pct", DEFAULT_PADDING_PCT))
    except Exception:
        padding_pct = DEFAULT_PADDING_PCT
    padding_pct = min(max(padding_pct, 0.0), 24.0)
    try:
        padding_x_pct = float(data.get("padding_x_pct", padding_pct if "padding_pct" in data else DEFAULT_PADDING_X_PCT))
    except Exception:
        padding_x_pct = DEFAULT_PADDING_X_PCT
    try:
        padding_y_pct = float(data.get("padding_y_pct", padding_pct if "padding_pct" in data else DEFAULT_PADDING_Y_PCT))
    except Exception:
        padding_y_pct = DEFAULT_PADDING_Y_PCT
    padding_x_pct = min(max(padding_x_pct, 0.0), 30.0)
    padding_y_pct = min(max(padding_y_pct, 0.0), 24.0)

    if not url or not imgbb_key:
        return jsonify({"error": "faltam 'url' e/ou 'imgbb_key'"}), 400
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
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

    # Auto-trim antes do upscale para reduzir espaco transparente e ganhar escala visual
    if auto_trim:
        img = _trim_and_pad(img, padding_pct=padding_pct, padding_x_pct=padding_x_pct, padding_y_pct=padding_y_pct, alpha_threshold=1)
        gc.collect()

    if scale and scale != 1.0:
        target_w, target_h = round(img.width * scale), round(img.height * scale)
        target_w, target_h = _fit_within(target_w, target_h, MAX_OUTPUT_PX)
        if (target_w, target_h) != (img.width, img.height):
            img = img.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)
            gc.collect()

    if threshold > 0 or erode_px > 0 or keyline_px > 0:
        import cv2
        arr = np.array(img)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        del arr
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
            out_rgb = rgb.copy()
            out_rgb[ring] = (255, 255, 255)
            out_alpha = (outer * 255).astype(np.uint8)
        else:
            out_rgb = rgb
            out_alpha = (opaque * 255).astype(np.uint8) if threshold > 0 else alpha
        img = Image.fromarray(np.dstack([out_rgb, out_alpha]), mode="RGBA")
        del rgb, alpha, opaque
        gc.collect()

    # Segunda passagem opcional de trim muito leve apos keyline/erode, para garantir margens consistentes
    if auto_trim:
        img = _trim_and_pad(img, padding_pct=padding_pct, padding_x_pct=padding_x_pct, padding_y_pct=padding_y_pct, alpha_threshold=1)
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
