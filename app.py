"""
padel-alpha-clean (v15 - fixed canvas safe area)
------------------------------------------------
Keeps the output PNG on a fixed transparent canvas instead of exporting a cropped
variable-size PNG. This avoids Gelato/mockup cropping when the generated artwork
is too close to the edge or when the placeholder uses a different fit mode.

New recommended body fields:
{
  "fit_to_canvas": true,
  "max_art_width_pct": 54,
  "max_art_height_pct": 68,
  "auto_trim": false
}
"""
import io
import gc
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)
Image.MAX_IMAGE_PIXELS = None

MAX_OUTPUT_PX = 4000
MAX_INPUT_PX = 3000


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


def _trim_and_pad(img, padding_pct=12.0, alpha_threshold=1):
    bbox = _alpha_bbox(img, alpha_threshold=alpha_threshold)
    if bbox is None:
        return img
    left, top, right, bottom = bbox
    cropped = img.crop((left, top, right, bottom))
    cw, ch = cropped.size
    longest = max(cw, ch)
    pad = max(1, round(longest * max(0.0, float(padding_pct)) / 100.0))
    out = Image.new("RGBA", (cw + pad * 2, ch + pad * 2), (0, 0, 0, 0))
    out.paste(cropped, (pad, pad), cropped)
    return out


def _fit_artwork_to_fixed_canvas(
    img,
    max_art_width_pct=54.0,
    max_art_height_pct=68.0,
    alpha_threshold=1,
):
    """
    Crop transparent excess around the actual art, then place the art back onto
    a fixed canvas with the SAME dimensions/aspect ratio as the input image.
    This is safer for Gelato templates than returning a variable-size cropped PNG.
    """
    bbox = _alpha_bbox(img, alpha_threshold=alpha_threshold)
    if bbox is None:
        return img

    canvas_w, canvas_h = img.size
    left, top, right, bottom = bbox
    art = img.crop((left, top, right, bottom))
    art_w, art_h = art.size

    safe_w = canvas_w * (max(1.0, min(float(max_art_width_pct), 95.0)) / 100.0)
    safe_h = canvas_h * (max(1.0, min(float(max_art_height_pct), 95.0)) / 100.0)

    factor = min(safe_w / art_w, safe_h / art_h)
    # Do not let accidental huge enlargement create soft artwork.
    factor = min(factor, 1.25)

    new_w = max(1, round(art_w * factor))
    new_h = max(1, round(art_h * factor))
    if (new_w, new_h) != art.size:
        art = art.resize((new_w, new_h), Image.LANCZOS)

    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    out.paste(art, (x, y), art)
    return out


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean", "ver": 15})


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

    fit_to_canvas = bool(data.get("fit_to_canvas", False))
    auto_trim = bool(data.get("auto_trim", False))

    try:
        max_art_width_pct = float(data.get("max_art_width_pct", 54))
    except Exception:
        max_art_width_pct = 54.0

    try:
        max_art_height_pct = float(data.get("max_art_height_pct", 68))
    except Exception:
        max_art_height_pct = 68.0

    try:
        padding_pct = float(data.get("padding_pct", 12))
    except Exception:
        padding_pct = 12.0

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
        gc.collect()

    showcase_bg = pick_showcase_bg(garment_set)

    # Input guard
    iw, ih = _fit_within(img.width, img.height, MAX_INPUT_PX)
    if (iw, ih) != img.size:
        img = img.resize((iw, ih), Image.LANCZOS)
        gc.collect()

    # Main safe mode for Gelato: keep fixed canvas, fit art inside safe center.
    if fit_to_canvas:
        img = _fit_artwork_to_fixed_canvas(
            img,
            max_art_width_pct=max_art_width_pct,
            max_art_height_pct=max_art_height_pct,
            alpha_threshold=1,
        )
        gc.collect()
    elif auto_trim:
        img = _trim_and_pad(img, padding_pct=padding_pct, alpha_threshold=1)
        gc.collect()

    # Optional alpha hardening/keyline.
    if threshold > 0 or erode_px > 0 or keyline_px > 0:
        import cv2

        arr = np.array(img)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        del arr

        opaque = (alpha >= threshold).astype(np.uint8) if threshold > 0 else (alpha > 0).astype(np.uint8)
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

        if fit_to_canvas:
            img = _fit_artwork_to_fixed_canvas(
                img,
                max_art_width_pct=max_art_width_pct,
                max_art_height_pct=max_art_height_pct,
                alpha_threshold=1,
            )
            gc.collect()

    # Upscale while preserving the fixed canvas ratio.
    if scale and scale != 1.0:
        target_w, target_h = round(img.width * scale), round(img.height * scale)
        target_w, target_h = _fit_within(target_w, target_h, MAX_OUTPUT_PX)
        if (target_w, target_h) != img.size:
            img = img.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)
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
