"""
padel-alpha-clean (v17 - product-specific safe canvases + larger fit + alpha bbox threshold)
-------------------------------------------------------
This version solves Gelato size/cropping problems by returning a different
transparent PNG per product/template.

Key idea:
- Do NOT export a tightly cropped variable-size PNG.
- Detect the actual artwork bounding box.
- Place it on a product-specific transparent canvas with a target aspect ratio.
- Fit the art inside product-specific safe percentages.
- Return multiple ImgBB URLs in one /clean call:
  url_tshirt_classic, url_tshirt_performance, url_hoodie, url_mug, etc.

Recommended /clean body:
{
  "url": "https://...png",
  "imgbb_key": "...",
  "scale": 2,
  "garment_set": "light",
  "multi_output": true,
  "fit_to_canvas": true,
  "bbox_alpha_threshold": 16,
  "outputs": {
    "tshirt_classic": {"target_aspect": 0.667, "max_art_width_pct": 74, "max_art_height_pct": 86},
    "tshirt_performance": {"target_aspect": 0.724, "max_art_width_pct": 72, "max_art_height_pct": 84}
  }
}
"""
import io
import gc
import time
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)
Image.MAX_IMAGE_PIXELS = None

MAX_OUTPUT_PX = 4000
MAX_INPUT_PX = 3000

DEFAULT_PROFILES = {"tshirt_classic": {"target_aspect": 0.667, "max_art_width_pct": 88, "max_art_height_pct": 93, "max_art_upscale": 8.0}, "tshirt_performance": {"target_aspect": 0.724, "max_art_width_pct": 88, "max_art_height_pct": 92, "max_art_upscale": 8.0}, "hoodie": {"target_aspect": 0.759, "max_art_width_pct": 82, "max_art_height_pct": 88, "max_art_upscale": 8.0}, "sweatshirt": {"target_aspect": 0.759, "max_art_width_pct": 84, "max_art_height_pct": 90, "max_art_upscale": 8.0}, "mug": {"target_aspect": 0.724, "max_art_width_pct": 94, "max_art_height_pct": 95, "max_art_upscale": 8.0}, "racerback_tank": {"target_aspect": 0.749, "max_art_width_pct": 78, "max_art_height_pct": 84, "max_art_upscale": 8.0}, "performance_woman_tank": {"target_aspect": 0.749, "max_art_width_pct": 80, "max_art_height_pct": 86, "max_art_upscale": 8.0}, "unisex_sports_jersey": {"target_aspect": 0.676, "max_art_width_pct": 88, "max_art_height_pct": 93, "max_art_upscale": 8.0}}


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


def _normalise_profile(profile):
    p = dict(profile or {})
    target_aspect = float(p.get("target_aspect", 0.70))
    target_aspect = min(max(target_aspect, 0.35), 2.5)

    max_art_width_pct = float(p.get("max_art_width_pct", 70))
    max_art_height_pct = float(p.get("max_art_height_pct", 82))
    max_art_width_pct = min(max(max_art_width_pct, 20), 95)
    max_art_height_pct = min(max(max_art_height_pct, 20), 95)

    max_art_upscale = float(p.get("max_art_upscale", 4.0))
    max_art_upscale = min(max(max_art_upscale, 1.0), 8.0)

    return {
        "target_aspect": target_aspect,
        "max_art_width_pct": max_art_width_pct,
        "max_art_height_pct": max_art_height_pct,
        "max_art_upscale": max_art_upscale,
    }


def _make_canvas_size(src_w, src_h, target_aspect):
    """Return a transparent canvas size with requested aspect, no smaller than source."""
    if target_aspect <= 0:
        target_aspect = src_w / float(src_h)

    # For portrait canvases, keep source height and widen if needed.
    if target_aspect <= 1:
        canvas_h = src_h
        canvas_w = round(canvas_h * target_aspect)
        if canvas_w < src_w:
            canvas_w = src_w
            canvas_h = round(canvas_w / target_aspect)
    else:
        canvas_w = src_w
        canvas_h = round(canvas_w / target_aspect)
        if canvas_h < src_h:
            canvas_h = src_h
            canvas_w = round(canvas_h * target_aspect)

    return max(1, canvas_w), max(1, canvas_h)


def _fit_artwork_to_product_canvas(img, profile, alpha_threshold=1):
    """
    Crop transparent excess around the actual artwork, then place it centered on
    a product-specific transparent canvas. The output keeps a stable canvas ratio,
    while the visible artwork is fitted to product-specific safe percentages.
    """
    profile = _normalise_profile(profile)
    bbox = _alpha_bbox(img, alpha_threshold=alpha_threshold)
    if bbox is None:
        return img.copy()

    left, top, right, bottom = bbox
    art = img.crop((left, top, right, bottom))
    art_w, art_h = art.size

    canvas_w, canvas_h = _make_canvas_size(img.width, img.height, profile["target_aspect"])

    max_w = canvas_w * profile["max_art_width_pct"] / 100.0
    max_h = canvas_h * profile["max_art_height_pct"] / 100.0

    factor = min(max_w / float(art_w), max_h / float(art_h), profile["max_art_upscale"])
    # Allow shrinking and enlargement; never zero.
    factor = max(factor, 0.01)

    new_w = max(1, round(art_w * factor))
    new_h = max(1, round(art_h * factor))
    if (new_w, new_h) != art.size:
        art = art.resize((new_w, new_h), Image.LANCZOS)

    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    out.paste(art, (x, y), art)
    return out


def _upload_imgbb(img, imgbb_key, filename="design.png"):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    try:
        up = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": imgbb_key},
            files={"image": (filename, buf, "image/png")},
            timeout=120,
        )
        up.raise_for_status()
        return up.json()["data"]["url"]
    finally:
        buf = None
        gc.collect()


def _apply_alpha_ops(img, threshold=0, erode_px=0, keyline_px=0):
    if threshold <= 0 and erode_px <= 0 and keyline_px <= 0:
        return img
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

    out = Image.fromarray(np.dstack([out_rgb, out_alpha]), mode="RGBA")
    del rgb, alpha, opaque
    gc.collect()
    return out


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean", "ver": 17})


@app.route("/clean", methods=["POST"])
def clean():
    data = request.get_json(force=True, silent=True) or {}

    url = data.get("url")
    imgbb_key = data.get("imgbb_key")
    garment_set = data.get("garment_set", "")
    scale = float(data.get("scale", 1))
    threshold = int(data.get("threshold", 0))
    erode_px = int(data.get("erode", 0))
    keyline_px = int(data.get("keyline", 0))
    try:
        bbox_alpha_threshold = int(data.get("bbox_alpha_threshold", 16))
    except Exception:
        bbox_alpha_threshold = 16
    bbox_alpha_threshold = min(max(bbox_alpha_threshold, 1), 64)

    multi_output = bool(data.get("multi_output", False))
    outputs = data.get("outputs") or {}

    if not url or not imgbb_key:
        return jsonify({"error": "faltam 'url' e/ou 'imgbb_key'"}), 400

    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"download falhou: {e}"}), 502

    try:
        base = Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception as e:
        return jsonify({"error": f"abrir imagem falhou: {e}"}), 422
    finally:
        r = None
        gc.collect()

    # Guard input size.
    iw, ih = _fit_within(base.width, base.height, MAX_INPUT_PX)
    if (iw, ih) != base.size:
        base = base.resize((iw, ih), Image.LANCZOS)
        gc.collect()

    base = _apply_alpha_ops(base, threshold=threshold, erode_px=erode_px, keyline_px=keyline_px)

    showcase_bg = pick_showcase_bg(garment_set)

    # Backwards-compatible single-output mode.
    if not multi_output:
        product_key = (data.get("product_key") or "tshirt_classic").strip()
        profile = data.get("profile") or DEFAULT_PROFILES.get(product_key) or DEFAULT_PROFILES["tshirt_classic"]
        img = _fit_artwork_to_product_canvas(base, profile, alpha_threshold=bbox_alpha_threshold)
        if scale and scale != 1.0:
            tw, th = round(img.width * scale), round(img.height * scale)
            tw, th = _fit_within(tw, th, MAX_OUTPUT_PX)
            if (tw, th) != img.size:
                img = img.resize((tw, th), Image.LANCZOS)
        out_url = _upload_imgbb(img, imgbb_key, filename=f"{product_key}.png")
        return jsonify({"url": out_url, "showcase_bg": showcase_bg})

    # Multi-output mode: one URL per product/template.
    if not outputs:
        outputs = DEFAULT_PROFILES

    result = {"showcase_bg": showcase_bg}
    first_url = None

    for key, raw_profile in outputs.items():
        profile = DEFAULT_PROFILES.get(key, {})
        merged = dict(profile)
        if isinstance(raw_profile, dict):
            merged.update(raw_profile)

        img = _fit_artwork_to_product_canvas(base, merged, alpha_threshold=bbox_alpha_threshold)

        if scale and scale != 1.0:
            tw, th = round(img.width * scale), round(img.height * scale)
            tw, th = _fit_within(tw, th, MAX_OUTPUT_PX)
            if (tw, th) != img.size:
                img = img.resize((tw, th), Image.LANCZOS)

        try:
            out_url = _upload_imgbb(img, imgbb_key, filename=f"{key}.png")
        except Exception as e:
            return jsonify({"error": f"upload ImgBB falhou para {key}: {e}"}), 502

        result[f"url_{key}"] = out_url
        if first_url is None:
            first_url = out_url

        img = None
        gc.collect()
        time.sleep(0.15)

    # Backwards-compatible aliases.
    if "url_tshirt_classic" in result:
        result["url"] = result["url_tshirt_classic"]
        result["url_tshirt"] = result["url_tshirt_classic"]
    elif first_url:
        result["url"] = first_url

    if "url_performance_woman_tank" in result:
        result["url_tank"] = result["url_performance_woman_tank"]
    elif "url_racerback_tank" in result:
        result["url_tank"] = result["url_racerback_tank"]

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
