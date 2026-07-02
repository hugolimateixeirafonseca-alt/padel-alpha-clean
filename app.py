"""
padel-alpha-clean (v20 - /personalize + /mockup_badge for personalized preview mockups)
-------------------------------------------------------
NEW in v20: POST /mockup_badge proxies the mockup renderer and overlays ADD YOUR NAME badge
NEW in v19: POST /personalize
  Draws a customer name into the reserved empty name zone of a __pname design
  (zone defined relative to the artwork alpha bounding box, lower area).
  Auto-fits font size, auto-detects dominant artwork ink colour when no colour
  is provided, uploads result to ImgBB and returns the final print URL.

Recommended /personalize body:
{
  "url": "https://...print.png",       # per-product print PNG (from PADEL_LOG col F)
  "imgbb_key": "...",
  "name": "MIGUEL",
  "font": "bebas",                      # preset (bebas|archivo|oswald) or direct TTF URL
  "color": "",                          # optional #RRGGBB; empty = auto dominant ink
  "zone": {                             # optional, relative to artwork bbox
    "y_top_pct": 82, "y_bottom_pct": 95, "width_pct": 68
  },
  "uppercase": true,
  "max_chars": 18
}
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from flask import Flask, request, jsonify, send_file

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


def _img_to_png_bytes(img):
    buf = io.BytesIO()
    # compress_level 4 is a good speed/size balance for Make/Render timeouts.
    img.save(buf, format="PNG", optimize=False, compress_level=4)
    return buf.getvalue()


def _upload_imgbb_bytes(payload, imgbb_key, filename="design.png", timeout=90):
    buf = io.BytesIO(payload)
    try:
        up = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": imgbb_key},
            files={"image": (filename, buf, "image/png")},
            timeout=timeout,
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


# ---------------------------------------------------------------------------
# /personalize — v19
# ---------------------------------------------------------------------------
import os
import re
import unicodedata
from PIL import ImageDraw, ImageFont

FONT_PRESETS = {
    "bebas": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "archivo": "https://raw.githubusercontent.com/google/fonts/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",
    "oswald": "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
}
_FONT_CACHE_DIR = "/tmp/pname_fonts"
_FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_NAME_ALLOWED = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9 '\-\.&]")


def _sanitize_name(raw, uppercase=True, max_chars=18):
    if not raw:
        return ""
    s = unicodedata.normalize("NFC", str(raw))
    s = _NAME_ALLOWED.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if uppercase:
        s = s.upper()
    return s[: max(1, int(max_chars))].strip()


def _get_font_path(font_param):
    """Resolve preset name or direct URL to a local TTF path (cached in /tmp)."""
    font_param = (font_param or "bebas").strip()
    url = FONT_PRESETS.get(font_param.lower(), font_param)
    if not url.lower().startswith("http"):
        url = FONT_PRESETS["bebas"]
    os.makedirs(_FONT_CACHE_DIR, exist_ok=True)
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", url.split("/")[-1]) or "font.ttf"
    path = os.path.join(_FONT_CACHE_DIR, fname)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        try:
            fr = requests.get(url, timeout=60)
            fr.raise_for_status()
            with open(path, "wb") as f:
                f.write(fr.content)
        except Exception:
            for fb in _FALLBACK_FONTS:
                if os.path.exists(fb):
                    return fb
            raise
    return path


def _hex_to_rgb(hex_str):
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return None


def _dominant_ink(img, bbox, alpha_min=200):
    """Most frequent quantised opaque colour inside the artwork bbox."""
    left, top, right, bottom = bbox
    art = img.crop((left, top, right, bottom))
    if max(art.size) > 160:
        art.thumbnail((160, 160), Image.LANCZOS)
    arr = np.asarray(art)
    mask = arr[:, :, 3] >= alpha_min
    if not mask.any():
        return (26, 26, 26)
    rgb = arr[:, :, :3][mask]
    q = (rgb // 24) * 24  # quantise to merge near-identical shades
    colors, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    best = colors[counts.argmax()]
    return tuple(int(min(v + 12, 255)) for v in best)  # centre of the bucket


def _load_sized_font(font_path, size):
    f = ImageFont.truetype(font_path, size=max(8, int(size)))
    # Variable fonts (e.g. Oswald[wght]) — pick a solid weight if supported.
    try:
        f.set_variation_by_axes([600])
    except Exception:
        pass
    return f


def _fit_text_font(draw, text, font_path, max_w, max_h):
    """Binary-search the largest font size whose rendered text fits max_w x max_h."""
    lo, hi, best = 8, int(max_h * 2.2) + 8, None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_sized_font(font_path, mid)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        w, h = r - l, b - t
        if w <= max_w and h <= max_h:
            best = (font, l, t, w, h)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        font = _load_sized_font(font_path, 8)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        best = (font, l, t, r - l, b - t)
    return best


@app.route("/personalize", methods=["POST"])
def personalize():
    data = request.get_json(force=True, silent=True) or {}

    url = data.get("url")
    imgbb_key = data.get("imgbb_key")
    raw_name = data.get("name", "")
    uppercase = bool(data.get("uppercase", True))
    max_chars = int(data.get("max_chars", 18))

    name = _sanitize_name(raw_name, uppercase=uppercase, max_chars=max_chars)
    if not url or not imgbb_key:
        return jsonify({"error": "faltam 'url' e/ou 'imgbb_key'"}), 400
    if not name:
        return jsonify({"error": "'name' vazio ou invalido apos sanitizacao"}), 400

    zone = data.get("zone") or {}
    try:
        y_top_pct = float(zone.get("y_top_pct", 82))
        y_bottom_pct = float(zone.get("y_bottom_pct", 95))
        width_pct = float(zone.get("width_pct", 68))
    except Exception:
        y_top_pct, y_bottom_pct, width_pct = 82.0, 95.0, 68.0
    y_top_pct = min(max(y_top_pct, 50), 97)
    y_bottom_pct = min(max(y_bottom_pct, y_top_pct + 3), 99)
    width_pct = min(max(width_pct, 20), 95)

    try:
        bbox_alpha_threshold = int(data.get("bbox_alpha_threshold", 16))
    except Exception:
        bbox_alpha_threshold = 16
    bbox_alpha_threshold = min(max(bbox_alpha_threshold, 1), 64)

    # --- download base print PNG ---
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

    bbox = _alpha_bbox(img, alpha_threshold=bbox_alpha_threshold)
    if bbox is None:
        return jsonify({"error": "imagem sem artwork visivel (alpha vazio)"}), 422
    left, top, right, bottom = bbox
    art_w = right - left
    art_h = bottom - top

    # --- resolve colour ---
    color = _hex_to_rgb(data.get("color"))
    color_source = "param"
    if color is None:
        color = _dominant_ink(img, bbox)
        color_source = "auto_dominant"

    # --- resolve font ---
    try:
        font_path = _get_font_path(data.get("font"))
    except Exception as e:
        return jsonify({"error": f"font indisponivel: {e}"}), 502

    # --- name zone in absolute pixels (relative to artwork bbox) ---
    zone_x0 = left + round(art_w * (1 - width_pct / 100.0) / 2.0)
    zone_x1 = right - round(art_w * (1 - width_pct / 100.0) / 2.0)
    zone_y0 = top + round(art_h * y_top_pct / 100.0)
    zone_y1 = top + round(art_h * y_bottom_pct / 100.0)
    zone_w = max(1, zone_x1 - zone_x0)
    zone_h = max(1, zone_y1 - zone_y0)

    draw = ImageDraw.Draw(img)
    font, off_l, off_t, text_w, text_h = _fit_text_font(draw, name, font_path, zone_w, zone_h)

    tx = zone_x0 + (zone_w - text_w) // 2 - off_l
    ty = zone_y0 + (zone_h - text_h) // 2 - off_t
    draw.text((tx, ty), name, font=font, fill=color + (255,))

    out_url = _upload_imgbb(img, imgbb_key, filename=f"pname_{name.replace(' ', '_')}.png")

    result = {
        "url": out_url,
        "name_rendered": name,
        "color_used": "#%02x%02x%02x" % color,
        "color_source": color_source,
        "font_px": font.size,
        "zone_px": {"x0": zone_x0, "y0": zone_y0, "x1": zone_x1, "y1": zone_y1},
        "artwork_bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
    }
    img = None
    gc.collect()
    return jsonify(result)



# ---------------------------------------------------------------------------
# /mockup_badge — v20
# ---------------------------------------------------------------------------
def _draw_badge(img, text="ADD YOUR NAME", position="top_right"):
    """Draw a clean Etsy-style badge on a final mockup image."""
    text = (text or "").strip()
    if not text:
        return img

    img = img.convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Responsive sizing.
    margin = max(18, int(min(w, h) * 0.035))
    pad_x = max(18, int(w * 0.018))
    pad_y = max(10, int(h * 0.012))
    radius = max(12, int(min(w, h) * 0.018))
    font_size = max(28, int(min(w, h) * 0.035))

    try:
        font_path = _get_font_path("archivo")
        font = _load_sized_font(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    text = text.upper()
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    bw = tw + pad_x * 2
    bh = th + pad_y * 2

    pos = (position or "top_right").strip().lower()
    if pos == "top_left":
        x0, y0 = margin, margin
    elif pos == "bottom_right":
        x0, y0 = w - bw - margin, h - bh - margin
    elif pos == "bottom_left":
        x0, y0 = margin, h - bh - margin
    else:
        x0, y0 = w - bw - margin, margin
    x1, y1 = x0 + bw, y0 + bh

    # Subtle shadow + dark badge with white text.
    shadow = max(2, int(min(w, h) * 0.004))
    draw.rounded_rectangle((x0 + shadow, y0 + shadow, x1 + shadow, y1 + shadow), radius=radius, fill=(0, 0, 0, 65))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=(20, 20, 20, 225))
    tx = x0 + pad_x - l
    ty = y0 + pad_y - t
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))
    return img


@app.route("/mockup_badge", methods=["POST"])
def mockup_badge():
    """
    Proxy to mockup-vg4o /render, then optionally overlay a badge.

    Accepts the same JSON body as mockup-vg4o /render plus optional:
      badge_text: "ADD YOUR NAME" or empty string
      badge_position: top_right|top_left|bottom_right|bottom_left
      render_url: override renderer URL
      output: png|jpg
    """
    data = request.get_json(force=True, silent=True) or {}
    badge_text = (data.pop("badge_text", "") or "").strip()
    badge_position = data.pop("badge_position", "top_right")
    output = (data.pop("output", "jpg") or "jpg").strip().lower()
    render_url = data.pop("render_url", "https://mockup-vg4o.onrender.com/render")

    # Ensure the proxied renderer returns an image.
    data["return_image"] = True

    try:
        rr = requests.post(render_url, json=data, timeout=int(data.get("timeout", 180)) if isinstance(data.get("timeout"), int) else 180)
        rr.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"mockup renderer falhou: {e}"}), 502

    try:
        img = Image.open(io.BytesIO(rr.content)).convert("RGBA")
    except Exception as e:
        return jsonify({"error": f"abrir mockup render falhou: {e}"}), 422
    finally:
        rr = None
        gc.collect()

    if badge_text:
        img = _draw_badge(img, badge_text, badge_position)

    buf = io.BytesIO()
    if output in ("jpg", "jpeg"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        bg.save(buf, format="JPEG", quality=94, optimize=True)
        mimetype = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=False, compress_level=4)
        mimetype = "image/png"
    buf.seek(0)
    img = None
    gc.collect()
    return send_file(buf, mimetype=mimetype)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean", "ver": 20})


@app.route("/clean", methods=["POST"])
def clean():
    data = request.get_json(force=True, silent=True) or {}

    url = data.get("url")
    imgbb_key = data.get("imgbb_key")
    garment_set = data.get("garment_set", "")
    scale = float(data.get("scale", 1))
    max_output_px = int(data.get("max_output_px", MAX_OUTPUT_PX))
    max_output_px = min(max(max_output_px, 2000), MAX_OUTPUT_PX)
    parallel_uploads = int(data.get("parallel_uploads", 3))
    parallel_uploads = min(max(parallel_uploads, 1), 4)
    upload_timeout = int(data.get("upload_timeout", 90))
    upload_timeout = min(max(upload_timeout, 30), 120)
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
            tw, th = _fit_within(tw, th, max_output_px)
            if (tw, th) != img.size:
                img = img.resize((tw, th), Image.LANCZOS)
        out_url = _upload_imgbb(img, imgbb_key, filename=f"{product_key}.png")
        return jsonify({"url": out_url, "showcase_bg": showcase_bg})

    # Multi-output mode: one URL per product/template.
    if not outputs:
        outputs = DEFAULT_PROFILES

    result = {"showcase_bg": showcase_bg}
    first_url = None

    # Prepare PNG payloads locally first, then upload to ImgBB concurrently.
    # This avoids Make HTTP timeout when 6-8 product-specific PNGs are uploaded sequentially.
    payloads = []
    for key, raw_profile in outputs.items():
        profile = DEFAULT_PROFILES.get(key, {})
        merged = dict(profile)
        if isinstance(raw_profile, dict):
            merged.update(raw_profile)

        img = _fit_artwork_to_product_canvas(base, merged, alpha_threshold=bbox_alpha_threshold)

        if scale and scale != 1.0:
            tw, th = round(img.width * scale), round(img.height * scale)
            tw, th = _fit_within(tw, th, max_output_px)
            if (tw, th) != img.size:
                img = img.resize((tw, th), Image.LANCZOS)

        try:
            payloads.append((key, _img_to_png_bytes(img)))
        except Exception as e:
            return jsonify({"error": f"gerar PNG falhou para {key}: {e}"}), 500
        finally:
            img = None
            gc.collect()

    try:
        with ThreadPoolExecutor(max_workers=parallel_uploads) as ex:
            futs = {
                ex.submit(_upload_imgbb_bytes, payload, imgbb_key, f"{key}.png", upload_timeout): key
                for key, payload in payloads
            }
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    out_url = fut.result()
                except Exception as e:
                    return jsonify({"error": f"upload ImgBB falhou para {key}: {e}"}), 502
                result[f"url_{key}"] = out_url
    finally:
        payloads = None
        gc.collect()

    # Keep a stable first URL preference.
    for key in outputs.keys():
        out_url = result.get(f"url_{key}")
        if out_url:
            first_url = out_url
            break

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
