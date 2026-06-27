"""
padel-alpha-clean  (v3)
-----------------------
POST /clean  (JSON body)
Para o pipeline de transparencia nativa (gpt-image-1.5 background=transparent):
  1) faz upscale alpha-aware (Lanczos, preserva a transparencia) -> resolucao de impressao
  2) (opcional) endurece o bordo: binariza alpha / erode / keyline branco
  3) re-upload ao ImgBB
  4) NOVO: analisa o brilho do design e escolhe a cor de fundo ideal para a montra
Body:
{
  "url":       "https://...png",   (obrig.) PNG (transparente) de origem
  "imgbb_key": "....",             (obrig.) chave ImgBB
  "scale":     4,                  (opc.) fator de upscale Lanczos (1 = nenhum)
  "threshold": 0,                  (opc.) 0 = NAO binariza (mantem o alpha nativo).
                                          >0 = binariza nesse corte
  "erode":     0,                  (opc.) px a comer ao bordo (so se >0)
  "keyline":   0                   (opc.) px de contorno branco (so se >0)
}
Resposta: {"url": "https://i.ibb.co/...", "showcase_bg": "#ffffff"}  ou  {"error": "..."}
"""
import io, base64
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
app = Flask(__name__)
Image.MAX_IMAGE_PIXELS = None


def pick_showcase_bg(img):
    """Decide a cor de fundo da montra a partir do brilho do design.
    Regra: a maioria dos designs tem partes escuras -> fundo branco (look limpo).
    So quando o design e quase todo claro/brilhante (desapareceria no branco)
    e que devolve fundo escuro."""
    try:
        im = img.convert("RGBA")
        im.thumbnail((200, 200))                     # analisa uma copia pequena (rapido)
        a = np.asarray(im, dtype=np.float32)
        rgb, alpha = a[..., :3], a[..., 3]
        mask = alpha > 40                            # ignora pixeis transparentes
        if mask.sum() < 50:
            return "#ffffff"
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[mask]
        dark_frac = float((lum < 90).mean())         # quanto do design e escuro
        light_frac = float((lum > 180).mean())       # quanto e muito claro
        # quase sem partes escuras E muito claro -> branco fa-lo sumir -> fundo escuro
        if dark_frac < 0.06 and light_frac > 0.5:
            return "#1a1a1a"
        return "#ffffff"                             # caso normal: branco limpo
    except Exception:
        return "#ffffff"                             # fallback seguro


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean", "ver": 3})


@app.route("/clean", methods=["POST"])
def clean():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    imgbb_key = data.get("imgbb_key")
    scale = float(data.get("scale", 1))
    threshold = int(data.get("threshold", 0))
    erode_px = int(data.get("erode", 0))
    keyline_px = int(data.get("keyline", 0))
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

    # NOVO: decide a cor de fundo da montra ANTES do upscale (analise na imagem pequena)
    showcase_bg = pick_showcase_bg(img)

    # 1) upscale alpha-aware (Lanczos preserva o canal alpha) - ideal para arte flat/vetorial
    if scale and scale != 1.0:
        nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
    # 2) endurecimento opcional (so se pedido) - usa numpy/cv2 apenas quando necessario
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
            # se binarizou, alpha 0/255; senao mantem o alpha nativo onde e opaco
            out_alpha = (opaque * 255).astype(np.uint8) if threshold > 0 else alpha
        img = Image.fromarray(np.dstack([out_rgb, out_alpha]), mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    del buf, img
    try:
        up = requests.post("https://api.imgbb.com/1/upload",
                           data={"key": imgbb_key, "image": png_b64}, timeout=120)
        up.raise_for_status()
        return jsonify({"url": up.json()["data"]["url"], "showcase_bg": showcase_bg})
    except Exception as e:
        return jsonify({"error": f"upload ImgBB falhou: {e}"}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
