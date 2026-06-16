"""
padel-alpha-clean
-----------------
POST /clean  (JSON body)

Limpa o recorte de um PNG transparente para ficar perfeito em peca escura:
  1) binariza o canal alpha (mata o fringe semitransparente / halo)
  2) erode opcional (come 1-2px de bordo sujo)
  3) keyline branco opcional (contorno solido de "sticker" -> some em peca clara,
     destaca em peca escura, e torna a qualidade do bordo irrelevante)
  4) re-upload ao ImgBB e devolve {"url": "..."}

Body:
{
  "url":       "https://...png",     (obrigatorio) PNG transparente de origem
  "imgbb_key": "....",               (obrigatorio) chave ImgBB
  "threshold": 128,                  (opcional) corte do alpha 0-255
  "erode":     1,                    (opcional) px a comer ao bordo (0 = nenhum)
  "keyline":   0                     (opcional) px de contorno branco (0 = nenhum)
}

Resposta: {"url": "https://i.ibb.co/..."}  ou  {"error": "..."}
"""

import io
import base64

import requests
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

Image.MAX_IMAGE_PIXELS = None  # nao abortar em imagens grandes (4096x6144)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean"})


@app.route("/clean", methods=["POST"])
def clean():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    imgbb_key = data.get("imgbb_key")
    threshold = int(data.get("threshold", 128))
    erode_px = int(data.get("erode", 0))
    keyline_px = int(data.get("keyline", 0))

    if not url or not imgbb_key:
        return jsonify({"error": "faltam 'url' e/ou 'imgbb_key'"}), 400

    # 1) download
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"download falhou: {e}"}), 502

    try:
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception as e:
        return jsonify({"error": f"abrir imagem falhou: {e}"}), 422

    arr = np.array(img)            # H x W x 4 (uint8)
    del img
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    del arr

    # 2) binariza alpha -> 0 ou 255
    opaque = (alpha >= threshold).astype(np.uint8)   # mascara 0/1
    del alpha

    kernel = np.ones((3, 3), np.uint8)               # iterations = px == raio

    # 3) erode (encolhe a regiao opaca, come o bordo sujo)
    if erode_px > 0:
        opaque = cv2.erode(opaque, kernel, iterations=erode_px)

    if keyline_px > 0:
        outer = cv2.dilate(opaque, kernel, iterations=keyline_px)
        ring = (outer & (1 - opaque)).astype(bool)   # so o anel novo
        out_rgb = rgb.copy()
        out_rgb[ring] = (255, 255, 255)               # pinta o anel a branco
        out_alpha = (outer * 255).astype(np.uint8)
        del rgb, outer, ring
    else:
        out_rgb = rgb
        out_alpha = (opaque * 255).astype(np.uint8)

    out = np.dstack([out_rgb, out_alpha])
    del out_rgb, out_alpha, opaque

    buf = io.BytesIO()
    Image.fromarray(out, mode="RGBA").save(buf, format="PNG", optimize=False)
    del out
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    del buf

    # 4) upload ImgBB
    try:
        up = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": imgbb_key, "image": png_b64},
            timeout=120,
        )
        up.raise_for_status()
        out_url = up.json()["data"]["url"]
    except Exception as e:
        return jsonify({"error": f"upload ImgBB falhou: {e}"}), 502

    return jsonify({"url": out_url})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
