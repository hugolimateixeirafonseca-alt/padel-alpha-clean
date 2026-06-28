"""
padel-alpha-clean  (v6 - memory safe + showcase_bg via garment_set, com "both")
-------------------------------------
POST /clean  (JSON body)
Mesma logica do v3, mas blindado contra OOM (out of memory) na instancia free:
  - MAX_OUTPUT_PX: limita a resolucao final independentemente do 'scale' pedido,
    para nunca criar buffers gigantes (era isto que rebentava a memoria).
  - Upload ao ImgBB via multipart (ficheiro) em vez de base64 string -> menos memoria.
  - Liberta buffers e forca garbage collection nos pontos criticos.
  - showcase_bg passa a seguir o garment_set recebido (coerente com a roupa),
    em vez de adivinhar pela luminosidade do design (que enganava com cores vivas).
Body:
{
  "url":       "https://...png",   (obrig.) PNG (transparente) de origem
  "imgbb_key": "....",             (obrig.) chave ImgBB
  "scale":     4,                  (opc.) fator de upscale Lanczos (1 = nenhum)
  "threshold": 0,                  (opc.) 0 = NAO binariza (mantem o alpha nativo).
                                          >0 = binariza nesse corte
  "erode":     0,                  (opc.) px a comer ao bordo (so se >0)
  "keyline":   0,                  (opc.) px de contorno branco (so se >0)
  "garment_set": "light"           (opc.) "light" ou "dark". Decide a cor de
                                          fundo da montra de forma coerente com
                                          a cor da roupa:
                                            light (roupa clara, design escuro) -> fundo BRANCO
                                            dark  (roupa escura, design claro) -> fundo ESCURO
                                          Se ausente, faz fallback a branco.
}
Resposta: {"url": "https://i.ibb.co/...", "showcase_bg": "#ffffff"}  ou  {"error": "..."}
"""
import io, gc
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
app = Flask(__name__)
Image.MAX_IMAGE_PIXELS = None

# --- BLINDAGEM DE MEMORIA --------------------------------------------------
# Lado maximo (px) da imagem final. 4000px chega e sobra para impressao Gelato
# (estampa ~30cm a >300 DPI). Acima disto, NAO ha ganho de qualidade visivel,
# so risco de OOM. Ajusta se precisares, mas mantem abaixo de ~4500.
MAX_OUTPUT_PX = 4000
# Lado maximo da imagem de ENTRADA antes de processar. Se o PNG de origem ja
# vier enorme, reduz-se primeiro para nao carregar buffers gigantes a toa.
MAX_INPUT_PX = 3000
# ---------------------------------------------------------------------------


def pick_showcase_bg(garment_set):
    """Decide a cor de fundo da montra a partir do garment_set (decisao unica
    e coerente com a cor da roupa).
      garment_set = "light" -> roupa clara         -> design escuro -> fundo BRANCO
      garment_set = "dark"  -> roupa escura         -> design claro  -> fundo ESCURO
      garment_set = "both"  -> ambas (design vivo)  -> montra sobre   -> fundo ESCURO
    Qualquer valor desconhecido ou ausente faz fallback a branco (mais seguro)."""
    gs = (garment_set or "").strip().lower()
    if gs in ("dark", "both"):
        return "#1a1a1a"
    return "#ffffff"


def _fit_within(w, h, max_side):
    """Devolve (nw, nh) reduzido para caber em max_side, mantendo o racio.
    Se ja couber, devolve o tamanho original."""
    longest = max(w, h)
    if longest <= max_side:
        return w, h
    f = max_side / float(longest)
    return max(1, round(w * f)), max(1, round(h * f))


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean", "ver": 6})


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
        r = None  # liberta os bytes do download

    # cor de fundo da montra: segue o garment_set (coerente com a cor da roupa)
    showcase_bg = pick_showcase_bg(garment_set)

    # 0) BLINDAGEM: se a imagem de entrada vier enorme, reduz primeiro
    iw, ih = _fit_within(img.width, img.height, MAX_INPUT_PX)
    if (iw, ih) != (img.size):
        img = img.resize((iw, ih), Image.LANCZOS)
        gc.collect()

    # 1) upscale alpha-aware (Lanczos), MAS com tecto em MAX_OUTPUT_PX
    if scale and scale != 1.0:
        target_w, target_h = round(img.width * scale), round(img.height * scale)
        # aplica o tecto: nunca passa de MAX_OUTPUT_PX no lado maior
        target_w, target_h = _fit_within(target_w, target_h, MAX_OUTPUT_PX)
        if (target_w, target_h) != (img.width, img.height):
            img = img.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)
            gc.collect()

    # 2) endurecimento opcional (so se pedido)
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

    # 3) grava PNG para um buffer e envia ao ImgBB como FICHEIRO (multipart),
    #    evitando a string base64 gigante (que era ~33% maior que o PNG).
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
